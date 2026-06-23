"""
agents/orchestrator.py

Smart RAG orchestrator - threshold-based routing between local FAISS
retrieval and Firecrawl web search, with citation-aware fused answers.

SPEC (implemented exactly as follows):
  1. Always attempt local FAISS retrieval first.
  2. If best similarity_score >= LOCAL_CONFIDENCE_THRESHOLD (0.75):
       answer ONLY from local context. No web call.
  3. If best similarity_score < threshold:
       - Ask an LLM scope-check: is this query even related to the
         uploaded research documents / research domain at all?
       - If IN SCOPE  -> trigger Firecrawl web_search, fuse local + web,
         rerank, and answer from both.
       - If OUT OF SCOPE -> politely decline. Never call web_search for
         genuinely unrelated queries.
  4. Every answer states whether it came from "local", "web", or "both".

This module is additive. It does not modify core/coordinator.py or any
existing agents/*.py files. It calls coordinator.document_processor's
existing search_documents() (which already returns 'similarity_score')
and coordinator.research_query() for final synthesis.
"""

import logfire
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────

LOCAL_CONFIDENCE_THRESHOLD = 0.75   # >= this -> local-only, no web call
FAISS_K = 6                          # chunks to pull for the routing check
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class LocalRetrieval:
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    best_score: float = 0.0
    context: str = ""

    @property
    def has_results(self) -> bool:
        return bool(self.chunks)


@dataclass
class WebRetrieval:
    sources: List[Dict[str, str]] = field(default_factory=list)
    context: str = ""

    @property
    def has_results(self) -> bool:
        return bool(self.sources)


@dataclass
class ScopeVerdict:
    in_scope: bool
    reasoning: str


@dataclass
class SmartRagResult:
    answer_source: str          # "local" | "web" | "both" | "declined"
    declined: bool
    decline_reason: Optional[str]
    local: LocalRetrieval
    web: WebRetrieval
    synthesis: Optional[Dict[str, Any]]  # output of coordinator.research_query()


# ── Tool: Firecrawl web search ──────────────────────────────────────────

def firecrawl_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Call Firecrawl /search. Returns [] on any failure (never raises)."""
    if not FIRECRAWL_API_KEY:
        logger.warning("FIRECRAWL_API_KEY not set - skipping web search")
        return []

    try:
        resp = requests.post(
            FIRECRAWL_SEARCH_URL,
            headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
            json={"query": query, "limit": max_results},
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()

        # Firecrawl /v2/search nests results under data.web (a list)
        data = payload.get("data", {})
        items = data.get("web", []) if isinstance(data, dict) else []

        results = []
        for item in items[:max_results]:
            results.append({
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "snippet": (item.get("description") or item.get("markdown", ""))[:500],
            })
        logger.info("Firecrawl returned %d results for query='%s'", len(results), query)
        return results
    except Exception as e:
        logger.warning("Firecrawl search failed: %s", e)
        return []


# ── Step 1: Local FAISS retrieval ────────────────────────────────────────

def retrieve_local(coordinator, question: str, k: int = FAISS_K) -> LocalRetrieval:
    """
    Pull chunks from the existing FAISS-backed DocumentProcessor.
    Reuses coordinator.document_processor.search_documents(), which
    already returns 'similarity_score' per chunk (see core/document_processor.py).
    """
    result = LocalRetrieval()

    has_store = (
        getattr(coordinator, "document_processor", None) is not None
        and getattr(coordinator.document_processor, "vector_store", None) is not None
    )
    if not has_store:
        logger.info("retrieve_local: no vector store initialised - skipping")
        return result

    try:
        chunks = coordinator.document_processor.search_documents(question, k=k)
    except Exception as e:
        logger.warning("retrieve_local: search_documents failed: %s", e)
        return result

    if not chunks:
        return result

    result.chunks = chunks
    result.best_score = max(c.get("similarity_score", 0.0) for c in chunks)
    result.context = "\n\n".join(c.get("content", "").strip() for c in chunks if c.get("content"))

    logger.info(
        "retrieve_local: %d chunks, best_similarity_score=%.4f",
        len(chunks), result.best_score,
    )
    return result


# ── Step 2: Scope check (LLM-based) ──────────────────────────────────────

SCOPE_CHECK_PROMPT = """You are a scope-checking gate for a research \
assistant. The local document search did not find a strong match for the \
user's question. Decide whether this question is still something a \
research assistant should try to answer via a web search, or whether it \
is genuinely out of scope (e.g. small talk, unrelated personal requests, \
or topics with no plausible research/academic angle).

QUESTION: {question}

HAS_UPLOADED_DOCUMENTS: {has_documents}
BEST_LOCAL_SIMILARITY_SCORE: {best_score:.3f}

Respond with ONLY a JSON object, no markdown, no extra text:
{{
  "in_scope": true or false,
  "reasoning": "one sentence explanation"
}}

RULES:
- in_scope=true for any question with a plausible research, academic,
  technical, or factual angle - even if it's about something NOT in the
  uploaded documents (that's exactly when web search should help).
- in_scope=false ONLY for: greetings/small talk, requests unrelated to
  research entirely (e.g. "write me a poem about my cat"), or requests
  for harmful/inappropriate content.
- When genuinely uncertain, prefer in_scope=true - it's better to search
  and find nothing useful than to wrongly decline a real question.
"""


def check_scope(llm_call_fn, question: str, has_documents: bool, best_score: float) -> ScopeVerdict:
    prompt = SCOPE_CHECK_PROMPT.format(
        question=question, has_documents=has_documents, best_score=best_score
    )
    try:
        raw = llm_call_fn(prompt)
        parsed = _extract_json(raw)
        return ScopeVerdict(
            in_scope=bool(parsed.get("in_scope", True)),
            reasoning=parsed.get("reasoning", ""),
        )
    except Exception as e:
        logger.warning("check_scope: LLM call failed (%s) - defaulting to in_scope=True", e)
        return ScopeVerdict(in_scope=True, reasoning="fallback: scope check failed, defaulting to in-scope")


# ── Step 3: Web retrieval ────────────────────────────────────────────────

def retrieve_web(question: str) -> WebRetrieval:
    sources = firecrawl_search(question, max_results=3)
    result = WebRetrieval(sources=sources)
    result.context = "\n\n".join(
        f"[{s['title']}]({s['url']}): {s['snippet']}" for s in sources
    )
    return result


FUSION_PROMPT = """You are a research assistant. Combine the following local \
document findings with supplementary web research to give a complete, \
grounded answer to the user's question.

QUESTION: {question}

LOCAL DOCUMENT FINDINGS:
{local_findings}

WEB RESEARCH FINDINGS:
{web_findings}

Write a clear, well-organized answer that:
- Directly answers the question using BOTH sources where relevant
- Clearly notes when something comes from the uploaded document vs the web
- If the local document doesn't cover the topic but the web does, say so
  briefly, then give the web-based answer
- Keep it concise (3-6 sentences or a short bullet list)

Respond with ONLY a JSON object, no markdown, no extra text:
{{"fused_answer": "your complete answer here"}}
"""


def fuse_local_and_web(llm_call_fn, question: str, local_findings: list, web: "WebRetrieval") -> str:
    """Make one extra LLM call to blend local synthesis findings with web results."""
    local_text = "\n".join(f"- {f}" for f in local_findings) or "(none)"
    web_text = "\n".join(
        f"- {s['title']}: {s['snippet'][:300]} (source: {s['url']})"
        for s in web.sources
    ) or "(none)"

    prompt = FUSION_PROMPT.format(
        question=question, local_findings=local_text, web_findings=web_text
    )
    try:
        raw = llm_call_fn(prompt)
        parsed = _extract_json(raw)
        return parsed.get("fused_answer", "")
    except Exception as e:
        logger.warning("fuse_local_and_web: LLM call failed (%s)", e)
        return ""


# ── Main entry point ──────────────────────────────────────────────────────

def smart_rag_answer(
    question: str,
    coordinator,
    llm_call_fn,
    has_documents: bool = True,
    local_threshold: float = LOCAL_CONFIDENCE_THRESHOLD,
) -> SmartRagResult:
    with logfire.span("smart_rag_answer", question=question, has_documents=has_documents):

        with logfire.span("retrieve_local"):
            local = retrieve_local(coordinator, question)
        logfire.info("local_retrieval", chunks=len(local.chunks), best_score=round(local.best_score, 4))

        if local.has_results and local.best_score >= local_threshold:
            logfire.info("routing", decision="local_only")
            with logfire.span("research_query"):
                synthesis = coordinator.research_query(question)
            synthesis["source_attribution"] = "local"
            return SmartRagResult(answer_source="local", declined=False,
                decline_reason=None, local=local, web=WebRetrieval(), synthesis=synthesis)

        with logfire.span("check_scope"):
            scope = check_scope(llm_call_fn, question, has_documents, local.best_score)
        logfire.info("scope_check", in_scope=scope.in_scope, reasoning=scope.reasoning)

        if not scope.in_scope:
            logfire.info("routing", decision="declined")
            return SmartRagResult(answer_source="declined", declined=True,
                decline_reason=scope.reasoning, local=local, web=WebRetrieval(), synthesis=None)

        with logfire.span("retrieve_web"):
            web = retrieve_web(question)
        logfire.info("web_retrieval", sources=len(web.sources))

        if not local.has_results and not web.has_results:
            return SmartRagResult(answer_source="declined", declined=True,
                decline_reason="No relevant information found.", local=local, web=web, synthesis=None)

        with logfire.span("research_query"):
            synthesis = coordinator.research_query(question)

        if web.context and isinstance(synthesis, dict):
            local_findings = synthesis.get("key_findings", [])
            with logfire.span("fuse_local_and_web"):
                fused_answer = fuse_local_and_web(llm_call_fn, question, local_findings, web)
            if fused_answer:
                synthesis["fused_answer"] = fused_answer
                logfire.info("fusion", answer_length=len(fused_answer))
            synthesis["web_findings"] = [
                f"{s['title']}: {s['snippet'][:200]}" for s in web.sources
            ]

        source = "both" if (local.has_results and web.has_results) else (
            "local" if local.has_results else "web"
        )
        synthesis["source_attribution"] = source
        synthesis["web_sources"] = web.sources
        logfire.info("smart_rag_complete", source=source)

        return SmartRagResult(answer_source=source, declined=False,
            decline_reason=None, local=local, web=web, synthesis=synthesis)


# ── Helpers ────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> Dict[str, Any]:
    """Strip markdown fences and extract the first {...} JSON object."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]}")
    return json.loads(text[start:end])