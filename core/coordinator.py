"""
Research Coordinator — Research Assistant System

Orchestrates the 3-agent pipeline deterministically.
Zero LLM calls in this file — all routing is pure logic.

Pipeline flow per query:
    1. Validate + classify domain          (deterministic)
    2. LiteratureScanner  — dynamic-k + re-rank  (0 LLM calls)
    3. CitationExtractor  — regex + parsing       (0 LLM calls)
    4. SynthesisAgent     — citation-aware prompt (1 LLM call)
    5. compute_confidence — FAISS score average   (deterministic)
    6. PipelineLogger     — async JSON write      (non-blocking)
"""

import os
import time
import logging
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from config.settings import DOMAIN_KEYWORDS, SystemConfig
from core.document_processor import DocumentProcessor
from core.llm_interface import LLMInterface
from core.memory import ResearchMemory
from core.models import (
    ResearchDomain,
    create_paper_summary,
    validate_research_query,
)
from core.pipeline_logger import PipelineLogger
from agents.citation_extractor import CitationExtractor
from agents.literature_scanner import LiteratureScanner
from agents.synthesis_agent import SynthesisAgent
from agents.verification_agent import VerificationAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level utility — lives here because it depends only on raw scores,
# not on any agent or coordinator state.
# ---------------------------------------------------------------------------

def compute_confidence(similarities: List[float]) -> float:
    """
    Compute a normalised RAG retrieval confidence score from similarity values.

    Expects values already in [0, 1] where 1.0 = perfect match.
    These are the combined_score / relevance_score values produced by
    rerank_chunks() — NOT raw FAISS L2 distances.

    Formula:  confidence = mean(similarities)

    Args:
        similarities: List of similarity floats in [0, 1] from retrieved papers.

    Returns:
        float in [0.0, 1.0].  Returns 0.0 for empty input.
    """
    if not similarities:
        return 0.0
    # Clamp defensively in case any value drifts outside [0, 1]
    clamped = [max(0.0, min(1.0, s)) for s in similarities]
    return round(sum(clamped) / len(clamped), 4)


# ---------------------------------------------------------------------------
# ResearchCoordinator
# ---------------------------------------------------------------------------

class ResearchCoordinator:
    """
    Orchestrates the 3-agent research pipeline.

    Responsibilities:
    - Validate and clean incoming queries
    - Classify research domain (keyword scoring, no LLM)
    - Run agents in order: Scanner → Extractor → Synthesiser
    - Compute retrieval confidence from FAISS scores
    - Write structured logs asynchronously
    - Expose system stats and reset controls to the UI
    """

    def __init__(self, api_key: str):
        """
        Initialise all subsystems with a single Gemini API key.
        The same key is used for both LLM calls and Google embeddings.
        """
        load_dotenv()

        # Core infrastructure
        self.llm              = LLMInterface(api_key)
        self.memory           = ResearchMemory()
        self.document_processor = DocumentProcessor(api_key)
        self.pipeline_logger  = PipelineLogger()

        # Agents — order reflects pipeline execution order
        self.literature_scanner  = LiteratureScanner()   # Step 1: 0 LLM calls
        self.citation_extractor  = CitationExtractor()   # Step 2: 0 LLM calls
        self.synthesis_agent     = SynthesisAgent(self.llm)  # Step 3: 1 LLM call
        self.verification_agent  = VerificationAgent()       # Step 4: 0 LLM calls

        # Session counter for logging
        self.total_queries_processed = 0
        self.created_at = time.time()

        logger.info(
            "ResearchCoordinator ready | target=%d LLM call(s) per query",
            SystemConfig.MAX_LLM_CALLS_PER_QUERY,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research_query(
        self,
        query: str,
        domain: str = "other",
        max_papers: int = 8,
        pdf_papers: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point — runs the full pipeline for one user query.

        Args:
            query:      Raw user question.
            domain:     Hint for domain classification (overridden by keyword scoring).
            max_papers: Upper bound on documents fed to the synthesiser.
            pdf_papers: Optional pre-loaded PDF data (bypasses FAISS when provided).

        Returns:
            Complete result dict including synthesis, metrics, and confidence score.
        """
        start_time = time.time()
        self.total_queries_processed += 1

        # --- Input validation (deterministic) ---
        validation = validate_research_query(query)
        if not validation["valid"]:
            return self._create_error_response("INVALID", validation["error"])

        query = validation["cleaned_query"]

        # --- Domain classification (keyword scoring, no LLM) ---
        research_domain = self._classify_domain(query, domain)

        # --- Create session for memory tracking ---
        session_id = self.memory.create_session(query, research_domain)
        logger.info(
            "Query #%d | domain=%s | session=%s",
            self.total_queries_processed, research_domain.value, session_id,
        )

        try:
            # --- Run the 3-agent pipeline ---
            # selected_k is the dynamic k chosen by LiteratureScanner for this query
            result, selected_k = self._execute_pipeline(
                session_id, query, research_domain, max_papers
            )

            # --- Finalise timing and efficiency rating ---
            processing_time = time.time() - start_time
            result["performance_metrics"]["processing_time"] = round(processing_time, 2)
            result["performance_metrics"]["efficiency_rating"] = (
                SystemConfig.get_efficiency_status(
                    result["performance_metrics"]["total_llm_calls"]
                )
            )

            # --- Persist session metrics ---
            self.memory.update_session(
                session_id,
                total_llm_calls=self.llm.call_count,
                total_cost=self.llm.total_cost,
                processing_time=processing_time,
            )

            # --- Async structured log (never blocks the pipeline) ---
            # Pass selected_k explicitly so the log records the real dynamic k,
            # not papers_analyzed (Issue 2 fix)
            self._log_pipeline_result(result, session_id, processing_time, selected_k)

            self._log_summary(result)
            return result

        except Exception as exc:
            processing_time = time.time() - start_time
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            return self._create_error_response(session_id, str(exc), processing_time)

    def search_uploaded_documents(
        self, query: str, max_results: int = 5
    ) -> Dict[str, Any]:
        """Direct FAISS search — used by the UI for quick lookups."""
        try:
            results = self.document_processor.search_documents(query, k=max_results)
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "llm_calls_made": self.document_processor.llm_call_count,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "query": query,
                "results": [],
                "total_results": 0,
                "llm_calls_made": self.document_processor.llm_call_count,
            }

    def get_document_processing_stats(self) -> Dict[str, Any]:
        """Proxy to document processor stats — used by the sidebar."""
        return self.document_processor.get_processing_stats()

    def process_document(
        self, file_path: str, file_type: str = None
    ) -> Dict[str, Any]:
        """Process a single document file and add it to the FAISS index."""
        try:
            result = self.document_processor.process_document(file_path, file_type)
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc), "file_path": file_path}

    def get_system_stats(self) -> Dict[str, Any]:
        """Aggregate stats for the sidebar metrics panel."""
        sys_metrics   = self.memory.get_system_metrics()
        doc_stats     = self.get_document_processing_stats()

        return {
            "total_research_sessions":        sys_metrics.total_sessions,
            "total_llm_calls":                sys_metrics.total_llm_calls,
            "total_cost":                     sys_metrics.total_cost,
            "average_llm_calls_per_session":  sys_metrics.average_llm_calls_per_session,
            "efficiency_score":               sys_metrics.efficiency_rating,
            "agent_performance": {
                name: {
                    "calls_made":          m.calls_made,
                    "llm_calls":           m.llm_calls_made,
                    "success_rate":        m.success_rate,
                    "avg_processing_time": round(
                        m.processing_time / max(1, m.calls_made), 3
                    ),
                }
                for name, m in sys_metrics.agent_metrics.items()
            },
            "domain_statistics":  self.memory.get_domain_statistics(),
            "efficiency_analysis": self.memory.get_efficiency_report(),
            "document_processing": {
                "total_documents":        doc_stats["total_documents"],
                "total_chunks":           doc_stats["total_chunks"],
                "vector_store_size":      doc_stats["vector_store_size"],
                "llm_calls_for_embeddings": doc_stats["llm_calls_made"],
            },
        }

    def reset_system(self):
        """Reset all counters and clear the vector store — used by the UI."""
        self.llm.reset_counters()
        self.memory.clear_history()
        self.literature_scanner.reset_metrics()
        self.citation_extractor.reset_metrics()
        self.synthesis_agent.reset_metrics()
        self.document_processor.reset_processor()
        self.total_queries_processed = 0
        logger.info("System reset complete")

    def research_query_with_pdfs(
        self,
        query: str,
        pdf_papers: List[Dict[str, Any]],
        domain: str = "other",
    ) -> Dict[str, Any]:
        """Convenience wrapper — routes PDF-based queries through the main pipeline."""
        if not pdf_papers:
            return {"success": False, "error": "No PDF papers provided"}
        return self.research_query(query, domain, len(pdf_papers), pdf_papers)

    # ------------------------------------------------------------------
    # Pipeline execution — each step is its own private method
    # ------------------------------------------------------------------

    def _execute_pipeline(
        self,
        session_id: str,
        query: str,
        domain: ResearchDomain,
        max_papers: int,
    ) -> tuple:  # returns (result_dict, selected_k)
        """
        Run the 3-agent pipeline in strict order.

        Step 1 — LiteratureScanner  (0 LLM calls)
            Selects dynamic-k based on query complexity, fetches chunks
            from FAISS, applies re-ranking, and groups into Paper objects.

        Step 2 — CitationExtractor  (0 LLM calls)
            Extracts citations, key quotes, and author metadata from the
            Paper objects using regex patterns.

        Step 3 — SynthesisAgent     (1 LLM call)
            Builds a citation index ([Paper N] → title), constructs the
            prompt with the SOURCE INDEX block, calls Gemini once, and
            validates the JSON response.  Falls back to deterministic
            sentence extraction if the LLM call fails.

        After the pipeline:
            compute_confidence() converts FAISS scores to a [0,1] score
            that is attached to performance_metrics.
        """

        # ── Step 1: Literature Scanner ────────────────────────────────
        logger.info("Step 1 | LiteratureScanner starting...")

        scanner_result = self.literature_scanner.process({
            "query":            query,
            "domain":           domain.value,
            "max_results":      max_papers,
            "coordinator":      self,       # gives scanner access to FAISS store
            "use_vector_store": True,       # prefer FAISS over raw PDF fallback
        })

        if not scanner_result.get("success"):
            raise RuntimeError(
                f"LiteratureScanner failed: {scanner_result.get('error')}"
            )

        papers_found = scanner_result["papers"]
        if not papers_found:
            raise RuntimeError("No relevant documents found for this query")

        # Compute retrieval confidence from re-ranked relevance scores
        # (combined_score = 0.7*similarity + 0.3*keyword_overlap)
        retrieval_confidence = compute_confidence(
            [p.relevance_score for p in papers_found]
        )
        logger.info(
            "Step 1 done | papers=%d dynamic_k=%d reranked=%s confidence=%.4f",
            len(papers_found),
            scanner_result.get("effective_k", "?"),
            scanner_result.get("reranked", False),
            retrieval_confidence,
        )

        self.memory.update_session(session_id, papers_found=papers_found)

        # ── Step 2: Citation Extractor ────────────────────────────────
        logger.info("Step 2 | CitationExtractor starting...")

        extraction_result = self.citation_extractor.process(papers_found)

        if not extraction_result.get("success"):
            raise RuntimeError(
                f"CitationExtractor failed: {extraction_result.get('error')}"
            )

        logger.info(
            "Step 2 done | citations=%d quotes=%d",
            extraction_result["citations_extracted"],
            extraction_result["quotes_extracted"],
        )

        # ── Step 3: Synthesis Agent ───────────────────────────────────
        # Uses enhanced_papers (metadata-enriched) from the extractor.
        # The agent internally builds the [Paper N] citation index and
        # injects it into the prompt — exactly 1 LLM call happens here.
        logger.info("Step 3 | SynthesisAgent starting (1 LLM call)...")

        synthesis_result = self.synthesis_agent.process({
            "query":          query,
            "papers":         extraction_result["enhanced_papers"],
            "extracted_data": extraction_result,
        })

        if not synthesis_result.get("success"):
            raise RuntimeError(
                f"SynthesisAgent failed: {synthesis_result.get('error')}"
            )

        synthesis = synthesis_result["synthesis"]
        logger.info(
            "Step 3 done | findings=%d confidence=%.2f fallback=%s",
            len(synthesis.key_findings),
            synthesis.confidence_score,
            synthesis_result.get("fallback_used", False),
        )

        self.memory.update_session(session_id, synthesis=synthesis)

        # ── Step 4: Verification Agent (0 LLM calls, embedding calls only) ──
        logger.info("Step 4 | VerificationAgent starting (0 LLM calls)...")

        verification_result = self.verification_agent.process({
            "synthesis":    synthesis,
            "citation_map": synthesis_result.get("citation_map", {}),
            "papers":       extraction_result["enhanced_papers"],
        })
        verification = verification_result.get("verification")

        if verification:
            logger.info(
                "Step 4 done | citations_valid=%s grounding_score=%.2f unsupported=%d",
                verification.citations_valid, verification.grounding_score,
                len(verification.unsupported_claims),
            )
        else:
            logger.warning(
                "Step 4 skipped/failed | error=%s",
                verification_result.get("error"),
            )

        # ── Compile final result ──────────────────────────────────────
        # Return selected_k alongside result so the caller can log it correctly
        selected_k = scanner_result.get("effective_k", scanner_result.get("dynamic_k", 0))
        return (
            self._compile_result(
                session_id, query, domain,
                papers_found, extraction_result, synthesis_result,
                retrieval_confidence, verification,
            ),
            selected_k,
        )

    def _compile_result(
        self,
        session_id: str,
        query: str,
        domain: ResearchDomain,
        papers: list,
        extraction_result: Dict[str, Any],
        synthesis_result: Dict[str, Any],
        retrieval_confidence: float,
        verification: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Assemble the final result dict returned to the UI.

        All five pipeline features are represented here:
        - papers_found          → dynamic-k + re-ranked documents
        - extracted_insights    → citation extractor output
        - research_synthesis    → LLM synthesis with [Paper N] citations
        - performance_metrics   → LLM calls, cost, retrieval_confidence
        """
        synthesis = synthesis_result["synthesis"]
        meta      = extraction_result["metadata_analysis"]

        # Augment key findings with their source page/section when the
        # LLM included explicit [Paper N] citation labels. We leave
        # findings untouched when no label is present.
        citation_map = synthesis_result.get("citation_map", {})

        def _attach_source_info(finding_text: str) -> Dict[str, Any]:
            """Return a dict with finding text and inferred source info."""
            m = re.search(r"\[Paper (\d+)\]", finding_text)
            label = None
            source = {"label": None, "title": None, "page": "N/A", "section": "N/A"}
            if m:
                try:
                    idx = int(m.group(1)) - 1
                    label = f"[Paper {m.group(1)}]"
                    paper = papers[idx] if 0 <= idx < len(papers) else None
                    if paper is not None:
                        source["label"] = label
                        source["title"] = getattr(paper, "title", None)
                        # Prefer explicit page metadata if available
                        pm = getattr(paper, "metadata", {}) or {}
                        source["page"] = pm.get("page", pm.get("page_range", "N/A"))
                        # Choose the first heading as the most relevant section
                        headings = pm.get("headings") or pm.get("sections") or []
                        source["section"] = headings[0] if headings else "N/A"
                except Exception:
                    # Be defensive — never raise while assembling response
                    pass

            # Clean finding text by removing the inline [Paper N] tag
            cleaned = re.sub(r"\s*\[Paper \d+\]", "", finding_text).strip()
            return {"text": cleaned, "source": source}

        return {
            "session_id": session_id,
            "success":    True,
            "query":      query,
            "domain":     domain.value,

            # Documents retrieved and re-ranked by LiteratureScanner
            "papers_found": {
                "count":  len(papers),
                "papers": [create_paper_summary(p) for p in papers],
            },

            # Citations, quotes, and author network from CitationExtractor
            "extracted_insights": {
                "total_citations":  extraction_result["citations_extracted"],
                "total_quotes":     extraction_result["quotes_extracted"],
                "key_quotes":       extraction_result["key_quotes"][:10],
                "top_authors":      meta["top_authors"],
                "venues":           list(meta["venue_distribution"].keys()),
                "year_span":        (
                    f"{meta['year_range']['min']}-{meta['year_range']['max']}"
                ),
                "citation_network": extraction_result["citation_network"],
                "research_insights": extraction_result["research_insights"],
            },

            # LLM synthesis with [Paper N] citation labels. Key findings are
            # augmented with source page/section where available. Research gaps
            # are placed last per user request.
            "research_synthesis": {
                "key_findings":        [
                    _attach_source_info(k) for k in getattr(synthesis, "key_findings", [])
                ],
                "methodology_insights": synthesis.methodology_insights,
                "recommended_papers":  synthesis.recommended_papers,
                "confidence":          synthesis.confidence_score,
                "completeness":        synthesis_result["synthesis_completeness"],
                "limitations":         getattr(synthesis, "limitations", []),
                "performance_metrics": getattr(synthesis, "performance_metrics", []),
                # research_gaps intentionally last
                "research_gaps":       synthesis.research_gaps,
            },

            # Observability: LLM calls, cost, retrieval confidence score
            "performance_metrics": {
                "total_llm_calls":          self.llm.call_count,
                "estimated_cost":           self.llm.total_cost,
                "papers_analyzed":          len(papers),
                "agents_used":              4,
                "llm_agent_calls":          synthesis_result["llm_calls_made"],
                "deterministic_agent_calls": 3,
                "retrieval_confidence":     retrieval_confidence,
            },

            # Citation-validity + grounding check on the synthesis output.
            # 0 LLM calls; embedding calls only. None if the agent failed
            # (never blocks the response -- verification is advisory).
            "verification": verification.to_dict() if verification else None,
        }

    # ------------------------------------------------------------------
    # Domain classification — deterministic keyword scoring
    # ------------------------------------------------------------------

    def _classify_domain(self, query: str, domain_hint: str) -> ResearchDomain:
        """
        Score each domain by keyword matches in the query.
        Longer keyword phrases score higher than single words.
        Falls back to domain_hint if no keywords match.
        """
        query_lower   = query.lower()
        domain_scores = {
            domain: sum(
                len(kw.split()) for kw in keywords if kw in query_lower
            )
            for domain, keywords in DOMAIN_KEYWORDS.items()
        }

        best_score = max(domain_scores.values(), default=0)
        if best_score > 0:
            best_domain = max(domain_scores, key=domain_scores.get)
            try:
                return ResearchDomain(best_domain)
            except ValueError:
                pass

        try:
            return ResearchDomain(domain_hint)
        except ValueError:
            return ResearchDomain.OTHER

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_pipeline_result(
        self,
        result: Dict[str, Any],
        session_id: str,
        processing_time: float,
        selected_k: int = 0,
    ) -> None:
        """
        Build the structured log entry and dispatch it asynchronously.

        selected_k is the dynamic k chosen by LiteratureScanner — passed
        explicitly from research_query() to avoid using papers_analyzed
        as a proxy (Issue 2 fix).
        """
        papers    = result.get("papers_found", {}).get("papers", [])
        synthesis = result.get("research_synthesis", {})
        metrics   = result.get("performance_metrics", {})

        # Build slim chunk list — use relevance_score as the similarity proxy
        # since paper summaries don't carry raw FAISS distances
        chunks = [
            {
                "content":          p.get("abstract_preview", "")[:300],
                "page":             p.get("metadata", {}).get("page", "N/A")
                                    if isinstance(p.get("metadata"), dict) else "N/A",
                "heading":          p.get("metadata", {}).get("heading", "N/A")
                                    if isinstance(p.get("metadata"), dict) else "N/A",
                "source_file":      p.get("venue", "Unknown"),
                "similarity_score": p.get("relevance_score", 0.0),
                "combined_score":   p.get("relevance_score", 0.0),
            }
            for p in papers
        ]

        self.pipeline_logger.log_query(
            user_query       = result["query"],
            selected_k       = selected_k,          # real dynamic k (Issue 2 fix)
            retrieved_chunks = chunks,
            final_response   = synthesis,
            confidence_score = synthesis.get("confidence", 0.0),
            response_time    = processing_time,
            domain           = result.get("domain", ""),
            session_id       = session_id,
            llm_calls        = metrics.get("total_llm_calls", 0),
            estimated_cost   = metrics.get("estimated_cost", 0.0),
            fallback_used    = synthesis.get("confidence", 1.0) <= 0.5,
        )

    def _log_summary(self, result: Dict[str, Any]) -> None:
        """Log a concise human-readable summary at INFO level."""
        if not result["success"]:
            logger.error("Research failed: %s", result.get("error"))
            return

        synthesis = result["research_synthesis"]
        metrics   = result["performance_metrics"]

        logger.info(
            "Pipeline complete | papers=%d domain=%s confidence=%.2f",
            result["papers_found"]["count"],
            result["domain"],
            metrics.get("retrieval_confidence", 0),
        )
        for i, finding in enumerate(synthesis["key_findings"][:3], 1):
            # `finding` may be a dict {text, source} after augmentation
            text = finding.get("text") if isinstance(finding, dict) else finding
            logger.info("  Finding %d: %.80s", i, text)
        logger.info(
            "LLM calls=%d cost=$%.4f time=%.2fs efficiency=%s",
            metrics["total_llm_calls"],
            metrics["estimated_cost"],
            metrics["processing_time"],
            metrics["efficiency_rating"]["status"],
        )

    # ------------------------------------------------------------------
    # Error response
    # ------------------------------------------------------------------

    def _create_error_response(
        self,
        session_id: str,
        error_msg: str,
        processing_time: float = 0.0,
    ) -> Dict[str, Any]:
        """Standardised error dict returned to the UI on any failure."""
        return {
            "session_id": session_id,
            "success":    False,
            "error":      error_msg,
            "performance_metrics": {
                "total_llm_calls":  self.llm.call_count,
                "estimated_cost":   self.llm.total_cost,
                "processing_time":  processing_time,
                "agents_used":      0,
            },
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Diagnostics (used by run_system_diagnostics in the UI)
    # ------------------------------------------------------------------

    def validate_system_architecture(self) -> Dict[str, Any]:
        """Check that the agent mix meets the project requirements."""
        agents = {
            "LiteratureScanner": self.literature_scanner,
            "CitationExtractor": self.citation_extractor,
            "SynthesisAgent":    self.synthesis_agent,
            "VerificationAgent": self.verification_agent,
        }
        total        = len(agents)
        llm_count    = sum(1 for a in agents.values() if a.uses_llm)
        det_count    = total - llm_count
        requirements = {
            "min_3_agents":               total >= 3,
            "max_2_llm_agents":           llm_count <= 2,
            "min_50_percent_deterministic": (det_count / total) >= 0.5,
            "deterministic_routing":      True,
        }
        return {
            "total_agents":            total,
            "llm_agents":              llm_count,
            "deterministic_agents":    det_count,
            "deterministic_percentage": round((det_count / total) * 100, 1),
            "requirements":            requirements,
            "overall_compliance":      all(requirements.values()),
        }

    def run_system_diagnostics(self) -> Dict[str, Any]:
        """Full health check — called from the UI diagnostics panel."""
        return {
            "timestamp": time.time(),
            "system_health": {
                "llm_interface_ready":  self.llm.client is not None,
                "api_key_configured":   SystemConfig.validate_api_key(),
                "memory_initialized":   self.memory is not None,
                "agents_initialized":   all([
                    self.literature_scanner is not None,
                    self.citation_extractor is not None,
                    self.synthesis_agent    is not None,
                    self.verification_agent is not None,
                ]),
            },
            "architecture": self.validate_system_architecture(),
            "performance":  self.get_system_stats(),
        }

    def get_agent_execution_flow(self) -> Dict[str, Any]:
        """Static description of the pipeline — used by the UI info panel."""
        return {
            "execution_order": [
                {
                    "step": 1, "agent": "LiteratureScanner",
                    "llm_calls": 0,
                    "features": "dynamic-k selection + FAISS search + re-ranking",
                },
                {
                    "step": 2, "agent": "CitationExtractor",
                    "llm_calls": 0,
                    "features": "regex citation extraction + key quote mining",
                },
                {
                    "step": 3, "agent": "SynthesisAgent",
                    "llm_calls": 1,
                    "features": "citation-aware prompt + Gemini synthesis",
                },
                {
                    "step": 4, "agent": "VerificationAgent",
                    "llm_calls": 0,
                    "features": "citation validity check + embedding-based grounding score",
                },
            ],
            "total_llm_calls":    1,
            "confidence_scoring": "FAISS similarity → normalised [0,1] score",
            "logging":            "async JSON append to logs/pipeline_logs.jsonl",
        }