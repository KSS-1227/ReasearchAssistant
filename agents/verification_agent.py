"""
agents/verification_agent.py

Agent 4 (optional, non-LLM by default): checks SynthesisAgent's output for
citation validity and grounding, before it reaches _compile_result().

Matches this repo's actual conventions:
- [Paper N] maps directly to papers[N-1] (LiteratureScanner sorts by
  relevance_score descending; that order survives through
  CitationExtractor.enhanced_papers into SynthesisAgent's top_papers slice).
- paper.full_text is already the chunk-derived, retrieval-relevant text
  (built by LiteratureScanner._group_chunks_into_papers), so no separate
  chunk store is needed here.
- Uses the same citation-label regex pattern coordinator.py already uses
  in _attach_source_info(), for consistency.

Cost note: this agent makes real embedding API calls -- embed_documents()
in GoogleEmbeddings loops one call per text, it is not batched. That is
an accepted tradeoff, NOT free, but it costs zero new dependencies.

Embedder choice: defaults to GoogleEmbeddings (already a required dep --
see requirements.txt: google-genai>=1.68.0). Deliberately NOT defaulting
to HuggingFaceEmbeddings here: sentence-transformers pulls in PyTorch,
which is a 200-700MB+ install and 300-500MB+ runtime memory footprint --
a real risk of OOM/build-timeout crashes on Render's 512MB free tier.
Pass a HuggingFaceEmbeddings instance explicitly ONLY if deploying
somewhere with headroom for that (paid tier, self-hosted, or local dev).
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from agents.base_agent import BaseAgent
from core.models import Paper, ResearchSynthesis

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[Paper (\d+)\]")

# Fields worth checking. research_gaps / recommended_papers are
# intentionally excluded -- they're not "claims about a source", they're
# meta-commentary or plain titles, so grounding doesn't apply to them.
CLAIM_FIELDS = [
    "key_findings",
    "methodology_insights",
    "technical_contributions",
    "comparative_analysis",
    "practical_implications",
    "limitations",
    "performance_metrics",
]


@dataclass
class UnsupportedClaim:
    field: str
    text: str
    cited_paper: int
    similarity: float


@dataclass
class VerificationReport:
    citations_valid: bool
    invalid_citations: List[int] = field(default_factory=list)
    grounding_score: float = 1.0
    unsupported_claims: List[UnsupportedClaim] = field(default_factory=list)
    total_claims_checked: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citations_valid": self.citations_valid,
            "invalid_citations": self.invalid_citations,
            "grounding_score": round(self.grounding_score, 3),
            "unsupported_claims": [
                {
                    "field": c.field,
                    "text": c.text,
                    "cited_paper": c.cited_paper,
                    "similarity": round(c.similarity, 3),
                }
                for c in self.unsupported_claims
            ],
            "total_claims_checked": self.total_claims_checked,
        }


class VerificationAgent(BaseAgent):
    """
    Runs after SynthesisAgent, before _compile_result(). Zero LLM calls.
    Embedding calls only (local by default -- see module docstring).
    """

    def __init__(self, embedder=None, grounding_threshold: float = 0.6):
        super().__init__("VerificationAgent", uses_llm=False)
        self.grounding_threshold = grounding_threshold

        if embedder is None:
            # Reuses the SAME dependency (google-genai) already required
            # for the rest of the app -- zero new install, zero Render
            # deployment risk. Costs real API calls; see module docstring.
            from core.google_embeddings import GoogleEmbeddings
            from config.settings import SystemConfig
            embedder = GoogleEmbeddings(
                api_key=SystemConfig.GEMINI_API_KEY,
                model=SystemConfig.DOCUMENT_CONFIG["embedding_model"],
            )
        self.embedder = embedder

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._execute_with_tracking(self._verify, input_data)

    # ------------------------------------------------------------------

    def _verify(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        synthesis: ResearchSynthesis = input_data["synthesis"]
        citation_map: Dict[int, str] = input_data["citation_map"]
        papers: List[Paper] = input_data["papers"]

        # citation_map was built from papers[:max_input_papers] in the
        # same order -- so N -> papers[N-1] holds for every valid N.
        valid_range = set(citation_map.keys())

        invalid_citations: List[int] = []
        claims: List[tuple] = []  # (field_name, cleaned_text, paper_num)

        for field_name in CLAIM_FIELDS:
            for item in getattr(synthesis, field_name, []) or []:
                match = CITATION_PATTERN.search(item)
                if not match:
                    continue  # uncited claim -- nothing to verify against
                n = int(match.group(1))
                if n not in valid_range:
                    invalid_citations.append(n)
                    continue
                cleaned = CITATION_PATTERN.sub("", item).strip()
                claims.append((field_name, cleaned, n))

        unsupported: List[UnsupportedClaim] = []
        similarities: List[float] = []

        if claims:
            # One embedding call per unique paper full_text (cached), one
            # per claim -- not one per (claim, paper) pair.
            unique_papers = sorted({n for _, _, n in claims})
            paper_texts = [papers[n - 1].full_text or papers[n - 1].abstract
                           for n in unique_papers]
            paper_vecs = np.array(self.embedder.embed_documents(paper_texts))
            paper_vec_by_num = dict(zip(unique_papers, paper_vecs))

            claim_texts = [c[1] for c in claims]
            claim_vecs = np.array(self.embedder.embed_documents(claim_texts))

            for (field_name, text, n), claim_vec in zip(claims, claim_vecs):
                sim = self._cosine_sim(claim_vec, paper_vec_by_num[n])
                similarities.append(sim)
                if sim < self.grounding_threshold:
                    unsupported.append(
                        UnsupportedClaim(field=field_name, text=text,
                                          cited_paper=n, similarity=sim)
                    )

        report = VerificationReport(
            citations_valid=len(invalid_citations) == 0,
            invalid_citations=sorted(set(invalid_citations)),
            grounding_score=float(np.mean(similarities)) if similarities else 1.0,
            unsupported_claims=unsupported,
            total_claims_checked=len(claims),
        )

        logger.info(
            "VerificationAgent | claims=%d invalid_citations=%d "
            "unsupported=%d grounding_score=%.3f",
            report.total_claims_checked, len(report.invalid_citations),
            len(report.unsupported_claims), report.grounding_score,
        )

        return {"success": True, "verification": report}

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        a_n = a / (np.linalg.norm(a) + 1e-10)
        b_n = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_n, b_n))

    def validate_input(self, input_data: Any) -> Dict[str, Any]:
        if not isinstance(input_data, dict):
            return {"valid": False, "error": "Input must be a dictionary"}
        for key in ("synthesis", "citation_map", "papers"):
            if key not in input_data:
                return {"valid": False, "error": f"Missing required key: {key}"}
        return {"valid": True}