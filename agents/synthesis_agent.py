"""
Synthesis Agent — Research Assistant System

Agent 3: The ONLY LLM-powered agent in the system.
Makes exactly 1 Gemini API call per query.

Integrated features:
- _build_citation_index()     : stable [Paper N] → title mapping (0 LLM calls)
- _prepare_papers_for_synthesis(): full-text context block for the prompt
- create_synthesis_prompt()   : citation-aware prompt with SOURCE INDEX
- _create_fallback_synthesis(): deterministic sentence extraction when LLM fails
"""

import json
import logging
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from config.settings import SystemConfig
from core.llm_interface import LLMInterface, validate_llm_response
from core.models import Paper, ResearchSynthesis
from core.prompts import create_synthesis_prompt

logger = logging.getLogger(__name__)


class SynthesisAgent(BaseAgent):
    """
    Agent 3: Research synthesis using a single Gemini LLM call.

    Responsibilities:
    1. Build a stable [Paper N] citation index from the top papers.
    2. Prepare a full-text context block (up to 30k chars per paper).
    3. Inject the citation index into the prompt as a SOURCE INDEX.
    4. Make exactly 1 LLM call and validate the JSON response.
    5. Fall back to deterministic sentence extraction if the LLM fails.
    """

    def __init__(self, llm: LLMInterface):
        super().__init__("SynthesisAgent", uses_llm=True)
        self.llm             = llm
        self.config          = SystemConfig.SYNTHESIS_CONFIG
        self.synthesis_calls = 0  # tracks total LLM calls made by this agent

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Entry point — wraps _synthesize_research with performance tracking."""
        return self._execute_with_tracking(self._synthesize_research, input_data)

    # ------------------------------------------------------------------
    # Core synthesis logic
    # ------------------------------------------------------------------

    def _synthesize_research(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full synthesis pipeline for one query.

        Steps:
        1. Sort papers by relevance and slice to max_input_papers.
        2. Build citation index: {1: "title", 2: "title", ...}
        3. Build papers_summary string (full text + quotes + metadata).
        4. Create citation-aware prompt with SOURCE INDEX block.
        5. Make the single LLM call.
        6. Validate JSON response → ResearchSynthesis Pydantic model.
        7. Fall back to deterministic extraction if validation fails.
        """
        query          = input_data.get("query", "")
        papers         = input_data.get("papers", [])
        extracted_data = input_data.get("extracted_data", {})

        logger.info("SynthesisAgent | synthesising from %d papers", len(papers))

        # Step 1: Sort by relevance and cap at max_input_papers
        # Sorting here ensures [Paper 1] is always the most relevant document.
        top_papers = sorted(papers, key=lambda p: p.relevance_score, reverse=True)[
            : self.config["max_input_papers"]
        ]

        # Step 2: Build citation index (deterministic, 0 LLM calls)
        # Maps paper number → clean title for [Paper N] notation in the prompt.
        citation_index     = self._build_citation_index(top_papers)
        citation_index_str = "\n".join(
            f"[Paper {n}] = {title}" for n, title in citation_index.items()
        )
        logger.info("citation_index | %d entries", len(citation_index))

        # Step 3: Build full-text context block for the LLM
        papers_summary = self._prepare_papers_for_synthesis(top_papers, extracted_data)
        logger.debug("papers_summary | %d chars", len(papers_summary))

        # Step 4: Build citation-aware prompt
        # The SOURCE INDEX block tells the LLM exactly which [Paper N] label
        # maps to which document — preventing hallucinated citations.
        messages = create_synthesis_prompt(query, papers_summary, citation_index_str)

        # Step 5: Single LLM call (the ONLY LLM call in the entire system)
        self.synthesis_calls += 1
        logger.info(
            "LLM call #%d | prompt_chars=%d | papers=%d | query='%.60s'",
            self.synthesis_calls, len(papers_summary), len(top_papers), query,
        )

        response = self.llm.make_call(messages, {"type": "json_object"})

        # Step 6: Validate JSON response
        if response and response.content:
            logger.info("LLM response | %d chars", len(response.content))

            # Log parse errors for debugging without crashing
            try:
                parsed = json.loads(response.content)
                logger.debug("JSON parsed OK | keys=%s", list(parsed.keys()))
            except json.JSONDecodeError as exc:
                logger.error("JSON parse error: %s | raw: %.500s", exc, response.content)

            synthesis = validate_llm_response(response.content, ResearchSynthesis)

            if synthesis:
                logger.info(
                    "Synthesis OK | findings=%d insights=%d gaps=%d confidence=%.2f",
                    len(synthesis.key_findings),
                    len(synthesis.methodology_insights),
                    len(synthesis.research_gaps),
                    synthesis.confidence_score,
                )
                return {
                    "success":              True,
                    "synthesis":            synthesis,
                    "llm_calls_made":       1,
                    "papers_analyzed":      len(papers),
                    "input_tokens_approx":  len(papers_summary.split()),
                    "synthesis_confidence": synthesis.confidence_score,
                    "synthesis_completeness": self._assess_completeness(synthesis),
                    "citation_map":         citation_index,
                    "used_llm":             True,
                    "fallback_used":        False,
                }

            # Step 7a: LLM response failed Pydantic validation → fallback
            logger.warning("LLM validation failed — using deterministic fallback")
            return self._create_fallback_synthesis(
                query, top_papers, extracted_data, citation_index
            )

        # Step 7b: LLM returned None (rate limit / network) → fallback
        logger.warning("LLM returned None — using deterministic fallback")
        return self._create_fallback_synthesis(
            query, top_papers, extracted_data, citation_index
        )

    # ------------------------------------------------------------------
    # Citation index
    # ------------------------------------------------------------------

    def _build_citation_index(self, papers: List[Paper]) -> Dict[int, str]:
        """
        Build a stable 1-based {paper_number: title} mapping.

        Papers must already be sorted by relevance (highest first) so that
        [Paper 1] always refers to the most relevant document for this query.

        Args:
            papers: Pre-sorted list of Paper objects.

        Returns:
            Dict[int, str] — e.g. {1: "Attention Is All You Need", 2: "Longformer"}
        """
        return {
            i: paper.title.replace("Document: ", "").strip()
            for i, paper in enumerate(papers, 1)
        }

    # ------------------------------------------------------------------
    # Context preparation
    # ------------------------------------------------------------------

    def _prepare_papers_for_synthesis(
        self, papers: List[Paper], extracted_data: Dict[str, Any]
    ) -> str:
        """
        Build the full-text context block sent to the LLM.

        Structure:
        1. Research landscape overview (year range, venues, authors, counts)
        2. Per-paper blocks: header + full_text (up to 30k chars) + key quotes
        3. Extracted insights summary (quote analysis, citation analysis)
        4. Citation network summary

        Papers are already sorted and sliced by the caller (_synthesize_research).
        """
        parts = []
        meta  = extracted_data.get("metadata_analysis", {})

        # --- Landscape overview ---
        parts += [
            "RESEARCH LANDSCAPE OVERVIEW:",
            f"• {len(papers)} papers | years: "
            f"{meta.get('year_range', {}).get('min', 'N/A')}–"
            f"{meta.get('year_range', {}).get('max', 'N/A')}",
            f"• Top venues: {', '.join(list(meta.get('venue_distribution', {}).keys())[:5])}",
            f"• Leading authors: "
            f"{', '.join(a for a, _ in meta.get('top_authors', [])[:5])}",
            f"• Citations analysed: {extracted_data.get('citations_extracted', 0)}",
            f"• Key quotes extracted: {extracted_data.get('quotes_extracted', 0)}",
            "",
            "DETAILED PAPER CONTENT:",
        ]

        # --- Per-paper blocks ---
        for i, paper in enumerate(papers, 1):
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += " et al."

            parts.append(f"\n{'=' * 60}")
            parts.append(f"[Paper {i}]: {paper.title}")
            parts.append(f"Authors: {authors_str} ({paper.year})")
            parts.append(f"Venue: {paper.venue}")
            parts.append(f"Relevance: {paper.relevance_score:.3f}")

            # Page range and section headings from metadata
            if hasattr(paper, "metadata") and paper.metadata:
                m = paper.metadata
                if "page_range" in m:
                    parts.append(f"Pages: {m['page_range']}")
                if m.get("headings"):
                    parts.append(f"Sections: {', '.join(list(m['headings'])[:8])}")

            parts.append("=" * 60)

            # Full document text (up to 30k chars to stay within token budget)
            if getattr(paper, "full_text", None):
                parts.append("\nDOCUMENT CONTENT:")
                parts.append(paper.full_text[:30_000])
                if len(paper.full_text) > 30_000:
                    parts.append(
                        f"[Truncated — total {len(paper.full_text)} chars]"
                    )
            else:
                parts.append("\nABSTRACT:")
                parts.append(paper.abstract)

            # Key quotes extracted by CitationExtractor
            if paper.key_quotes:
                parts.append("\nKEY QUOTES:")
                for j, quote in enumerate(paper.key_quotes[:5], 1):
                    text       = quote.get("text", "")       if isinstance(quote, dict) else getattr(quote, "text", "")
                    quote_type = quote.get("quote_type", "")  if isinstance(quote, dict) else getattr(quote, "quote_type", "")
                    confidence = quote.get("confidence", 0.5) if isinstance(quote, dict) else getattr(quote, "confidence", 0.5)
                    parts.append(f"  {j}. [{quote_type}, conf={confidence:.2f}] \"{text}\"")

            parts.append(f"\n{'-' * 60}\n")

        # --- Extracted insights summary ---
        insights = extracted_data.get("research_insights", {})
        if insights:
            parts.append("EXTRACTED INSIGHTS SUMMARY:")
            qa = insights.get("quote_analysis", {})
            if qa:
                parts.append(f"• Quotes: {qa.get('total_quotes', 0)} "
                              f"(avg confidence {qa.get('average_confidence', 0):.3f})")
            ca = insights.get("citation_analysis", {})
            if ca:
                parts.append(f"• Citations: {ca.get('total_citations', 0)} "
                              f"over {ca.get('citation_time_span', 0)} years")

        # --- Citation network ---
        cn = extracted_data.get("citation_network", {})
        if cn:
            parts.append("\nCITATION NETWORK:")
            parts.append(f"• Connections: {cn.get('total_connections', 0)}")
            parts.append(f"• Unique authors: {cn.get('unique_authors', 0)}")
            parts.append(f"• Clusters: {len(cn.get('clusters', {}))}")

        summary = "\n".join(parts)
        logger.debug(
            "papers_summary | %d chars ~%d tokens %d papers",
            len(summary), len(summary.split()), len(papers),
        )
        return summary

    # ------------------------------------------------------------------
    # Deterministic fallback
    # ------------------------------------------------------------------

    def _create_fallback_synthesis(
        self,
        query: str,
        papers: List[Paper],
        extracted_data: Dict[str, Any],
        citation_index: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Deterministic fallback when the LLM call fails or returns invalid JSON.

        Extracts the most query-relevant sentences from each paper's full_text
        using keyword overlap scoring, then injects [Paper N] citation labels
        from the same citation_index used in the prompt.

        Confidence score is fixed at 0.5 to signal fallback was used.
        """
        logger.info("Fallback synthesis | query='%.60s'", query)

        citation_index = citation_index or {}
        # Reverse map: clean_title → [Paper N] label
        title_to_label = {
            title: f"[Paper {n}]" for n, title in citation_index.items()
        }

        query_keywords       = set(query.lower().split())
        key_findings         = []
        methodology_insights = []
        technical_contributions = []

        for paper in papers:
            clean_title    = paper.title.replace("Document: ", "")
            citation_label = title_to_label.get(clean_title, f"[{clean_title}]")

            # Extract and score sentences by keyword overlap with the query
            source = (getattr(paper, "full_text", "") or "") or paper.abstract or ""
            sentences = [
                s.strip()
                for s in source.replace("\n", " ").split(".")
                if len(s.strip()) > 40
            ]
            scored = sorted(
                [(len(query_keywords & set(s.lower().split())), s)
                 for s in sentences if query_keywords & set(s.lower().split())],
                reverse=True,
            )

            # Top 3 sentences → key findings
            for _, sentence in scored[:3]:
                key_findings.append(f"{sentence.strip()} {citation_label}")

            # Next 2 sentences → methodology insights
            for _, sentence in scored[3:5]:
                methodology_insights.append(f"{sentence.strip()} {citation_label}")

            # Key quotes → technical contributions
            for quote in paper.key_quotes[:2]:
                qt = (
                    quote.get("text", "") if isinstance(quote, dict)
                    else getattr(quote, "text", "")
                )
                if len(qt) > 30:
                    technical_contributions.append(f"{qt} {citation_label}")

        # If no keyword-matching sentences found, use top sentences from best paper
        if not key_findings and papers:
            best   = papers[0]
            source = (getattr(best, "full_text", "") or best.abstract or "")
            for s in [
                s.strip()
                for s in source.replace("\n", " ").split(".")
                if len(s.strip()) > 40
            ][:5]:
                key_findings.append(
                    f"{s} [{best.title.replace('Document: ', '')}]"
                )

        # Honest fallback messages when no content matches at all
        if not key_findings:
            key_findings = [
                f'The uploaded documents do not contain information about: "{query}"',
                "Try uploading documents specifically related to your question.",
            ]
            methodology_insights = [
                "No relevant methodology found in the uploaded documents."
            ]
            research_gaps = [f'The document set does not cover: "{query}"']
        else:
            research_gaps = [
                "The documents may not fully cover all aspects of this question.",
                "Consider uploading additional sources for a more complete answer.",
            ]

        if not methodology_insights:
            methodology_insights = [
                "No specific methodology details found matching this query."
            ]

        recommended_papers = [
            f"{p.title.replace('Document: ', '')} (relevance: {p.relevance_score:.2f})"
            for p in papers[:5]
        ]

        fallback = ResearchSynthesis(
            research_question=query,
            key_findings=key_findings[:12],
            methodology_insights=methodology_insights[:8],
            research_gaps=research_gaps[:5],
            recommended_papers=recommended_papers,
            confidence_score=0.5,   # fixed at 0.5 to signal fallback
            technical_contributions=technical_contributions[:6],
            comparative_analysis=[],
            practical_implications=[],
        )

        logger.info(
            "Fallback done | findings=%d insights=%d gaps=%d",
            len(fallback.key_findings),
            len(fallback.methodology_insights),
            len(fallback.research_gaps),
        )

        return {
            "success":              True,
            "synthesis":            fallback,
            "llm_calls_made":       0,
            "papers_analyzed":      len(papers),
            "fallback_used":        True,
            "synthesis_confidence": 0.5,
            "synthesis_completeness": self._assess_completeness(fallback),
            "citation_map":         citation_index,
            "used_llm":             False,
        }

    # ------------------------------------------------------------------
    # Completeness assessment
    # ------------------------------------------------------------------

    def _assess_completeness(self, synthesis: ResearchSynthesis) -> Dict[str, Any]:
        """
        Score synthesis completeness against quality criteria.
        Used by the UI Performance Metrics tab.
        """
        scores = {
            "key_findings":          len(synthesis.key_findings) >= 5,
            "methodology_insights":  len(synthesis.methodology_insights) >= 3,
            "research_gaps":         len(synthesis.research_gaps) >= 2,
            "recommended_papers":    len(synthesis.recommended_papers) >= 2,
            "sufficient_confidence": synthesis.confidence_score >= 0.7,
            "technical_contributions": len(getattr(synthesis, "technical_contributions", [])) >= 2,
            "comparative_analysis":  len(getattr(synthesis, "comparative_analysis", [])) >= 1,
            "practical_implications": len(getattr(synthesis, "practical_implications", [])) >= 1,
            "comprehensive_findings": len(synthesis.key_findings) >= 8,
            "detailed_insights":     len(synthesis.methodology_insights) >= 5,
        }

        met        = sum(scores.values())
        total      = len(scores)
        percentage = (met / total) * 100

        richness = (
            len(synthesis.key_findings)
            + len(synthesis.methodology_insights)
            + len(synthesis.research_gaps)
            + len(getattr(synthesis, "technical_contributions", []))
            + len(getattr(synthesis, "comparative_analysis", []))
            + len(getattr(synthesis, "practical_implications", []))
        )

        return {
            "criteria_met":           met,
            "total_criteria":         total,
            "completeness_percentage": round(percentage, 1),
            "quality_rating":         self._quality_rating(percentage),
            "detailed_scores":        scores,
            "content_richness_score": richness,
            "is_comprehensive":       richness >= 20,
        }

    def _quality_rating(self, pct: float) -> str:
        """Convert completeness percentage to a human-readable rating."""
        if pct >= 90:   return "EXCELLENT"
        if pct >= 75:   return "GOOD"
        if pct >= 60:   return "ACCEPTABLE"
        return "NEEDS_IMPROVEMENT"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_input(self, input_data: Any) -> Dict[str, Any]:
        """Validate required keys before synthesis begins."""
        if not isinstance(input_data, dict):
            return {"valid": False, "error": "Input must be a dictionary"}

        for key in ("query", "papers", "extracted_data"):
            if key not in input_data:
                return {"valid": False, "error": f"Missing required key: {key}"}

        if not input_data.get("query", "").strip() or len(input_data["query"].strip()) < 3:
            return {"valid": False, "error": "Query must be at least 3 characters"}

        if not isinstance(input_data.get("papers"), list) or not input_data["papers"]:
            return {"valid": False, "error": "Papers list cannot be empty"}

        return {"valid": True}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """Performance stats for the sidebar."""
        return {
            **self.get_performance_metrics(),
            "synthesis_specific_metrics": {
                "total_synthesis_calls":  self.synthesis_calls,
                "llm_calls_per_synthesis": 1,
                "max_input_papers":       self.config["max_input_papers"],
                "max_tokens":             self.config["max_tokens"],
                "temperature":            self.config["temperature"],
                "citation_aware":         True,
                "fallback_enabled":       True,
            },
        }
