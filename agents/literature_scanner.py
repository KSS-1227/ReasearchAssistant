"""
Literature Scanner Agent — Research Assistant System

Agent 1: Retrieves and ranks documents using deterministic methods only.
Zero LLM calls — all logic is pure Python.

Integrated features:
- determine_k()         : dynamic top-k based on query complexity
- rerank_chunks()       : combined score = 0.7*similarity + 0.3*keyword_overlap
- keyword_overlap_score(): normalised token overlap between query and chunk
"""

import re
import logging
from typing import Any, Dict, List, Set

from agents.base_agent import BaseAgent
from config.settings import DOMAIN_KEYWORDS, SystemConfig
from core.models import Paper, ResearchDomain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dynamic-k thresholds
_SIMPLE_K  = 5   # short / simple queries
_MEDIUM_K  = 8   # medium-length or descriptive queries
_COMPLEX_K = 12   # long or analytical queries

# Re-ranking weights (must sum to 1.0)
_SIMILARITY_WEIGHT = 0.7   # FAISS cosine similarity contribution
_KEYWORD_WEIGHT    = 0.3   # keyword overlap contribution

# Keywords that signal query complexity — checked before word-count heuristic
_COMPLEX_KEYWORDS = {
    "compare", "contrast", "analyze", "analyse", "evaluate", "assess",
    "explain in detail", "critically", "differentiate", "relationship between",
    "pros and cons", "advantages and disadvantages", "in depth", "thoroughly",
}
_MEDIUM_KEYWORDS = {
    "explain", "describe", "summarize", "summarise", "overview", "discuss",
    "how does", "what is", "why does", "impact of", "effect of",
}

# Stop words removed before keyword overlap scoring
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "this", "that", "it", "its", "as", "from", "not", "also", "which",
}


# ---------------------------------------------------------------------------
# Module-level functions (pure, stateless — easy to unit-test)
# ---------------------------------------------------------------------------

def determine_k(query: str) -> int:
    """
    Choose FAISS top-k based on query complexity (deterministic, 0 LLM calls).

    Rules (checked in order):
    1. Complex keyword present OR word count > 12  → k = 7
    2. Medium keyword present  OR 7 ≤ words ≤ 12  → k = 5
    3. Everything else                             → k = 3

    Args:
        query: Raw user query string.

    Returns:
        int: One of 3, 5, or 7.
    """
    q          = query.lower().strip()
    word_count = len(q.split())

    if any(kw in q for kw in _COMPLEX_KEYWORDS) or word_count > 12:
        k, reason = _COMPLEX_K, "complex keyword or long query"
    elif any(kw in q for kw in _MEDIUM_KEYWORDS) or 7 <= word_count <= 12:
        k, reason = _MEDIUM_K, "medium keyword or medium-length query"
    else:
        k, reason = _SIMPLE_K, "simple/short query"

    logger.info(
        "determine_k | words=%d k=%d reason='%s' query='%.60s'",
        word_count, k, reason, query,
    )
    return k


def keyword_overlap_score(query: str, chunk_text: str) -> float:
    """
    Normalised keyword overlap between query and a chunk (precision-style).

    Algorithm (pure Python, zero API calls):
    1. Tokenise both strings to lowercase alpha tokens.
    2. Remove stop words and single-character tokens.
    3. score = |query_terms ∩ chunk_terms| / |query_terms|

    Args:
        query:      Raw user query string.
        chunk_text: Text content of a retrieved chunk.

    Returns:
        float in [0.0, 1.0].  Returns 0.0 when query has no meaningful terms.
    """
    def _tokenise(text: str) -> Set[str]:
        tokens = re.findall(r"[a-z]+", text.lower())
        return {t for t in tokens if len(t) > 1 and t not in _STOP_WORDS}

    query_terms = _tokenise(query)
    if not query_terms:
        return 0.0

    chunk_terms = _tokenise(chunk_text)
    return min(1.0, len(query_terms & chunk_terms) / len(query_terms))


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    """
    Re-rank FAISS chunks using a combined score.

    Formula:
        combined_score = (0.7 × similarity_score) + (0.3 × keyword_overlap_score)

    Each chunk dict is mutated in-place to add:
        - keyword_overlap : raw keyword overlap score
        - combined_score  : final re-ranking score

    Args:
        query:  Raw user query string.
        chunks: Chunk dicts from DocumentProcessor.search_documents().
                Must contain 'content' and 'similarity_score' keys.
        top_n:  Number of top chunks to return.

    Returns:
        List of top_n chunk dicts sorted by combined_score descending.
    """
    if not chunks:
        return []

    for chunk in chunks:
        # similarity_score is already in [0, 1] (converted from FAISS L2 distance
        # inside DocumentProcessor.search_documents). Clamp defensively.
        sim   = max(0.0, min(1.0, float(chunk.get("similarity_score", 0.0))))
        kw    = keyword_overlap_score(query, chunk.get("content", ""))
        score = (_SIMILARITY_WEIGHT * sim) + (_KEYWORD_WEIGHT * kw)

        chunk["keyword_overlap"] = round(kw,    4)
        chunk["combined_score"]  = round(score, 4)

    reranked = sorted(chunks, key=lambda c: c["combined_score"], reverse=True)

    logger.info(
        "rerank | in=%d top_n=%d best=%.4f worst=%.4f",
        len(chunks), top_n,
        reranked[0]["combined_score"]  if reranked else 0,
        reranked[-1]["combined_score"] if reranked else 0,
    )
    return reranked[:top_n]


# ---------------------------------------------------------------------------
# LiteratureScanner agent
# ---------------------------------------------------------------------------

class LiteratureScanner(BaseAgent):
    """
    Agent 1: Document retrieval and ranking — zero LLM calls.

    Primary path  (FAISS vector store available):
        1. determine_k()    — pick k based on query complexity
        2. search_documents() — fetch k*3 chunks with page diversity
        3. rerank_chunks()  — re-score with combined similarity + keyword overlap
        4. Group chunks by source document → Paper objects

    Fallback path (raw PDF data provided, no vector store):
        Score papers by term overlap + recency + citation boost.
    """

    def __init__(self):
        super().__init__("LiteratureScanner", uses_llm=False)
        self.config = SystemConfig.LITERATURE_CONFIG

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Entry point — wraps _search_papers with performance tracking."""
        return self._execute_with_tracking(self._search_papers, input_data)

    # ------------------------------------------------------------------
    # Core search logic
    # ------------------------------------------------------------------

    def _search_papers(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute document retrieval with dynamic-k and re-ranking.

        Returns a result dict containing:
        - papers       : List[Paper] sorted by relevance (highest first)
        - dynamic_k    : k chosen by determine_k()
        - effective_k  : min(dynamic_k, max_results)
        - reranked     : True when FAISS path was used
        - search_method: 'vector_store' | 'pdf_analysis' | 'none'
        """
        query            = input_data.get("query", "")
        domain           = ResearchDomain(input_data.get("domain", "other"))
        max_results      = input_data.get("max_results", 10)
        pdf_papers       = input_data.get("pdf_papers", [])
        use_vector_store = input_data.get("use_vector_store", False)

        # Validate before doing any work
        validation = self.validate_input(input_data)
        if not validation["valid"]:
            return {"success": False, "error": validation["error"], "papers": []}

        # --- Dynamic k selection (deterministic, 0 LLM calls) ---
        dynamic_k   = determine_k(query)
        effective_k = min(dynamic_k, max_results)

        logger.info(
            "LiteratureScanner | query='%.60s' dynamic_k=%d effective_k=%d domain=%s",
            query, dynamic_k, effective_k, domain.value,
        )

        # ── Primary path: FAISS vector store ─────────────────────────
        if use_vector_store and hasattr(input_data.get("coordinator"), "document_processor"):
            coordinator = input_data["coordinator"]
            if coordinator.document_processor.vector_store:
                return self._search_via_vector_store(
                    query, coordinator, effective_k, dynamic_k
                )

        # ── Fallback path: raw PDF data ───────────────────────────────
        if pdf_papers:
            return self._search_via_pdf_data(
                query, domain, pdf_papers, effective_k, dynamic_k
            )

        # ── No data available ─────────────────────────────────────────
        return {
            "success":       False,
            "error":         "No documents available. Please upload documents first.",
            "papers":        [],
            "search_method": "none",
        }

    def _search_via_vector_store(
        self,
        query: str,
        coordinator: Any,
        effective_k: int,
        dynamic_k: int,
    ) -> Dict[str, Any]:
        """
        Retrieve chunks from FAISS, re-rank them, then group into Paper objects.

        The page-diversity filter in DocumentProcessor.search_documents() ensures
        chunks are spread across the document rather than clustering on page 1-3.
        Re-ranking then promotes chunks that are both semantically similar AND
        contain the actual query keywords.
        """
        # Fetch chunks (page-diversity filter applied inside search_documents)
        raw_chunks = coordinator.document_processor.search_documents(
            query, k=effective_k
        )

        # Re-rank: combined_score = 0.7*similarity + 0.3*keyword_overlap
        ranked_chunks = rerank_chunks(query, raw_chunks, top_n=effective_k)

        if ranked_chunks:
            logger.info(
                "Re-rank complete | top_score=%.4f",
                ranked_chunks[0]["combined_score"],
            )

        # Group chunks by source document to build Paper objects
        papers = self._group_chunks_into_papers(ranked_chunks)

        # Sort papers by their best chunk's combined_score (highest first)
        papers.sort(key=lambda p: p.relevance_score, reverse=True)

        return {
            "success":       True,
            "papers":        papers[:effective_k],
            "search_method": "vector_store",
            "total_papers":  len(papers),
            "vector_store_used": True,
            "reranked":      True,
            "dynamic_k":     dynamic_k,
            "effective_k":   effective_k,
        }

    def _group_chunks_into_papers(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Paper]:
        """
        Merge chunk-level results into document-level Paper objects.

        Each unique source_file becomes one Paper.  Chunks are sorted by
        page number so the full_text reads in document order.
        The Paper's relevance_score is the best combined_score across its chunks.
        """
        document_groups: Dict[str, Dict] = {}

        for chunk in chunks:
            meta        = chunk.get("metadata", {})
            source_file = meta.get("source_file", chunk.get("source", "unknown"))
            page        = meta.get("page", 0)
            heading     = meta.get("heading", "N/A")
            # Use combined_score if available (re-ranked), else fall back to
            # similarity_score which is already in [0,1] from document_processor
            relevance   = chunk.get("combined_score",
                                    chunk.get("similarity_score", 0.0))

            if source_file not in document_groups:
                document_groups[source_file] = {
                    "chunks":        [],
                    "best_relevance": 0.0,
                    "metadata":      meta,
                    "pages":         set(),
                    "headings":      set(),
                }

            document_groups[source_file]["chunks"].append({
                "content":  chunk.get("content", ""),
                "relevance": relevance,
                "page":     page,
                "heading":  heading,
            })
            document_groups[source_file]["best_relevance"] = max(
                document_groups[source_file]["best_relevance"], relevance
            )
            document_groups[source_file]["pages"].add(page)
            document_groups[source_file]["headings"].add(heading)

        papers = []
        for doc_id, (source_file, doc_data) in enumerate(document_groups.items()):
            # Sort chunks by page so full_text reads in document order
            sorted_chunks = sorted(
                doc_data["chunks"], key=lambda c: c.get("page", 0)
            )

            # Combine chunks preserving page and section context
            combined_content = "\n\n".join(
                f"[Page {c['page']}, Section: {c['heading']}]\n{c['content']}"
                for c in sorted_chunks
            )

            original_filename = doc_data["metadata"].get("original_filename", source_file)
            display_name = (
                original_filename.split("/")[-1].split("\\")[-1]
                if "/" in original_filename or "\\" in original_filename
                else original_filename
            )

            pages      = sorted(doc_data["pages"])
            page_range = (
                f"{pages[0]}-{pages[-1]}" if len(pages) > 1
                else str(pages[0]) if pages else "N/A"
            )
            headings_list = list(doc_data["headings"])

            logger.debug(
                "Paper built | file=%s pages=%s sections=%d",
                display_name, page_range, len(headings_list),
            )

            paper = Paper(
                id=f"doc_{doc_id}_{original_filename}",
                title=f"Document: {display_name}",
                authors=["Research Paper"],
                abstract=(
                    combined_content[:500] + "..."
                    if len(combined_content) > 500
                    else combined_content
                ),
                year=doc_data["metadata"].get("year", 2024),
                venue=f"Research Document ({display_name})",
                citations=[],
                key_quotes=[],
            )
            paper.relevance_score = doc_data["best_relevance"]
            paper.full_text       = combined_content
            paper.metadata        = {
                **doc_data["metadata"],
                "source_file":   source_file,
                "chunk_count":   len(doc_data["chunks"]),
                "document_type": "uploaded",
                "page_range":    page_range,
                "pages":         pages,
                "headings":      headings_list,
                "total_pages":   len(pages),
            }
            papers.append(paper)

        return papers

    def _search_via_pdf_data(
        self,
        query: str,
        domain: ResearchDomain,
        pdf_papers: List[Dict],
        effective_k: int,
        dynamic_k: int,
    ) -> Dict[str, Any]:
        """
        Score raw PDF paper dicts by term overlap + recency + citation boost.
        Used when no FAISS vector store is available.
        """
        expanded_query = self._expand_query(query, domain)
        query_terms    = self._extract_query_terms(expanded_query)
        scored_papers  = []

        for paper_data in pdf_papers:
            paper = Paper(
                id=paper_data.get("id", "unknown"),
                title=paper_data.get("title", "Unknown Title"),
                authors=paper_data.get("authors", ["Unknown Author"]),
                abstract=paper_data.get("abstract", ""),
                year=paper_data.get("year", 2024),
                venue=paper_data.get("venue", "Uploaded PDF"),
                citations=paper_data.get("citations", []),
                key_quotes=[],
            )
            relevance = self._calculate_relevance_score(paper, query_terms)
            if relevance > self.config["min_relevance_threshold"]:
                paper.relevance_score = relevance
                scored_papers.append(paper)

        scored_papers.sort(key=lambda p: p.relevance_score, reverse=True)

        return {
            "success":       True,
            "papers":        scored_papers[:effective_k],
            "search_method": "pdf_analysis",
            "total_papers":  len(scored_papers),
            "dynamic_k":     dynamic_k,
            "effective_k":   effective_k,
            "reranked":      False,
        }

    # ------------------------------------------------------------------
    # Scoring helpers (PDF fallback path only)
    # ------------------------------------------------------------------

    def _expand_query(self, query: str, domain: ResearchDomain) -> str:
        """Append up to 3 domain keywords that overlap with query terms."""
        query_lower = query.lower()
        relevant    = [
            kw for kw in DOMAIN_KEYWORDS.get(domain.value, [])
            if any(term in kw for term in query_lower.split())
        ]
        return query + (" " + " ".join(relevant[:3]) if relevant else "")

    def _extract_query_terms(self, query: str) -> Set[str]:
        """Tokenise query, remove stop words, add meaningful bigrams."""
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "is", "are", "was", "were", "be",
            "been", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should",
        }
        words = query.lower().split()
        terms = {w for w in words if len(w) > 2 and w not in stop_words}

        # Add meaningful bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i + 1]}"
            if len(bigram) > 8:
                terms.add(bigram)

        return terms

    def _calculate_relevance_score(self, paper: Paper, query_terms: Set[str]) -> float:
        """
        Multi-factor relevance score for PDF fallback path.

        Factors:
        - Term overlap (title weighted 2×, abstract 1×)
        - Recency boost  (papers after 2020 get a small boost)
        - Citation boost (well-cited papers get a small boost)
        - Venue boost    (ACM/IEEE/Nature/Science venues get +0.1)
        """
        title_text    = paper.title.lower()
        abstract_text = paper.abstract.lower()

        title_matches    = len(query_terms & set(title_text.split()))
        abstract_matches = len(query_terms & set(abstract_text.split()))
        base_relevance   = (
            ((title_matches * 2) + abstract_matches) / len(query_terms)
            if query_terms else 0
        )

        recency_boost  = min(
            self.config["max_recency_boost"],
            max(0, paper.year - 2020) * self.config["recency_boost_factor"],
        )
        citation_boost = min(
            self.config["max_citation_boost"],
            len(paper.citations) * self.config["citation_boost_factor"],
        )
        venue_boost = (
            0.1 if any(
                t in paper.venue.lower()
                for t in ["nature", "science", "acm", "ieee"]
            ) else 0
        )

        return min(1.0, base_relevance + recency_boost + citation_boost + venue_boost)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_input(self, input_data: Any) -> Dict[str, Any]:
        """Validate required keys and value ranges."""
        if not isinstance(input_data, dict):
            return {"valid": False, "error": "Input must be a dictionary"}

        for key in ("query", "domain"):
            if key not in input_data:
                return {"valid": False, "error": f"Missing required key: {key}"}

        if not input_data.get("query", "").strip() or len(input_data["query"].strip()) < 3:
            return {"valid": False, "error": "Query must be at least 3 characters"}

        try:
            ResearchDomain(input_data["domain"])
        except ValueError:
            return {"valid": False, "error": f"Invalid domain: {input_data['domain']}"}

        max_results = input_data.get("max_results", 10)
        if not isinstance(max_results, int) or not (1 <= max_results <= 50):
            return {"valid": False, "error": "max_results must be an integer between 1 and 50"}

        return {"valid": True}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_search_statistics(self) -> Dict[str, Any]:
        """Performance metrics for the sidebar."""
        return {
            **self.get_performance_metrics(),
            "search_specific_metrics": {
                "domains_supported":      len(list(ResearchDomain)),
                "relevance_threshold":    self.config["min_relevance_threshold"],
                "max_results_per_search": self.config.get("max_papers_per_domain", 20),
                "reranking_enabled":      True,
                "dynamic_k_enabled":      True,
            },
        }
