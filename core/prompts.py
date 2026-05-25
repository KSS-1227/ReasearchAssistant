"""
Prompt templates for the Research Assistant System

Single source of truth for all LLM prompt construction.
Kept separate from llm_interface.py so prompts can be
edited without touching API/retry logic.
"""


def create_synthesis_prompt(query: str, papers_summary: str, citation_index: str = ""):
    """
    Build the citation-aware synthesis prompt sent to Gemini.

    Args:
        query:          User research question.
        papers_summary: Full paper content block built by SynthesisAgent.
        citation_index: Pre-built string listing [Paper N] → title mappings.
                        Injected at the top of the user prompt so the LLM
                        knows exactly which label maps to which source.
    """

    system_prompt = """You are an expert research analyst. Your job is to answer the user's question using ONLY the content from the provided documents.

CRITICAL RULES:
1. Answer the EXACT question asked — do not give generic research summaries
2. Use ONLY information found in the provided documents
3. If the documents do not contain relevant information to answer the question, say so clearly
4. Include specific details: page numbers, section names, numbers, metrics when available
5. Be direct and specific — avoid vague or generic statements
6. CITATION RULE — you MUST cite every claim using [Paper N] notation where N matches
   the paper number in the SOURCE INDEX provided. Use the provided context and cite
   sources using [Paper 1], [Paper 2], etc. notation. Every key finding, methodology
   insight, and technical contribution MUST end with at least one [Paper N] citation.
   Do NOT invent citation labels — only use the numbers listed in the SOURCE INDEX.

You MUST return a valid JSON object with EXACTLY these fields (no extra fields):
{
    "research_question": "restate the exact question asked",
    "key_findings": [
        "Direct answer to the question from the document [Paper 1]",
        "Supporting evidence or detail from the document [Paper 2]",
        "Additional relevant finding [Paper 1, Paper 3]"
    ],
    "methodology_insights": [
        "How the document approaches this topic [Paper 2]",
        "Specific method or technique mentioned [Paper 1]"
    ],
    "research_gaps": [
        "What the documents do NOT cover related to this question",
        "Limitations mentioned by the authors [Paper 3]"
    ],
    "recommended_papers": [
        "Exact document title that best answers this question"
    ],
    "confidence_score": 0.85,
    "technical_contributions": [
        "Specific technical detail relevant to the question [Paper 1]"
    ],
    "comparative_analysis": [
        "Any comparisons made in the documents [Paper 1, Paper 2]"
    ],
    "practical_implications": [
        "Practical takeaway from the documents [Paper 2]"
    ]
}"""

    citation_block = f"""SOURCE INDEX (use these labels for citations):
{citation_index}

""" if citation_index else ""

    user_prompt = f"""Question to answer: {query}

{citation_block}Documents to search through:
{papers_summary}

Instructions:
- Find the direct answer to "{query}" in the documents above
- Cite every claim with [Paper N] using the SOURCE INDEX above
- Quote specific sections, pages, or passages that answer the question
- If the documents contain a direct answer, extract it precisely with its citation
- If the documents are not relevant to this question, state that clearly in key_findings
- Do NOT generate generic research summaries — answer THIS specific question
- Do NOT use citation labels not listed in the SOURCE INDEX

Return ONLY the JSON object, no other text:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]