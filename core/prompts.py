"""
Prompt templates for the Research Assistant System

Single source of truth for all LLM prompt construction.
Kept separate from llm_interface.py so prompts can be
edited without touching API/retry logic.
"""


def create_synthesis_prompt(query: str, papers_summary: str):
    """Build the synthesis prompt sent to Gemini for research analysis."""

    system_prompt = """You are an expert research analyst. Your job is to answer the user's question using ONLY the content from the provided documents.

CRITICAL RULES:
1. Answer the EXACT question asked — do not give generic research summaries
2. Use ONLY information found in the provided documents
3. If the documents do not contain relevant information to answer the question, say so clearly
4. Include specific details: page numbers, section names, numbers, metrics when available
5. Be direct and specific — avoid vague or generic statements

You MUST return a valid JSON object with EXACTLY these fields (no extra fields):
{
    "research_question": "restate the exact question asked",
    "key_findings": [
        "Direct answer to the question from the document (Page X, Section Y)",
        "Supporting evidence or detail from the document",
        "Additional relevant finding from the document"
    ],
    "methodology_insights": [
        "How the document approaches or explains this topic",
        "Specific method or technique mentioned in the document"
    ],
    "research_gaps": [
        "What the document does NOT cover related to this question",
        "Limitations mentioned by the authors"
    ],
    "recommended_papers": [
        "Exact document title that best answers this question"
    ],
    "confidence_score": 0.85,
    "technical_contributions": [
        "Specific technical detail from the document relevant to the question"
    ],
    "comparative_analysis": [
        "Any comparisons made in the document relevant to the question"
    ],
    "practical_implications": [
        "Practical takeaway from the document relevant to the question"
    ]
}"""

    user_prompt = f"""Question to answer: {query}

Documents to search through:
{papers_summary}

Instructions:
- Find the direct answer to "{query}" in the documents above
- Quote specific sections, pages, or passages that answer the question
- If the documents contain a direct answer, extract it precisely
- If the documents are not relevant to this question, state that clearly in key_findings
- Do NOT generate generic research summaries — answer THIS specific question

Return ONLY the JSON object, no other text:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]