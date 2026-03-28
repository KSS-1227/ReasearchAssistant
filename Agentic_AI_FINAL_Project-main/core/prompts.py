"""
Prompt templates for the Research Assistant System
CSYE 7374 Final Project - Summer 2025

Single source of truth for all LLM prompt construction.
Kept separate from llm_interface.py so prompts can be
edited without touching API/retry logic.
"""


def create_synthesis_prompt(query: str, papers_summary: str):
    """Build the synthesis prompt sent to Gemini for research analysis."""

    system_prompt = """You are a research analyst. Extract information ONLY from the documents provided.

Rules:
- Use ONLY document content
- Include page and section: (Page X, Section: Y)
- Extract specific numbers and metrics
- Identify limitations mentioned in paper
- Answer the specific question asked

Return JSON:
{
    "research_question": "the question",
    "key_findings": ["finding 1 (Page X, Section: Y)", "finding 2 (Page Z, Section: W)"],
    "methodology_insights": ["insight (Page X, Section: Y)"],
    "research_gaps": ["gap"],
    "recommended_papers": ["title"],
    "confidence_score": 0.8,
    "limitations": ["limitation 1 mentioned in paper (Page X)", "limitation 2 (Page Y)"],
    "performance_metrics": ["Model achieved 90% accuracy (Page X, Section: Results)", "F1-score: 0.85 (Page Y)"]
}"""

    user_prompt = f"""Question: {query}

Documents:
{papers_summary}

Extract from documents:
1. Answer the question with page numbers
2. Find ALL performance metrics (accuracy, F1, precision, recall, etc.)
3. Find ALL limitations mentioned by authors
4. Include specific numbers and baselines

Generate JSON:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
