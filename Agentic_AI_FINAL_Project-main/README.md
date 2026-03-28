# 🔬 Research Assistant AI
**CSYE 7374 Final Project — Summer 2025**

A hybrid multi-agent RAG system that processes academic documents and generates research synthesis using **exactly 1 LLM call per query**.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│           ResearchCoordinator               │
│         (deterministic routing)             │
└──────┬──────────────┬──────────────┬────────┘
       │              │              │
       ▼              ▼              ▼
 LiteratureScanner  CitationExtractor  SynthesisAgent
  (0 LLM calls)     (0 LLM calls)    (1 LLM call ←only one)
  FAISS vector       Regex/parsing    Gemini 2.5 Flash
  similarity         citation &       structured JSON
  search             quote extract    synthesis
```

### Agent breakdown

| Agent | LLM Calls | Method | Output |
|---|---|---|---|
| Document Processor | 0 | LangChain + FAISS | Vector store |
| Literature Scanner | 0 | Cosine similarity | Ranked chunks |
| Citation Extractor | 0 | Regex patterns | Citations + quotes |
| Synthesis Agent | **1** | Gemini 2.5 Flash | Research synthesis |

**Total: exactly 1 LLM call per research query.**

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Gemini API key → [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) (free)

### 2. Install
```bash
git clone <repo-url>
cd Agentic_AI_FINAL_Project-main

pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
# Open .env and set your key:
# GEMINI_API_KEY=your-actual-key-here
```

### 4. Run
```bash
python -m streamlit run streamlit_app.py
```

Open → `http://localhost:8501`

---

## Usage

**Step 1 — Upload documents**
Upload PDF, TXT, or MD research papers (max 50 MB each).
The system chunks them and builds a FAISS vector store.

**Step 2 — Ask a research question**
Type any research question. The 3-agent pipeline runs automatically.

**Step 3 — Review results across 6 tabs**
- Literature Discovery — documents found and relevance scores
- Citation Analysis — extracted authors and citation network
- Key Quotes — important sentences with page/section references
- Research Gaps — areas needing further investigation
- Limitations — constraints mentioned by the paper authors
- Performance Metrics — accuracy, F1-score, and baselines

---

## Project Structure

```
├── streamlit_app.py          # Web UI entry point
├── agents/
│   ├── base_agent.py         # Abstract base with performance tracking
│   ├── literature_scanner.py # Agent 1: FAISS vector search (0 LLM)
│   ├── citation_extractor.py # Agent 2: Regex citation extraction (0 LLM)
│   └── synthesis_agent.py    # Agent 3: Gemini synthesis (1 LLM)
├── core/
│   ├── coordinator.py        # Orchestrates the 3-agent pipeline
│   ├── document_processor.py # PDF/TXT loading + FAISS indexing
│   ├── google_embeddings.py  # Google embedding-001 wrapper
│   ├── llm_interface.py      # Gemini API client with retry + cost tracking
│   ├── prompts.py            # LLM prompt templates
│   ├── memory.py             # Session state and metrics
│   └── models.py             # Pydantic data models
├── config/
│   └── settings.py           # All configuration and constants
├── data/                     # 13 sample transformer/attention PDFs
├── .env.example              # Environment variable template
└── requirements.txt          # Pinned dependencies
```

---

## Configuration (`config/settings.py`)

| Setting | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `gemini-2.5-flash` | Gemini model for synthesis |
| `MAX_LLM_CALLS_PER_QUERY` | `2` | Efficiency target |
| `DOCUMENT_CONFIG.chunk_size` | `1000` | RAG chunk size in chars |
| `DOCUMENT_CONFIG.chunk_overlap` | `200` | Overlap between chunks |
| `RAG_CONFIG.max_full_text_chars` | `30000` | Max chars sent to LLM per doc |
| `SYNTHESIS_CONFIG.max_tokens` | `6000` | Max LLM output tokens |

---

## Cost

Real token-based pricing using Gemini's `usage_metadata`:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| gemini-2.5-flash | $0.15 | $0.60 |
| gemini-1.5-pro | $1.25 | $5.00 |

A typical research query costs **< $0.001**.

---

## Troubleshooting

**`streamlit` not found**
```bash
python -m streamlit run streamlit_app.py
```

**`FAISS` install issues**
```bash
pip install faiss-cpu
```

**API key error**
- Check `.env` file exists and contains `GEMINI_API_KEY=...`
- Verify the key at [aistudio.google.com](https://aistudio.google.com)

**Document processing fails**
- Supported formats: `.pdf`, `.txt`, `.md`
- Max file size: 50 MB
- File must not be password-protected

---

## Requirements
- Python 3.10+
- 4 GB+ RAM (for FAISS vector operations)
- Internet connection (Gemini API calls)
