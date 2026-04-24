# 🔬 Research Assistant AI

> A hybrid multi-agent RAG system that processes academic documents and generates research synthesis using **exactly 1 LLM call per query**.

---

## How It Works

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│              ResearchCoordinator                │
│   Deterministic routing — zero LLM calls       │
└──────┬──────────────┬──────────────┬────────────┘
       │              │              │
       ▼              ▼              ▼
 LiteratureScanner  CitationExtractor  SynthesisAgent
  (0 LLM calls)      (0 LLM calls)    (1 LLM call)
  FAISS cosine        Regex + parsing   Gemini 2.5 Flash
  similarity          citations &       structured JSON
  search              quote extract     synthesis
```

### Agent Breakdown

| Agent | LLM Calls | Method | Output |
|---|---|---|---|
| Document Processor | 0 | LangChain + FAISS | Vector store |
| Literature Scanner | 0 | FAISS cosine similarity | Ranked document chunks |
| Citation Extractor | 0 | Regex patterns | Citations, quotes, author network |
| Synthesis Agent | **1** | Gemini 2.5 Flash | Structured JSON synthesis |

**Total: exactly 1 LLM call per research query.**

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash (`google-genai` SDK) |
| Embeddings | Google `text-embedding-004` (768-dim) |
| Vector Store | FAISS (in-memory, CPU) |
| Document Loading | LangChain `PyPDFLoader` / `TextLoader` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Data Validation | Pydantic v2 |
| Web UI | Streamlit |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Gemini API key → [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) (free tier available)

### Install

```bash
git clone <repo-url>
cd Agentic_AI_FINAL_Project-main
pip install -r requirements.txt
```

### Configure

```bash
cp env_template.txt .env
# Edit .env and set:
# GEMINI_API_KEY=your-actual-key-here
```

### Run

```bash
python -m streamlit run streamlit_app.py
```

Open → `http://localhost:8501`

---

## Usage

**Step 1 — Upload documents**
Upload PDF, TXT, or MD research papers (max 50 MB each). The system chunks them into 1000-char segments with 200-char overlap and builds a FAISS vector store using Google embeddings.

**Step 2 — Process**
Click "Process Documents". Each file is loaded, split, embedded, and indexed. Zero LLM calls at this stage.

**Step 3 — Ask a research question**
The system automatically generates **6 suggested questions** from your uploaded documents using an LLM call. These appear as clickable buttons — click any to pre-fill the input, or type your own question. The 3-agent pipeline then runs and returns results in under 30 seconds.

**Step 4 — Review results across 6 tabs**

| Tab | Content |
|---|---|
| Literature Discovery | Documents found with relevance scores |
| Citation Analysis | Extracted authors and citation network |
| Key Quotes | Important sentences with page/section references |
| Research Gaps | Areas needing further investigation |
| Limitations | Constraints mentioned by the authors |
| Performance Metrics | Accuracy, F1-score, and baselines |

---

## Project Structure

```
├── streamlit_app.py              # Streamlit UI — 3-step wizard + results display
├── agents/
│   ├── base_agent.py             # Abstract base: performance tracking, error handling
│   ├── literature_scanner.py     # Agent 1: FAISS vector search + relevance scoring (0 LLM)
│   ├── citation_extractor.py     # Agent 2: Regex citation + quote extraction (0 LLM)
│   └── synthesis_agent.py        # Agent 3: Gemini synthesis + deterministic fallback (1 LLM)
├── core/
│   ├── coordinator.py            # Orchestrates 3-agent pipeline, deterministic domain routing
│   ├── document_processor.py     # PDF/TXT loading, chunking, FAISS indexing, upload validation
│   ├── google_embeddings.py      # google.genai embedding wrapper (LangChain-compatible)
│   ├── llm_interface.py          # Gemini API client: retry, backoff, real token cost tracking
│   ├── prompts.py                # All LLM prompt templates (single source of truth)
│   ├── memory.py                 # Session state, agent metrics, efficiency reporting
│   └── models.py                 # Pydantic + dataclass models for all data structures
├── config/
│   └── settings.py               # All constants, pricing table, domain keywords, logging
├── data/                         # 13 sample transformer/attention PDFs
├── env_template.txt              # Environment variable template
├── QUICK_FIX_GUIDE.md            # Known issues and patches
├── USECASE_DIAGRAM.md            # Use case diagram (Markdown)
├── usecase_diagram.puml          # PlantUML source
└── requirements.txt              # Pinned dependencies
```

---

## Configuration (`config/settings.py`)

| Setting | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `gemini-2.5-flash` | Gemini model used for synthesis |
| `MAX_LLM_CALLS_PER_QUERY` | `2` | Efficiency budget (target is 1) |
| `DOCUMENT_CONFIG.chunk_size` | `1000` | RAG chunk size in characters |
| `DOCUMENT_CONFIG.chunk_overlap` | `200` | Overlap between adjacent chunks |
| `DOCUMENT_CONFIG.embedding_model` | `models/text-embedding-004` | Google embedding model |
| `RAG_CONFIG.max_full_text_chars` | `30,000` | Max chars sent per document to LLM |
| `RAG_CONFIG.embedding_dimension` | `768` | Output dimension of embedding model |
| `SYNTHESIS_CONFIG.max_tokens` | `6,000` | Max LLM output tokens per synthesis |
| `SYNTHESIS_CONFIG.temperature` | `0.3` | LLM temperature for synthesis |
| `SYNTHESIS_CONFIG.max_input_papers` | `8` | Max documents fed into synthesis prompt |
| `CITATION_CONFIG.min_quote_length` | `20` | Minimum characters for a valid quote |
| `CITATION_CONFIG.max_quotes_per_paper` | `3` | Quotes extracted per document |

---

## Cost

Real token-based pricing via Gemini's `usage_metadata` — no estimates, actual counts:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| `gemini-2.5-flash` | $0.15 | $0.60 |
| `gemini-2.5-pro` | $1.25 | $10.00 |
| `gemini-1.5-pro` | $1.25 | $5.00 |
| `gemini-1.5-flash` | $0.075 | $0.30 |

A typical research query costs **< $0.001**.

---

## Key Design Decisions

**Why exactly 1 LLM call?**
The Literature Scanner uses FAISS cosine similarity (deterministic, free). The Citation Extractor uses compiled regex patterns (deterministic, free). Only the final synthesis step — which requires cross-document reasoning — uses Gemini. This keeps costs near zero while maintaining quality.

**Why FAISS over a hosted vector DB?**
No external service dependency, no latency, no cost. FAISS runs in-process and is rebuilt per session from uploaded documents.

**Why `google-genai` instead of `google-generativeai`?**
`google-generativeai` is deprecated. The project uses the new `google-genai` SDK throughout — both for LLM calls and embeddings.

**Fallback synthesis**
If the Gemini API call fails (rate limit, quota, network), `SynthesisAgent._create_fallback_synthesis` extracts the most query-relevant sentences directly from document content using keyword overlap scoring. The UI never crashes.

---

## Supported Research Domains

Domain classification is deterministic (keyword scoring, no LLM):

`machine_learning` · `computer_vision` · `natural_language` · `robotics` · `cybersecurity` · `software_engineering` · `other`

---

## Troubleshooting

**`streamlit` not found**
```bash
python -m streamlit run streamlit_app.py
```

**FAISS install fails**
```bash
pip install faiss-cpu
```

**API key error**
- Confirm `.env` exists and contains `GEMINI_API_KEY=...`
- Verify the key at [aistudio.google.com](https://aistudio.google.com)
- The app reads the key via `python-dotenv` on startup

**Document processing fails**
- Supported formats: `.pdf`, `.txt`, `.md`
- Max file size: 50 MB per file
- File must not be password-protected
- Filename must not contain special characters (`\ / : * ? " < > |`)

**Empty synthesis / fallback used**
- The document may not contain content relevant to your query
- Try a more specific question that matches terms in the document
- Check the "Performance Metrics" tab — if `confidence_score` is 0.5, the fallback was used

**Suggested questions not generating**
- Click "Force Regenerate Questions" in the debug expander
- This uses an additional LLM call to generate 6 questions from document content

---

## Requirements

- Python 3.10+
- 4 GB+ RAM (FAISS vector operations)
- Internet connection (Gemini API + Google Embeddings API)
