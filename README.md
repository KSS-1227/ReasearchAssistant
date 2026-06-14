# 📚 Research Assistant AI

A production-grade, citation-aware research synthesis system built on a **FastAPI + Streamlit** architecture with a deterministic 3-agent RAG pipeline powered by Google Gemini.

> Live demo: https://reasearchassistant.streamlit.app/
>
> Backend (deployed): https://reasearchassistant.onrender.com/

---

## What It Does

- Upload PDF, TXT, or Markdown research documents (up to 50 MB each)
- Build an in-session FAISS vector index using Google `text-embedding-004` embeddings
- Auto-navigate through a guided 3-step workflow: Upload → Process → Ask
- Generate suggested research questions directly from your uploaded content
- Click any suggested question to auto-paste it into the custom question input
- Ask research questions and receive structured, citation-aware synthesis
- View results across tabbed panels: Findings, Gaps, Contributions, Analysis, Metrics, Raw JSON
- Download any synthesis result as a structured JSON file
- Track LLM calls, cost, retrieval confidence, and agent performance in real time

---

## Architecture

```
User uploads documents
        │
        ▼
┌─────────────────────┐
│   Streamlit UI      │  ← 3-step guided flow (Upload → Process → Ask)
│   (streamlit_app.py)│     session_state-driven tab navigation
└────────┬────────────┘
         │ HTTP (requests)
         ▼
┌─────────────────────┐
│   FastAPI Backend   │  ← REST API (fastapi_app.py)
│   /upload           │     per-session coordinator instances
│   /process          │     in-memory session management
│   /ask              │
│   /suggested-questions
│   /health  /status  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│               ResearchCoordinator                   │
│  (core/coordinator.py)                              │
│                                                     │
│  Step 1 → LiteratureScanner     (0 LLM calls)       │
│           FAISS search, dynamic-k, re-ranking       │
│                                                     │
│  Step 2 → CitationExtractor     (0 LLM calls)       │
│           regex citations, key quotes, author stats │
│                                                     │
│  Step 3 → SynthesisAgent        (1 LLM call)        │
│           citation-aware Gemini prompt              │
│           [Paper N] citation index injected         │
│           deterministic fallback on LLM failure     │
└─────────────────────────────────────────────────────┘
```

**Design principle:** exactly **1 Gemini LLM call per user query**. Document retrieval, citation extraction, and domain classification are all deterministic — zero LLM calls.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit ≥ 1.35 |
| API backend | FastAPI + Uvicorn |
| LLM | Google Gemini (`gemini-2.5-flash`) via `google-genai` |
| Embeddings | Google `models/text-embedding-004` (768-dim) |
| Vector search | FAISS CPU |
| Document loading | LangChain (`PyPDFLoader`, `TextLoader`) |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Validation | Pydantic v2 |
| Environment | `python-dotenv` |
| Containerisation | Docker |
| Orchestration | Kubernetes (k8s/) |
| CI/CD | GitHub Actions |

---

## Project Structure

```
ReasearchAssistant/
│
├── streamlit_app.py              # Streamlit UI — 3-step workflow, session_state navigation
├── fastapi_app.py                # FastAPI REST backend — session management, endpoints
│
├── agents/
│   ├── base_agent.py             # Abstract base class, performance tracking, execute_with_tracking
│   ├── literature_scanner.py     # FAISS retrieval, dynamic-k selection, re-ranking (0 LLM calls)
│   ├── citation_extractor.py     # Regex citation parsing, key quote mining, author stats (0 LLM calls)
│   └── synthesis_agent.py        # Citation-aware Gemini synthesis, [Paper N] index, fallback (1 LLM call)
│
├── core/
│   ├── coordinator.py            # 3-agent pipeline orchestration, domain classification, confidence scoring
│   ├── document_processor.py     # File validation, PDF/TXT/MD loading, chunking, FAISS indexing
│   ├── google_embeddings.py      # Google text-embedding-004 wrapper
│   ├── llm_interface.py          # Gemini client, retry logic, token counting, cost tracking
│   ├── llm_interface_fixed.py    # Patched LLM interface variant
│   ├── memory.py                 # In-session ResearchSession and SystemMetrics tracking
│   ├── models.py                 # Pydantic + dataclass models (Paper, ResearchSynthesis, etc.)
│   ├── pipeline_logger.py        # Async JSON structured logging to logs/pipeline_logs.jsonl
│   └── prompts.py                # Citation-aware synthesis prompt builder
│
├── config/
│   └── settings.py               # SystemConfig, DOMAIN_KEYWORDS, pricing table, LOGGING_CONFIG
│
├── data/                         # Sample research papers (13 transformer / attention PDFs)
│   ├── Attention Is All You Need.pdf
│   ├── Longformer The Long-Document Transformer.pdf
│   └── ...
│
├── k8s/
│   ├── deployment.yaml           # Kubernetes Deployment — 2 replicas, resource limits, health probes
│   ├── api-service.yaml          # Kubernetes Service for FastAPI
│   └── service.yaml              # Kubernetes Service for Streamlit
│
├── tests/
│   ├── __init__.py
│   └── test_app.py               # Pytest test suite
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                # CI: install → test → docker build → health check
│   │   ├── deploy.yml            # CD: deploy pipeline
│   │   └── openhands.yml         # OpenHands agent workflow
│   └── scripts/
│       └── issue_agent.py        # GitHub issue automation agent
│
├── logs/
│   └── pipeline_logs.jsonl       # Runtime structured query logs (auto-generated)
│
├── Dockerfile                    # Multi-service image — exposes :8501 (Streamlit) + :8000 (FastAPI)
├── entrypoint.sh                 # Container startup — launches both FastAPI and Streamlit
├── requirements.txt              # Python dependencies
├── env_template.txt              # Environment variable template
├── .env                          # Local secrets (git-ignored)
├── .dockerignore
├── .gitignore
├── USECASE_DIAGRAM.md
├── usecase_diagram.puml
└── Documentation.pdf
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/KSS-1227/ReasearchAssistant.git
cd ReasearchAssistant
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp env_template.txt .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your-gemini-api-key-here
```

Get a key at: https://aistudio.google.com/app/apikey

### 5. Start FastAPI Backend

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start Streamlit UI

In a second terminal (with venv active):
```bash
python -m streamlit run streamlit_app.py
```

Open: http://localhost:8501

---

## How To Use

| Step | Action |
|---|---|
| **1. Upload** | Select `.pdf`, `.txt`, or `.md` files → click **Upload** → auto-navigates to Process |
| **2. Process** | Click **Process Documents** → builds FAISS index → auto-navigates to Ask |
| **3. Ask** | Type a question or click **Generate Suggestions** → pick a suggestion to auto-paste it → click **Ask & Synthesize** |
| **4. Review** | Browse Findings, Gaps, Contributions, Analysis, Metrics, Raw JSON tabs |
| **5. Download** | Click **Download as JSON** to save the full synthesis result |
| **Reset** | Click **New Session** in the sidebar to start fresh |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns status and version |
| `GET` | `/status` | System stats — active sessions, docs processed, questions asked, uptime |
| `POST` | `/upload` | Upload documents — creates a new session, returns `session_id` |
| `POST` | `/process` | Process uploaded documents — builds FAISS vector index |
| `POST` | `/ask` | Ask a research question — runs the 3-agent pipeline, returns synthesis |
| `GET` | `/suggested-questions` | Generate suggested questions from indexed documents |
| `DELETE` | `/reset` | Delete session and clean up all resources |

Interactive docs: http://localhost:8000/docs

---

## Docker

### Build

```bash
docker build -t research-assistant:latest .
```

### Run

```bash
docker run -d \
  -p 8501:8501 \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your-key-here \
  research-assistant:latest
```

Both services start via `entrypoint.sh`. Streamlit is available at `:8501`, FastAPI at `:8000`.

---

## Kubernetes

### Prerequisites

- A running Kubernetes cluster (e.g. minikube, kind, or cloud provider)
- `kubectl` configured

### Create Secret

```bash
kubectl create secret generic research-assistant-secrets \
  --from-literal=GEMINI_API_KEY=your-key-here
```

### Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/service.yaml
```

### Check Status

```bash
kubectl get pods
kubectl get services
```

The deployment runs **2 replicas** with liveness and readiness probes on `GET /health`.

---

## CI/CD

GitHub Actions runs on every push and pull request to `main`:

1. Checkout code
2. Set up Python 3.12
3. `pip install -r requirements.txt`
4. `pytest tests/ -v --tb=short`
5. `docker build`
6. Start container, wait for `GET /health` → 200
7. Cleanup container

Workflow file: `.github/workflows/ci.yml`

---

## Configuration Reference

All settings are in `config/settings.py`.

| Setting | Value |
|---|---|
| Default model | `gemini-2.5-flash` |
| Embedding model | `models/text-embedding-004` |
| Embedding dimension | `768` |
| Target LLM calls per query | `1` |
| Max LLM calls budget | `2` |
| Chunk size | `1000` characters |
| Chunk overlap | `200` characters |
| Max documents | `100` |
| Max synthesis input papers | `8` |
| Max synthesis output tokens | `6000` |
| Synthesis temperature | `0.3` |
| Max full-text chars per paper | `30,000` |
| Supported file formats | `.pdf`, `.txt`, `.md` |
| Max file size | `50 MB` |

---

## Supported Research Domains

Domain classification is purely deterministic keyword scoring — no LLM call.

| Domain | Key terms |
|---|---|
| `machine_learning` | neural, training, deep learning, classification, optimization |
| `computer_vision` | image, detection, segmentation, convolutional, vision |
| `natural_language` | transformer, attention, tokenization, embedding, sentiment |
| `robotics` | navigation, manipulation, kinematics, motion planning |
| `cybersecurity` | encryption, vulnerability, threat, cryptography |
| `software_engineering` | architecture, design patterns, agile, devops, testing |
| `other` | fallback for unmatched queries |

---

## Synthesis Output Structure

Each `/ask` response returns a `synthesis` object with the following fields under `research_synthesis`:

| Field | Description |
|---|---|
| `key_findings` | Core technical discoveries with `[Paper N]` citation labels |
| `research_gaps` | Areas identified as needing further investigation |
| `methodology_insights` | Research method observations across papers |
| `technical_contributions` | Specific algorithmic innovations |
| `comparative_analysis` | Cross-paper comparisons |
| `practical_implications` | Real-world applications |
| `performance_metrics` | Quantitative results (accuracy, F1, baselines) |
| `confidence` | Retrieval confidence score `[0.0 – 1.0]` |
| `recommended_papers` | Ranked list of most relevant documents |

---

## Troubleshooting

**`streamlit` command not found**
```bash
python -m streamlit run streamlit_app.py
```

**Gemini API key error**
- Confirm `.env` exists and contains `GEMINI_API_KEY`
- On Streamlit Cloud: add the key under App Settings → Secrets
- Regenerate at https://aistudio.google.com/app/apikey

**FAISS installation fails**
```bash
pip install --upgrade faiss-cpu
```

**"No key findings in synthesis"**
- This means the LLM returned empty fields — ask a fresh question after re-uploading
- Check the **Raw JSON** tab to inspect the actual API response
- Use the fallback: the system auto-extracts sentences deterministically when Gemini fails

**Document fails to process**
- Must be `.pdf`, `.txt`, or `.md`
- Must not be password-protected or empty
- Must be under 50 MB
- Avoid special characters (`\ / : * ? " < > |`) in filenames

**Answer quality is low**
- Upload documents that contain terms relevant to your question
- Use the suggested questions — they are generated from your actual document content
- Check the **Metrics** tab for retrieval confidence score

---

## Requirements

- Python 3.10+
- Google Gemini API key
- Internet connection (for Gemini LLM and embedding API calls)
- Sufficient RAM for FAISS in-session indexing (512 MB minimum recommended)

---

## License

MIT
