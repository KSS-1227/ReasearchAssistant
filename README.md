# Research Assistant AI

A production-grade, full-stack research synthesis platform. Upload academic documents, authenticate via Supabase, and receive citation-aware synthesis through a deterministic multi-agent RAG pipeline powered by Google Gemini — all through a guided, zero-friction 4-step UI.

> Live demo: https://reasearchassistant.streamlit.app/

---

## Overview

| What it does | How |
|---|---|
| Authenticates users | Supabase Auth (JWT, email/password) |
| Accepts PDF, TXT, Markdown documents | FastAPI `/upload` endpoint, temp file storage |
| Builds a searchable vector index | FAISS + Google `text-embedding-004` |
| Retrieves relevant document chunks | Deterministic dynamic-k FAISS search + re-ranking |
| Extracts citations and key quotes | Regex-based CitationExtractor (0 LLM calls) |
| Synthesises a structured research answer | Single Gemini LLM call with `[Paper N]` citation index |
| Persists query history per user | Supabase `queries` table, JWT-gated `/history` endpoint |
| Exposes everything as a REST API | FastAPI with interactive Swagger docs at `/docs` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  streamlit_app.py                                               │
│                                                                 │
│  Auth Gate (Supabase anon key)                                  │
│  ├── Login / Sign Up page (JWT stored in session_state)         │
│  └── Authenticated app                                          │
│       ├── Step 1 — Upload                                       │
│       ├── Step 2 — Process                                      │
│       ├── Step 3 — Ask  (Custom + Suggested Questions)          │
│       └── Step 4 — History (JWT → Bearer token → /history)     │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP  (requests library)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│  fastapi_app.py                                                 │
│                                                                 │
│  In-memory SessionManager (per-user coordinator instances)      │
│  Supabase service-key client (server-side only)                 │
│                                                                 │
│  POST   /upload              Create session, save temp files    │
│  POST   /process             Chunk → embed → FAISS index        │
│  POST   /ask                 Run 3-agent pipeline, save to DB   │
│  GET    /suggested-questions FAISS search → question templates  │
│  GET    /history             JWT-verified query history         │
│  GET    /health              Liveness check                     │
│  GET    /status              Active sessions + system stats     │
│  DELETE /reset               Destroy session + clean up files   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ResearchCoordinator                           │
│  core/coordinator.py                                            │
│                                                                 │
│  Step 1 ── LiteratureScanner      0 LLM calls                  │
│            Dynamic-k FAISS search, relevance re-ranking         │
│                                                                 │
│  Step 2 ── CitationExtractor      0 LLM calls                  │
│            Regex citations, key quotes, author/venue stats      │
│                                                                 │
│  Step 3 ── SynthesisAgent         1 LLM call                   │
│            [Paper N] citation index → Gemini prompt             │
│            Deterministic fallback on API failure                │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  Google Gemini API         Supabase
  (LLM + Embeddings)        (Auth + queries table)
```

**Design rule:** exactly **1 Gemini LLM call per user query**. Retrieval, citation extraction, and domain classification are fully deterministic.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend UI | Streamlit ≥ 1.35 |
| Backend API | FastAPI + Uvicorn |
| Authentication | Supabase Auth (JWT, email/password) |
| Database | Supabase (PostgreSQL — `queries` table) |
| LLM | Google Gemini via `google-genai` |
| Default model | `gemini-2.5-flash` |
| Embeddings | Google `models/text-embedding-004` (768-dim) |
| Vector search | FAISS CPU |
| Document loading | LangChain `PyPDFLoader`, `TextLoader` |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Data validation | Pydantic v2 |
| Environment config | `python-dotenv` |
| Containerisation | Docker (Python 3.11-slim) |
| Orchestration | Kubernetes |
| CI/CD | GitHub Actions |
| Testing | Pytest |

---

## Project Structure

```
ReasearchAssistant/
│
├── streamlit_app.py              # Full-stack UI — auth gate, 4-step workflow,
│                                 # Supabase login/signup, JWT session, history tab
│
├── fastapi_app.py                # REST API — session management, pipeline trigger,
│                                 # Supabase service-key client, /history endpoint
│
├── agents/
│   ├── base_agent.py             # Abstract agent base — execute_with_tracking,
│   │                             # performance metrics, success rate tracking
│   ├── literature_scanner.py     # FAISS retrieval, dynamic-k selection,
│   │                             # relevance re-ranking (0 LLM calls)
│   ├── citation_extractor.py     # Regex citation parsing, key quote mining,
│   │                             # author/venue/year statistics (0 LLM calls)
│   └── synthesis_agent.py        # [Paper N] citation index builder, Gemini
│                                 # synthesis prompt, deterministic fallback (1 LLM call)
│
├── core/
│   ├── coordinator.py            # 3-agent pipeline orchestrator — domain
│   │                             # classification, confidence scoring, async logging
│   ├── document_processor.py     # File validation, PDF/TXT/MD loading,
│   │                             # 1000-char chunking, FAISS index construction
│   ├── google_embeddings.py      # Google text-embedding-004 wrapper
│   ├── llm_interface.py          # Gemini client — retry logic, token counting,
│   │                             # real-time cost tracking from API metadata
│   ├── llm_interface_fixed.py    # Patched LLM interface variant
│   ├── memory.py                 # In-session ResearchSession and SystemMetrics
│   ├── models.py                 # Pydantic + dataclass models — Paper,
│   │                             # ResearchSynthesis, AgentMetrics, etc.
│   ├── pipeline_logger.py        # Async structured JSON logging →
│   │                             # logs/pipeline_logs.jsonl
│   └── prompts.py                # Citation-aware synthesis prompt builder —
│                                 # injects SOURCE INDEX block into Gemini prompt
│
├── config/
│   └── settings.py               # SystemConfig, DOMAIN_KEYWORDS, Gemini
│                                 # pricing table, chunk/synthesis parameters,
│                                 # structured logging configuration
│
├── data/                         # 13 sample transformer/attention research PDFs
│   ├── Attention Is All You Need.pdf
│   ├── Longformer The Long-Document Transformer.pdf
│   ├── AN IMAGE IS WORTH 16X16 WORDS.pdf
│   └── ...                       # (10 additional papers)
│
├── k8s/
│   ├── deployment.yaml           # 2-replica Deployment — resource limits,
│   │                             # liveness + readiness probes on /health
│   ├── api-service.yaml          # Kubernetes Service for FastAPI (:8000)
│   └── service.yaml              # Kubernetes Service for Streamlit (:8501)
│
├── tests/
│   ├── __init__.py
│   └── test_app.py               # Pytest suite — env template, SystemConfig,
│                                 # coordinator imports, requirements, project structure
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                # CI: checkout → Python 3.12 → install →
│   │   │                         # pytest → docker build → health check → cleanup
│   │   ├── deploy.yml            # CD: production deployment pipeline
│   │   └── openhands.yml         # OpenHands autonomous agent workflow
│   └── scripts/
│       └── issue_agent.py        # GitHub issue triage automation
│
├── logs/
│   └── pipeline_logs.jsonl       # Runtime query logs — auto-generated, git-ignored
│
├── Dockerfile                    # Python 3.11-slim, exposes :8501 + :8000,
│                                 # HEALTHCHECK on Streamlit /_stcore/health
├── entrypoint.sh                 # Startup script — launches FastAPI, waits for
│                                 # /health readiness, then launches Streamlit
├── requirements.txt              # All Python dependencies with pinned minimums
├── env_template.txt              # Environment variable reference (no secrets)
├── .env                          # Local secrets — git-ignored, never committed
├── .gitignore
├── .dockerignore
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
```powershell
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

Fill in your values — see the [Environment Variables](#environment-variables) section for what each key does. The `.env` file is git-ignored and must never be committed.

### 5. Start FastAPI Backend

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start Streamlit UI

Open a second terminal with the venv active:

```bash
python -m streamlit run streamlit_app.py
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

---

## User Workflow

```
Login / Sign Up
      │
      ▼
Step 1 — Upload
  Select .pdf / .txt / .md files → Upload
  → Auto-navigates to Step 2
      │
      ▼
Step 2 — Process
  Click "Process Documents"
  Chunks text → generates embeddings → builds FAISS index
  → Auto-navigates to Step 3
      │
      ▼
Step 3 — Ask
  ┌── Custom Question: type your own question
  └── Suggested Questions: AI-generated from your docs
       → click any suggestion to auto-paste it
  Click "Ask & Synthesize"
  → Structured result: Findings / Gaps / Contributions /
    Analysis / Metrics / Raw JSON + Download as JSON
      │
      ▼
Step 4 — History
  Past queries with key findings and full synthesis JSON
  Gated by JWT — expires gracefully with re-login prompt
```

---

## API Reference

All endpoints are documented interactively at `http://localhost:8000/docs`.

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | None | Liveness check — returns status and version |
| `GET` | `/status` | None | Active sessions, docs processed, uptime |
| `POST` | `/upload` | None | Upload documents — creates session, returns `session_id` |
| `POST` | `/process` | None | Chunk, embed, and build FAISS index |
| `POST` | `/ask` | None | Run 3-agent pipeline, return structured synthesis |
| `GET` | `/suggested-questions` | None | Generate questions from indexed documents |
| `GET` | `/history` | Bearer JWT | Return last 20 queries for the authenticated user |
| `DELETE` | `/reset` | None | Delete session and remove temp files |

### Example `/ask` request

```json
{
  "session_id": "a1b2c3d4",
  "question": "What are the key advances in sparse attention mechanisms?",
  "domain": "natural_language"
}
```

### Example `/ask` response structure

```json
{
  "session_id": "a1b2c3d4",
  "question": "...",
  "synthesis": {
    "research_synthesis": {
      "key_findings": [{ "text": "...", "source": { "label": "[Paper 1]", "title": "..." } }],
      "research_gaps": ["..."],
      "technical_contributions": ["..."],
      "comparative_analysis": ["..."],
      "performance_metrics": ["..."],
      "methodology_insights": ["..."],
      "confidence": 0.87
    },
    "performance_metrics": {
      "total_llm_calls": 1,
      "papers_analyzed": 5,
      "retrieval_confidence": 0.83,
      "estimated_cost": 0.0004
    }
  },
  "processing_time_seconds": 8.4
}
```

---

## Environment Variables

Copy `env_template.txt` to `.env` and fill in the values below. The template file contains descriptions for each variable and is safe to commit — it contains no secrets.

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google AI Studio API key for LLM and embeddings |
| `SUPABASE_URL` | For auth | Your Supabase project URL |
| `SUPABASE_ANON_KEY` | For auth | Public anon key — safe for frontend use |
| `SUPABASE_SERVICE_KEY` | For history | Service role key — **server-side only**, never in UI |
| `API_BASE_URL` | No | FastAPI base URL (default: `http://localhost:8000`) |
| `API_HOST` | No | FastAPI bind host (default: `0.0.0.0`) |
| `API_PORT` | No | FastAPI port (default: `8000`) |

> The Supabase service key is used exclusively in `fastapi_app.py` to verify JWTs and write to the `queries` table. It is never loaded in `streamlit_app.py`.

---

## Supabase Setup

1. Create a project at https://supabase.com
2. Create a `queries` table in your Supabase database with the following columns:
   - `id` — uuid, primary key
   - `user_id` — uuid, references `auth.users`
   - `session_id` — text
   - `question` — text
   - `synthesis` — jsonb
   - `processing_time_seconds` — float
   - `created_at` — timestamptz, default `now()`
3. Enable Row Level Security on the `queries` table so users can only read their own rows
4. Copy the Project URL, anon key, and service role key from Settings → API into your `.env`

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
  -e GEMINI_API_KEY=<your-key> \
  -e SUPABASE_URL=<your-url> \
  -e SUPABASE_ANON_KEY=<your-anon-key> \
  -e SUPABASE_SERVICE_KEY=<your-service-key> \
  research-assistant:latest
```

`entrypoint.sh` starts FastAPI first, waits for `/health` to return 200, then starts Streamlit. Both services run in the same container. If either process exits, the container shuts down cleanly.

---

## Kubernetes

### Create Secret

```bash
kubectl create secret generic research-assistant-secrets \
  --from-literal=GEMINI_API_KEY=<your-key> \
  --from-literal=SUPABASE_URL=<your-url> \
  --from-literal=SUPABASE_SERVICE_KEY=<your-service-key>
```

### Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/service.yaml
```

### Verify

```bash
kubectl get pods
kubectl get services
kubectl describe deployment research-assistant
```

The deployment runs 2 replicas. Liveness and readiness probes hit `GET /health` on port 8000. Resource limits: 512Mi–1Gi memory, 250m–1000m CPU.

---

## CI/CD

GitHub Actions triggers on every push and pull request to `main`.

**Pipeline steps (`.github/workflows/ci.yml`):**

1. Checkout code
2. Set up Python 3.12
3. `pip install -r requirements.txt`
4. `pytest tests/ -v --tb=short`
5. `docker build -t research-assistant:<sha> .`
6. Start container with a test API key
7. Poll `GET /health` up to 60 seconds
8. Assert HTTP 200 — dump container logs on any failure
9. Stop and remove test container (`if: always()`)

---

## Configuration Reference

All values are in `config/settings.py`.

| Parameter | Value |
|---|---|
| Default LLM | `gemini-2.5-flash` |
| Embedding model | `models/text-embedding-004` |
| Embedding dimension | 768 |
| Target LLM calls per query | 1 |
| Max LLM calls budget | 2 |
| Chunk size | 1000 characters |
| Chunk overlap | 200 characters |
| Max documents per session | 100 |
| Max file size | 50 MB |
| Max synthesis input papers | 8 |
| Max full-text chars per paper | 30,000 |
| Max synthesis output tokens | 6,000 |
| Synthesis temperature | 0.3 |
| Supported formats | `.pdf` `.txt` `.md` |
| Session timeout | 30 minutes |

---

## Supported Research Domains

Domain classification uses deterministic keyword scoring — no LLM call.

| Domain key | Example keywords |
|---|---|
| `machine_learning` | neural, training, deep learning, classification, optimization |
| `computer_vision` | image, detection, segmentation, convolutional, visual |
| `natural_language` | transformer, attention, tokenization, embedding, translation |
| `robotics` | navigation, manipulation, motion planning, kinematics |
| `cybersecurity` | encryption, vulnerability, threat, intrusion, cryptography |
| `software_engineering` | architecture, design patterns, agile, devops, testing |
| `other` | fallback when no domain keywords match |

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

The test suite covers:
- `env_template.txt` existence and content
- `SystemConfig` imports and method presence
- `ResearchCoordinator` import and instantiation behaviour
- `requirements.txt` existence and required package presence
- Project folder structure (`core/`, `agents/`, `config/`)

---

## Troubleshooting

**`streamlit` command not found**
```bash
python -m streamlit run streamlit_app.py
```

**Gemini API errors**
- Confirm `GEMINI_API_KEY` is set in `.env`
- On Streamlit Cloud: add it under App Settings → Secrets
- Regenerate at https://aistudio.google.com/app/apikey

**Supabase auth not working**
- Confirm `SUPABASE_URL` and `SUPABASE_ANON_KEY` are set
- The anon key must be the public key, not the service role key
- Check that email confirmation settings in Supabase Auth match your expectations

**"No key findings in synthesis"**
- Ask a fresh question — the previous result in session may be stale
- Open the Raw JSON tab to inspect the actual API response
- The fallback synthesiser activates automatically when Gemini fails

**FAISS installation fails**
```bash
pip install --upgrade faiss-cpu
```

**Document fails to process**
- Must be `.pdf`, `.txt`, or `.md`
- Must not be password-protected or empty
- Must be under 50 MB
- Avoid special characters in filenames

**History tab shows "Session expired"**
- This is expected behaviour when the JWT has expired
- Log out and log in again to get a fresh token

---

## Requirements

- Python 3.10+
- Google Gemini API key
- Supabase project (for auth and history — optional for local testing)
- Internet connection for Gemini LLM and embedding API calls
- 512 MB RAM minimum for FAISS in-session indexing

---

## License

MIT
