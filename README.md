# Research Assistant AI

A Streamlit research assistant that lets you upload academic documents, ask questions, and receive citation-aware synthesis from a deterministic multi-agent RAG pipeline.

Live demo: https://reasearchassistant.streamlit.app/

## What It Does

- Upload PDF, TXT, or Markdown research documents.
- Build an in-session FAISS vector index using Google embeddings.
- Generate suggested research questions from uploaded content.
- Ask custom or follow-up questions through a Streamlit inbox.
- Use browser speech-to-text from the question inbox microphone button.
- Retrieve relevant chunks with deterministic ranking and reranking.
- Extract citations, key quotes, source metadata, and citation-network signals.
- Generate a structured synthesis with Gemini and source labels such as `[Paper 1]`.
- Show traceable results across literature discovery, citation analysis, quotes, gaps, limitations, and metrics.

## Current Architecture

```text
Uploaded documents
      |
      v
DocumentProcessor
  - validates uploads
  - loads PDF/TXT/MD files
  - chunks text
  - creates Google embeddings
  - builds FAISS index
      |
      v
User question
      |
      v
ResearchCoordinator
  - validates query
  - classifies domain with keyword rules
  - runs the agent pipeline
      |
      +--> LiteratureScanner      0 LLM calls
      |    FAISS search, dynamic-k retrieval, reranking
      |
      +--> CitationExtractor      0 LLM calls
      |    regex citation parsing, key quote extraction, author stats
      |
      +--> SynthesisAgent         1 LLM call
           Gemini synthesis with a citation-aware prompt
```

The main research answer is designed around one synthesis LLM call per user query. Suggested-question generation is a separate helper step and may use an additional LLM call when recommendations are generated.

## Tech Stack

| Layer | Technology |
| --- | --- |
| UI | Streamlit |
| LLM | Google Gemini via `google-genai` |
| Default model | `gemini-2.5-flash` |
| Embeddings | `models/text-embedding-004` |
| Vector search | FAISS CPU |
| Document loading | LangChain loaders |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Validation | Pydantic v2 |
| Environment | `python-dotenv` |

## Project Structure

```text
.
|-- streamlit_app.py              # Streamlit UI, 3-step workflow, speech-to-text inbox
|-- agents/
|   |-- base_agent.py             # Agent base class and metrics
|   |-- literature_scanner.py     # FAISS retrieval and deterministic reranking
|   |-- citation_extractor.py     # Citation, quote, and metadata extraction
|   `-- synthesis_agent.py        # Gemini synthesis and fallback synthesis
|-- core/
|   |-- coordinator.py            # Pipeline orchestration
|   |-- document_processor.py     # Upload validation, loading, chunking, indexing
|   |-- google_embeddings.py      # Google embedding wrapper
|   |-- llm_interface.py          # Gemini client, retries, token/cost tracking
|   |-- memory.py                 # Session and agent metrics
|   |-- models.py                 # Shared data models
|   |-- pipeline_logger.py        # Structured query logging
|   `-- prompts.py                # Citation-aware synthesis prompts
|-- config/
|   `-- settings.py               # Models, limits, pricing, domain keywords
|-- data/                         # Sample/reference documents
|-- logs/                         # Runtime logs
|-- env_template.txt              # Environment variable template
|-- requirements.txt              # Python dependencies
|-- Documentation.pdf
|-- USECASE_DIAGRAM.md
`-- usecase_diagram.puml
```

## Quick Start

### 1. Clone and Enter the Project

```bash
git clone https://github.com/KSS-1227/ReasearchAssistant.git
cd ReasearchAssistant
```

If you are using the downloaded project folder directly, run commands from the project root.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```bash
cp env_template.txt .env
```

Set your Gemini key:

```env
GEMINI_API_KEY=your-gemini-api-key
```

You can create a key from Google AI Studio: https://aistudio.google.com/app/apikey

### 5. Run the App

```bash
python -m streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## How To Use

1. Upload documents in Step 1.
   Supported formats are `.pdf`, `.txt`, and `.md`.

2. Process the documents in Step 2.
   The app validates files, extracts text, chunks content, creates embeddings, and builds the FAISS index.

3. Ask a question in Step 3.
   Use a suggested question, type your own, or click the microphone button in the top-right of the inbox to dictate your question.

4. Review the result tabs.
   The app shows literature matches, citations, key quotes, research gaps, limitations, and performance metrics.

5. Ask follow-up questions.
   After the first answer, a follow-up inbox appears with the same speech-to-text support.

## Speech-To-Text Notes

The microphone feature uses the browser Web Speech API through a lightweight Streamlit component. It does not require server-side audio packages, so it is suitable for Streamlit Cloud.

Important browser notes:

- Works best in Chrome or Edge.
- Requires microphone permission from the browser.
- Streamlit Cloud works because it is served over HTTPS.
- Some browsers, especially Firefox, may not support the Web Speech API.

## Configuration Highlights

Most settings live in `config/settings.py`.

| Setting | Current value |
| --- | --- |
| Default model | `gemini-2.5-flash` |
| Target LLM calls per query | `1` |
| Max LLM calls per query budget | `2` |
| Chunk size | `1000` characters |
| Chunk overlap | `200` characters |
| Supported formats | `.pdf`, `.txt`, `.md` |
| Vector store | `FAISS` |
| Embedding model | `models/text-embedding-004` |
| Embedding dimension | `768` |
| Max documents | `100` |
| Max synthesis input papers | `8` |
| Max synthesis output tokens | `6000` |
| Synthesis temperature | `0.3` |

## Supported Research Domains

Domain classification is deterministic keyword scoring, not an LLM call.

- `machine_learning`
- `computer_vision`
- `natural_language`
- `robotics`
- `cybersecurity`
- `software_engineering`
- `other`

## Cost Behavior

The app tracks Gemini token usage through API metadata and calculates cost using the pricing table in `config/settings.py`. Document processing and retrieval do not use synthesis LLM calls, but embeddings do use the Google embeddings API.

Typical research questions are intended to stay very low cost because only the final synthesis step uses the LLM.

## Deployment On Streamlit Cloud

1. Push the repository to GitHub.
2. Create a Streamlit Cloud app from the repo.
3. Set the main file to:

```text
streamlit_app.py
```

4. Add this secret in Streamlit Cloud:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

5. Deploy.

## Troubleshooting

### `streamlit` command not found

Run Streamlit through Python:

```bash
python -m streamlit run streamlit_app.py
```

### Gemini API key error

- Confirm `.env` exists locally.
- Confirm `GEMINI_API_KEY` is set.
- On Streamlit Cloud, confirm the key is configured in app secrets.
- Regenerate or verify the key in Google AI Studio.

### FAISS installation fails

Install or upgrade the FAISS CPU package:

```bash
pip install --upgrade faiss-cpu
```

### Uploaded document fails to process

Check that:

- The file is PDF, TXT, or MD.
- The file is not empty.
- The file is not password-protected.
- The filename avoids characters such as `\ / : * ? " < > |`.

### Microphone appears but does not type

- Refresh the page after deployment.
- Use Chrome or Edge.
- Allow microphone permission in the browser prompt.
- Make sure the page is served over HTTPS when deployed.

### Answer is weak or fallback-like

- Ask a question that uses terms present in the uploaded documents.
- Upload more relevant documents.
- Increase the max result slider before asking.
- Check the Performance Metrics tab for confidence and retrieval details.

## Development Commands

Syntax check:

```bash
python -m py_compile streamlit_app.py
```

Run locally:

```bash
python -m streamlit run streamlit_app.py
```

Check git changes:

```bash
git status --short
```

## Requirements

- Python 3.10+
- Gemini API key
- Internet connection for Gemini and embedding calls
- Enough memory for in-session FAISS indexing

# #   C I / C D   P i p e l i n e   A c t i v e  
 