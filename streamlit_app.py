"""
Research Assistant AI — Streamlit UI
Communicates with FastAPI backend at API_BASE_URL.
"""

import streamlit as st
import requests
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = 300

st.set_page_config(
    page_title="Research Assistant AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stTabs [data-baseweb="tab-list"] button { min-width: 150px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "api_session_id": None,
        "vector_store_ready": False,
        "documents_info": {},
        "last_answer": None,
        "api_healthy": None,
        "api_checked_at": None,
        "active_tab": 0,
        "selected_question": "",
        "ask_inner_tab": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ── API Helpers ───────────────────────────────────────────────────────────

def health_check() -> bool:
    """Check FastAPI health — cached for 10 seconds to avoid hammering API"""
    now = datetime.now()
    if (
        st.session_state.api_checked_at is not None
        and (now - st.session_state.api_checked_at).seconds < 10
        and st.session_state.api_healthy is not None
    ):
        return st.session_state.api_healthy

    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        result = response.status_code == 200
    except Exception:
        result = False

    st.session_state.api_healthy = result
    st.session_state.api_checked_at = now
    return result


def upload_documents(uploaded_files: list) -> Optional[Dict[str, Any]]:
    if not uploaded_files:
        st.error("No files selected")
        return None
    try:
        files = [("files", (f.name, f.getbuffer(), f.type)) for f in uploaded_files]
        with st.spinner("Uploading documents..."):
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=files,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            st.session_state.api_session_id = data["session_id"]
            st.session_state.documents_info = {
                "count": data["file_count"],
                "size_mb": data["total_size_mb"],
                "files": data["supported_files"],
            }
            return data
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"Upload failed: {detail}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE_URL}. Is FastAPI running?")
        return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        logger.error(f"Upload error: {e}", exc_info=True)
        return None


def process_documents() -> Optional[Dict[str, Any]]:
    if not st.session_state.api_session_id:
        st.error("No session found. Upload documents first.")
        return None
    try:
        with st.spinner("Processing documents — chunking, embedding, indexing..."):
            response = requests.post(
                f"{API_BASE_URL}/process",
                json={"session_id": st.session_state.api_session_id},
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                st.session_state.vector_store_ready = True
            return data
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"Processing failed: {detail}")
        return None
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        logger.error(f"Processing error: {e}", exc_info=True)
        return None


def ask_question(question: str, domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not st.session_state.vector_store_ready:
        st.error("Vector store not ready. Process documents first.")
        return None
    if not question.strip():
        st.error("Question cannot be empty")
        return None
    try:
        payload: Dict[str, Any] = {
            "session_id": st.session_state.api_session_id,
            "question": question,
        }
        if domain:
            payload["domain"] = domain

        with st.spinner("Generating synthesis..."):
            response = requests.post(
                f"{API_BASE_URL}/ask",
                json=payload,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            st.session_state.last_answer = data
            return data
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"Question failed: {detail}")
        return None
    except Exception as e:
        st.error(f"Question error: {str(e)}")
        logger.error(f"Question error: {e}", exc_info=True)
        return None


def get_suggested_questions() -> Optional[Dict[str, Any]]:
    if not st.session_state.vector_store_ready:
        st.error("Vector store not ready. Process documents first.")
        return None
    try:
        with st.spinner("Generating suggested questions..."):
            response = requests.get(
                f"{API_BASE_URL}/suggested-questions",
                params={"session_id": st.session_state.api_session_id},
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"Failed to get suggestions: {str(e)}")
        logger.error(f"Suggestions error: {e}", exc_info=True)
        return None


def reset_session() -> bool:
    if not st.session_state.api_session_id:
        return False
    try:
        requests.delete(
            f"{API_BASE_URL}/reset",
            params={"session_id": st.session_state.api_session_id},
            timeout=10
        )
    except Exception as e:
        logger.warning(f"Reset API call failed (ignoring): {e}")
    # Always clear local state regardless of API response
    for key in ["api_session_id", "vector_store_ready", "documents_info", "last_answer", "active_tab", "selected_question", "suggestions", "ask_inner_tab"]:
        st.session_state[key] = None if key == "api_session_id" else (
            False if key == "vector_store_ready" else (
                {} if key == "documents_info" else (
                    0 if key in ("active_tab", "ask_inner_tab") else (
                        [] if key == "suggestions" else (
                            "" if key == "selected_question" else None
                        )
                    )
                )
            )
        )
    return True


def get_system_status() -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


# ── UI Components ─────────────────────────────────────────────────────────

def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📚 Research Assistant AI")
        st.markdown("Upload research documents, ask questions, get citation-aware synthesis")
    with col2:
        if health_check():
            st.success("✅ API Connected")
        else:
            st.error(f"❌ API Offline ({API_BASE_URL})")


def render_sidebar():
    with st.sidebar:
        st.header("📋 Session Info")

        if st.session_state.api_session_id:
            st.info(f"**Session:** `{st.session_state.api_session_id}`")

            info = st.session_state.documents_info
            if info:
                st.markdown(f"**Files:** {info.get('count', 0)}")
                st.markdown(f"**Size:** {info.get('size_mb', 0):.2f} MB")

            if st.session_state.vector_store_ready:
                st.success("✅ Vector store ready")

            if st.button("🔄 New Session", use_container_width=True):
                reset_session()
                st.rerun()
        else:
            st.info("No active session. Upload documents to start.")

        st.divider()
        st.header("📊 System Stats")
        status = get_system_status()
        if status:
            st.metric("Active Sessions", status.get("active_sessions", 0))
            st.metric("Docs Processed", status.get("total_documents_processed", 0))
            st.metric("Questions Asked", status.get("total_questions_asked", 0))
            uptime_hrs = status.get("system_uptime_seconds", 0) / 3600
            st.metric("Uptime (hrs)", f"{uptime_hrs:.1f}")
        else:
            st.warning("Stats unavailable")


def render_upload_tab():
    st.header("📤 Step 1: Upload Research Documents")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        Supported formats: **PDF**, **TXT**, **Markdown**  
        Max size: 50 MB per file  
        Recommended: 3–10 related papers
        """)
    with col2:
        st.info("💡 More papers = better synthesis")

    uploaded_files = st.file_uploader(
        "Choose research documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📬 Upload", type="primary", use_container_width=True):
            if uploaded_files:
                result = upload_documents(uploaded_files)
                if result:
                    if result.get("unsupported_files"):
                        st.warning(f"Skipped: {result['unsupported_files']}")
                    st.session_state.active_tab = 1  # auto-switch to Process tab
                    st.rerun()
            else:
                st.warning("Please select at least one file")
    with col2:
        if st.session_state.api_session_id:
            st.info(f"✅ Session: `{st.session_state.api_session_id}`")


def render_process_tab():
    st.header("⚙️ Step 2: Process & Index Documents")

    if not st.session_state.api_session_id:
        st.warning("⬅️ Upload documents first (Step 1)")
        return

    if st.session_state.vector_store_ready:
        st.success("✅ Vector store ready")
        count = st.session_state.documents_info.get("count", 0)
        st.write(f"**{count}** document(s) indexed and ready for querying.")
        return

    st.markdown("""
    This step will:
    1. Extract text from your documents
    2. Split into chunks (1000 chars, 200 overlap)
    3. Generate Google embeddings
    4. Build FAISS vector index
    """)

    if st.button("🔨 Process Documents", type="primary", use_container_width=True):
        result = process_documents()
        if result and result.get("status") == "success":
            st.success(result.get("message", "Processing complete"))
            st.json({
                "documents_processed": result["documents_processed"],
                "time_seconds": result["processing_time_seconds"],
            })
            st.session_state.active_tab = 2  # jump to Ask tab
            st.rerun()


def render_ask_tab():
    st.header("❓ Step 3: Ask Research Questions")

    if not st.session_state.vector_store_ready:
        st.warning("⬅️ Process documents first (Step 2)")
        return

    # Inner tab toggle buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "✏️ Custom Question",
            type="primary" if st.session_state.ask_inner_tab == 0 else "secondary",
            use_container_width=True,
        ):
            st.session_state.ask_inner_tab = 0
            st.rerun()
    with col2:
        if st.button(
            "💡 Suggested Questions",
            type="primary" if st.session_state.ask_inner_tab == 1 else "secondary",
            use_container_width=True,
        ):
            st.session_state.ask_inner_tab = 1
            st.rerun()

    st.divider()

    if st.session_state.ask_inner_tab == 0:
        question = st.text_area(
            "Enter your research question",
            value=st.session_state.selected_question,
            placeholder="e.g., What are the key advances in transformer architectures?",
            height=120,
            max_chars=500,
            key="question_input",
        )
        domain = st.selectbox(
            "Research Domain (optional)",
            options=[None, "machine_learning", "computer_vision",
                     "natural_language", "robotics", "cybersecurity",
                     "software_engineering"],
            format_func=lambda x: "Auto-detect" if x is None
                                  else x.replace("_", " ").title()
        )
        if st.button("🔍 Ask & Synthesize", type="primary", use_container_width=True):
            if question.strip():
                st.session_state.selected_question = ""
                result = ask_question(question, domain)
                if result:
                    render_answer(result, key_suffix="current")
            else:
                st.warning("Please enter a question")

    else:
        if st.button("🔄 Generate Suggestions", use_container_width=True):
            suggestions = get_suggested_questions()
            if suggestions:
                st.session_state.suggestions = suggestions.get("questions", [])

        suggestions = st.session_state.get("suggestions", [])
        if suggestions:
            st.markdown(f"**{len(suggestions)} suggestions — click any to use it:**")
            for idx, q in enumerate(suggestions):
                if st.button(f"▶ {q}", key=f"sug_{idx}", use_container_width=True):
                    st.session_state.selected_question = q
                    st.session_state.ask_inner_tab = 0  # switch to Custom Question
                    st.rerun()
        else:
            st.info("Click 'Generate Suggestions' to get question ideas from your documents.")

    # Show previous answer
    if st.session_state.last_answer:
        st.divider()
        st.subheader("📖 Previous Result")
        render_answer(st.session_state.last_answer, key_suffix="previous")


def render_answer(result: Dict[str, Any], key_suffix: str = ""):
    """Render research answer with metrics and synthesis tabs."""
    st.divider()
    st.subheader("📖 Research Synthesis")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Processing Time (s)", f"{result.get('processing_time_seconds', 0):.1f}")
    with col2:
        q = result.get("question", "")
        st.metric("Question", (q[:40] + "...") if len(q) > 40 else q)

    # The API wraps everything under result["synthesis"]["research_synthesis"]
    raw = result.get("synthesis", {})
    synthesis = raw.get("research_synthesis", raw)  # fallback to raw if flat

    if not synthesis:
        st.warning("No synthesis returned")
        return

    # key_findings may be list of dicts {text, source} or plain strings
    def extract_text(item):
        return item.get("text", str(item)) if isinstance(item, dict) else str(item)

    fields = {
        "Findings":      [extract_text(f) for f in synthesis.get("key_findings", [])],
        "Gaps":          synthesis.get("research_gaps", []),
        "Contributions": synthesis.get("technical_contributions", []) or synthesis.get("methodology_insights", []),
        "Analysis":      synthesis.get("comparative_analysis", []) or synthesis.get("methodology_insights", []),
        "Metrics":       synthesis.get("performance_metrics", []),
    }

    syn_tabs = st.tabs([*fields.keys(), "Raw JSON"])

    for tab, (label, items) in zip(syn_tabs[:5], fields.items()):
        with tab:
            if items:
                for i, item in enumerate(items, 1):
                    st.write(f"{i}. {item}")
            else:
                st.info(f"No {label.lower()} in synthesis")

    with syn_tabs[5]:
        st.json(synthesis)

    st.download_button(
        label="💾 Download as JSON",
        data=json.dumps(result, indent=2),
        file_name=f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key=f"download_btn_{key_suffix}",
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    render_header()
    render_sidebar()

    active = st.session_state.active_tab

    # Step progress indicator
    cols = st.columns(3)
    steps = [("📤 Upload", 0), ("⚙️ Process", 1), ("❓ Ask", 2)]
    for col, (label, idx) in zip(cols, steps):
        with col:
            if idx == active:
                st.success(f"▶ {label}")
            elif idx < active:
                st.info(f"✅ {label}")
            else:
                st.warning(f"○ {label}")

    st.divider()

    if active == 0:
        render_upload_tab()
    elif active == 1:
        render_process_tab()
    elif active == 2:
        render_ask_tab()


if __name__ == "__main__":
    main()