"""
Research Assistant AI — Streamlit UI
Communicates with FastAPI backend at API_BASE_URL.
Auth via Supabase (anon key, client-side only).
"""

import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from streamlit_cookies_manager import EncryptedCookieManager

# ── Cookie Manager ────────────────────────────────────────────────────────
# Must be initialised before set_page_config and any st calls
cookies = EncryptedCookieManager(
    prefix="research_assistant_",
    password=os.getenv("COOKIE_SECRET", "research-assistant-cookie-secret-key")
)
if not cookies.ready():
    st.stop()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = 300

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")


st.set_page_config(
    page_title="Research Assistant AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { padding-top: 1rem; }

    /* Kill red border and Press Enter tooltip on password/text inputs */
    div[data-baseweb="input"] {
        border-color: #3a3a3a !important;
        box-shadow: none !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #555 !important;
        box-shadow: none !important;
    }
    [data-testid="InputInstructions"],
    div[data-testid="InputInstructions"] {
        display: none !important;
    }

    /* Replace Streamlit's default red primary button with professional blue */
    .stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Supabase Client ────────────────────────────────────────────────────────

def init_supabase_client():
    """Initialise Supabase client using anon key only (safe for frontend)."""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except ImportError:
        logger.error("supabase package not installed — run: pip install supabase")
        return None
    except Exception as e:
        logger.error(f"Supabase init failed: {e}")
        return None

supabase = init_supabase_client()

# ── Session State ──────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        # auth
        "is_authenticated": False,
        "access_token": None,
        "user_email": None,
        # research workflow
        "api_session_id": None,
        "vector_store_ready": False,
        "documents_info": {},
        "last_answer": None,
        "api_healthy": None,
        "api_checked_at": None,
        "active_tab": 0,
        "selected_question": "",
        "ask_inner_tab": 0,
        "suggestions": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


def restore_from_cookies():
    """Restore auth session from browser cookies on page refresh."""
    token = cookies.get("access_token", "")
    email = cookies.get("user_email", "")
    if token and not st.session_state.get("is_authenticated"):
        try:
            if supabase:
                user = supabase.auth.get_user(token)
                if user and user.user:
                    st.session_state.is_authenticated = True
                    st.session_state.access_token = token
                    st.session_state.user_email = email
        except Exception:
            # Token invalid or expired — clear cookies silently
            cookies["access_token"] = ""
            cookies["user_email"] = ""
            cookies.save()

restore_from_cookies()

# ── Auth Helpers ───────────────────────────────────────────────────────────

def logout():
    """Sign out from Supabase, clear session state and cookies."""
    if supabase and st.session_state.access_token:
        try:
            supabase.auth.sign_out()
        except Exception as e:
            logger.warning(f"Supabase sign_out error (ignoring): {e}")

    # Clear cookies
    cookies["access_token"] = ""
    cookies["user_email"] = ""
    cookies.save()
    for key in [
        "is_authenticated", "access_token", "user_email",
        "api_session_id", "vector_store_ready", "documents_info",
        "last_answer", "api_healthy", "api_checked_at",
        "active_tab", "selected_question", "ask_inner_tab", "suggestions",
    ]:
        default_map = {
            "is_authenticated": False,
            "vector_store_ready": False,
            "documents_info": {},
            "active_tab": 0,
            "ask_inner_tab": 0,
            "suggestions": [],
            "selected_question": "",
        }
        st.session_state[key] = default_map.get(key, None)

def _auth_headers() -> Dict[str, str]:
    """Return Authorization header if a token is present."""
    token = st.session_state.access_token
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

# ── Auth UI ────────────────────────────────────────────────────────────────

def render_login_form():
    """Email + password login form."""
    st.subheader("Login")
    email = st.text_input("Email", key="login_email", placeholder="you@example.com")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", type="primary", use_container_width=True, key="login_btn"):
        if not email.strip() or not password:
            st.error("Please enter email and password.")
            return
        if supabase is None:
            st.error("Cannot connect to authentication service. Check SUPABASE_URL and SUPABASE_ANON_KEY.")
            return
        try:
            with st.spinner("Signing in..."):
                resp = supabase.auth.sign_in_with_password(
                    {"email": email.strip(), "password": password}
                )
            if resp.user and resp.session:
                st.session_state.is_authenticated = True
                st.session_state.access_token = resp.session.access_token
                st.session_state.user_email = resp.user.email
                cookies["access_token"] = resp.session.access_token
                cookies["user_email"] = resp.user.email
                cookies.save()
                with st.spinner("Loading your workspace..."):
                    import time
                    time.sleep(0.5)
                st.rerun()
            else:
                st.error("Login failed. Please check your credentials.")
        except Exception as e:
            msg = str(e).lower()
            if "invalid" in msg or "credentials" in msg or "password" in msg:
                st.error("Invalid email or password.")
            elif "not confirmed" in msg or "email" in msg and "confirm" in msg:
                st.warning("Please confirm your email before logging in.")
            elif "network" in msg or "connection" in msg:
                st.error("Cannot connect to authentication service.")
            else:
                st.error(f"Login error: {e}")


def render_signup_form():
    """Email + password sign-up form with validation."""
    st.subheader("Create Account")
    email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
    password = st.text_input("Password", type="password", key="signup_password",
                              help="Minimum 6 characters")
    confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")

    if st.button("Sign Up", type="primary", use_container_width=True, key="signup_btn"):
        if not email.strip() or not password or not confirm:
            st.error("All fields are required.")
            return
        if len(password) < 6:
            st.error("Password must be at least 6 characters.")
            return
        if password != confirm:
            st.error("Passwords do not match.")
            return
        if supabase is None:
            st.error("Cannot connect to authentication service. Check SUPABASE_URL and SUPABASE_ANON_KEY.")
            return
        try:
            with st.spinner("Creating account..."):
                resp = supabase.auth.sign_up(
                    {"email": email.strip(), "password": password}
                )
            if resp.user:
                # If email confirmation is enabled, session will be None
                if resp.session:
                    st.session_state.is_authenticated = True
                    st.session_state.access_token = resp.session.access_token
                    st.session_state.user_email = resp.user.email
                    st.rerun()
                else:
                    st.success("Account created! Check your email to confirm before logging in.")
            else:
                st.error("Sign up failed. Please try again.")
        except Exception as e:
            msg = str(e).lower()
            if "already registered" in msg or "already exists" in msg:
                st.error("An account with this email already exists. Please log in.")
            elif "network" in msg or "connection" in msg:
                st.error("Cannot connect to authentication service.")
            elif "password" in msg:
                st.error(f"Password error: {e}")
            else:
                st.error(f"Sign up error: {e}")


def render_login_page():
    """Centered login/signup card shown when not authenticated."""
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        st.markdown("## 📚 Research Assistant AI")
        st.markdown("Citation-aware research synthesis powered by Gemini")
        st.divider()

        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
        with tab_login:
            render_login_form()
        with tab_signup:
            render_signup_form()

        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            st.warning(
                "⚠️ SUPABASE_URL or SUPABASE_ANON_KEY not set. "
                "Auth is disabled — set these in your .env file."
            )

# ── History Tab ────────────────────────────────────────────────────────────

def render_history_tab():
    """Fetch and display past queries from the /history endpoint."""
    st.header("📜 Query History")

    if not st.session_state.access_token:
        st.warning("Not authenticated.")
        return

    try:
        with st.spinner("Loading history..."):
            resp = requests.get(
                f"{API_BASE_URL}/history",
                headers=_auth_headers(),
                timeout=15,
            )

        if resp.status_code == 401:
            st.error("Session expired. Please log in again.")
            logout()
            st.rerun()
            return

        resp.raise_for_status()
        data = resp.json()
        queries = data.get("queries", [])

    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE_URL}.")
        return
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return

    if not queries:
        st.info("No history yet. Ask your first research question to see it here.")
        return

    st.markdown(f"**{len(queries)} past queries**")

    for i, entry in enumerate(queries):
        question = entry.get("question", "Unknown question")
        timestamp = entry.get("created_at", "")
        if timestamp:
            try:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                timestamp = ts.strftime("%d %b %Y, %H:%M")
            except Exception:
                pass

        synthesis = entry.get("synthesis", {})
        rs = synthesis.get("research_synthesis", synthesis) if isinstance(synthesis, dict) else {}
        findings = rs.get("key_findings", []) if isinstance(rs, dict) else []

        def extract_text(item):
            return item.get("text", str(item)) if isinstance(item, dict) else str(item)

        with st.expander(f"**{question[:80]}{'...' if len(question) > 80 else ''}** — {timestamp}", expanded=False):
            st.markdown(f"**Question:** {question}")
            if timestamp:
                st.caption(f"Asked: {timestamp}")

            if findings:
                st.markdown("**Key Findings:**")
                for finding in findings[:3]:
                    st.write(f"• {extract_text(finding)}")

            with st.expander("View Full Synthesis JSON"):
                st.json(synthesis if synthesis else {"message": "No synthesis data stored."})

# ── API Helpers ────────────────────────────────────────────────────────────

def health_check() -> bool:
    now = datetime.now()
    if (
        st.session_state.api_checked_at is not None
        and (now - st.session_state.api_checked_at).seconds < 30
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
                timeout=API_TIMEOUT,
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
                timeout=API_TIMEOUT,
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
        if st.session_state.get("access_token"):
            payload["user_token"] = st.session_state.access_token
        with st.spinner("Generating synthesis..."):
            response = requests.post(
                f"{API_BASE_URL}/ask",
                json=payload,
                timeout=API_TIMEOUT,
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
                timeout=API_TIMEOUT,
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
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"Reset API call failed (ignoring): {e}")
    for key in [
        "api_session_id", "vector_store_ready", "documents_info",
        "last_answer", "active_tab", "selected_question",
        "suggestions", "ask_inner_tab",
    ]:
        default_map = {
            "vector_store_ready": False,
            "documents_info": {},
            "active_tab": 0,
            "ask_inner_tab": 0,
            "suggestions": [],
            "selected_question": "",
        }
        st.session_state[key] = default_map.get(key, None)
    return True


@st.cache_data(ttl=30)
def get_system_status() -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

# ── UI Components ──────────────────────────────────────────────────────────

def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📚 Research Assistant AI")
        st.markdown("Upload research documents, ask questions, get citation-aware synthesis")
    with col2:
        st.caption(f"Backend: `{API_BASE_URL}`")


def render_sidebar():
    with st.sidebar:
        # ── User info + logout ──
        st.markdown(f"👤 **{st.session_state.user_email or 'User'}**")
        if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            logout()
            st.rerun()

        st.divider()
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
        key="file_uploader",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📬 Upload", type="primary", use_container_width=True):
            if uploaded_files:
                result = upload_documents(uploaded_files)
                if result:
                    if result.get("unsupported_files"):
                        st.warning(f"Skipped: {result['unsupported_files']}")
                    st.session_state.active_tab = 1
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
            st.session_state.active_tab = 2
            st.rerun()


def render_ask_tab():
    st.header("❓ Step 3: Ask Research Questions")

    if not st.session_state.vector_store_ready:
        st.warning("⬅️ Process documents first (Step 2)")
        return

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
                    st.session_state.ask_inner_tab = 0
                    st.rerun()
        else:
            st.info("Click 'Generate Suggestions' to get question ideas from your documents.")

    if st.session_state.last_answer:
        st.divider()
        st.subheader("📖 Previous Result")
        render_answer(st.session_state.last_answer, key_suffix="previous")


def render_answer(result: Dict[str, Any], key_suffix: str = ""):
    st.divider()
    st.subheader("📖 Research Synthesis")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Processing Time (s)", f"{result.get('processing_time_seconds', 0):.1f}")
    with col2:
        q = result.get("question", "")
        st.metric("Question", (q[:40] + "...") if len(q) > 40 else q)

    raw = result.get("synthesis", {})
    synthesis = raw.get("research_synthesis", raw) if isinstance(raw, dict) else {}

    if not synthesis:
        st.warning("No synthesis returned")
        return

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

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # ── Auth gate ──
    if not st.session_state.is_authenticated:
        render_login_page()
        return

    render_header()
    render_sidebar()

    active = st.session_state.active_tab

    # Step progress indicator (Upload=0, Process=1, Ask=2, History=3)
    cols = st.columns(4)
    steps = [("📤 Upload", 0), ("⚙️ Process", 1), ("❓ Ask", 2), ("📜 History", 3)]
    for col, (label, idx) in zip(cols, steps):
        with col:
            if idx == active:
                st.success(f"▶ {label}")
            elif idx < active:
                st.info(f"✅ {label}")
            else:
                st.warning(f"○ {label}")
            # Clicking the progress indicator also switches tabs
            if st.button(label, key=f"nav_{idx}", use_container_width=True):
                st.session_state.active_tab = idx
                st.rerun()

    st.divider()

    if active == 0:
        render_upload_tab()
    elif active == 1:
        render_process_tab()
    elif active == 2:
        render_ask_tab()
    elif active == 3:
        render_history_tab()


if __name__ == "__main__":
    main()
