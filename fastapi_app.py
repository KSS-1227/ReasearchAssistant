"""
Endpoints:
  POST   /upload                - Upload research documents (PDF, TXT, MD)
  POST   /process               - Build vector index from uploaded docs
  POST   /ask                   - Ask a research question
  GET    /suggested-questions   - Get suggested research questions
  GET    /health                - Health check
  GET    /status                - System statistics and session info
  DELETE /reset                 - Clear session and vector index
"""
import re
import os
import logging
import uuid
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv

try:
    from supabase import create_client, Client as SupabaseClient
    _supabase_available = True
except ImportError:
    _supabase_available = False

from core.coordinator import ResearchCoordinator

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Supabase client — optional, only initialised when env vars are present
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

supabase: Optional[Any] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")
    except Exception as e:
        logger.warning(f"Supabase initialization failed: {e}")
        supabase = None
else:
    logger.info("Supabase not configured — running without auth/history")

security = HTTPBearer(auto_error=False)

# ── Pydantic Models ───────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id: str
    file_count: int
    total_size_mb: float
    supported_files: List[str]
    unsupported_files: List[str]
    message: str


class ProcessRequest(BaseModel):
    session_id: str


class ProcessResponse(BaseModel):
    session_id: str
    status: str
    documents_processed: int
    chunks_created: int = 0
    vector_store_size: int = 0
    processing_time_seconds: float
    message: str


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=5, max_length=500)
    domain: Optional[str] = Field(None)
    user_token: Optional[str] = Field(None, description="Supabase JWT token for history saving")


class AskResponse(BaseModel):
    session_id: str
    question: str
    synthesis: Dict[str, Any]
    processing_time_seconds: float
    matching_papers: int = 0
    confidence_score: float = 0.0
    llm_calls: int = 0
    total_cost: float = 0.0


class SuggestedQuestionsResponse(BaseModel):
    session_id: str
    questions: List[str]
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    api_ready: bool


class StatusResponse(BaseModel):
    active_sessions: int
    total_documents_processed: int
    total_questions_asked: int
    system_uptime_seconds: float


# ── Guardrails ────────────────────────────────────────────────────────────

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above|prior) instructions",
    r"disregard (all )?(previous|above|prior)",
    r"new instructions?:",
    r"system prompt:",
    r"you are now",
    r"forget (everything|all) (above|before)",
    r"act as if",
    r"override your instructions",
]

def detect_prompt_injection(text: str) -> List[str]:
    """Scan text for potential prompt injection patterns. Returns list of matches found."""
    findings = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            findings.append(pattern)
    return findings


# ── Session Management ────────────────────────────────────────────────────

class SessionState:
    """In-memory session state for a single client"""

    def __init__(self, session_id: str, api_key: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.coordinator = ResearchCoordinator(api_key)
        self.documents_uploaded = 0
        self.documents_processed = 0
        self.questions_asked = 0
        self.vector_store_ready = False
        self.temp_dir: Optional[str] = None

    def update_access_time(self):
        self.last_accessed = datetime.now()

    def is_expired(self, timeout_minutes: int = 30) -> bool:
        return datetime.now() - self.last_accessed > timedelta(minutes=timeout_minutes)

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp dir for session {self.session_id}")


class SessionManager:
    """Manages session lifecycle"""

    def __init__(self, max_sessions: int = 100):
        self.sessions: Dict[str, SessionState] = {}
        self.max_sessions = max_sessions
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        self.stats = {
            "total_sessions_created": 0,
            "total_documents_processed": 0,
            "total_questions_asked": 0,
        }
        self.startup_time = datetime.now()

    def create_session(self) -> SessionState:
        if not self.api_key:
            raise HTTPException(
                status_code=503,
                detail="GEMINI_API_KEY GOOGLE_API_KEY not configured on server"
            )
        if len(self.sessions) >= self.max_sessions:
            self._cleanup_expired_sessions()

        session_id = str(uuid.uuid4())[:8]
        session = SessionState(session_id, self.api_key)
        self.sessions[session_id] = session
        self.stats["total_sessions_created"] += 1
        logger.info(f"Created session: {session_id}")
        return session

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found or expired"
            )
        session = self.sessions[session_id]
        session.update_access_time()
        return session

    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            self.sessions[session_id].cleanup()
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def _cleanup_expired_sessions(self):
        expired = [
            sid for sid, s in self.sessions.items()
            if s.is_expired()
        ]
        for sid in expired:
            self.delete_session(sid)
            logger.info(f"Auto-cleaned expired session: {sid}")

    def get_active_sessions_count(self) -> int:
        self._cleanup_expired_sessions()
        return len(self.sessions)

    def get_uptime_seconds(self) -> float:
        return (datetime.now() - self.startup_time).total_seconds()


# ── Global session manager ────────────────────────────────────────────────

session_manager: Optional[SessionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_manager
    session_manager = SessionManager()
    logger.info("Session manager initialized")
    yield
    logger.info("Shutting down — cleaning sessions...")
    for sid in list(session_manager.sessions.keys()):
        session_manager.delete_session(sid)


# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Research Assistant API",
    description="REST API for the Research Assistant AI system",
    version="1.0.0",
    lifespan=lifespan,
)
# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_session_manager() -> SessionManager:
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return session_manager


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Verify Supabase JWT token and return user."""
    if supabase is None:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    
    if credentials is None:
        raise HTTPException(status_code=401, detail="No authorization token provided")
    
    try:
        user = supabase.auth.get_user(credentials.credentials)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user.user
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        api_ready=session_manager is not None,
    )


@app.get("/history", tags=["Research"])
async def get_history(
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    """Get query history for authenticated user."""
    if supabase is None:
        raise HTTPException(status_code=503, detail="Supabase not configured")

    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        user = supabase.auth.get_user(credentials.credentials)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid token")

        result = supabase.table("queries")\
            .select("*")\
            .eq("user_id", str(user.user.id))\
            .order("created_at", desc=True)\
            .limit(20)\
            .execute()

        return {
            "user_id": str(user.user.id),
            "queries": result.data,
            "count": len(result.data)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"History fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Research Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "workflow": [
            "1. POST /upload",
            "2. POST /process",
            "3. POST /ask",
            "4. GET /suggested-questions",
            "5. DELETE /reset",
        ]
    }


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
@limiter.limit("5/minute")
async def upload_documents(
    request: Request,
    files: List[UploadFile] = File(...),
    sm: SessionManager = Depends(get_session_manager),
) -> UploadResponse:
    """Upload research documents. Creates a new session."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    session = sm.create_session()
    session.temp_dir = tempfile.mkdtemp(prefix=f"research_{session.session_id}_")

    supported_files = []
    unsupported_files = []
    total_size_mb = 0.0
    allowed_extensions = {".pdf", ".txt", ".md"}

    try:
        for file in files:
            file_ext = Path(file.filename).suffix.lower()

            if file_ext not in allowed_extensions:
                unsupported_files.append(file.filename)
                continue

            contents = await file.read()
            size_mb = len(contents) / (1024 * 1024)

            if size_mb > 50:
                unsupported_files.append(f"{file.filename} (too large: {size_mb:.1f}MB)")
                continue

            file_path = os.path.join(session.temp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(contents)

            supported_files.append(file.filename)
            total_size_mb += size_mb
            session.documents_uploaded += 1

        if not supported_files:
            sm.delete_session(session.session_id)
            raise HTTPException(
                status_code=400,
                detail="No valid files. Supported: .pdf, .txt, .md"
            )

        logger.info(
            f"Session {session.session_id}: uploaded {len(supported_files)} files "
            f"({total_size_mb:.2f} MB)"
        )

        return UploadResponse(
            session_id=session.session_id,
            file_count=len(supported_files),
            total_size_mb=round(total_size_mb, 3),
            supported_files=supported_files,
            unsupported_files=unsupported_files,
            message=f"Uploaded {len(supported_files)} file(s). Next: POST /process",
        )

    except HTTPException:
        raise
    except Exception as e:
        sm.delete_session(session.session_id)
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/process", response_model=ProcessResponse, tags=["Documents"])
async def process_documents(
    req: ProcessRequest,
    background_tasks: BackgroundTasks,                   # ✅ fixed — no default
    sm: SessionManager = Depends(get_session_manager),
) -> ProcessResponse:
    """Process uploaded documents and build FAISS vector index."""
    session = sm.get_session(req.session_id)

    if session.documents_uploaded == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded for this session"
        )

    if not session.temp_dir or not os.path.exists(session.temp_dir):
        raise HTTPException(
            status_code=400,
            detail="Session temp directory missing — please re-upload"
        )

    start_time = datetime.now()

    try:
        docs_processed = 0
        for filename in os.listdir(session.temp_dir):
            file_path = os.path.join(session.temp_dir, filename)
            if not os.path.isfile(file_path):
                continue

           # ✅ Using actual coordinator method: process_document()
            result = session.coordinator.process_document(file_path)
            docs_processed += 1
            logger.info(f"Processed: {filename} → {result}")

            # Guardrail: check for prompt injection patterns
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content_sample = f.read(5000)  # check first 5000 chars
                injection_findings = detect_prompt_injection(content_sample)
                if injection_findings:
                    logger.warning(
                        f"⚠️ Potential prompt injection in '{filename}': "
                        f"{len(injection_findings)} pattern(s) matched"
                    )
            except Exception:
                pass  # binary files (PDFs) can't be read as text directly — skip

        session.documents_processed = docs_processed
        session.vector_store_ready = True
        sm.stats["total_documents_processed"] += docs_processed

        processing_time = (datetime.now() - start_time).total_seconds()

        # Clean up temp files after response is sent
        background_tasks.add_task(session.cleanup)

        logger.info(
            f"Session {req.session_id}: processed {docs_processed} docs "
            f"in {processing_time:.2f}s"
        )

        stats = session.coordinator.document_processor.processing_stats
        return ProcessResponse(
            session_id=req.session_id,
            status="success",
            documents_processed=docs_processed,
            chunks_created=stats.get("total_chunks", 0),
            vector_store_size=stats.get("vector_store_size", 0),
            processing_time_seconds=round(processing_time, 2),
            message=f"Processed {docs_processed} document(s). Next: POST /ask",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/ask", response_model=AskResponse, tags=["Research"])
@limiter.limit("10/minute")
async def ask_question(
    request: Request,
    req: AskRequest,
    sm: SessionManager = Depends(get_session_manager),
) -> AskResponse:
    """Ask a research question. Returns structured synthesis."""

    # Guardrail: check question FIRST before any session lookup
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    injection_findings = detect_prompt_injection(req.question)
    if injection_findings:
        logger.warning(
            f"⚠️ Potential prompt injection in question: {req.question[:100]}"
        )
        raise HTTPException(
            status_code=400,
            detail="Question contains disallowed content and cannot be processed."
        )

    session = sm.get_session(req.session_id)

    if not session.vector_store_ready:
        raise HTTPException(
            status_code=400,
            detail="Vector store not ready. Run POST /process first."
        )

    start_time = datetime.now()

    try:
        # ✅ Using actual coordinator method: research_query()
        result = session.coordinator.research_query(req.question)

        processing_time = (datetime.now() - start_time).total_seconds()
        session.questions_asked += 1
        sm.stats["total_questions_asked"] += 1

        # Normalize result to dict
        if hasattr(result, "dict"):
            synthesis = result.dict()
        elif isinstance(result, dict):
            synthesis = result
        else:
            synthesis = {"raw": str(result)}

        logger.info(
            f"Session {req.session_id}: answered in {processing_time:.2f}s"
        )
        # Save query to Supabase if token provided
        if supabase and req.user_token:
            try:
                user_response = supabase.auth.get_user(req.user_token)
                if user_response and user_response.user:
                    user_id = str(user_response.user.id)
                    supabase.table("queries").insert({
                        "user_id": user_id,
                        "question": req.question,
                        "synthesis": synthesis,
                        "processing_time_seconds": round(processing_time, 2)
                    }).execute()
                    logger.info(f"Query saved for user {user_id[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to save query: {e}")

        perf = synthesis.get("performance_metrics", {}) if isinstance(synthesis, dict) else {}
        return AskResponse(
            session_id=req.session_id,
            question=req.question,
            synthesis=synthesis,
            processing_time_seconds=round(processing_time, 2),
            matching_papers=perf.get("papers_analyzed", 0),
            confidence_score=perf.get("retrieval_confidence", 0.0),
            llm_calls=perf.get("total_llm_calls", 0),
            total_cost=perf.get("estimated_cost", 0.0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Question failed: {str(e)}")


@app.get("/suggested-questions", response_model=SuggestedQuestionsResponse, tags=["Research"])
async def get_suggested_questions(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
) -> SuggestedQuestionsResponse:
    """Get suggested research questions from uploaded documents."""
    session = sm.get_session(session_id)

    if not session.vector_store_ready:
        raise HTTPException(
            status_code=400,
            detail="Vector store not ready. Run POST /process first."
        )

    try:
        docs_response = session.coordinator.search_uploaded_documents("research methodology findings")
        questions = []
        results = []
        if isinstance(docs_response, dict):
            results = docs_response.get("results", [])

        for doc in results[:5]:
            title = None
            preview = None
            if hasattr(doc, "metadata") and isinstance(getattr(doc, "metadata"), dict):
                title = doc.metadata.get("title") or doc.metadata.get("source")
            if hasattr(doc, "page_content"):
                preview = getattr(doc, "page_content").strip().replace("\n", " ")[:80]
            elif isinstance(doc, str):
                preview = doc.strip().replace("\n", " ")[:80]
            else:
                preview = str(doc).strip().replace("\n", " ")[:80]

            if title:
                questions.append(f"What are the key contributions of '{title}'?")
            elif preview:
                questions.append(f"What are the main insights from this excerpt: {preview}...")

        if not questions:
            questions = [
                "What are the main contributions of these papers?",
                "What methodologies are used across these studies?",
                "What research gaps are identified?",
                "How do the results compare across papers?",
                "What are the practical implications?",
            ]

        logger.info(f"Generated {len(questions)} suggestions for session {session_id}")

        return SuggestedQuestionsResponse(
            session_id=session_id,
            questions=questions,
            message=f"Generated {len(questions)} suggested questions.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Suggestions failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status(
    sm: SessionManager = Depends(get_session_manager)
) -> StatusResponse:
    """Get system status and stats."""
    return StatusResponse(
        active_sessions=sm.get_active_sessions_count(),
        total_documents_processed=sm.stats["total_documents_processed"],
        total_questions_asked=sm.stats["total_questions_asked"],
        system_uptime_seconds=round(sm.get_uptime_seconds(), 1),
    )


@app.delete("/reset", tags=["Session"])
async def reset_session(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
) -> Dict[str, str]:
    """Delete session and clean up all resources."""
    success = sm.delete_session(session_id)
    if success:
        return {"message": f"Session {session_id} reset successfully"}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


# ── Run directly ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        workers=1,                          # ✅ 1 worker — safe for containers
        reload=False,
        log_level="info",
    )