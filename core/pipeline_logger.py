"""
Pipeline Logger - Research Assistant System

Structured JSON logger for every research query.
Appends one JSON object per line to logs/pipeline_logs.jsonl
(newline-delimited JSON — easy to parse, grep, and stream).

Design goals:
- Zero impact on pipeline speed  (writes happen in a background thread)
- Never crashes the pipeline     (all errors are swallowed silently)
- One log entry per query        (complete picture in a single record)
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

# Default log file location — sits next to the project root
_DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "pipeline_logs.jsonl"
_MAX_LOG_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB — rotate when exceeded


class PipelineLogger:
    """
    Structured JSON logger for the research pipeline.

    Each call to log() appends one JSON line to the log file.
    Writes are dispatched to a background daemon thread so the
    pipeline never waits for disk I/O.

    Usage:
        logger = PipelineLogger()          # uses default path
        logger.log({...})                  # fire-and-forget
    """

    def __init__(self, log_path: Optional[Path] = None):
        self._path = Path(log_path) if log_path else _DEFAULT_LOG_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Single background thread + queue keeps writes ordered and non-blocking
        self._lock = threading.Lock()
        _log.info("PipelineLogger ready | path=%s", self._path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, data: dict) -> None:
        """
        Append a structured log entry asynchronously.

        Dispatches the write to a daemon thread — returns immediately
        so the pipeline is never blocked by disk I/O.

        Args:
            data: Arbitrary dict.  A 'timestamp' key is injected
                  automatically if not already present.
        """
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()

        thread = threading.Thread(
            target=self._write,
            args=(data,),
            daemon=True,          # dies with the main process — no cleanup needed
        )
        thread.start()

    def log_query(
        self,
        *,
        user_query: str,
        selected_k: int,
        retrieved_chunks: List[Dict[str, Any]],
        final_response: Dict[str, Any],
        confidence_score: float,
        response_time: float,
        domain: str = "",
        session_id: str = "",
        llm_calls: int = 0,
        estimated_cost: float = 0.0,
        fallback_used: bool = False,
    ) -> None:
        """
        Convenience wrapper that builds the standard pipeline log entry.

        Args:
            user_query:       Raw query string from the user.
            selected_k:       Dynamic k value chosen by LiteratureScanner.
            retrieved_chunks: List of chunk dicts (content + metadata).
                              Only text and key metadata fields are stored
                              to keep log size manageable.
            final_response:   research_synthesis dict from the coordinator.
            confidence_score: synthesis.confidence_score from SynthesisAgent.
            response_time:    Wall-clock seconds for the full pipeline.
            domain:           Classified research domain.
            session_id:       Coordinator session ID.
            llm_calls:        Total LLM calls made.
            estimated_cost:   Estimated USD cost.
            fallback_used:    True if deterministic fallback was triggered.
        """
        # Trim chunks to avoid bloating the log file
        slim_chunks = [
            {
                "content":          c.get("content", "")[:300],   # first 300 chars
                "page":             c.get("page", "N/A"),
                "heading":          c.get("heading", "N/A"),
                "source_file":      c.get("source_file", "Unknown"),
                "similarity_score": round(float(c.get("similarity_score", 0)), 4),
                "combined_score":   round(float(c.get("combined_score",   0)), 4),
            }
            for c in (retrieved_chunks or [])
        ]

        entry = {
            "session_id":       session_id,
            "user_query":       user_query,
            "domain":           domain,
            "selected_k":       selected_k,
            "retrieved_chunks": slim_chunks,
            "chunks_count":     len(slim_chunks),
            "final_response": {
                "key_findings":        (final_response or {}).get("key_findings",        [])[:5],
                "methodology_insights":(final_response or {}).get("methodology_insights",[])[:3],
                "research_gaps":       (final_response or {}).get("research_gaps",       [])[:3],
            },
            "confidence_score": round(confidence_score, 4),
            "response_time_s":  round(response_time,    4),
            "llm_calls":        llm_calls,
            "estimated_cost_usd": round(estimated_cost, 6),
            "fallback_used":    fallback_used,
        }
        self.log(entry)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write(self, data: dict) -> None:
        """Write one JSON line to the log file (runs in background thread).

        Rotates the log file when it exceeds _MAX_LOG_SIZE_BYTES by renaming
        the current file to pipeline_logs.jsonl.1 before opening a fresh one.
        Only one rotated backup is kept to bound disk usage.
        """
        try:
            line = json.dumps(data, ensure_ascii=False, default=str)
            with self._lock:
                # Rotate if the file has grown too large
                if self._path.exists() and self._path.stat().st_size >= _MAX_LOG_SIZE_BYTES:
                    rotated = self._path.with_suffix(".jsonl.1")
                    if rotated.exists():
                        rotated.unlink()          # remove old backup
                    self._path.rename(rotated)    # current → backup
                    _log.info(
                        "PipelineLogger rotated | backup=%s", rotated
                    )
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:
            # Never let logging crash the pipeline
            _log.warning("PipelineLogger write failed: %s", exc)
