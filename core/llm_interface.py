"""
Centralized LLM Interface for Research Assistant System
CSYE 7374 Final Project - Summer 2025

Centralized LLM management with:
- google.genai SDK (replaces deprecated google.generativeai)
- Real token-based cost tracking
- Exponential backoff retry with jitter for transient API errors
- Retryable vs non-retryable error classification
- Full call history for the sidebar metrics panel
"""

import json
import time
import random
import logging
from typing import List, Dict, Optional, Any

from google import genai
from google.genai import types as genai_types
from config.settings import SystemConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level response wrapper — defined once, not re-created per call
# ---------------------------------------------------------------------------

class GeminiResponse:
    """Wraps a Gemini API response to match the interface expected by agents."""

    def __init__(self, content: str):
        self.content = content
        self.role    = "assistant"


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

_RETRYABLE_MESSAGES = (
    "rate limit",
    "quota",
    "resource exhausted",
    "overloaded",
    "internal error",
    "service unavailable",
    "timeout",
    "deadline exceeded",
)

_MAX_RETRIES   = 3
_BASE_DELAY    = 1.0
_MAX_DELAY     = 16.0
_JITTER_FACTOR = 0.25


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception represents a transient API error."""
    msg         = str(exc).lower()
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code in _RETRYABLE_STATUS_CODES:
        return True
    return any(phrase in msg for phrase in _RETRYABLE_MESSAGES)


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with ±25 % jitter. attempt=0 → ~1 s, 1 → ~2 s, 2 → ~4 s."""
    base   = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
    jitter = base * _JITTER_FACTOR * (2 * random.random() - 1)
    return max(0.0, base + jitter)


# ---------------------------------------------------------------------------
# LLM Interface
# ---------------------------------------------------------------------------

class LLMInterface:
    """
    Single entry point for all Gemini API calls.

    Uses the new google.genai SDK (google-genai package).
    Handles retries, real token-based cost tracking, and call history.
    """

    def __init__(self, api_key: str):
        self.client       = genai.Client(api_key=api_key)
        self.call_count   = 0
        self.total_cost   = 0.0
        self.call_history: List[Dict[str, Any]] = []
        self.start_time   = time.time()
        self.api_key      = api_key

        if not api_key or api_key == "your-gemini-api-key-here":
            logger.warning("API key not configured — set GEMINI_API_KEY in .env")
        else:
            logger.info("Gemini API ready (%s...)", api_key[:10])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make_call(
        self,
        messages: List[Dict],
        response_format: Optional[Dict] = None,  # kept for interface compatibility
        model: str = None,
        json_mode: bool = True,
        max_tokens: Optional[int] = None,  # New parameter
    ) -> Optional[GeminiResponse]:
        """
        Make a single LLM call with automatic retry on transient errors.

        Args:
            messages:        OpenAI-style list of {role, content} dicts.
            response_format: Ignored — JSON mode is always enabled.
            model:           Override the default model from SystemConfig.

        Returns:
            GeminiResponse on success, None after all retries are exhausted.
        """
        model      = model or SystemConfig.DEFAULT_MODEL
        call_start = time.time()
        self.call_count += 1

        logger.info("LLM Call #%d | model=%s", self.call_count, model)

        # Separate system instructions from conversation messages.
        # Gemini does not support a "system" role in contents — it must be
        # passed via system_instruction in GenerateContentConfig instead.
        system_parts = [
            msg["content"] for msg in messages if msg["role"] == "system"
        ]
        conversation = [msg for msg in messages if msg["role"] != "system"]

        # Build contents using only user / model turns
        contents = [
            genai_types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[genai_types.Part(text=msg["content"])],
            )
            for msg in conversation
        ]

        config = genai_types.GenerateContentConfig(
            system_instruction="\n\n".join(system_parts) if system_parts else None,
            temperature=0.2,
            max_output_tokens=max_tokens or SystemConfig.SYNTHESIS_CONFIG["max_tokens"],
            response_mime_type="application/json" if json_mode else "text/plain",
        )

        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                duration      = time.time() - call_start
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count",     0)
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
                total_tokens  = getattr(response.usage_metadata, "total_token_count",      0)
                cost          = SystemConfig.calculate_real_cost(model, prompt_tokens, output_tokens)

                self.total_cost += cost
                self._log_success(model, cost, duration, prompt_tokens, output_tokens, total_tokens)

                logger.info(
                    "LLM done %.2fs | in:%d out:%d tokens | $%.6f",
                    duration, prompt_tokens, output_tokens, cost,
                )
                return GeminiResponse(response.text)

            except Exception as exc:
                last_exc        = exc
                is_last_attempt = attempt == _MAX_RETRIES - 1

                if not _is_retryable(exc) or is_last_attempt:
                    reason = "non-retryable" if not _is_retryable(exc) else "retries exhausted"
                    logger.error("LLM call failed (%s): %s", reason, exc)
                    self._log_failure(model, call_start, str(exc))
                    return None

                delay = _backoff_delay(attempt)
                logger.warning(
                    "Attempt %d/%d failed (retryable) — retry in %.1fs: %s",
                    attempt + 1, _MAX_RETRIES, delay, exc,
                )
                time.sleep(delay)

        self._log_failure(model, call_start, str(last_exc))
        return None

    def estimate_query_cost(self, num_queries: int) -> Dict[str, float]:
        """Estimate cost for N queries using typical token assumptions."""
        cost_per_query = SystemConfig.get_cost_estimate()
        return {
            "queries":        num_queries,
            "total_calls":    num_queries,
            "cost_per_query": cost_per_query,
            "total_cost":     num_queries * cost_per_query,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return aggregated stats for the sidebar metrics panel."""
        runtime    = time.time() - self.start_time
        successful = [c for c in self.call_history if c["success"]]
        failed     = [c for c in self.call_history if not c["success"]]

        return {
            "total_calls":           self.call_count,
            "successful_calls":      len(successful),
            "failed_calls":          len(failed),
            "success_rate":          len(successful) / max(1, self.call_count),
            "total_cost":            self.total_cost,
            "average_cost_per_call": self.total_cost / max(1, self.call_count),
            "total_runtime":         runtime,
            "average_call_duration": (
                sum(c["duration"] for c in successful) / max(1, len(successful))
            ),
            "total_tokens_used": sum(
                c.get("token_usage", {}).get("total_tokens", 0) for c in successful
            ),
            "calls_per_minute": (self.call_count / runtime) * 60 if runtime > 0 else 0,
        }

    def reset_counters(self):
        """Reset all counters — used by the Reset System button."""
        self.call_count   = 0
        self.total_cost   = 0.0
        self.call_history = []
        self.start_time   = time.time()
        logger.info("LLM counters reset")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_success(
        self,
        model: str,
        cost: float,
        duration: float,
        prompt_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ):
        self.call_history.append({
            "call_number": self.call_count,
            "model":       model,
            "cost":        cost,
            "duration":    duration,
            "success":     True,
            "timestamp":   time.time(),
            "token_usage": {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens":      total_tokens,
            },
        })

    def _log_failure(self, model: str, call_start: float, error: str):
        self.call_history.append({
            "call_number": self.call_count,
            "model":       model,
            "cost":        0.0,
            "duration":    time.time() - call_start,
            "success":     False,
            "error":       error,
            "timestamp":   time.time(),
        })


# ---------------------------------------------------------------------------
# Validation utility (used by SynthesisAgent)
# ---------------------------------------------------------------------------

_vlog = logging.getLogger(__name__)


def validate_llm_response(response_content: str, expected_model: type) -> Optional[Any]:
    """
    Parse and validate a JSON LLM response against a Pydantic model.
    Returns the validated model instance, or None on any failure.
    """
    try:
        parsed = json.loads(response_content)
        return expected_model(**parsed)
    except json.JSONDecodeError as exc:
        _vlog.error("JSON parse error: %s", exc)
    except ValueError as exc:
        _vlog.error("Pydantic validation error: %s", exc)
    except Exception as exc:
        _vlog.error("Unexpected validation error: %s", exc)
    return None