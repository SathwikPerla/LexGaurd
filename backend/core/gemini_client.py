"""
gemini_client.py — Anthropic Claude SDK wrapper for LEXGUARD.

Design decisions:
  - Uses anthropic.Anthropic (sync) in asyncio.to_thread — keeps retry logic
    simple (tenacity works cleanly with sync functions) while keeping the
    event loop unblocked.
  - Prompt caching via cache_control on system prompts — reduces cost and latency
    when the same agent system prompt is reused across multiple documents.
  - JSON extraction handles markdown code fences as a fallback — Claude follows
    JSON-only instructions reliably but this guard is cheap insurance.
  - ANTHROPIC_API_KEY validated at instantiation time (not import time) so
    unit tests can monkeypatch env vars before calling LLMClient().
  - Retry on RateLimitError, InternalServerError, APIConnectionError with
    exponential backoff via tenacity.
  - Default model: claude-sonnet-4-6 (quality/speed balance for legal analysis).
    Override with CLAUDE_MODEL_NAME env var.

Env vars:
  ANTHROPIC_API_KEY  — required
  CLAUDE_MODEL_NAME  — optional, default: claude-sonnet-4-6

Common failure points:
  - Missing ANTHROPIC_API_KEY → LLMConfigurationError at instantiation
  - Rate limit (429) → retried up to 3 times with backoff
  - Claude wraps JSON in markdown fences → stripped by _extract_json()
  - Response exceeds max_tokens → LLMCallError with clear message
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import tenacity

logger = logging.getLogger(__name__)

CALL_TIMEOUT_SECONDS: int = 180  # Agent 4 with 6+ clauses + full alt wording can take >120s
_MAX_ATTEMPTS: int = 3
_BACKOFF_MIN: float = 2.0
_BACKOFF_MAX: float = 30.0

# Max tokens — negotiator output for 5+ complex clauses needs 7000+ tokens
MAX_TOKENS_JSON: int = 8192
MAX_TOKENS_TEXT: int = 2048


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class LLMConfigurationError(RuntimeError):
    """Missing or invalid API key / configuration."""


class LLMCallError(RuntimeError):
    """API call failed after all retries."""


class LLMParseError(ValueError):
    """Response could not be parsed as JSON."""


# Backward-compatible aliases (agents imported these names)
GeminiConfigurationError = LLMConfigurationError
GeminiCallError = LLMCallError
GeminiParseError = LLMParseError


# ─────────────────────────────────────────────────────────────────────────────
# Retry predicate
# ─────────────────────────────────────────────────────────────────────────────


def _is_retryable(exc: BaseException) -> bool:
    try:
        import anthropic as _anthropic  # type: ignore[import-untyped]
        return isinstance(
            exc,
            (
                _anthropic.RateLimitError,
                _anthropic.InternalServerError,
                _anthropic.APIConnectionError,
                _anthropic.APITimeoutError,
            ),
        )
    except ImportError:
        return isinstance(exc, (ConnectionError, TimeoutError))


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction helper
# ─────────────────────────────────────────────────────────────────────────────


def _extract_json(text: str) -> Any:
    """
    Parse JSON from Claude's response text.

    Tries in order:
      1. Direct JSON parse
      2. Strip markdown fences (```json ... ```)
      3. Find first { or [ to last } or ]
      4. Truncation recovery: if the JSON object starts but is cut off,
         close all open arrays/objects and re-parse the salvaged prefix.

    Raises:
        LLMParseError if all attempts fail.
    """
    stripped = text.strip()
    if not stripped:
        raise LLMParseError("Claude returned an empty response.")

    # 1. Direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    working = stripped
    if stripped.startswith("```"):
        newline_pos = stripped.find("\n")
        if newline_pos != -1:
            inner = stripped[newline_pos + 1:]
            last_fence = inner.rfind("```")
            working = inner[:last_fence].strip() if last_fence != -1 else inner.strip()
            try:
                return json.loads(working)
            except json.JSONDecodeError:
                pass

    # 3. Find outermost { ... } or [ ... ]
    for open_c, close_c in [('{', '}'), ('[', ']')]:
        start = working.find(open_c)
        end = working.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(working[start:end + 1])
            except json.JSONDecodeError:
                pass

    # 4. Truncation recovery — append missing closing brackets.
    # When Claude hits the token limit mid-JSON (e.g. 47-clause documents),
    # the output is valid JSON up to the cut point. We close all open
    # brackets, strip any trailing comma, and parse the repaired string.
    candidate = working if (working.startswith("{") or working.startswith("[")) else stripped
    if candidate:
        closers: list[str] = []
        in_str = False
        esc = False
        for ch in candidate:
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                closers.append('}')
            elif ch == '[':
                closers.append(']')
            elif ch in ('}', ']') and closers:
                closers.pop()

        if closers:
            # Strip trailing whitespace and dangling comma before closing
            tail = candidate.rstrip()
            if tail.endswith(','):
                tail = tail[:-1]
            repaired = tail + ''.join(reversed(closers))
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    raise LLMParseError(
        f"Claude returned non-JSON despite JSON-only instruction. "
        f"First 500 chars: {stripped[:500]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────


class LLMClient:
    """
    Thin async wrapper around the Anthropic Claude SDK.

    Usage:
        client = LLMClient()
        result: dict = await client.generate_json(user_prompt, system_prompt)
        text:   str  = await client.generate_text(user_prompt, system_prompt)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._model_name = model_name or os.getenv("CLAUDE_MODEL_NAME", "claude-sonnet-4-6")

        if not self._api_key:
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )

        self._client = self._build_client()

        logger.info(
            "LLMClient initialised",
            extra={"model": self._model_name},
        )

    def _build_client(self):  # type: ignore[return]
        """Build the Anthropic client — separated for easy mocking in tests."""
        import anthropic  # type: ignore[import-untyped]
        return anthropic.Anthropic(api_key=self._api_key)

    # ── Sync call (runs in asyncio.to_thread) ────────────────────────────────

    def _call_json_sync(self, system_prompt: str, user_prompt: str) -> Any:
        """
        Synchronous Claude call requesting JSON output, with retry and caching.

        Prompt caching is applied to the system prompt — when the same agent
        processes multiple documents, the system prompt is served from cache
        after the first call (saves ~90% of input token cost for that portion).
        """

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(_MAX_ATTEMPTS),
            wait=tenacity.wait_exponential(
                multiplier=1, min=_BACKOFF_MIN, max=_BACKOFF_MAX
            ),
            retry=tenacity.retry_if_exception(_is_retryable),
            before_sleep=lambda s: logger.warning(
                "Claude JSON call retrying",
                extra={"attempt": s.attempt_number, "error": str(s.outcome.exception())},
            ),
            reraise=True,
        )
        def _call() -> str:
            response = self._client.beta.prompt_caching.messages.create(
                model=self._model_name,
                max_tokens=MAX_TOKENS_JSON,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
                betas=["prompt-caching-2024-07-31"],
            )
            if not response.content:
                raise LLMCallError("Claude returned empty content.")
            return response.content[0].text

        try:
            raw_text = _call()
        except (LLMCallError, LLMParseError):
            raise
        except tenacity.RetryError as exc:
            raise LLMCallError(
                f"Claude JSON call failed after {_MAX_ATTEMPTS} attempts."
            ) from exc
        except Exception as exc:
            raise LLMCallError(f"Claude JSON call failed: {exc}") from exc

        logger.debug(
            "Claude JSON response received",
            extra={"chars": len(raw_text), "preview": raw_text[:200]},
        )

        return _extract_json(raw_text)

    def _call_text_sync(self, system_prompt: str, user_prompt: str) -> str:
        """Synchronous Claude call returning plain text."""

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(_MAX_ATTEMPTS),
            wait=tenacity.wait_exponential(
                multiplier=1, min=_BACKOFF_MIN, max=_BACKOFF_MAX
            ),
            retry=tenacity.retry_if_exception(_is_retryable),
            before_sleep=lambda s: logger.warning(
                "Claude text call retrying",
                extra={"attempt": s.attempt_number},
            ),
            reraise=True,
        )
        def _call() -> str:
            response = self._client.beta.prompt_caching.messages.create(
                model=self._model_name,
                max_tokens=MAX_TOKENS_TEXT,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
                betas=["prompt-caching-2024-07-31"],
            )
            if not response.content:
                raise LLMCallError("Claude returned empty content.")
            return response.content[0].text

        try:
            return _call()
        except (LLMCallError,):
            raise
        except tenacity.RetryError as exc:
            raise LLMCallError(f"Claude text call failed after {_MAX_ATTEMPTS} attempts.") from exc
        except Exception as exc:
            raise LLMCallError(f"Claude text call failed: {exc}") from exc

    # ── Public async interface ────────────────────────────────────────────────

    async def generate_json(
        self,
        user_prompt: str,
        system_prompt: str,
    ) -> Any:
        """
        Call Claude and return the response parsed as JSON (dict or list).

        Raises:
            LLMConfigurationError:  API key invalid.
            LLMCallError:           API failed after retries.
            LLMParseError:          Response was not valid JSON.
            asyncio.TimeoutError:   Call exceeded CALL_TIMEOUT_SECONDS.
        """
        logger.info(
            "LLM JSON call starting",
            extra={"model": self._model_name, "prompt_chars": len(user_prompt)},
        )
        result = await asyncio.wait_for(
            asyncio.to_thread(self._call_json_sync, system_prompt, user_prompt),
            timeout=CALL_TIMEOUT_SECONDS,
        )
        logger.info("LLM JSON call complete", extra={"model": self._model_name})
        return result

    async def generate_text(
        self,
        user_prompt: str,
        system_prompt: str,
    ) -> str:
        """Call Claude and return plain text response."""
        logger.info(
            "LLM text call starting",
            extra={"model": self._model_name, "prompt_chars": len(user_prompt)},
        )
        result = await asyncio.wait_for(
            asyncio.to_thread(self._call_text_sync, system_prompt, user_prompt),
            timeout=CALL_TIMEOUT_SECONDS,
        )
        logger.info("LLM text call complete", extra={"model": self._model_name})
        return result


# Backward-compatible alias
GeminiClient = LLMClient
