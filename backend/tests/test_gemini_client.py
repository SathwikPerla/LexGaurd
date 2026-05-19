"""
test_gemini_client.py — Unit tests for LLMClient (Anthropic SDK wrapper).

ALL tests are mocked — zero API calls, zero cost.

Run: pytest backend/tests/test_gemini_client.py -v
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gemini_client import (
    LLMCallError,
    LLMClient,
    LLMConfigurationError,
    LLMParseError,
    _extract_json,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _patched_client(monkeypatch, api_key: str = "sk-ant-test-key") -> LLMClient:
    """Build an LLMClient with the Anthropic SDK call patched out."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)
    monkeypatch.setenv("CLAUDE_MODEL_NAME", "claude-sonnet-4-6")
    with patch("core.gemini_client.LLMClient._build_client", return_value=MagicMock()):
        client = LLMClient()
    return client


def _make_response_mock(text: str) -> MagicMock:
    """Return a mock Anthropic response whose content[0].text = text."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


# ─────────────────────────────────────────────────────────────────────────────
# _extract_json tests — pure function, no mocking needed
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractJson:
    def test_pure_json_object(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_pure_json_list(self):
        result = _extract_json('[{"id": 1}, {"id": 2}]')
        assert result == [{"id": 1}, {"id": 2}]

    def test_json_fenced_with_json_label(self):
        text = '```json\n{"clauses": []}\n```'
        result = _extract_json(text)
        assert result == {"clauses": []}

    def test_json_fenced_without_label(self):
        text = '```\n{"document_type": "employment"}\n```'
        result = _extract_json(text)
        assert result == {"document_type": "employment"}

    def test_empty_string_raises_parse_error(self):
        with pytest.raises(LLMParseError, match="empty"):
            _extract_json("")

    def test_whitespace_only_raises_parse_error(self):
        with pytest.raises(LLMParseError, match="empty"):
            _extract_json("   ")

    def test_plain_text_raises_parse_error(self):
        with pytest.raises(LLMParseError, match="non-JSON"):
            _extract_json("This is just a sentence.")

    def test_nested_json_parsed(self):
        payload = {"clauses": [{"id": "c1", "score": 8.5}]}
        result = _extract_json(json.dumps(payload))
        assert result["clauses"][0]["score"] == 8.5


# ─────────────────────────────────────────────────────────────────────────────
# LLMClient — credential validation
# ─────────────────────────────────────────────────────────────────────────────


class TestCredentialValidation:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
            LLMClient()

    def test_explicit_api_key_accepted(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("core.gemini_client.LLMClient._build_client", return_value=MagicMock()):
            client = LLMClient(api_key="sk-ant-explicit")
        assert client._api_key == "sk-ant-explicit"

    def test_env_api_key_accepted(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        with patch("core.gemini_client.LLMClient._build_client", return_value=MagicMock()):
            client = LLMClient()
        assert client._api_key == "sk-ant-from-env"

    def test_default_model_is_sonnet(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("CLAUDE_MODEL_NAME", raising=False)
        with patch("core.gemini_client.LLMClient._build_client", return_value=MagicMock()):
            client = LLMClient()
        assert client._model_name == "claude-sonnet-4-6"

    def test_model_name_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("CLAUDE_MODEL_NAME", "claude-haiku-4-5-20251001")
        with patch("core.gemini_client.LLMClient._build_client", return_value=MagicMock()):
            client = LLMClient()
        assert client._model_name == "claude-haiku-4-5-20251001"

    def test_explicit_model_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("CLAUDE_MODEL_NAME", "claude-opus-4-7")
        with patch("core.gemini_client.LLMClient._build_client", return_value=MagicMock()):
            client = LLMClient(model_name="claude-haiku-4-5-20251001")
        assert client._model_name == "claude-haiku-4-5-20251001"


# ─────────────────────────────────────────────────────────────────────────────
# generate_json — success paths
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateJsonSuccess:
    def test_returns_dict(self, monkeypatch):
        client = _patched_client(monkeypatch)
        payload = {"clauses": [], "total": 0}
        response_mock = _make_response_mock(json.dumps(payload))
        client._client.beta.prompt_caching.messages.create.return_value = response_mock

        result = asyncio.get_event_loop().run_until_complete(
            client.generate_json("prompt", "system")
        )
        assert result == payload

    def test_returns_list(self, monkeypatch):
        client = _patched_client(monkeypatch)
        payload = [{"id": 1}, {"id": 2}]
        response_mock = _make_response_mock(json.dumps(payload))
        client._client.beta.prompt_caching.messages.create.return_value = response_mock

        result = asyncio.get_event_loop().run_until_complete(
            client.generate_json("prompt", "system")
        )
        assert result == payload

    def test_markdown_fenced_json_extracted(self, monkeypatch):
        client = _patched_client(monkeypatch)
        response_mock = _make_response_mock('```json\n{"ok": true}\n```')
        client._client.beta.prompt_caching.messages.create.return_value = response_mock

        result = asyncio.get_event_loop().run_until_complete(
            client.generate_json("prompt", "system")
        )
        assert result == {"ok": True}

    def test_correct_model_used(self, monkeypatch):
        client = _patched_client(monkeypatch)
        response_mock = _make_response_mock('{"ok": true}')
        client._client.beta.prompt_caching.messages.create.return_value = response_mock

        asyncio.get_event_loop().run_until_complete(
            client.generate_json("prompt", "system")
        )
        call_kwargs = client._client.beta.prompt_caching.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"


# ─────────────────────────────────────────────────────────────────────────────
# generate_json — error paths
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateJsonErrors:
    def test_non_json_response_raises_parse_error(self, monkeypatch):
        client = _patched_client(monkeypatch)
        response_mock = _make_response_mock("This is plain text, not JSON.")
        client._client.beta.prompt_caching.messages.create.return_value = response_mock

        with pytest.raises(LLMParseError):
            asyncio.get_event_loop().run_until_complete(
                client.generate_json("prompt", "system")
            )

    def test_empty_content_raises_call_error(self, monkeypatch):
        client = _patched_client(monkeypatch)
        response_mock = MagicMock()
        response_mock.content = []  # empty content list
        client._client.beta.prompt_caching.messages.create.return_value = response_mock

        with pytest.raises(LLMCallError):
            asyncio.get_event_loop().run_until_complete(
                client.generate_json("prompt", "system")
            )

    def test_api_exception_raises_call_error(self, monkeypatch):
        client = _patched_client(monkeypatch)
        client._client.beta.prompt_caching.messages.create.side_effect = RuntimeError("API down")

        with pytest.raises(LLMCallError):
            asyncio.get_event_loop().run_until_complete(
                client.generate_json("prompt", "system")
            )


# ─────────────────────────────────────────────────────────────────────────────
# generate_text — success and error
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateText:
    def test_returns_string(self, monkeypatch):
        client = _patched_client(monkeypatch)
        response_mock = _make_response_mock("Plain text response from Claude.")
        client._client.beta.prompt_caching.messages.create.return_value = response_mock

        result = asyncio.get_event_loop().run_until_complete(
            client.generate_text("prompt", "system")
        )
        assert isinstance(result, str)
        assert "Plain text" in result

    def test_api_error_raises_call_error(self, monkeypatch):
        client = _patched_client(monkeypatch)
        client._client.beta.prompt_caching.messages.create.side_effect = ConnectionError("no net")

        with pytest.raises(LLMCallError):
            asyncio.get_event_loop().run_until_complete(
                client.generate_text("prompt", "system")
            )


# ─────────────────────────────────────────────────────────────────────────────
# Retry logic
# ─────────────────────────────────────────────────────────────────────────────


class TestRetryLogic:
    def test_retries_on_connection_error_then_succeeds(self, monkeypatch):
        import anthropic as _anthropic
        client = _patched_client(monkeypatch)
        call_count = {"n": 0}

        def flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise _anthropic.APIConnectionError(request=MagicMock())
            return _make_response_mock('{"ok": true}')

        client._client.beta.prompt_caching.messages.create.side_effect = flaky

        result = asyncio.get_event_loop().run_until_complete(
            client.generate_json("prompt", "system")
        )
        assert result == {"ok": True}
        assert call_count["n"] == 3

    def test_exhausts_retries_raises_call_error(self, monkeypatch):
        import anthropic as _anthropic
        client = _patched_client(monkeypatch)
        client._client.beta.prompt_caching.messages.create.side_effect = (
            _anthropic.APIConnectionError(request=MagicMock())
        )

        with pytest.raises(LLMCallError):
            asyncio.get_event_loop().run_until_complete(
                client.generate_json("prompt", "system")
            )


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible aliases
# ─────────────────────────────────────────────────────────────────────────────


class TestAliases:
    def test_gemini_client_alias(self):
        from core.gemini_client import GeminiClient
        assert GeminiClient is LLMClient

    def test_gemini_call_error_alias(self):
        from core.gemini_client import GeminiCallError
        assert GeminiCallError is LLMCallError

    def test_gemini_parse_error_alias(self):
        from core.gemini_client import GeminiParseError
        assert GeminiParseError is LLMParseError
