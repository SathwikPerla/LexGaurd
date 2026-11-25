"""
test_gemini_fallback.py — Unit tests for the optional Gemini fallback in LLMClient.

ALL tests are mocked — zero API calls, zero cost, no real keys.

Run: pytest backend/tests/test_gemini_fallback.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gemini_client import LLMCallError, LLMClient, LLMConfigurationError


def _gemini_response(text: str, status: int = 200) -> httpx.Response:
    body = {"choices": [{"message": {"content": text}}]} if status == 200 else {"error": "x"}
    return httpx.Response(status, json=body, request=httpx.Request("POST", "http://test"))


def _client(monkeypatch, *, anthropic_key: str | None = "sk-ant-test", gemini_key: str | None = "gm-test"):
    if anthropic_key:
        monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_key)
    else:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    if gemini_key:
        monkeypatch.setenv("GEMINI_API_KEY", gemini_key)
    with patch("core.gemini_client.LLMClient._build_client", return_value=MagicMock()):
        return LLMClient()


class TestConfiguration:
    def test_no_keys_at_all_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(LLMConfigurationError):
            LLMClient()

    def test_gemini_only_is_allowed(self, monkeypatch):
        client = _client(monkeypatch, anthropic_key=None)
        assert client._client is None
        assert client._gemini_enabled()

    def test_no_gemini_key_means_no_fallback(self, monkeypatch):
        client = _client(monkeypatch, gemini_key=None)
        assert not client._gemini_enabled()


class TestFallbackBehaviour:
    def test_claude_failure_without_gemini_still_raises(self, monkeypatch):
        client = _client(monkeypatch, gemini_key=None)
        client._client.beta.prompt_caching.messages.create.side_effect = RuntimeError("credit balance is too low")
        with pytest.raises(LLMCallError, match="credit balance"):
            client._call_text_sync("sys", "hello")

    def test_claude_failure_falls_back_to_gemini_text(self, monkeypatch):
        client = _client(monkeypatch)
        client._client.beta.prompt_caching.messages.create.side_effect = RuntimeError("credit balance is too low")
        with patch("httpx.post", return_value=_gemini_response("from gemini")) as post:
            assert client._call_text_sync("sys", "hello") == "from gemini"
        sent = post.call_args.kwargs["json"]
        assert sent["messages"][0] == {"role": "system", "content": "sys"}
        assert sent["messages"][1] == {"role": "user", "content": "hello"}
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer gm-test"

    def test_claude_failure_falls_back_to_gemini_json(self, monkeypatch):
        client = _client(monkeypatch)
        client._client.beta.prompt_caching.messages.create.side_effect = RuntimeError("401")
        with patch("httpx.post", return_value=_gemini_response('```json\n{"ok": true}\n```')):
            assert client._call_json_sync("sys", "hello") == {"ok": True}

    def test_chat_fallback_merges_context_into_system(self, monkeypatch):
        client = _client(monkeypatch)
        client._client.beta.prompt_caching.messages.create.side_effect = RuntimeError("down")
        msgs = [{"role": "user", "content": "q"}]
        with patch("httpx.post", return_value=_gemini_response("answer")) as post:
            assert client._call_chat_sync("SYS", "CTX", msgs) == "answer"
        sent = post.call_args.kwargs["json"]
        assert "SYS" in sent["messages"][0]["content"] and "CTX" in sent["messages"][0]["content"]
        assert sent["messages"][1:] == msgs

    def test_claude_success_never_calls_gemini(self, monkeypatch):
        client = _client(monkeypatch)
        block = MagicMock()
        block.text = "from claude"
        client._client.beta.prompt_caching.messages.create.return_value = MagicMock(content=[block])
        with patch("httpx.post") as post:
            assert client._call_text_sync("sys", "hello") == "from claude"
        post.assert_not_called()

    def test_gemini_only_mode_uses_gemini_directly(self, monkeypatch):
        client = _client(monkeypatch, anthropic_key=None)
        with patch("httpx.post", return_value=_gemini_response("gemini only")):
            assert client._call_text_sync("sys", "hi") == "gemini only"


class TestGeminiErrors:
    def test_non_retryable_http_error_raises_call_error(self, monkeypatch):
        client = _client(monkeypatch, anthropic_key=None)
        with patch("httpx.post", return_value=_gemini_response("", status=400)):
            with pytest.raises(LLMCallError, match="HTTP 400"):
                client._call_text_sync("sys", "hi")

    def test_retries_on_429_then_succeeds(self, monkeypatch):
        client = _client(monkeypatch, anthropic_key=None)
        responses = [_gemini_response("", status=429), _gemini_response("ok")]
        with patch("httpx.post", side_effect=responses), patch("time.sleep"):
            assert client._call_text_sync("sys", "hi") == "ok"

    def test_gives_up_after_max_attempts(self, monkeypatch):
        client = _client(monkeypatch, anthropic_key=None)
        with patch("httpx.post", return_value=_gemini_response("", status=503)), patch("time.sleep"):
            with pytest.raises(LLMCallError, match="after 3 attempts"):
                client._call_text_sync("sys", "hi")


class TestExactCreditError:
    """The real error seen in production when Anthropic credits run out."""

    def _credit_error(self):
        import anthropic

        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx.Response(
            400,
            request=req,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Your credit balance is too low to access the Anthropic API. "
                    "Please go to Plans & Billing to upgrade or purchase credits.",
                },
            },
        )
        return anthropic.BadRequestError(
            "Error code: 400 - credit balance is too low", response=resp, body=resp.json()
        )

    def test_credit_balance_error_switches_json_call_to_gemini(self, monkeypatch):
        client = _client(monkeypatch)
        client._client.beta.prompt_caching.messages.create.side_effect = self._credit_error()
        with patch("httpx.post", return_value=_gemini_response('{"risk": "HIGH"}')) as post:
            assert client._call_json_sync("sys", "contract text") == {"risk": "HIGH"}
        post.assert_called_once()

    def test_credit_balance_error_without_gemini_key_still_reports_clearly(self, monkeypatch):
        client = _client(monkeypatch, gemini_key=None)
        client._client.beta.prompt_caching.messages.create.side_effect = self._credit_error()
        with pytest.raises(LLMCallError, match="credit balance is too low"):
            client._call_json_sync("sys", "contract text")


class TestProviderOverride:
    """FALLBACK_* env vars point the same adapter at any OpenAI-compatible provider."""

    def test_fallback_env_overrides_url_model_and_key(self, monkeypatch):
        monkeypatch.setenv("FALLBACK_API_KEY", "mistral-key")
        monkeypatch.setenv("FALLBACK_BASE_URL", "https://api.mistral.ai/v1/")
        monkeypatch.setenv("FALLBACK_MODEL_NAME", "mistral-large-latest")
        client = _client(monkeypatch, anthropic_key=None, gemini_key="gm-ignored")
        with patch("httpx.post", return_value=_gemini_response("hi")) as post:
            assert client._call_text_sync("sys", "hello") == "hi"
        assert post.call_args.args[0] == "https://api.mistral.ai/v1/chat/completions"
        assert post.call_args.kwargs["json"]["model"] == "mistral-large-latest"
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer mistral-key"

    def test_default_still_points_at_gemini(self, monkeypatch):
        client = _client(monkeypatch, anthropic_key=None)
        with patch("httpx.post", return_value=_gemini_response("hi")) as post:
            client._call_text_sync("sys", "hello")
        assert post.call_args.args[0].startswith("https://generativelanguage.googleapis.com/v1beta/openai/")


class TestFallbackChain:
    """Gemini first, then Groq: the next provider is tried when one fails."""

    def _two_providers(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gq-test")
        return _client(monkeypatch)  # also sets ANTHROPIC + GEMINI keys

    def test_chain_order_is_gemini_then_groq(self, monkeypatch):
        client = self._two_providers(monkeypatch)
        assert [f["label"] for f in client._fallbacks] == ["gemini", "groq"]

    def test_groq_only_works_without_gemini_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gq-test")
        client = _client(monkeypatch, gemini_key=None)
        assert [f["label"] for f in client._fallbacks] == ["groq"]
        with patch("httpx.post", return_value=_gemini_response("from groq")) as post:
            client._client.beta.prompt_caching.messages.create.side_effect = RuntimeError("credits")
            assert client._call_text_sync("sys", "hi") == "from groq"
        assert post.call_args.args[0] == "https://api.groq.com/openai/v1/chat/completions"
        assert post.call_args.kwargs["json"]["model"] == "openai/gpt-oss-120b"

    def test_gemini_403_falls_through_to_groq(self, monkeypatch):
        client = self._two_providers(monkeypatch)
        client._client.beta.prompt_caching.messages.create.side_effect = RuntimeError("credits")
        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            if "generativelanguage" in url:
                return _gemini_response("", status=403)
            return _gemini_response("groq answer")

        with patch("httpx.post", side_effect=fake_post):
            assert client._call_text_sync("sys", "hi") == "groq answer"
        assert len(calls) == 2 and "groq" in calls[1]

    def test_all_providers_failing_reports_each(self, monkeypatch):
        client = self._two_providers(monkeypatch)
        client._client.beta.prompt_caching.messages.create.side_effect = RuntimeError("credits")
        with patch("httpx.post", return_value=_gemini_response("", status=403)):
            with pytest.raises(LLMCallError, match="gemini:.*groq:"):
                client._call_text_sync("sys", "hi")
