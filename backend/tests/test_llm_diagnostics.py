from __future__ import annotations

import io
import json
import sys
import tempfile
import traceback
import types
import unittest
import urllib.error
from copy import deepcopy
from email.message import Message
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.defaults import DEFAULT_CONFIG
from app.llm import (
    LLM_AUTH_ERROR,
    LLM_EMPTY_RESPONSE,
    LLM_HTTP_ERROR,
    LLM_INVALID_RESPONSE,
    LLM_RATE_LIMIT,
    LLMError,
    LLMMessage,
    LLMRateLimitError,
    create_llm_client,
    sanitize_diagnostic_data,
)
from app.pipeline import PipelineError, TaskContext, translate_segments


def chat_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    refusal: str | None = None,
    tool_calls: list[types.SimpleNamespace] | None = None,
    response_id: str = "chatcmpl-test",
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        id=response_id,
        choices=[
            types.SimpleNamespace(
                delta=types.SimpleNamespace(
                    content=content,
                    refusal=refusal,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
    )


def http_error(*, status: int, body: str, request_id: str = "req-test-123") -> urllib.error.HTTPError:
    headers = Message()
    headers["x-request-id"] = request_id
    return urllib.error.HTTPError(
        "https://user:password@example.test/v1/messages?api_key=sk-query-secret",
        status,
        "upstream error",
        headers,
        io.BytesIO(body.encode("utf-8")),
    )


class LLMDiagnosticsTests(unittest.TestCase):
    @patch("app.llm.providers.OpenAI")
    def test_chat_empty_content_filter_records_finish_reason_and_shape(self, mock_openai) -> None:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = [
            chat_chunk(content=None, finish_reason="content_filter")
        ]
        client = create_llm_client(
            "openai-chat",
            "https://api.openai.test",
            "sk-test-secret",
            "gpt-test",
            30,
        )

        with self.assertRaises(LLMError) as raised:
            client.complete([LLMMessage(role="user", content="secret subtitle transcript")])

        error = raised.exception
        self.assertEqual(error.error_code, LLM_EMPTY_RESPONSE)
        self.assertEqual(error.details["finish_reason"], "content_filter")
        self.assertEqual(error.details["choices_count"], 1)
        self.assertEqual(error.details["response_id"], "chatcmpl-test")
        self.assertIn("delta_keys", error.details)
        self.assertIn("elapsed_ms", error.details)

    @patch("app.llm.providers.OpenAI")
    def test_chat_refusal_without_content_is_diagnosed(self, mock_openai) -> None:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = [
            chat_chunk(content=None, finish_reason="stop", refusal="I cannot translate this.")
        ]
        client = create_llm_client("openai-chat", "https://api.openai.test", "sk-test-secret", "gpt-test", 30)

        with self.assertRaises(LLMError) as raised:
            client.complete([LLMMessage(role="user", content="hello")])

        self.assertEqual(raised.exception.error_code, LLM_EMPTY_RESPONSE)
        self.assertTrue(raised.exception.details["refusal_present"])
        self.assertIn("cannot translate", raised.exception.details["refusal_preview"])

    @patch("app.llm.providers.OpenAI")
    def test_chat_tool_calls_without_content_are_diagnosed(self, mock_openai) -> None:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = [
            chat_chunk(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    types.SimpleNamespace(
                        id="call_1",
                        type="function",
                        function=types.SimpleNamespace(name="lookup_translation"),
                    )
                ],
            )
        ]
        client = create_llm_client("openai-chat", "https://api.openai.test", "sk-test-secret", "gpt-test", 30)

        with self.assertRaises(LLMError) as raised:
            client.complete([LLMMessage(role="user", content="hello")])

        self.assertEqual(raised.exception.error_code, LLM_EMPTY_RESPONSE)
        self.assertEqual(raised.exception.details["tool_call_count"], 1)
        self.assertEqual(raised.exception.details["tool_call_names"], ["lookup_translation"])

    @patch("app.llm.providers.OpenAI")
    def test_responses_incomplete_details_is_structured(self, mock_openai) -> None:
        mock_client = mock_openai.return_value
        mock_client.responses.create.return_value = types.SimpleNamespace(
            id="resp-test",
            status="incomplete",
            incomplete_details=types.SimpleNamespace(reason="max_output_tokens"),
            output=[
                types.SimpleNamespace(
                    type="message",
                    content=[types.SimpleNamespace(type="output_text", text="")],
                )
            ],
            output_text="",
        )
        client = create_llm_client(
            "openai-responses",
            "https://api.openai.test",
            "sk-test-secret",
            "gpt-test",
            30,
        )

        with self.assertRaises(LLMError) as raised:
            client.complete([LLMMessage(role="user", content="hello")])

        details = raised.exception.details
        self.assertEqual(raised.exception.error_code, LLM_EMPTY_RESPONSE)
        self.assertEqual(details["response_status"], "incomplete")
        self.assertEqual(details["incomplete_details"]["reason"], "max_output_tokens")
        self.assertEqual(details["incomplete_reason"], "max_output_tokens")
        self.assertEqual(details["output_item_types"], ["message"])
        self.assertEqual(details["content_block_types"], ["output_text"])

    def test_anthropic_and_ollama_structure_errors_are_distinguishable(self) -> None:
        anthropic = create_llm_client(
            "anthropic",
            "https://api.anthropic.test",
            "client-secret",
            "claude-test",
            30,
        )
        with patch.object(
            anthropic,
            "_post_json",
            return_value={"content": "not-a-list", "stop_reason": "end_turn"},
        ):
            with self.assertRaises(LLMError) as raised_anthropic:
                anthropic.complete([LLMMessage(role="user", content="hello")])

        self.assertEqual(raised_anthropic.exception.error_code, LLM_INVALID_RESPONSE)
        self.assertEqual(raised_anthropic.exception.details["stop_reason"], "end_turn")
        self.assertIn("response_preview", raised_anthropic.exception.details)

        ollama = create_llm_client("ollama", "http://ollama.test", "", "qwen-test", 30)
        with patch.object(
            ollama,
            "_post_json",
            return_value={"message": [], "done_reason": "length"},
        ):
            with self.assertRaises(LLMError) as raised_ollama:
                ollama.complete([LLMMessage(role="user", content="hello")])

        self.assertEqual(raised_ollama.exception.error_code, LLM_INVALID_RESPONSE)
        self.assertEqual(raised_ollama.exception.details["done_reason"], "length")
        self.assertEqual(raised_ollama.exception.details["message_type"], "list")

    def test_anthropic_and_ollama_valid_empty_structures_are_empty_responses(self) -> None:
        anthropic = create_llm_client(
            "anthropic",
            "https://api.anthropic.test",
            "client-secret",
            "claude-test",
            30,
        )
        with patch.object(anthropic, "_post_json", return_value={"content": [], "stop_reason": "max_tokens"}):
            with self.assertRaises(LLMError) as raised_anthropic:
                anthropic.complete([LLMMessage(role="user", content="hello")])

        self.assertEqual(raised_anthropic.exception.error_code, LLM_EMPTY_RESPONSE)
        self.assertEqual(raised_anthropic.exception.details["content_block_types"], [])

        ollama = create_llm_client("ollama", "http://ollama.test", "", "qwen-test", 30)
        with patch.object(ollama, "_post_json", return_value={"message": {"role": "assistant"}, "done_reason": "stop"}):
            with self.assertRaises(LLMError) as raised_ollama:
                ollama.complete([LLMMessage(role="user", content="hello")])

        self.assertEqual(raised_ollama.exception.error_code, LLM_EMPTY_RESPONSE)
        self.assertEqual(raised_ollama.exception.details["message_keys"], ["role"])

    def test_http_429_and_500_are_classified_and_preview_is_sanitized(self) -> None:
        secret = "plain-client-secret"
        anthropic = create_llm_client(
            "anthropic",
            "https://api.anthropic.test",
            secret,
            "claude-test",
            30,
        )
        long_body = (
            '{"error":{"code":"rate_limit_exceeded"},'
            '"authorization":"Bearer sk-live-secret-123456",'
            f'"api_key":"{secret}",'
            '"url":"https://user:pass@example.test/hook?token=query-secret",'
            '"padding":"' + ("x" * 900) + '","tail":"SHOULD_NOT_APPEAR"}'
        )
        with patch("app.llm.providers.urllib.request.urlopen", side_effect=http_error(status=429, body=long_body)):
            with self.assertRaises(LLMRateLimitError) as raised_429:
                anthropic.complete([LLMMessage(role="user", content="hello")])

        error_429 = raised_429.exception
        preview_429 = error_429.details["upstream_body_preview"]
        serialized_429 = json.dumps(error_429.to_dict(), ensure_ascii=False)
        formatted_429 = "".join(traceback.format_exception(error_429))
        self.assertEqual(error_429.error_code, LLM_RATE_LIMIT)
        self.assertEqual(error_429.details["status_code"], 429)
        self.assertEqual(error_429.details["request_id"], "req-test-123")
        self.assertLessEqual(len(preview_429), 800)
        self.assertTrue(error_429.details["upstream_body_truncated"])
        self.assertNotIn("SHOULD_NOT_APPEAR", preview_429)
        for leaked in ("sk-live-secret", secret, "query-secret", "user:pass", "api_key=sk-query-secret"):
            self.assertNotIn(leaked, serialized_429)
            self.assertNotIn(leaked, formatted_429)
        self.assertNotIn("token=query-secret", serialized_429)

        with patch("app.llm.providers.urllib.request.urlopen", side_effect=http_error(status=500, body='{"error":"boom"}')):
            with self.assertRaises(LLMError) as raised_500:
                anthropic.complete([LLMMessage(role="user", content="hello")])

        self.assertEqual(raised_500.exception.error_code, LLM_HTTP_ERROR)
        self.assertEqual(raised_500.exception.details["status_code"], 500)
        self.assertEqual(raised_500.exception.details["request_id"], "req-test-123")

    def test_sanitizer_removes_credentials_and_url_queries(self) -> None:
        payload = {
            "authorization": "Bearer sk-secret-token-123456",
            "x-api-key": "plain-secret",
            "nested": {
                "api_key": "sk-nested-secret-123456",
                "endpoint": "https://user:password@example.test/v1/chat?api_key=sk-query-secret&foo=bar",
            },
            "text": "Authorization: Bearer sk-body-secret-123456 https://example.test/x?token=abc",
            "prompt": "private prompt",
            "subtitle": "private subtitle text",
        }
        sanitized = sanitize_diagnostic_data(payload, secrets=("plain-secret",))
        serialized = json.dumps(sanitized, ensure_ascii=False)

        self.assertEqual(sanitized["authorization"], "[REDACTED]")
        self.assertEqual(sanitized["x-api-key"], "[REDACTED]")
        self.assertNotIn("sk-secret-token", serialized)
        self.assertNotIn("plain-secret", serialized)
        self.assertNotIn("sk-query-secret", serialized)
        self.assertNotIn("foo=bar", serialized)
        self.assertNotIn("token=abc", serialized)
        self.assertNotIn("user:password", serialized)
        self.assertEqual(sanitized["prompt"], "[REDACTED]")
        self.assertEqual(sanitized["subtitle"], "[REDACTED]")

    def test_translation_error_preserves_target_language_and_chunk_range(self) -> None:
        class FailingClient:
            def complete(self, messages, cancellation_check=None):
                raise LLMError(
                    "翻译服务未返回文本内容",
                    {"provider": "openai-chat", "model": "gpt-test", "finish_reason": "stop"},
                    error_code=LLM_EMPTY_RESPONSE,
                )

            def resolved_base_url(self) -> str:
                return "https://api.openai.test/v1"

        snapshot = deepcopy(DEFAULT_CONFIG)
        snapshot["translation"].update(
            {
                "enabled": True,
                "target_languages": ["zh-CN"],
                "max_retries": 1,
                "llm_type": "openai-chat",
                "api_base_url": "https://api.openai.test",
                "api_key": "sk-test-secret",
                "model": "gpt-test",
                "timeout_seconds": 30,
            }
        )
        segments = [
            {"start": float(index), "end": float(index + 1), "text": f"line {index}"}
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            context = TaskContext(
                task_id=1,
                file_path=str(Path(temp_dir) / "video.mp4"),
                config_snapshot=snapshot,
                work_dir=Path(temp_dir) / "work",
            )
            with patch("app.pipeline.create_llm_client", return_value=FailingClient()):
                with self.assertLogs("app.pipeline", level="ERROR"):
                    with self.assertRaises(PipelineError) as raised:
                        translate_segments(context, segments)

        error = raised.exception
        self.assertEqual(error.error_code, LLM_EMPTY_RESPONSE)
        self.assertEqual(error.details["provider"], "openai-chat")
        self.assertEqual(error.details["model"], "gpt-test")
        self.assertEqual(error.details["target_language"], "zh-CN")
        self.assertEqual(error.details["chunk_start_index"], 0)
        self.assertEqual(error.details["chunk_end_index"], 14)
        self.assertEqual(error.details["attempt"], 1)
        self.assertIn("elapsed_ms", error.details)

    def test_auth_error_code_is_stable(self) -> None:
        client = create_llm_client("openai-chat", "https://api.openai.test", "", "gpt-test", 30)
        with self.assertRaises(LLMError) as raised:
            client.complete([LLMMessage(role="user", content="hello")])
        self.assertEqual(raised.exception.error_code, LLM_AUTH_ERROR)


if __name__ == "__main__":
    unittest.main()
