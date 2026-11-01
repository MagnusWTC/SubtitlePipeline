from __future__ import annotations

import asyncio
import json
import re
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import CancelledError
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import openai
from openai import OpenAI


SUPPORTED_LLM_TYPES = {
    "openai-chat": {
        "label": "OpenAI Chat Completions",
        "default_base_url": "https://api.openai.com",
        "requires_api_key": True,
    },
    "openai-responses": {
        "label": "OpenAI Responses",
        "default_base_url": "https://api.openai.com",
        "requires_api_key": True,
    },
    "openai-compatible": {
        "label": "OpenAI Compatible",
        "default_base_url": "https://api.openai.com",
        "requires_api_key": True,
    },
    "anthropic": {
        "label": "Anthropic Messages",
        "default_base_url": "https://api.anthropic.com",
        "requires_api_key": True,
    },
    "lmstudio": {
        "label": "LM Studio",
        "default_base_url": "http://localhost:1234",
        "requires_api_key": False,
    },
    "ollama": {
        "label": "Ollama",
        "default_base_url": "http://localhost:11434",
        "requires_api_key": False,
    },
}


LLM_EMPTY_RESPONSE = "LLM_EMPTY_RESPONSE"
LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"
LLM_HTTP_ERROR = "LLM_HTTP_ERROR"
LLM_AUTH_ERROR = "LLM_AUTH_ERROR"
LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
LLM_CONNECTION_ERROR = "LLM_CONNECTION_ERROR"
LLM_TIMEOUT = "LLM_TIMEOUT"
LLM_CANCELLED = "LLM_CANCELLED"
LLM_UNKNOWN_ERROR = "LLM_UNKNOWN_ERROR"

UPSTREAM_PREVIEW_LIMIT = 800
REFUSAL_PREVIEW_LIMIT = 500

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "body",
    "bearer",
    "content",
    "cookie",
    "credential",
    "credentials",
    "input",
    "lines",
    "messages",
    "password",
    "prompt",
    "raw_body",
    "request_body",
    "response_body",
    "secret",
    "segments",
    "set_cookie",
    "source_text",
    "subtitle",
    "subtitles",
    "system_prompt",
    "text",
    "token",
    "transcript",
    "x_api_key",
}
_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)([\"']?authorization[\"']?\s*[:=]\s*)(?:[\"']?bearer\s+)?[^\"'\r\n,;}]+"
)
_API_KEY_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)([\"']?(?:x-api-key|api[-_]?key)[\"']?\s*[:=]\s*)[\"']?[^\"'\s,;}]+"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{4,}")
_SK_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b", re.IGNORECASE)


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[\s-]+", "_", key.strip().lower())
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.endswith("_token")
        or normalized.endswith("_secret")
        or normalized.endswith("_password")
        or normalized.endswith("_credential")
    )


def sanitize_url(url: str) -> str:
    """Remove credentials, query parameters, and fragments from a URL."""
    value = str(url or "").strip()
    if not value:
        return ""
    try:
        parts = urlsplit(value)
        if parts.scheme and parts.hostname:
            hostname = parts.hostname
            if ":" in hostname and not hostname.startswith("["):
                hostname = f"[{hostname}]"
            try:
                port = f":{parts.port}" if parts.port else ""
            except ValueError:
                port = ""
            return urlunsplit((parts.scheme, f"{hostname}{port}", parts.path, "", ""))
    except ValueError:
        pass
    return value.split("?", 1)[0].split("#", 1)[0]


def _sanitize_string(value: str, secrets: tuple[str, ...]) -> str:
    sanitized = value
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, "[REDACTED]")
    sanitized = _AUTHORIZATION_PATTERN.sub(r"\1[REDACTED]", sanitized)
    sanitized = _API_KEY_ASSIGNMENT_PATTERN.sub(r"\1[REDACTED]", sanitized)
    sanitized = _BEARER_PATTERN.sub("Bearer [REDACTED]", sanitized)
    sanitized = _SK_KEY_PATTERN.sub("[REDACTED_API_KEY]", sanitized)

    def replace_url(match: re.Match[str]) -> str:
        candidate = match.group(0)
        trailing = ""
        while candidate and candidate[-1] in ".,);]":
            trailing = candidate[-1] + trailing
            candidate = candidate[:-1]
        return f"{sanitize_url(candidate)}{trailing}"

    return _URL_PATTERN.sub(replace_url, sanitized)


def _plain_diagnostic(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_diagnostic(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain_diagnostic(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _plain_diagnostic(model_dump())
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _plain_diagnostic(to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {str(key): _plain_diagnostic(item) for key, item in vars(value).items()}
    return value


def sanitize_diagnostic_data(
    value: Any,
    *,
    secrets: Iterable[str] = (),
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    """Return JSON-safe diagnostic data with credentials and URL queries removed."""
    secret_values = tuple(sorted({str(item) for item in secrets if item}, key=len, reverse=True))
    if _seen is None:
        _seen = set()
    if _depth > 8:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _sanitize_string(value, secret_values)
    if isinstance(value, bytes):
        return _sanitize_string(value.decode("utf-8", errors="replace"), secret_values)
    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in _seen:
            return "[CIRCULAR]"
        _seen.add(object_id)
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            result[key_text] = (
                "[REDACTED]"
                if _is_sensitive_key(key_text)
                else sanitize_diagnostic_data(item, secrets=secret_values, _depth=_depth + 1, _seen=_seen)
            )
        _seen.discard(object_id)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        object_id = id(value)
        if object_id in _seen:
            return "[CIRCULAR]"
        _seen.add(object_id)
        result = [
            sanitize_diagnostic_data(item, secrets=secret_values, _depth=_depth + 1, _seen=_seen)
            for item in value
        ]
        _seen.discard(object_id)
        return result
    plain = _plain_diagnostic(value)
    if plain is not value:
        return sanitize_diagnostic_data(plain, secrets=secret_values, _depth=_depth + 1, _seen=_seen)
    return _sanitize_string(str(value), secret_values)


def _safe_preview(value: Any, *, api_key: str = "", limit: int = UPSTREAM_PREVIEW_LIMIT) -> str:
    plain = _plain_diagnostic(value)
    if isinstance(plain, str):
        rendered = plain
    else:
        try:
            rendered = json.dumps(plain, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rendered = str(plain)
    return _sanitize_string(rendered, (api_key,))[:limit]


class LLMError(RuntimeError):
    """Provider error with a stable code and structured, sanitized diagnostics."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        *,
        error_code: str = LLM_UNKNOWN_ERROR,
    ) -> None:
        super().__init__(message)
        self.error_code = str(error_code)
        self.details = sanitize_diagnostic_data(details or {})

    def with_context(self, **context: Any) -> "LLMError":
        self.details = sanitize_diagnostic_data({**self.details, **context})
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": str(self),
            "details": self.details,
        }


class LLMRateLimitError(LLMError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details, error_code=LLM_RATE_LIMIT)


def _llm_details(
    provider: str,
    model: str,
    *,
    error_code: str | None = None,
    status_code: int | None = None,
    finish_reason: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    details: dict[str, Any] = {"provider": provider, "model": model}
    if error_code:
        details["error_code"] = error_code
    if status_code is not None:
        details["status_code"] = int(status_code)
    if finish_reason:
        details["finish_reason"] = str(finish_reason)
    details.update({key: value for key, value in extra.items() if value is not None})
    return details


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def _exception_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _header_value(headers: Any, names: tuple[str, ...]) -> str | None:
    if headers is None:
        return None
    for name in names:
        try:
            value = headers.get(name)
        except AttributeError:
            value = None
        if value:
            return str(value)
    return None


def _request_id_from_headers(headers: Any) -> str | None:
    return _header_value(
        headers,
        ("x-request-id", "request-id", "x-amzn-requestid", "x-correlation-id", "cf-ray"),
    )


def _exception_request_id(exc: BaseException) -> str | None:
    request_id = getattr(exc, "request_id", None)
    if request_id:
        return str(request_id)
    response = getattr(exc, "response", None)
    return _request_id_from_headers(getattr(response, "headers", None)) or _request_id_from_headers(
        getattr(exc, "headers", None)
    )


def _exception_endpoint(exc: BaseException, fallback: str) -> str:
    request = getattr(exc, "request", None)
    if request is None:
        response = getattr(exc, "response", None)
        request = getattr(response, "request", None)
    return sanitize_url(str(getattr(request, "url", "") or fallback))


def _exception_preview(exc: BaseException, *, api_key: str) -> str:
    response = getattr(exc, "response", None)
    candidates = [
        getattr(response, "text", None),
        getattr(exc, "body", None),
        getattr(exc, "message", None),
    ]
    for candidate in candidates:
        if candidate:
            return _safe_preview(candidate, api_key=api_key)
    return ""


def _openai_exception_to_error(
    exc: BaseException,
    *,
    provider: str,
    model: str,
    fallback_url: str,
    api_key: str,
    started_at: float,
) -> LLMError:
    status_code = _exception_status(exc)
    common = {
        "status_code": status_code,
        "endpoint": _exception_endpoint(exc, fallback_url),
        "request_id": _exception_request_id(exc),
        "response_preview": _exception_preview(exc, api_key=api_key),
        "exception_type": type(exc).__name__,
        "elapsed_ms": _elapsed_ms(started_at),
    }
    if isinstance(exc, openai.AuthenticationError) or status_code in {401, 403}:
        return LLMError(
            "翻译服务鉴权失败，请检查 API Key",
            _llm_details(provider, model, error_code=LLM_AUTH_ERROR, **common),
            error_code=LLM_AUTH_ERROR,
        )
    if isinstance(exc, openai.RateLimitError) or status_code == 429:
        return LLMRateLimitError(
            "翻译服务触发限流，请稍后重试",
            _llm_details(provider, model, error_code=LLM_RATE_LIMIT, **common),
        )
    if isinstance(exc, openai.APITimeoutError) or isinstance(exc, (TimeoutError, socket.timeout)):
        return LLMError(
            "翻译服务请求超时，请稍后重试",
            _llm_details(provider, model, error_code=LLM_TIMEOUT, **common),
            error_code=LLM_TIMEOUT,
        )
    if isinstance(exc, openai.APIConnectionError):
        return LLMError(
            "翻译服务连接失败，请检查 API 地址、网络或服务状态",
            _llm_details(provider, model, error_code=LLM_CONNECTION_ERROR, **common),
            error_code=LLM_CONNECTION_ERROR,
        )
    if status_code is not None:
        return LLMError(
            f"翻译服务返回异常状态: {status_code}",
            _llm_details(provider, model, error_code=LLM_HTTP_ERROR, **common),
            error_code=LLM_HTTP_ERROR,
        )
    return LLMError(
        "翻译服务调用失败",
        _llm_details(provider, model, error_code=LLM_UNKNOWN_ERROR, **common),
        error_code=LLM_UNKNOWN_ERROR,
    )


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


CancellationCheck = Callable[[], None]


def _check_cancelled(cancellation_check: CancellationCheck | None) -> None:
    if cancellation_check is not None:
        cancellation_check()


class LLMClient:
    def complete(self, messages: list[LLMMessage], cancellation_check: CancellationCheck | None = None) -> str:
        raise NotImplementedError

    def resolved_base_url(self) -> str:
        raise NotImplementedError


def normalize_llm_type(value: str | None) -> str:
    normalized = str(value or "openai-compatible").strip().lower()
    legacy_mapping = {
        "openai": "openai-chat",
        "openai-chat": "openai-chat",
        "openai_chat": "openai-chat",
        "chat": "openai-chat",
        "chat-completions": "openai-chat",
        "chat_completions": "openai-chat",
        "openai-responses": "openai-responses",
        "openai_responses": "openai-responses",
        "responses": "openai-responses",
        "openai_compatible": "openai-compatible",
        "openai-compatible": "openai-compatible",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "lmstudio": "lmstudio",
        "lm-studio": "lmstudio",
        "lm_studio": "lmstudio",
        "ollama": "ollama",
    }
    return legacy_mapping.get(normalized, "openai-compatible")


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


class OpenAICompatibleLLMClient(LLMClient):
    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int,
        *,
        default_base_url: str | None = None,
        requires_api_key: bool = True,
        provider_name: str = "openai-compatible",
    ):
        self.api_base_url = (api_base_url or default_base_url or SUPPORTED_LLM_TYPES["openai-compatible"]["default_base_url"]).rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.requires_api_key = requires_api_key
        self.provider_name = provider_name
        self.client = OpenAI(
            api_key=self.api_key or "missing-api-key",
            base_url=self.resolved_base_url(),
            timeout=float(self.timeout_seconds),
            max_retries=0,
        )

    def resolved_base_url(self) -> str:
        if self.api_base_url.endswith("/v1"):
            return self.api_base_url
        return f"{self.api_base_url}/v1"

    def complete(self, messages: list[LLMMessage], cancellation_check: CancellationCheck | None = None) -> str:
        _check_cancelled(cancellation_check)
        started_at = time.perf_counter()
        if self.requires_api_key and not self.api_key:
            details = _llm_details(
                self.provider_name,
                self.model,
                error_code=LLM_AUTH_ERROR,
                endpoint=sanitize_url(self.resolved_base_url()),
                elapsed_ms=_elapsed_ms(started_at),
            )
            raise LLMError(
                "translation.api_key 未配置，无法调用翻译服务",
                details,
                error_code=LLM_AUTH_ERROR,
            )

        parts: list[str] = []
        finish_reason: str | None = None
        response_id: str | None = None
        choices_count = 0
        chunk_count = 0
        delta_keys: set[str] = set()
        refusal_present = False
        refusal_preview = ""
        tool_call_count = 0
        tool_call_names: list[str] = []
        invalid_shape = False
        invalid_content_type: str | None = None

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": item.role, "content": item.content} for item in messages],
                max_tokens=8192,
                stream=True,
                temperature=0.3,
                frequency_penalty=1.2,
                presence_penalty=0.8,
            )
            for chunk in stream:
                _check_cancelled(cancellation_check)
                chunk_count += 1
                if response_id is None:
                    candidate_id = _field(chunk, "id")
                    response_id = str(candidate_id) if candidate_id else None
                choices = _field(chunk, "choices", None)
                if choices is None or not isinstance(choices, list):
                    invalid_shape = True
                    continue
                choices_count = max(choices_count, len(choices))
                if not choices:
                    continue
                choice = choices[0]
                candidate_finish_reason = _field(choice, "finish_reason")
                if candidate_finish_reason:
                    finish_reason = str(candidate_finish_reason)
                delta = _field(choice, "delta")
                message = _field(choice, "message")
                structure = delta if delta is not None else message
                if structure is None:
                    continue
                if isinstance(structure, Mapping):
                    delta_keys.update(str(key) for key in structure.keys())
                elif hasattr(structure, "__dict__"):
                    delta_keys.update(str(key) for key in vars(structure).keys())
                content_value = _field(structure, "content")
                if isinstance(content_value, str):
                    if content_value:
                        parts.append(content_value)
                elif content_value is not None:
                    invalid_shape = True
                    invalid_content_type = type(content_value).__name__
                refusal = _field(structure, "refusal")
                if refusal:
                    refusal_present = True
                    refusal_preview = _safe_preview(refusal, api_key=self.api_key, limit=REFUSAL_PREVIEW_LIMIT)
                tool_calls = _field(structure, "tool_calls")
                if isinstance(tool_calls, list):
                    tool_call_count += len(tool_calls)
                    for tool_call in tool_calls:
                        function = _field(tool_call, "function")
                        name = _field(function, "name") or _field(tool_call, "name")
                        if name:
                            tool_call_names.append(str(name))
                elif tool_calls is not None:
                    invalid_shape = True
        except (CancelledError, asyncio.CancelledError) as exc:
            details = _llm_details(
                self.provider_name,
                self.model,
                error_code=LLM_CANCELLED,
                endpoint=sanitize_url(self.resolved_base_url()),
                elapsed_ms=_elapsed_ms(started_at),
                exception_type=type(exc).__name__,
            )
            raise LLMError(
                "翻译服务调用已取消",
                details,
                error_code=LLM_CANCELLED,
            ) from None
        except openai.APIError as exc:
            raise _openai_exception_to_error(
                exc,
                provider=self.provider_name,
                model=self.model,
                fallback_url=self.resolved_base_url(),
                api_key=self.api_key,
                started_at=started_at,
            ) from None

        content = "".join(parts)
        if content:
            _check_cancelled(cancellation_check)
            return content

        error_code = LLM_INVALID_RESPONSE if invalid_shape or invalid_content_type else LLM_EMPTY_RESPONSE
        details = _llm_details(
            self.provider_name,
            self.model,
            error_code=error_code,
            finish_reason=finish_reason,
            response_id=response_id,
            choices_count=choices_count,
            chunk_count=chunk_count,
            delta_keys=sorted(delta_keys),
            delta_content_chars=len(content),
            invalid_content_type=invalid_content_type,
            refusal_present=refusal_present,
            refusal_preview=refusal_preview or None,
            tool_call_count=tool_call_count,
            tool_call_names=sorted(set(tool_call_names)),
            elapsed_ms=_elapsed_ms(started_at),
        )
        message = "翻译服务返回内容格式无效" if error_code == LLM_INVALID_RESPONSE else "翻译服务未返回文本内容"
        raise LLMError(message, details, error_code=error_code)


class OpenAIResponsesLLMClient(LLMClient):
    def __init__(self, api_base_url: str, api_key: str, model: str, timeout_seconds: int):
        self.api_base_url = (api_base_url or SUPPORTED_LLM_TYPES["openai-responses"]["default_base_url"]).rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.client = OpenAI(
            api_key=self.api_key or "missing-api-key",
            base_url=self.resolved_base_url(),
            timeout=float(self.timeout_seconds),
            max_retries=0,
        )

    def resolved_base_url(self) -> str:
        if self.api_base_url.endswith("/v1"):
            return self.api_base_url
        return f"{self.api_base_url}/v1"

    def complete(self, messages: list[LLMMessage], cancellation_check: CancellationCheck | None = None) -> str:
        _check_cancelled(cancellation_check)
        started_at = time.perf_counter()
        if not self.api_key:
            details = _llm_details(
                "openai-responses",
                self.model,
                error_code=LLM_AUTH_ERROR,
                endpoint=sanitize_url(self.resolved_base_url()),
                elapsed_ms=_elapsed_ms(started_at),
            )
            raise LLMError(
                "translation.api_key 未配置，无法调用翻译服务",
                details,
                error_code=LLM_AUTH_ERROR,
            )
        instructions = "\n\n".join(item.content for item in messages if item.role == "system")
        input_text = "\n\n".join(item.content for item in messages if item.role != "system")
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions or None,
                input=input_text,
                max_output_tokens=8192,
                temperature=0.3,
            )
        except (CancelledError, asyncio.CancelledError) as exc:
            details = _llm_details(
                "openai-responses",
                self.model,
                error_code=LLM_CANCELLED,
                endpoint=sanitize_url(self.resolved_base_url()),
                elapsed_ms=_elapsed_ms(started_at),
                exception_type=type(exc).__name__,
            )
            raise LLMError(
                "翻译服务调用已取消",
                details,
                error_code=LLM_CANCELLED,
            ) from None
        except openai.APIError as exc:
            raise _openai_exception_to_error(
                exc,
                provider="openai-responses",
                model=self.model,
                fallback_url=self.resolved_base_url(),
                api_key=self.api_key,
                started_at=started_at,
            ) from None
        output_text = _field(response, "output_text", "")
        content = output_text.strip() if isinstance(output_text, str) else ""
        diagnostics, structurally_valid = self._response_diagnostics(response)
        extracted_text = diagnostics.pop("_text", "")
        if not content and structurally_valid:
            content = extracted_text.strip()
        diagnostics.update(
            {
                "response_id": diagnostics.get("response_id") or (_field(response, "id") and str(_field(response, "id"))),
                "elapsed_ms": _elapsed_ms(started_at),
            }
        )
        if content:
            _check_cancelled(cancellation_check)
            return content

        error_code = LLM_EMPTY_RESPONSE if structurally_valid else LLM_INVALID_RESPONSE
        diagnostics.update(
            _llm_details(
                "openai-responses",
                self.model,
                error_code=error_code,
                finish_reason=diagnostics.get("incomplete_reason"),
                response_preview=None if structurally_valid else _safe_preview(response, api_key=self.api_key),
            )
        )
        message = "OpenAI Responses 翻译服务未返回文本内容" if structurally_valid else "OpenAI Responses 翻译服务返回内容格式无效"
        raise LLMError(message, diagnostics, error_code=error_code)

    def _response_diagnostics(self, response: Any) -> tuple[dict[str, Any], bool]:
        output = _field(response, "output", None)
        structurally_valid = output is None or isinstance(output, list)
        output_item_types: list[str] = []
        content_block_types: list[str] = []
        refusal_present = False
        text_parts: list[str] = []
        if isinstance(output, list):
            for item in output:
                item_type = _field(item, "type")
                output_item_types.append(str(item_type) if item_type else "unknown")
                content = _field(item, "content")
                if content is None:
                    continue
                if not isinstance(content, list):
                    structurally_valid = False
                    continue
                for part in content:
                    part_type = _field(part, "type")
                    content_block_types.append(str(part_type) if part_type else "unknown")
                    if _field(part, "refusal"):
                        refusal_present = True
                    part_text = _field(part, "text")
                    if isinstance(part_text, str):
                        text_parts.append(part_text)
                    elif part_text is not None:
                        structurally_valid = False
        incomplete_details = _plain_diagnostic(_field(response, "incomplete_details"))
        incomplete_reason = incomplete_details.get("reason") if isinstance(incomplete_details, Mapping) else None
        response_keys: list[str] = []
        if hasattr(response, "__dict__"):
            response_keys = sorted(str(key) for key in vars(response).keys())
        elif isinstance(response, Mapping):
            response_keys = sorted(str(key) for key in response.keys())
        diagnostics = {
            "_text": "".join(text_parts),
            "response_status": _field(response, "status"),
            "incomplete_details": incomplete_details,
            "incomplete_reason": str(incomplete_reason) if incomplete_reason else None,
            "response_id": _field(response, "id"),
            "output_item_types": output_item_types,
            "content_block_types": content_block_types,
            "refusal_present": refusal_present,
            "response_keys": response_keys,
        }
        return diagnostics, structurally_valid


class JsonHttpLLMClient(LLMClient):
    provider_name = "http"

    def __init__(self, api_base_url: str, api_key: str, model: str, timeout_seconds: int):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._last_request_id: str | None = None

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        started_at = time.perf_counter()
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                **headers,
            },
            method="POST",
        )
        endpoint = sanitize_url(url)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                self._last_request_id = _request_id_from_headers(getattr(response, "headers", None))
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="replace")
            upstream_error_code: str | None = None
            try:
                error_payload = json.loads(raw_body)
                if isinstance(error_payload, dict):
                    error_value = error_payload.get("error")
                    candidate = error_value.get("code") or error_value.get("type") if isinstance(error_value, dict) else None
                    upstream_error_code = str(candidate)[:100] if candidate else None
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            status_code = int(exc.code)
            common = {
                "status_code": status_code,
                "endpoint": endpoint,
                "request_id": _request_id_from_headers(getattr(exc, "headers", None)),
                "upstream_error_code": upstream_error_code,
                "upstream_body_preview": _safe_preview(raw_body, api_key=self.api_key),
                "upstream_body_length": len(raw_body),
                "upstream_body_truncated": len(raw_body) > UPSTREAM_PREVIEW_LIMIT,
                "elapsed_ms": _elapsed_ms(started_at),
                "exception_type": type(exc).__name__,
            }
            if status_code in {401, 403}:
                details = _llm_details(self.provider_name, self.model, error_code=LLM_AUTH_ERROR, **common)
                raise LLMError("翻译服务鉴权失败，请检查 API Key", details, error_code=LLM_AUTH_ERROR) from None
            if status_code == 429:
                raise LLMRateLimitError(
                    "翻译服务触发限流，请稍后重试",
                    _llm_details(self.provider_name, self.model, error_code=LLM_RATE_LIMIT, **common),
                ) from None
            details = _llm_details(self.provider_name, self.model, error_code=LLM_HTTP_ERROR, **common)
            raise LLMError(
                f"翻译服务返回异常状态: {status_code}",
                details,
                error_code=LLM_HTTP_ERROR,
            ) from None
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            is_timeout = isinstance(reason, (TimeoutError, socket.timeout)) or isinstance(exc, TimeoutError)
            error_code = LLM_TIMEOUT if is_timeout else LLM_CONNECTION_ERROR
            details = _llm_details(
                self.provider_name,
                self.model,
                error_code=error_code,
                endpoint=endpoint,
                elapsed_ms=_elapsed_ms(started_at),
                exception_type=type(exc).__name__,
            )
            message = "翻译服务请求超时，请稍后重试" if is_timeout else "翻译服务连接失败，请检查 API 地址、网络或服务状态"
            raise LLMError(
                message,
                details,
                error_code=error_code,
            ) from None
        except (TimeoutError, socket.timeout) as exc:
            details = _llm_details(
                self.provider_name,
                self.model,
                error_code=LLM_TIMEOUT,
                endpoint=endpoint,
                elapsed_ms=_elapsed_ms(started_at),
                exception_type=type(exc).__name__,
            )
            raise LLMError("翻译服务请求超时，请稍后重试", details, error_code=LLM_TIMEOUT) from None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            details = _llm_details(
                self.provider_name,
                self.model,
                error_code=LLM_INVALID_RESPONSE,
                endpoint=endpoint,
                request_id=self._last_request_id,
                response_preview=_safe_preview(raw, api_key=self.api_key),
                elapsed_ms=_elapsed_ms(started_at),
                exception_type=type(exc).__name__,
            )
            raise LLMError(
                "翻译服务返回了无法解析的 JSON",
                details,
                error_code=LLM_INVALID_RESPONSE,
            ) from None
        if not isinstance(parsed, dict):
            details = _llm_details(
                self.provider_name,
                self.model,
                error_code=LLM_INVALID_RESPONSE,
                endpoint=endpoint,
                request_id=self._last_request_id,
                response_preview=_safe_preview(parsed, api_key=self.api_key),
                elapsed_ms=_elapsed_ms(started_at),
            )
            raise LLMError(
                "翻译服务返回内容格式无效",
                details,
                error_code=LLM_INVALID_RESPONSE,
            )
        return parsed


class AnthropicLLMClient(JsonHttpLLMClient):
    provider_name = "anthropic"

    def __init__(self, api_base_url: str, api_key: str, model: str, timeout_seconds: int):
        super().__init__(api_base_url or SUPPORTED_LLM_TYPES["anthropic"]["default_base_url"], api_key, model, timeout_seconds)

    def resolved_base_url(self) -> str:
        return self.api_base_url

    def _messages_url(self) -> str:
        if self.api_base_url.endswith("/v1"):
            return _join_url(self.api_base_url, "/messages")
        return _join_url(self.api_base_url, "/v1/messages")

    def complete(self, messages: list[LLMMessage], cancellation_check: CancellationCheck | None = None) -> str:
        _check_cancelled(cancellation_check)
        started_at = time.perf_counter()
        if not self.api_key:
            details = _llm_details(
                "anthropic",
                self.model,
                error_code=LLM_AUTH_ERROR,
                endpoint=sanitize_url(self._messages_url()),
                elapsed_ms=_elapsed_ms(started_at),
            )
            raise LLMError(
                "translation.api_key 未配置，无法调用翻译服务",
                details,
                error_code=LLM_AUTH_ERROR,
            )
        system_prompt = "\n\n".join(item.content for item in messages if item.role == "system")
        user_messages = [
            {"role": "assistant" if item.role == "assistant" else "user", "content": item.content}
            for item in messages
            if item.role != "system"
        ]
        if system_prompt:
            if user_messages:
                user_messages[0]["content"] = f"{system_prompt}\n\n{user_messages[0]['content']}"
            else:
                user_messages.append({"role": "user", "content": system_prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "temperature": 0.3,
            "messages": user_messages,
        }
        response = self._post_json(
            self._messages_url(),
            payload,
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        parts = response.get("content")
        block_types: list[str] = []
        structurally_valid = isinstance(parts, list)
        text_parts: list[str] = []
        if isinstance(parts, list):
            for item in parts:
                if not isinstance(item, dict):
                    structurally_valid = False
                    block_types.append(type(item).__name__)
                    continue
                block_type = item.get("type")
                block_types.append(str(block_type) if block_type else "unknown")
                if block_type == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                    elif text is not None:
                        structurally_valid = False
        error_code = LLM_EMPTY_RESPONSE if structurally_valid else LLM_INVALID_RESPONSE
        diagnostics = _llm_details(
            "anthropic",
            self.model,
            error_code=error_code,
            finish_reason=response.get("stop_reason"),
            stop_reason=response.get("stop_reason"),
            request_id=self._last_request_id,
            content_block_types=block_types,
            content_block_count=len(parts) if isinstance(parts, list) else None,
            response_preview=None if structurally_valid else _safe_preview(response, api_key=self.api_key),
            elapsed_ms=_elapsed_ms(started_at),
        )
        text = "".join(text_parts)
        if not structurally_valid:
            raise LLMError("Anthropic 翻译服务返回内容格式无效", diagnostics, error_code=LLM_INVALID_RESPONSE)
        if not text:
            raise LLMError("Anthropic 翻译服务未返回文本内容", diagnostics, error_code=LLM_EMPTY_RESPONSE)
        _check_cancelled(cancellation_check)
        return text


class OllamaLLMClient(JsonHttpLLMClient):
    provider_name = "ollama"

    def __init__(self, api_base_url: str, api_key: str, model: str, timeout_seconds: int):
        super().__init__(api_base_url or SUPPORTED_LLM_TYPES["ollama"]["default_base_url"], api_key, model, timeout_seconds)

    def resolved_base_url(self) -> str:
        return self.api_base_url

    def _chat_url(self) -> str:
        if self.api_base_url.endswith("/api"):
            return _join_url(self.api_base_url, "/chat")
        return _join_url(self.api_base_url, "/api/chat")

    def complete(self, messages: list[LLMMessage], cancellation_check: CancellationCheck | None = None) -> str:
        _check_cancelled(cancellation_check)
        started_at = time.perf_counter()
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "options": {
                "temperature": 0.3,
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = self._post_json(self._chat_url(), payload, headers)
        message = response.get("message")
        message_mapping = message if isinstance(message, dict) else None
        message_keys = sorted(str(key) for key in message_mapping.keys()) if message_mapping is not None else []
        content = message_mapping.get("content") if message_mapping is not None else None
        tool_calls = message_mapping.get("tool_calls") if message_mapping is not None else None
        structurally_valid = message_mapping is not None and (content is None or isinstance(content, str))
        error_code = LLM_EMPTY_RESPONSE if structurally_valid else LLM_INVALID_RESPONSE
        diagnostics = _llm_details(
            "ollama",
            self.model,
            error_code=error_code,
            finish_reason=response.get("done_reason"),
            done_reason=response.get("done_reason"),
            request_id=self._last_request_id,
            message_type=type(message).__name__,
            message_keys=message_keys,
            message_role=message_mapping.get("role") if message_mapping is not None else None,
            content_type=type(content).__name__ if content is not None else None,
            content_chars=len(content) if isinstance(content, str) else None,
            tool_call_count=len(tool_calls) if isinstance(tool_calls, list) else (1 if tool_calls is not None else 0),
            response_keys=sorted(str(key) for key in response.keys()),
            response_preview=None if structurally_valid else _safe_preview(response, api_key=self.api_key),
            elapsed_ms=_elapsed_ms(started_at),
        )
        if not structurally_valid:
            raise LLMError("Ollama 翻译服务返回内容格式无效", diagnostics, error_code=LLM_INVALID_RESPONSE)
        content_text = content.strip() if isinstance(content, str) else ""
        if not content_text:
            raise LLMError("Ollama 翻译服务未返回文本内容", diagnostics, error_code=LLM_EMPTY_RESPONSE)
        _check_cancelled(cancellation_check)
        return content_text


def create_llm_client(
    llm_type: str | None,
    api_base_url: str,
    api_key: str,
    model: str,
    timeout_seconds: int,
) -> LLMClient:
    normalized = normalize_llm_type(llm_type)
    if normalized == "openai-chat":
        return OpenAICompatibleLLMClient(
            api_base_url,
            api_key,
            model,
            timeout_seconds,
            default_base_url=SUPPORTED_LLM_TYPES["openai-chat"]["default_base_url"],
            requires_api_key=True,
            provider_name="openai-chat",
        )
    if normalized == "openai-responses":
        return OpenAIResponsesLLMClient(api_base_url, api_key, model, timeout_seconds)
    if normalized == "anthropic":
        return AnthropicLLMClient(api_base_url, api_key, model, timeout_seconds)
    if normalized == "lmstudio":
        return OpenAICompatibleLLMClient(
            api_base_url,
            api_key,
            model,
            timeout_seconds,
            default_base_url=SUPPORTED_LLM_TYPES["lmstudio"]["default_base_url"],
            requires_api_key=False,
            provider_name="lmstudio",
        )
    if normalized == "ollama":
        return OllamaLLMClient(api_base_url, api_key, model, timeout_seconds)
    return OpenAICompatibleLLMClient(
        api_base_url,
        api_key,
        model,
        timeout_seconds,
        provider_name="openai-compatible",
    )
