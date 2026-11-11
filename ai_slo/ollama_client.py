"""Minimal client for a local Ollama server (POST /api/chat, non-streaming).

Host defaults to $OLLAMA_HOST or http://localhost:11434.
Ollama reports durations in nanoseconds; this client converts them to ms.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import requests

DEFAULT_HOST = "http://localhost:11434"
_NS_PER_MS = 1_000_000


class OllamaError(RuntimeError):
    """The server was unreachable, returned an error, or sent an unexpected body."""


@dataclass
class ChatResult:
    message: dict[str, Any]
    total_duration_ms: float
    load_duration_ms: Optional[float]
    prompt_eval_duration_ms: Optional[float]
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def content(self) -> str:
        return self.message.get("content") or ""

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        # The message is the single source of truth for tool calls.
        return list(self.message.get("tool_calls") or [])


def parse_chat_response(data: dict[str, Any]) -> ChatResult:
    """Turn a raw /api/chat JSON body into a ChatResult. Pure; no network."""
    if "message" not in data:
        raise OllamaError(f"unexpected response body: {json.dumps(data)[:200]}")

    def ms(key: str) -> Optional[float]:
        value = data.get(key)
        return None if value is None else value / _NS_PER_MS

    return ChatResult(
        message=data["message"],
        total_duration_ms=ms("total_duration") or 0.0,
        load_duration_ms=ms("load_duration"),
        prompt_eval_duration_ms=ms("prompt_eval_duration"),
        prompt_tokens=int(data.get("prompt_eval_count") or 0),
        completion_tokens=int(data.get("eval_count") or 0),
    )


class OllamaClient:
    def __init__(
        self,
        model: str,
        host: Optional[str] = None,
        timeout: float = 300.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not model:
            raise ValueError("model is required (e.g. the name shown by `ollama list`)")
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()

    def check(self) -> list[str]:
        """Preflight: confirm the server is up and the model is installed.
        Returns the installed model names. Raises OllamaError otherwise."""
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=10)
        except requests.RequestException as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host}: {exc}") from exc
        if resp.status_code != 200:
            raise OllamaError(f"Ollama at {self.host} returned HTTP {resp.status_code}")
        names = [m.get("name", "") for m in (resp.json().get("models") or [])]
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            raise OllamaError(
                f"model {self.model!r} is not installed on {self.host}. "
                f"Installed: {', '.join(names) or 'none'}. Run `ollama pull {self.model}`."
            )
        return names

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        try:
            resp = self._session.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host}: {exc}") from exc
        if resp.status_code != 200:
            raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned non-JSON body") from exc
        return parse_chat_response(body)
