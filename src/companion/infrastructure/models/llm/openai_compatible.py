"""OpenAI-compatible LLM provider for remote/local GPU servers.

Lets the same application talk to llama.cpp server, Ollama, LM Studio or a
future GPU box without any code change — only config. Uses stdlib urllib so no
extra dependencies. Only used when the user opts in via configuration.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import AsyncIterator

from companion.core.clock import SystemClock
from companion.core.contracts import (
    GenerationRequest,
    GenerationResult,
    ModelCapability,
)
from companion.core.errors import NetworkAccessDeniedError, ProviderTimeoutError
from companion.infrastructure.models.base import BaseAdapter

log = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseAdapter):
    provider_name = "openai_compatible"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self.base_url = str(config.get("base_url", "http://127.0.0.1:8080")).rstrip("/")
        self.api_key = str(config.get("api_key", ""))
        self.cloud_allowed = bool(config.get("cloud_allowed", False))
        self.timeout = float(config.get("timeout", 120.0))

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(
            name=self.model_id,
            supports_streaming=True,
            supports_json=True,
            supports_tools=True,
            estimated_ram_mb=0,  # lives on the server
            supports_gpu=True,
        )

    def _do_load(self) -> None:
        # Connection is established lazily per call; nothing to preload.
        return None

    def _check_network(self) -> None:
        if not self.cloud_allowed and self.base_url.startswith(("http://", "https://")):
            host = urllib.request.urlparse(self.base_url).hostname
            if host not in ("127.0.0.1", "localhost"):
                raise NetworkAccessDeniedError(
                    f"external endpoint {self.base_url} requires privacy.cloud_enabled / cloud_allowed"
                )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        import asyncio

        self.mark_used()
        self._check_network()
        payload = {
            "model": self.model_id,
            "messages": self._messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if request.json_schema:
            payload["response_format"] = {"type": "json_object", "schema": request.json_schema}
        if request.stop:
            payload["stop"] = request.stop
        started = SystemClock().monotonic()
        try:
            body = await asyncio.to_thread(self._post, "/v1/chat/completions", payload)
        except NetworkAccessDeniedError:
            raise
        except Exception as exc:
            raise ProviderTimeoutError(f"openai-compatible call failed: {exc}") from exc
        latency = (SystemClock().monotonic() - started) * 1000.0
        choice = body["choices"][0]
        usage = body.get("usage", {})
        return GenerationResult(
            text=choice["message"]["content"],
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        import asyncio

        self.mark_used()
        self._check_network()
        payload = {
            "model": self.model_id,
            "messages": self._messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }
        try:
            stream = await asyncio.to_thread(self._stream_body, payload)
        except Exception as exc:
            raise ProviderTimeoutError(f"openai-compatible stream failed: {exc}") from exc
        for line in stream:
            if line.startswith("data: "):
                data = line[6:]
            elif line == "data: [DONE]":
                break
            else:
                continue
            try:
                obj = json.loads(data)
                delta = obj["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, json.JSONDecodeError):
                delta = None
            if delta:
                yield delta

    def _post(self, path: str, payload: dict) -> dict:
        req = self._build_request(path, payload)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _stream_body(self, payload: dict):
        req = self._build_request("/v1/chat/completions", payload)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw in resp:
                yield raw.decode("utf-8").strip()

    def _build_request(self, path: str, payload: dict) -> urllib.request.Request:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    @staticmethod
    def _messages(request: GenerationRequest) -> list[dict]:
        messages: list[dict] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for ctx in request.context:
            messages.append({"role": "user", "content": ctx})
        messages.append({"role": "user", "content": request.prompt})
        return messages
