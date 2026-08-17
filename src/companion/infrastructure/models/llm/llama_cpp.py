"""Llama.cpp LLM provider.

Wraps llama-cpp-python. Loads a quantized GGUF locally, supports streaming and
JSON-schema-constrained output when the backend exposes it. Fully offline.
"""

from __future__ import annotations

import logging
import re
from typing import AsyncIterator

from companion.core.clock import SystemClock
from companion.core.contracts import (
    GenerationRequest,
    GenerationResult,
    ModelCapability,
)
from companion.core.errors import ProviderNotAvailableError, ProviderTimeoutError
from companion.infrastructure.models.base import BaseAdapter

log = logging.getLogger(__name__)


def _fmt_json_schema(schema: dict) -> str:
    import json

    return json.dumps(schema)


class LlamaCppProvider(BaseAdapter):
    provider_name = "llama_cpp"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self.require(
            "llama_cpp",
            "llama-cpp-python is not installed. Run: pip install 'human-companion[llm]'",
        )
        self._llm = None
        self._params = self.config
        self._path = config.get("path", "")

    @property
    def capability(self) -> ModelCapability:
        llm = self._llm
        ctx = getattr(llm, "n_ctx", 0) if llm else int(self._params.get("n_ctx", 4096))
        return ModelCapability(
            name=self.model_id,
            quantization=self._params.get("quantization", ""),
            context_length=ctx,
            supports_streaming=True,
            supports_json=True,
            supports_tools=False,
            estimated_ram_mb=self.estimate_ram_mb(),
            supports_gpu=False,  # default CPU-first; config may override
        )

    def estimate_ram_mb(self) -> int:
        # rough: params_billions * 0.6 GB for Q4 + ctx overhead
        return int(self._params.get("estimated_ram_mb", 1400))

    def _do_load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ProviderNotAvailableError(
                "llama-cpp-python is not installed. Run: pip install 'human-companion[llm]'"
            ) from exc
        if not self._path and not self._params.get("model_path"):
            raise ProviderNotAvailableError(
                f"llama_cpp slot needs a GGUF 'path' for model {self.model_id}"
            )
        started = SystemClock().monotonic()
        kwargs = {
            "model_path": self._path or self._params["model_path"],
            "n_ctx": int(self._params.get("n_ctx", 4096)),
            "n_batch": int(self._params.get("n_batch", 256)),
            "n_threads": int(self._params.get("threads", 4)),
            "n_gpu_layers": int(self._params.get("gpu_layers", 0)),
            "verbose": False,
        }
        if self._params.get("chat_format"):
            # e.g. "chatml" bypasses a GGUF Jinja template that forces chain-of-thought.
            kwargs["chat_format"] = self._params["chat_format"]
        try:
            self._llm = Llama(**kwargs)
        except Exception as exc:
            raise ProviderNotAvailableError(f"llama.cpp failed to init: {exc}") from exc
        if not self._params.get("enable_thinking", True):
            self._disable_thinking(self._llm)
        self._load_time_ms = (SystemClock().monotonic() - started) * 1000.0

    @staticmethod
    def _disable_thinking(llm) -> None:
        """Inject enable_thinking=False into the model's chat-template handler.

        Qwen3 GGUF templates default to chain-of-thought; disabling it keeps
        responses direct and fast. Guarded per-handler so only templates that
        honor the variable change behavior.
        """
        handlers = getattr(llm, "_chat_handlers", {}) or {}
        for key, base in list(handlers.items()):
            if key.startswith("chat_template") and callable(base):
                handlers[key] = (
                    lambda *args, _base=base, **kwargs: _base(
                        *args, **{**{"enable_thinking": False}, **kwargs}
                    )
                )
        chat_handler = getattr(llm, "chat_handler", None)
        if callable(chat_handler):
            llm.chat_handler = (
                lambda *args, _base=chat_handler, **kwargs: _base(
                    *args, **{**{"enable_thinking": False}, **kwargs}
                )
            )

    def unload(self) -> None:
        self._llm = None
        super().unload()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        llm = self._llm
        if llm is None:
            raise ProviderNotAvailableError(f"model {self.model_id} not loaded")
        import asyncio

        self.mark_used()
        started = SystemClock().monotonic()
        try:
            result = await asyncio.to_thread(
                llm.create_chat_completion,
                messages=self._messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop or None,
                response_format=(
                    {"type": "json_object", "schema": request.json_schema}
                    if request.json_schema
                    else None
                ),
            )
        except Exception as exc:
            raise ProviderTimeoutError(f"llama.cpp generation failed: {exc}") from exc
        latency = (SystemClock().monotonic() - started) * 1000.0
        choice = result["choices"][0]
        usage = result.get("usage", {})
        return GenerationResult(
            text=self._clean(choice["message"]["content"]),
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        llm = self._llm
        if llm is None:
            raise ProviderNotAvailableError(f"model {self.model_id} not loaded")
        import asyncio

        self.mark_used()
        try:
            stream = await asyncio.to_thread(
                llm.create_chat_completion,
                messages=self._messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop or None,
                stream=True,
            )
        except Exception as exc:
            raise ProviderTimeoutError(f"llama.cpp streaming failed: {exc}") from exc
        for chunk in stream:
            try:
                delta = chunk["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, TypeError):
                delta = None
            if delta:
                yield delta

    def _clean(self, text: str) -> str:
        """Strip chain-of-thought blocks when configured (default on)."""
        if not self._params.get("strip_think_blocks", True):
            return text
        # remove complete <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # remove an unterminated block (max_tokens cut it off mid-thought)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _messages(request: GenerationRequest) -> list[dict]:
        messages: list[dict] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for ctx in request.context:
            messages.append({"role": "user", "content": ctx})
        messages.append({"role": "user", "content": request.prompt})
        return messages
