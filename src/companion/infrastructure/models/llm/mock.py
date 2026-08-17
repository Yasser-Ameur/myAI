"""Mock language model.

Deterministic, dependency-free. Used for tests, simulation, evaluation and
offline demos when no weights are installed. It can serve canned text
responses and shape-minimal JSON for structured tasks.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from companion.core.contracts import (
    GenerationRequest,
    GenerationResult,
    ModelCapability,
)
from companion.infrastructure.models.base import BaseAdapter


class MockLLMProvider(BaseAdapter):
    provider_name = "mock"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        # List of (substring, response). First match wins.
        self.script: list[tuple[str, str]] = [
            (t["match"], t["response"]) for t in config.get("script", [])
        ]
        self.default_response = config.get(
            "default_response", "I'm running in mock mode — no local model weights are loaded yet."
        )
        self.delay_s = float(config.get("delay_s", 0.0))

    def _do_load(self) -> None:
        return None

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(
            name=self.model_id,
            parameter_count="mock",
            context_length=4096,
            supports_streaming=True,
            supports_json=True,
            estimated_ram_mb=0,
            supports_gpu=False,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        import asyncio

        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        self.mark_used()
        if request.json_schema:
            return GenerationResult(text=self._json_response(request), finish_reason="stop")
        return GenerationResult(text=self._text_response(request), finish_reason="stop")

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        text = (await self.generate(request)).text
        for tok in _chunks(text, 12):
            yield tok

    # -- internals ------------------------------------------------------

    def _text_response(self, request: GenerationRequest) -> str:
        for match, response in self.script:
            if match.lower() in request.prompt.lower():
                return response
        return self.default_response

    def _json_response(self, request: GenerationRequest) -> str:
        schema = request.json_schema or {}
        for match, response in self.script:
            if match.lower() in request.prompt.lower():
                return response
        return json.dumps(_schema_defaults(schema))


def _chunks(text: str, n: int):
    for i in range(0, len(text), n):
        yield text[i : i + n]


def _schema_defaults(schema: dict) -> dict:
    """Build a shape-minimal dict matching the schema's properties."""
    props = schema.get("properties", schema.get("items", {}).get("properties", {}))
    out: dict = {}
    for name, spec in props.items():
        if isinstance(spec, list):  # some schemas list possible values
            out[name] = spec[0] if spec else ""
            continue
        enum = spec.get("enum")
        if enum:
            out[name] = enum[0]
            continue
        t = spec.get("type", "string")
        if t == "array":
            item_spec = spec.get("items", {})
            out[name] = [_schema_defaults(item_spec)] if isinstance(item_spec, dict) and item_spec.get("type") == "object" else []
        elif t == "number" or t == "integer":
            out[name] = 0
        elif t == "boolean":
            out[name] = False
        else:
            out[name] = ""
    return out
