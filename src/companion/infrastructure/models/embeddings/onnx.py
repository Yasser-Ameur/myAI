"""Embedding providers.

Embeddings are stored with their model_id and dimension as a separate
namespace, so swapping models never corrupts existing vectors.
"""

from __future__ import annotations

import hashlib
import logging
import os

from companion.core.errors import ProviderNotAvailableError
from companion.core.types import Vector
from companion.infrastructure.models.base import BaseAdapter

log = logging.getLogger(__name__)


class MockEmbeddingProvider(BaseAdapter):
    """Deterministic hash-based embeddings (no weights).

    Not semantically meaningful, but stable and useful for tests/simulation of
    the retrieval pipeline (dedup, namespaces, dimension changes).
    """

    provider_name = "mock"
    dimension: int = 64

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self.dimension = int(config.get("dimension", 64))

    def _do_load(self) -> None:
        return None

    @property
    def capability(self):
        from companion.core.contracts import ModelCapability

        return ModelCapability(name=self.model_id, estimated_ram_mb=0)

    async def embed(self, texts: list[str]) -> list[Vector]:
        import asyncio

        self.mark_used()
        await asyncio.sleep(0)
        return [self._hash_vector(t) for t in texts]

    def _hash_vector(self, text: str) -> Vector:
        d = self.dimension
        vec = [0.0] * d
        for token in _tokens(text):
            h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % d
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        norm = sum(v * v for v in vec) ** 0.5
        if norm:
            vec = [v / norm for v in vec]
        return vec


class OnnxEmbeddingProvider(BaseAdapter):
    """Sentence embeddings via onnxruntime (e.g. bge-small-en).

    The actual onnx file is installed via `companion models install`. Smallest
    strong local sentence model fits comfortably under the memory budget.
    """

    provider_name = "onnx"
    dimension: int = 384

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self.require(
            "onnxruntime",
            "onnxruntime is not installed. Run: pip install 'human-companion[embeddings]'",
        )
        self.dimension = int(config.get("dimension", 384))
        self._session = None
        self._tokenizer = None

    def _do_load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ProviderNotAvailableError(
                "onnxruntime is not installed. Run: pip install 'human-companion[embeddings]'"
            ) from exc
        model_path = self._params.get("path") or self._params.get("model_path")
        if not model_path:
            raise ProviderNotAvailableError("onnx embedding provider requires 'path' (model.onnx)")
        tokenizer_path = self._params.get("tokenizer_path") or os.path.join(
            os.path.dirname(model_path), "tokenizer.json"
        )
        try:
            from tokenizers import Tokenizer

            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            if os.path.isdir(tokenizer_path):
                tokenizer_path = os.path.join(tokenizer_path, "tokenizer.json")
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
        except Exception as exc:
            raise ProviderNotAvailableError(f"onnx embeddings failed to load: {exc}") from exc

    @property
    def capability(self):
        from companion.core.contracts import ModelCapability

        return ModelCapability(name=self.model_id, estimated_ram_mb=self.estimate_ram_mb())

    def estimate_ram_mb(self) -> int:
        return int(self._params.get("estimated_ram_mb", 300))

    async def embed(self, texts: list[str]) -> list[Vector]:
        import asyncio

        self.mark_used()
        if self._session is None:
            raise ProviderNotAvailableError(f"model {self.model_id} not loaded")
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[Vector]:
        import numpy as np

        encodings = self._tokenizer.encode_batch(
            texts, add_special_tokens=True, is_pretokenized=False
        )
        max_len = min(512, max((len(e.ids) for e in encodings), default=1))
        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention = np.zeros((len(encodings), max_len), dtype=np.int64)
        token_type = np.zeros((len(encodings), max_len), dtype=np.int64)
        for i, e in enumerate(encodings):
            ids = e.ids[:max_len]
            input_ids[i, : len(ids)] = ids
            attention[i, : len(ids)] = e.attention_mask[:max_len]
            token_type[i, : len(ids)] = e.type_ids[:max_len]
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "token_type_ids": token_type,
        }
        out = self._session.run(None, feed)[0]
        mask = attention[:, :, None].astype(np.float32)
        summed = np.sum(out * mask, axis=1)
        counts = np.clip(np.sum(mask, axis=1), 1, None)
        vectors = summed / counts
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-8, None)
        return [v.tolist() for v in vectors]


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[\w']+", text.lower())
