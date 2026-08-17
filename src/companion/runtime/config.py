"""Configuration loading.

All provider/model selection is configuration-driven. YAML file + env override.
The runtime builds a typed Config dataclass; nothing in the domain cares.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from companion.core.errors import ConfigurationError


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None, env_prefix: str = "COMPANION_") -> dict:
    path = path or os.environ.get(f"{env_prefix}CONFIG", "config/companion.yaml")
    if not os.path.exists(path):
        raise ConfigurationError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # env overrides: COMPANION_<SECTION>_<KEY>
    prefix = env_prefix.lower()
    for full_key, value in os.environ.items():
        if not full_key.lower().startswith(prefix):
            continue
        parts = full_key.lower()[len(prefix):].split("_")
        if len(parts) < 2:
            continue
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return cfg


def _coerce(value: str):
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("auto", "none", "null"):
        return None if lowered in ("none", "null") else "auto"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        return cls(raw=raw)

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        return cls.from_dict(load_config(path))

    # -- accessors --------------------------------------------------------

    def section(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
            if node is default and k in node if isinstance(node, dict) else node is default:
                break
        return node

    @property
    def system_mode(self) -> str:
        return str(self.section("system", "mode", default="local"))

    @property
    def hardware_profile(self) -> str:
        return str(self.section("hardware", "profile", default="balanced"))

    @property
    def db_path(self) -> str:
        return str(self.section("memory", "database", default="data/cognitive.db"))

    @property
    def max_prompt_tokens(self) -> int:
        return int(self.section("memory", "max_prompt_tokens", default=3500))

    @property
    def context_budget(self) -> dict:
        return dict(self.section("context_budget", default={}))

    @property
    def models(self) -> dict:
        return dict(self.section("models", default={}))

    @property
    def max_ai_ram_mb(self) -> int:
        return int(self.section("runtime", "memory", "max_ai_ram_mb", default=8192))

    @property
    def max_heavy_models_resident(self) -> int:
        return int(self.section("runtime", "concurrency", "max_heavy_models_resident", default=1))

    @property
    def cloud_enabled(self) -> bool:
        return bool(self.section("privacy", "cloud_enabled", default=False))

    @property
    def api_port(self) -> int:
        return int(self.section("api", "port", default=8377))

    @property
    def api_host(self) -> str:
        return str(self.section("api", "host", default="127.0.0.1"))

    @property
    def logging_level(self) -> str:
        return str(self.section("logging", "level", default="warning"))

    @property
    def logging_file(self) -> str | None:
        return self.section("logging", "file", default=None)

    @property
    def barge_in_enabled(self) -> bool:
        return bool(self.section("runtime", "audio", "barge_in", default=True))

    @property
    def idle_unload_s(self) -> int:
        return int(self.section("runtime", "concurrency", "idle_unload_s", default=300))

    @property
    def personality_update_mode(self) -> str:
        return str(self.section("personality", "update_mode", default="conservative"))

    @property
    def embedding_backend(self) -> str:
        return str(self.section("memory", "vector_index", default="sqlite_bruteforce"))
