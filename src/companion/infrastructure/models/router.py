"""TaskRouter: routes work to the cheapest suitable model.

The primary LLM is not used for trivial work. Classification and extraction go
to a small model; normal conversation to the default model; explicit reasoning
to the reasoning model (if available). Embeds always use the embedding model.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

REASONING_KEYWORDS = (
    "compare", "contrast", "analyze", "why", "how does", "explain the reasoning",
    "prove", "solve", "work through", "logical", "step by step",
)


class TaskRouter:
    def __init__(self, registry) -> None:
        self._registry = registry

    def route(self, task: str, text: str = "") -> str:
        """Return the model slot for a task kind."""
        slot = self._registry.get_optional(f"llm.{task}")
        if slot is not None:
            return f"llm.{task}"
        return "llm.default"

    def llm_for_task(self, task: str, text: str = "") -> str:
        if task in ("classification", "extraction", "planning"):
            return self._prefer("llm.fast", "llm.default")
        if task == "reasoning":
            if any(k in text.lower() for k in REASONING_KEYWORDS):
                return self._prefer("llm.reasoning", "llm.default")
            return "llm.default"
        return "llm.default"

    def _prefer(self, preferred: str, fallback: str) -> str:
        if self._registry.has(preferred):
            return preferred
        return fallback

    def capabilities_report(self) -> list[dict]:
        report: list[dict] = []
        for slot in self._registry.slots():
            provider = self._registry.get_optional(slot)
            if provider is None:
                continue
            try:
                cap = provider.capability
            except Exception as exc:  # provider may not be constructible cheaply
                cap = None
                log.debug("no capability for %s: %s", slot, exc)
            report.append(
                {
                    "slot": slot,
                    "provider": getattr(provider, "provider_name", "?"),
                    "model": getattr(provider, "model_id", ""),
                    "capability": cap.__dict__ if cap else {},
                }
            )
        return report
