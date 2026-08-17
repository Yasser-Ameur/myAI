"""SkillRouter: decides whether a turn belongs to a skill or to conversation.

Routing is a poll, not a classification prompt. Every available skill is asked
``can_handle`` — a cheap, deterministic method — and the most confident answer
above a threshold wins. This keeps routing predictable, testable and free of
model latency, and it means a skill can never be invoked for a turn it did not
claim.

Skills that fail are recorded and, for this turn, skipped: a broken skill
degrades to normal conversation rather than breaking the companion.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from companion.skills.base import (
    SkillContext,
    SkillInput,
    SkillMemory,
    SkillResult,
)
from companion.skills.permissions import PermissionDenied, PermissionManager
from companion.skills.registry import SkillRegistry

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.6
DEFAULT_SKILL_TIMEOUT_S = 10.0


@dataclass
class SkillOutcome:
    handled: bool = False
    text: str = ""
    skill_id: str = ""
    confidence: float = 0.0
    data: dict = field(default_factory=dict)
    error: str = ""
    considered: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "handled": self.handled,
            "skill_id": self.skill_id,
            "confidence": self.confidence,
            "error": self.error,
            "considered": list(self.considered),
        }


class SkillRouter:
    def __init__(
        self,
        registry: SkillRegistry,
        permissions: PermissionManager | None = None,
        graph=None,
        fact_writer=None,
        self_model=None,
        tools=None,
        clock=None,
        runtime=None,
        threshold: float = DEFAULT_THRESHOLD,
        timeout_s: float = DEFAULT_SKILL_TIMEOUT_S,
    ) -> None:
        self._registry = registry
        self._permissions = permissions or PermissionManager()
        self._graph = graph
        self._writer = fact_writer
        self._self_model = self_model
        self._tools = tools
        self._clock = clock
        self._runtime = runtime
        self._threshold = threshold
        self._timeout_s = timeout_s

    def context_for(self, skill_id: str, user_state=None) -> SkillContext:
        memory = None
        if self._graph is not None and self._writer is not None:
            memory = SkillMemory(
                skill_id=skill_id, graph=self._graph, permissions=self._permissions,
                fact_writer=self._writer, self_model=self._self_model, clock=self._clock,
            )
        return SkillContext(
            skill_id=skill_id, memory=memory, self_model=self._self_model,
            user_state=user_state, clock=self._clock, tools=self._tools,
            registry=self._registry, runtime=self._runtime,
        )

    async def handle(self, text: str, user_state=None, episode_id: str = "",
                     session_id: str = "", turn_id: str = "") -> SkillOutcome:
        payload = SkillInput(text=text, episode_id=episode_id,
                             session_id=session_id, turn_id=turn_id)
        outcome = SkillOutcome()

        best = None
        best_decision = None
        for record in self._registry.available():
            context = self.context_for(record.manifest.id, user_state)
            try:
                decision = await asyncio.wait_for(
                    record.skill.can_handle(context, payload), timeout=2.0
                )
            except asyncio.CancelledError:
                raise
            except PermissionDenied as exc:
                log.warning("skill %s denied during routing: %s", record.manifest.id, exc)
                continue
            except Exception:
                log.exception("skill %s can_handle() failed", record.manifest.id)
                continue
            if decision is None or not decision.can_handle:
                continue
            outcome.considered.append(
                {"skill": record.manifest.id, "confidence": decision.confidence}
            )
            if best_decision is None or decision.confidence > best_decision.confidence:
                best, best_decision = record, decision

        if best is None or best_decision.confidence < self._threshold:
            return outcome

        payload.args = dict(best_decision.args or {})
        context = self.context_for(best.manifest.id, user_state)
        try:
            result: SkillResult = await asyncio.wait_for(
                best.skill.execute(context, payload), timeout=self._timeout_s
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self._registry.record_invocation(best.manifest.id, False)
            outcome.error = f"skill {best.manifest.id} timed out"
            log.warning(outcome.error)
            return outcome
        except PermissionDenied as exc:
            self._registry.record_invocation(best.manifest.id, False)
            outcome.error = str(exc)
            log.warning("skill %s permission denied at execute: %s", best.manifest.id, exc)
            return outcome
        except Exception as exc:
            self._registry.record_invocation(best.manifest.id, False)
            outcome.error = f"{type(exc).__name__}: {exc}"
            log.exception("skill %s execute() failed", best.manifest.id)
            return outcome

        self._registry.record_invocation(best.manifest.id, bool(result and result.success))
        if result is None or not result.success or not (result.text or "").strip():
            outcome.error = result.error if result else "skill returned nothing"
            return outcome

        outcome.handled = True
        outcome.text = result.text
        outcome.skill_id = best.manifest.id
        outcome.confidence = best_decision.confidence
        outcome.data = result.data
        log.info("skill %s handled turn (confidence %.2f)",
                 best.manifest.id, best_decision.confidence)
        return outcome
