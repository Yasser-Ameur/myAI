"""CompanionApp: the runtime orchestrator.

Wires configuration, hardware profile, storage/graph, model registry + lifecycle,
perception, memory, personality, retrieval, conversation, avatar, reflection,
memory guard and metrics into one application. This is the only place that
instantiates infrastructure; everything above uses interfaces.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from companion.application.avatar import AvatarService, ConsoleAvatarDriver, ExpressionController
from companion.application.conversation import (
    ContextAssembler,
    ConversationService,
    LLMResponsePlanner,
)
from companion.application.extraction import StructuredExtractor
from companion.application.facts import FactWriter
from companion.application.identity import SelfModelService
from companion.application.memory import MemoryPipeline, MemoryService
from companion.application.perception import PerceptionService
from companion.application.personality import PersonalityEngine
from companion.application.ports import NullGraphStore
from companion.application.reflection import IdleDetector, ReflectionService
from companion.application.relationship import RelationshipEngine
from companion.application.retrieval import HybridRetriever
from companion.application.salience import TurnCommitter
from companion.application.speech_output import SpeechOutputService
from companion.core.clock import SystemClock
from companion.core.errors import MemoryUnavailableError
from companion.core.events import EventBus
from companion.core.ids import new_session_id, new_turn_id
from companion.domain.agent import AgentIdentity, AgentPersonality, AgentState
from companion.infrastructure.models.factory import build_provider
from companion.infrastructure.models.lifecycle import ModelLifecycleManager
from companion.infrastructure.models.registry import ModelRegistry
from companion.infrastructure.models.router import TaskRouter
from companion.infrastructure.sqlite_graph import CognitiveGraph
from companion.infrastructure.storage import SqliteStorage
from companion.infrastructure.vector import build_vector_store
from companion.runtime.config import Config
from companion.runtime.hardware import build_hardware_profile
from companion.runtime.memory_guard import MemoryGuard
from companion.runtime.metrics import Metrics
from companion.runtime.scheduler import Scheduler
from companion.skills.permissions import PermissionManager
from companion.skills.registry import SkillLoader, SkillRegistry
from companion.skills.router import SkillRouter
from companion.tools.builtin import default_tools
from companion.tools.registry import ToolInvoker, ToolRegistry

log = logging.getLogger(__name__)


@dataclass
class CompanionComponents:
    app: "CompanionApp"
    config: Config = field(default_factory=lambda: Config.load())
    storage: SqliteStorage | None = None
    graph: object | None = None
    registry: ModelRegistry | None = None
    lifecycle: ModelLifecycleManager | None = None
    router: TaskRouter | None = None
    personality: PersonalityEngine | None = None
    relationships: RelationshipEngine | None = None
    retriever: HybridRetriever | None = None
    conversation: ConversationService | None = None
    memory: MemoryService | None = None
    perception: PerceptionService | None = None
    avatar: AvatarService | None = None
    speech: SpeechOutputService | None = None
    reflection: ReflectionService | None = None
    metrics: Metrics | None = None
    scheduler: Scheduler | None = None
    memory_guard: MemoryGuard | None = None
    agent_identity: AgentIdentity | None = None
    agent_personality: AgentPersonality | None = None
    agent_state: AgentState | None = None
    self_model: SelfModelService | None = None
    fact_writer: FactWriter | None = None
    turn_committer: TurnCommitter | None = None
    skills: SkillRegistry | None = None
    skill_router: SkillRouter | None = None
    tools: ToolRegistry | None = None
    tool_invoker: ToolInvoker | None = None
    permissions: PermissionManager | None = None
    degraded: str | None = None

    def to_dict(self) -> dict:
        return {
            "degraded": self.degraded,
            "hardware": self.app.hardware.to_dict() if self.app.hardware else {},
            "registry": self.registry.health() if self.registry else {},
            "metrics": self.metrics.snapshot() if self.metrics else {},
        }


class CompanionApp:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.load()
        self.clock = SystemClock()
        self.bus = EventBus(clock=self.clock)
        self.components: CompanionComponents | None = None
        self.hardware = None
        self._tasks: list[asyncio.Task] = []
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self.session_id = new_session_id()

    # -- build -----------------------------------------------------------

    def build(self) -> CompanionComponents:
        cfg = self.config
        self.hardware = build_hardware_profile(
            profile_name=cfg.hardware_profile,
            max_ai_mb=cfg.max_ai_ram_mb,
        )
        metrics = Metrics()
        scheduler = Scheduler(max_background_workers=int(
            cfg.section("runtime", "concurrency", "max_background_workers", default=1)
        ))
        memory_guard = MemoryGuard(
            threshold_elevated_mb=max(4096, self.hardware.ram_budget_mb + 2048),
            threshold_critical_mb=max(6144, self.hardware.ram_budget_mb + 3072),
        )

        # Storage / graph. Degradation must be *visible*: previously a failure
        # to build the graph still left `storage` non-None, so diagnostics
        # reported "ok" while every memory write went to a null store.
        storage = None
        graph = None
        degraded = None
        try:
            storage = SqliteStorage(cfg.db_path)
            graph = CognitiveGraph(storage)
        except MemoryUnavailableError as exc:
            degraded = f"cognitive store unavailable: {exc}"
            log.error(degraded)
            graph = NullGraphStore(reason=degraded)
            if storage is not None:
                storage.close()
            storage = None
        except Exception as exc:  # schema/migration failures are just as fatal
            degraded = f"cognitive store unusable: {type(exc).__name__}: {exc}"
            log.error(degraded)
            graph = NullGraphStore(reason=degraded)
            if storage is not None:
                storage.close()
            storage = None

        # model registry + lifecycle
        registry = ModelRegistry(cfg.models, cfg.raw.get("providers") or {})
        try:
            registry.build_from_config(build_provider)
        except Exception as exc:
            log.warning("model registry build incomplete: %s", exc)
        lifecycle = ModelLifecycleManager(
            registry,
            max_heavy_resident=cfg.max_heavy_models_resident,
            max_ai_ram_mb=cfg.max_ai_ram_mb,
        )
        router = TaskRouter(registry)

        # embedding + vector store
        embedding_model_id = cfg.section("models", "embeddings", "default", "model_id", default="default") or "default"
        embeddings = registry.get_optional("embeddings.default")
        vector_backend = cfg.embedding_backend
        vectors = None
        if storage is not None:
            vectors = build_vector_store(vector_backend, storage)

        # agent identity: configuration is only the bootstrap. The persisted
        # self-model wins if the user has ever named the companion.
        fact_writer = FactWriter(graph, self.clock)
        self_model = SelfModelService(
            graph, self.clock,
            configured_name=str(cfg.section("system", "name", default="Companion")),
            writer=fact_writer,
        )
        self_model.load()
        agent_identity = self_model.model().identity
        agent_personality = AgentPersonality()
        agent_state = AgentState()

        # skills + tools
        permissions = PermissionManager(cfg.section("skills", default={}) or {})
        tool_registry = ToolRegistry()
        for tool in default_tools():
            try:
                tool_registry.register(tool)
            except Exception as exc:
                log.warning("could not register tool: %s", exc)
        tool_invoker = ToolInvoker(tool_registry, permissions)
        skill_registry = SkillRegistry(permissions, tools=tool_registry)
        SkillLoader(skill_registry).load_package()
        self_model.set_capability_provider(skill_registry.capability_names)

        # personality + relationships (work even in degraded mode, in-memory)
        personality = PersonalityEngine(
            graph, self.clock,
            update_mode=cfg.personality_update_mode,
            decay_days=int(cfg.section("personality", "decay_days", default=180)),
        ).load()
        relationships = RelationshipEngine(graph, self.clock)

        # memory pipeline + service. The extractor was never wired to a model,
        # so consolidation silently ran rules-only; give it the fast model
        # (config: llm.fast is designated for classification/extraction) and
        # let it fall back to rules when the model is absent or misbehaves.
        extraction_llm = registry.get_optional("llm.fast") or registry.get_optional("llm.default")
        pipeline = MemoryPipeline(
            graph=graph,
            vector_store=vectors if vectors is not None else _NullVectors(),
            embeddings=embeddings,
            clock=self.clock,
            extractor=StructuredExtractor(extraction_llm, router),
            personality=personality,
            relationships=relationships,
            embedding_model_id=embedding_model_id,
        )
        memory = MemoryService(graph, pipeline, self.clock)

        # retrieval
        retriever = HybridRetriever(
            graph=graph,
            vector_store=vectors if vectors is not None else _NullVectors(),
            embeddings=embeddings,
            clock=self.clock,
            embedding_model_id=embedding_model_id,
        )

        # turn-level durable commit (runs inline on every user turn)
        turn_committer = TurnCommitter(graph, self_model, writer=fact_writer,
                                       clock=self.clock)

        skill_router = SkillRouter(
            skill_registry, permissions, graph=graph, fact_writer=fact_writer,
            self_model=self_model, tools=tool_invoker, clock=self.clock,
            runtime=self.subsystem_health,
            threshold=float(cfg.section("skills", "route_threshold", default=0.6)),
        )

        # conversation
        llm = registry.get_optional("llm.default")
        assembler = ContextAssembler(
            budgets=cfg.context_budget or None,
            total_budget=cfg.max_prompt_tokens,
            self_model=self_model,
        )
        planner = LLMResponsePlanner(
            llm, router,
            mode=str(cfg.section("conversation", "planner", default="rules")),
        )
        conversation = ConversationService(
            llm=llm,
            retriever=retriever,
            assembler=assembler,
            planner=planner,
            graph=graph,
            memory=memory,
            personality=personality,
            relationships=relationships,
            agent_profile=agent_personality.profile,
            agent_state=agent_state,
            bus=self.bus,
            clock=self.clock,
            router=router,
            self_model=self_model,
            turn_committer=turn_committer,
            skills=skill_router,
        )

        # perception (face/vad/stt)
        face = registry.get_optional("vision.face")
        vad = registry.get_optional("vad.default")
        stt = registry.get_optional("stt.default")
        perception = PerceptionService(
            face_provider=face,
            vad_provider=vad,
            stt_provider=stt,
            bus=self.bus,
            clock=self.clock,
            face_fps=int(cfg.section("runtime", "perception", "face_fps", default=10)),
            sample_rate=int(cfg.section("runtime", "audio", "sample_rate", default=16000)),
        )

        # avatar
        driver = ConsoleAvatarDriver()
        controller = ExpressionController(driver=driver)
        avatar = AvatarService(controller, bus=self.bus, agent_state=agent_state)

        # speech output
        tts = registry.get_optional("tts.default")
        playback = None
        try:
            from companion.infrastructure.audio.playback import SoundDevicePlaybackSink
            playback = SoundDevicePlaybackSink()
        except Exception as exc:
            log.warning("audio playback unavailable: %s", exc)
        speech = SpeechOutputService(tts, bus=self.bus, clock=self.clock, playback=playback)

        # reflection
        idle = IdleDetector(
            clock=self.clock,
            quiet_after_s=float(cfg.section("memory", "consolidation", "quiet_after_seconds", default=30)),
            idle_after_s=float(cfg.section("memory", "consolidation", "min_idle_seconds", default=180)),
        )
        reflection = ReflectionService(
            graph=graph,
            personality=personality,
            relationships=relationships,
            lifecycle=lifecycle,
            clock=self.clock,
            idle=idle,
            scheduler=scheduler,
            max_runs_per_hour=int(cfg.section("memory", "consolidation", "max_runs_per_hour", default=2)),
            pipeline=pipeline,
        )

        comp = CompanionComponents(
            app=self, config=cfg, storage=storage, graph=graph, registry=registry,
            lifecycle=lifecycle, router=router, personality=personality,
            relationships=relationships, retriever=retriever, conversation=conversation,
            memory=memory, perception=perception, avatar=avatar, speech=speech,
            reflection=reflection, metrics=metrics, scheduler=scheduler,
            memory_guard=memory_guard, agent_identity=agent_identity,
            agent_personality=agent_personality, agent_state=agent_state, degraded=degraded,
            self_model=self_model, fact_writer=fact_writer, turn_committer=turn_committer,
            skills=skill_registry, skill_router=skill_router, tools=tool_registry,
            tool_invoker=tool_invoker, permissions=permissions,
        )
        self.components = comp
        return comp

    # -- runtime ----------------------------------------------------------

    async def start(self) -> None:
        comp = self.components
        if comp is None:
            comp = self.build()
        if self._closed:
            raise RuntimeError("companion runtime has been shut down")
        self._loop = asyncio.get_running_loop()
        # build() is deliberately synchronous so diagnostic commands can use
        # it.  Subscription activation belongs to the async runtime boundary.
        comp.avatar.attach_bus(self.bus)
        if not self._tasks:
            self._tasks.append(asyncio.create_task(self._reflection_loop(), name="reflection"))
            self._tasks.append(asyncio.create_task(self._memory_loop(), name="memory_guard"))
            self._tasks.append(asyncio.create_task(self._idle_unload_loop(), name="idle_unload"))
        log.info("companion runtime started (profile=%s)", self.hardware.profile_name if self.hardware else "?")

    async def _reflection_loop(self) -> None:
        comp = self.components
        await asyncio.sleep(10)
        while True:
            try:
                await comp.reflection.run_once()
            except Exception:
                log.exception("reflection loop error")
            await asyncio.sleep(20)

    async def _memory_loop(self) -> None:
        comp = self.components
        while True:
            await asyncio.sleep(5)
            try:
                comp.memory_guard.check(lifecycle=comp.lifecycle, scheduler=comp.scheduler)
            except Exception:
                log.exception("memory guard error")

    async def _idle_unload_loop(self) -> None:
        """Free heavy models after the companion has been idle for a while.

        respond() re-loads whatever it needs on the next turn, so unloading
        only costs a load (~1-4 s) when the user comes back.
        """
        comp = self.components
        interval = 30
        unload_after_s = self.config.idle_unload_s
        while True:
            await asyncio.sleep(interval)
            try:
                victims = self._idle_unload_once(comp, unload_after_s)
                if victims:
                    log.info("idle unload: freed %d heavy model(s): %s", len(victims), victims)
            except Exception:
                log.exception("idle unload loop error")

    @staticmethod
    def _idle_unload_once(comp, unload_after_s: int) -> list[str]:
        """Unload heavy idle models; returns the slots that were unloaded."""
        if unload_after_s <= 0 or comp.reflection is None or comp.lifecycle is None:
            return []
        if comp.app.clock.monotonic() - comp.reflection.idle.last_activity < unload_after_s:
            return []
        victims = [s for s in comp.lifecycle.stats()["loaded"]
                   if s.startswith("llm.") or s.startswith("stt.")]
        for slot in victims:
            comp.lifecycle.unload(slot)
        return victims

    async def respond(self, text: str, source: str = "text", speak: bool = False) -> dict:
        comp = self.components
        comp.scheduler.begin_interactive()
        comp.reflection.idle.register_activity()
        turn_id = new_turn_id()
        trace = {"session_id": self.session_id, "turn_id": turn_id}
        log.info("turn_start", extra={
            "session_id": self.session_id, "turn_id": turn_id,
            "source": source, "text_len": len(text)})
        try:
            for slot in ("llm.fast", "llm.default", "embeddings.default"):
                if comp.registry.has(slot):
                    try:
                        comp.lifecycle.load(slot)
                    except Exception as exc:
                        log.warning("could not load %s: %s", slot, exc)
            import time as _t

            started = _t.monotonic()
            result = await comp.conversation.respond(text, source=source,
                                                     user_state=comp.perception.current_user_state(),
                                                     **trace)
            latency = (_t.monotonic() - started) * 1000.0
            comp.metrics.llm_latency_ms(latency, len(result.text.split()))
            comp.metrics.inc("user_turns")
            log.info("turn_complete", extra={
                "session_id": self.session_id, "turn_id": turn_id,
                "source": source, "latency_ms": round(latency, 1),
                "tokens": len(result.text.split()), "text_len": len(result.text)})
            if speak:
                await comp.speech.speak(result.text, result.plan)
            return {"text": result.text, "plan": result.plan.to_dict(),
                    "skill": getattr(result, "skill_id", "")}
        finally:
            comp.scheduler.end_interactive()

    async def aclose(self, consolidate: bool = True,
                     consolidate_timeout_s: float = 20.0) -> None:
        """Consolidate, then cancel and await runtime work, then release storage.

        Closing the episode is what turns a session's raw turns into extracted
        memories. It used to be skipped entirely on shutdown, so consolidation
        never ran in the real runtime. It is bounded by a timeout because a
        slow LLM must not be able to hang exit — the salient facts are already
        durable from the per-turn commit either way.
        """
        if self._closed:
            return
        self._closed = True
        comp = self.components
        if consolidate and comp is not None and comp.memory is not None:
            try:
                await asyncio.wait_for(comp.memory.close_episode(),
                                       timeout=consolidate_timeout_s)
            except asyncio.TimeoutError:
                log.warning("episode consolidation timed out after %.0fs; "
                            "per-turn facts are already durable", consolidate_timeout_s)
            except Exception:
                log.exception("episode consolidation failed during shutdown")
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self.components:
            self.components.avatar.stop()
        await self.bus.aclose()
        self._loop = None
        if self.components and self.components.storage:
            self.components.storage.close()

    def shutdown(self) -> None:
        """Compatibility cleanup for non-async diagnostic command paths.

        Active runtime owners must use :meth:`aclose` so cancellation is
        awaited on the CLI-owned event loop.
        """
        for t in self._tasks:
            t.cancel()
        if self.components:
            self.components.avatar.stop()
        if self.components and self.components.storage:
            self.components.storage.close()

    def submit_from_thread(self, awaitable):
        """Schedule work from an adapter thread on the runtime-owned loop."""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("companion runtime is not running")
        return asyncio.run_coroutine_threadsafe(awaitable, self._loop)

    def subsystem_health(self) -> dict:
        """Ground truth about every subsystem, for diagnostics and the doctor.

        Distinguishes the states that matter and were previously conflated:
        not configured / configured but unloaded / loaded / running on a
        fallback / failed to load.
        """
        comp = self.components
        slots: dict[str, dict] = {}
        if comp is not None and comp.registry is not None:
            health = comp.registry.health()
            for slot in ("llm.default", "llm.fast", "stt.default", "tts.default",
                         "vad.default", "vision.face", "embeddings.default"):
                info = health.get(slot)
                if info is None:
                    slots[slot] = {"present": False}
                    continue
                provider = comp.registry.get_optional(slot)
                slots[slot] = {
                    "present": True,
                    "provider": info.get("provider"),
                    "model": info.get("model_id"),
                    "loaded": bool(info.get("loaded")),
                    "load_error": info.get("load_error") or "",
                    "fallback": bool(getattr(provider, "_fallback", False)),
                    "ram_mb": info.get("ram_mb"),
                }
        storage_ok = comp is not None and comp.storage is not None and not comp.degraded
        return {
            "storage": {
                "ok": bool(storage_ok),
                "reason": (comp.degraded if comp is not None else "not built") or "",
                "path": self.config.db_path,
            },
            "slots": slots,
            "degraded": comp.degraded if comp is not None else "not built",
            "playback": bool(comp is not None and comp.speech is not None
                             and getattr(comp.speech, "_playback", None) is not None),
            "skills": comp.skills.snapshot() if comp is not None and comp.skills else {},
        }

    def health(self) -> dict:
        comp = self.components
        return {
            "status": "degraded" if comp.degraded else "ok",
            "degraded": comp.degraded,
            "hardware": self.hardware.to_dict() if self.hardware else {},
            "loaded_models": comp.lifecycle.stats() if comp.lifecycle else {},
            "registry_health": comp.registry.health() if comp.registry else {},
            "idle_level": comp.reflection.idle.level.value if comp.reflection else "?",
            "memory_guard": comp.memory_guard.state.__dict__ if comp.memory_guard else {},
        }

    def runtime_report(self) -> dict:
        comp = self.components
        active_models = {}
        for slot, provider in {s: comp.registry.get_optional(s) for s in comp.registry.slots()}.items():
            if provider is None:
                continue
            active_models[slot] = {
                "provider": provider.provider_name,
                "model": provider.model_id,
                "loaded": provider.is_loaded(),
                "fallback": getattr(provider, "_fallback", False),
                "ram_mb": provider.estimate_ram_mb(),
            }
        return {
            "hardware": self.hardware.to_dict() if self.hardware else {},
            "execution_backend": (
                self.hardware.preferred_execution_mode if self.hardware else "unknown"
            ),
            "active_models": active_models,
            "models": comp.registry.health() if comp.registry else {},
            "lifecycle": comp.lifecycle.stats() if comp.lifecycle else {},
            "metrics": comp.metrics.snapshot() if comp.metrics else {},
            "memory_guard": comp.memory_guard.state.__dict__ if comp.memory_guard else {},
            "memory_stats": comp.memory.stats() if comp.memory else {},
            "profile_snapshot": comp.personality.snapshot() if comp.personality else {},
            "relationships": comp.relationships.snapshot() if comp.relationships else [],
            "degraded": comp.degraded,
        }


class _NullVectors:
    """Vector-store no-op for degraded mode."""

    def upsert(self, *a, **k) -> None: ...
    def remove(self, *a, **k) -> None: ...
    def search(self, model_id, query, top_k) -> list:
        return []
    def clear_namespace(self, *a, **k) -> None: ...
    def count(self, *a, **k) -> int:
        return 0
