"""Command-line interface (the `companion` command).

Commands:
  run            interactive REPL conversation
  interact       interactive session: --text | --voice | --multimodal
  doctor         check config, hardware and model slots
  benchmark      measure model load/latency/tokens-per-sec (offline)
  models         list / install / remove / switch
  memory         inspect / forget / lock / correct memories
  personality    inspect trait/value profile
  graph          inspect entities / facts / goals
  runtime        dump health + metrics
  api            start the HTTP API server
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from companion.runtime.config import Config
from companion.runtime.logging import configure_logging
from companion.runtime.orchestration import CompanionApp

_cfg = Config.load()
configure_logging(level=_cfg.logging_level, file=_cfg.logging_file)


def _app(args) -> CompanionApp:
    cfg = Config.load()
    app = CompanionApp(cfg)
    app.build()
    return app


def _json(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_run(args: argparse.Namespace) -> int:
    async def loop() -> int:
        app = _app(args)
        comp = app.components
        if comp.degraded:
            print(f"! running in degraded mode: {comp.degraded}")
        for slot in ("llm.default", "embeddings.default"):
            try:
                comp.lifecycle.load(slot)
            except Exception as exc:
                print(f"! could not load {slot}: {exc}")
        print("Companion interactive session (type 'quit' to exit)")
        print(f"  profile={app.hardware.profile_name}  llm={comp.registry.get_optional('llm.default').model_id if comp.registry.has('llm.default') else 'none'}")
        await app.start()
        try:
            while True:
                try:
                    line = (await asyncio.to_thread(input, "you> ")).strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not line:
                    continue
                if line.lower() in ("quit", "exit"):
                    break
                try:
                    result = await app.respond(line)
                    print(f"companion> {result['text']}")
                except Exception as exc:
                    print(f"! error: {exc}")
        finally:
            await app.aclose()
        return 0

    return asyncio.run(loop())


def cmd_interact(args: argparse.Namespace) -> int:
    from companion.application.interact import InteractSession

    voice = bool(args.voice or args.multimodal)
    camera = bool(args.multimodal)
    async def run() -> int:
        app = _app(args)
        if app.components.degraded:
            print(f"! running in degraded mode: {app.components.degraded}")
        await app.start()
        session = InteractSession(app, voice=voice, camera=camera, record=args.record)
        await session.run()
        return 0

    return asyncio.run(run())


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what is actually true, not what is merely configured.

    This command previously reported OK whenever a storage object existed,
    even when the graph behind it had failed and every memory write was going
    to a null store. Diagnostics that can say OK while the system is broken
    are worse than no diagnostics, so each check now has to earn its status.
    """
    cfg = Config.load()
    app = _app(args)
    comp = app.components
    problems: list[str] = []
    warnings: list[str] = []

    print("config:")
    print(f"  db: {cfg.db_path}")
    print(f"  hardware profile: {cfg.hardware_profile}")
    print(f"  ai ram budget: {cfg.max_ai_ram_mb} MB")
    print(f"  vector backend: {cfg.embedding_backend}")
    print("hardware:")
    for k, v in app.hardware.to_dict().items():
        print(f"  {k}: {v}")

    print("model slots:")
    for slot, h in comp.registry.health().items():
        provider = comp.registry.get_optional(slot)
        fallback = bool(getattr(provider, "_fallback", False))
        state = "FALLBACK (not a real model)" if fallback else (
            "loaded" if h["loaded"] else "installed")
        print(f"  {slot:24} provider={h['provider']:12} model={h['model_id']} "
              f"{state} ram={h['ram_mb']}MB"
              f"{'  ERROR: ' + h['load_error'] if h['load_error'] else ''}")
        if h["load_error"]:
            problems.append(f"{slot}: {h['load_error']}")
        elif fallback:
            warnings.append(f"{slot} is running on a deterministic fallback")

    health = app.subsystem_health()
    store = health["storage"]
    print("store:")
    print(f"  writable graph: {'yes' if store['ok'] else 'NO'}")
    if not store["ok"]:
        problems.append(f"cognitive store unavailable: {store['reason']}")
    else:
        # Prove durability rather than assuming it: a committed round-trip is
        # the only evidence that memory will survive a restart.
        try:
            probe_key = "doctor_write_probe"
            comp.graph.set_system_state(probe_key, app.clock.now_iso())
            print(f"  durability probe: {'ok' if comp.graph.get_system_state(probe_key) else 'FAILED'}")
            if not comp.graph.get_system_state(probe_key):
                problems.append("graph write probe did not read back")
        except Exception as exc:
            problems.append(f"graph write probe failed: {exc}")
            print(f"  durability probe: FAILED ({exc})")

    print("skills:")
    for record in comp.skills.all():
        status = "ok" if record.available else f"UNAVAILABLE — {record.reason}"
        print(f"  {record.manifest.id:16} {status}")
        if not record.available:
            warnings.append(f"skill {record.manifest.id}: {record.reason}")
    print(f"tools: {', '.join(comp.tools.ids())}")
    print(f"identity: {comp.self_model.name} (source: {comp.self_model.model().name_source})")

    if warnings:
        print("\nwarnings:")
        for w in warnings:
            print(f"  - {w}")
    if problems:
        print("\nproblems:")
        for p in problems:
            print(f"  - {p}")
    print("\n" + ("DOCTOR OK" if not problems else "DOCTOR: ISSUES FOUND"))
    return 0 if not problems else 1


def _ensure_models(app, slots) -> None:
    comp = app.components
    for slot in slots:
        try:
            comp.lifecycle.load(slot)
        except Exception as exc:
            print(f"! could not load {slot}: {exc}")


def cmd_benchmark(args: argparse.Namespace) -> int:
    from companion.runtime.benchmarks import (
        benchmark_app,
        e2e_benchmark,
        resource_benchmark,
        speech_input_benchmark,
    )

    async def run() -> int:
        results = await benchmark_app(max_tokens=args.max_tokens)
        print("== model benchmarks ==")
        for r in results:
            print(f"{r.slot:24} provider={r.provider:10} loaded={r.loaded} "
                  f"load={r.load_ms:8.1f}ms first={r.first_token_ms:8.1f}ms "
                  f"tps={r.tokens_per_s:8.1f} ram={r.est_ram_mb}MB {r.notes}")
        if args.full:
            rb = await resource_benchmark(iterations=args.turns, max_tokens=args.max_tokens)
            print("\n== end-to-end turns (text) ==")
            print(json.dumps(rb, indent=2))
        if args.e2e or args.audio:
            cfg = Config.load()
            app = CompanionApp(cfg)
            app.build()
            try:
                if args.audio:
                    _ensure_models(app, ("llm.default", "embeddings.default",
                                         "stt.default", "vad.default", "tts.default"))
                    print(f"\n== speech-input pipeline (audio: {args.audio}) ==")
                    print(json.dumps(await speech_input_benchmark(app, args.audio), indent=2))
                else:
                    _ensure_models(app, ("llm.default", "embeddings.default", "tts.default"))
                    print("\n== e2e pipeline latency (real models, silent) ==")
                    print(json.dumps(await e2e_benchmark(app, iterations=args.turns, mute=True),
                                     indent=2))
            finally:
                await app.aclose()
        return 0

    return asyncio.run(run())


def cmd_models(args: argparse.Namespace) -> int:
    from companion.runtime import model_installer as mi

    if args.action == "list":
        print("model manifest (offline-cacheable):")
        for m in mi.status():
            status = "installed" if m["installed"] else ("remote" if m.get("url") else "local")
            print(f"  {m['id']:16} {m['kind']:10} {m['quantization']:8} "
                  f"{m['size_mb']:7}MB  ram~{m['estimated_ram_mb']}MB  {status}"
                  f"{'  verified' if m['verified'] else '  sha256 TBD'}")
        return 0
    if args.action == "status":
        import json as _json

        from companion.runtime.orchestration import CompanionApp

        app = CompanionApp(Config.load())
        app.build()
        registry = app.components.registry
        active = {}
        for slot, provider in {s: registry.get_optional(s) for s in registry.slots()}.items():
            if provider is None:
                continue
            active[slot] = {
                "provider": provider.provider_name,
                "model": provider.model_id,
                "loaded": provider.is_loaded(),
                "fallback": getattr(provider, "_fallback", False),
                "ram_mb": provider.estimate_ram_mb(),
                "capabilities": provider.capability.__dict__ if hasattr(provider, "capability") else {},
            }
        print(_json.dumps({"installed": [m for m in mi.status() if m["installed"]],
                           "active_slots": active}, indent=2, default=str))
        app.shutdown()
        return 0
    if args.action == "install":
        try:
            result = mi.install(args.model)
            print(json.dumps(result, indent=2))
            return 0
        except Exception as exc:
            print(f"install failed: {exc}")
            return 1
    if args.action == "remove":
        removed = mi.remove(args.model)
        print(f"removed {args.model}: {removed}")
        return 0
    return 1


def cmd_memory(args: argparse.Namespace) -> int:
    app = _app(args)
    comp = app.components
    mem = comp.memory
    graph = comp.graph
    if args.action == "stats":
        _json(mem.stats())
        return 0
    if args.action == "list":
        for m in graph.list_memories(status=args.status or "", limit=args.limit):
            print(f"{m.id}  [{m.status}] {m.type.value}  {m.content[:80]}")
        return 0
    if args.action == "forget":
        mem.forget_memory(args.memory_id)
        print(f"forgot {args.memory_id}")
        return 0
    if args.action == "lock":
        mem.lock_memory(args.memory_id, True)
        print(f"locked {args.memory_id}")
        return 0
    if args.action == "unlock":
        mem.lock_memory(args.memory_id, False)
        print(f"unlocked {args.memory_id}")
        return 0
    if args.action == "correct":
        mem.correct_memory(args.memory_id, args.text)
        print(f"corrected {args.memory_id}")
        return 0
    return 1


def cmd_skills(args: argparse.Namespace) -> int:
    app = _app(args)
    comp = app.components
    if args.action == "list":
        for record in comp.skills.all():
            mark = "ok " if record.available else "OFF"
            print(f"[{mark}] {record.manifest.id:14} v{record.manifest.version}  "
                  f"{record.manifest.description}")
            if not record.available:
                print(f"        reason: {record.reason}")
            if record.manifest.required_permissions:
                print(f"        permissions: {', '.join(record.manifest.required_permissions)}")
        return 0
    if args.action == "describe":
        record = comp.skills.get(args.skill_id)
        if record is None:
            print(f"no such skill: {args.skill_id}")
            return 1
        _json(record.to_dict())
        return 0
    if args.action == "permissions":
        _json(comp.permissions.snapshot())
        return 0
    if args.action == "tools":
        _json(comp.tools.manifests())
        return 0
    return 1


def cmd_identity(args: argparse.Namespace) -> int:
    app = _app(args)
    self_model = app.components.self_model
    if args.action == "show":
        _json(self_model.model().to_dict())
        return 0
    if args.action == "history":
        for entry in self_model.name_history():
            state = "current" if entry["current"] else f"until {entry['to'][:19]}"
            print(f"{entry['name']:20} from {entry['from'][:19]}  [{state}]  "
                  f"via {entry['provenance']}")
        return 0
    if args.action == "set":
        result = self_model.set_name(args.name)
        _json(result)
        return 0 if result.get("changed") else 1
    return 1


def cmd_why(args: argparse.Namespace) -> int:
    """Explain why a fact or memory is believed: evidence, source, history."""
    app = _app(args)
    comp = app.components
    graph = comp.graph
    target = args.target

    fact = graph.get_fact(target)
    if fact is not None:
        _json({
            "fact": fact.to_dict(),
            "status": "active" if not fact.valid_to else f"superseded at {fact.valid_to}",
            "evidence": _observations_for(graph, fact.id),
        })
        return 0
    memory = graph.get_memory(target)
    if memory is not None:
        _json({"memory": memory.to_dict(),
               "evidence": _observations_for(graph, memory.id)})
        return 0
    print(f"no fact or memory with id {target!r}")
    return 1


def _observations_for(graph, object_id: str) -> list[dict]:
    out = []
    try:
        rows = graph.list_observations(limit=2000)
    except Exception:
        return out
    for obs in rows:
        payload = obs.payload or {}
        if payload.get("fact_id") == object_id or payload.get("memory_id") == object_id:
            out.append({"kind": obs.kind, "at": obs.timestamp,
                        "confidence": obs.confidence,
                        "utterance": payload.get("utterance", ""),
                        "superseded": payload.get("superseded", [])})
    return out


def cmd_data(args: argparse.Namespace) -> int:
    from companion.runtime import portability

    app = _app(args)
    if args.action == "export":
        result = portability.export_state(app.components.graph, args.path)
        _json(result)
        return 0
    if args.action == "import":
        result = portability.import_state(app.components.graph, args.path,
                                          replace=args.replace)
        _json(result)
        return 0
    return 1


def cmd_personality(args: argparse.Namespace) -> int:
    app = _app(args)
    comp = app.components
    profile = comp.personality.profile()
    print("traits:")
    for name, t in profile.traits.items():
        print(f"  {name:24} {t.value:+.2f}  confidence={t.confidence:.2f}  stable={t.is_stable}")
    print("values:")
    for name, v in profile.values.items():
        print(f"  {name:24} {v.value:+.2f}  confidence={v.confidence:.2f}")
    print("preferences:")
    for name, p in profile.preferences.items():
        print(f"  {name:24} {p.value:+.2f}  confidence={p.confidence:.2f}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    app = _app(args)
    comp = app.components
    g = comp.graph
    if args.action == "entities":
        for e in g.list_entities(limit=args.limit):
            print(f"{e.id}  [{e.type}] {e.name}")
        return 0
    if args.action == "facts":
        for f in g.list_facts():
            print(f"{f.id}  {f.subject_id} -{f.predicate}-> {f.object_id or f.value}  "
                  f"conf={f.confidence} valid={f.valid_from}..{f.valid_to or 'now'}")
        return 0
    if args.action == "goals":
        for goal in g.list_goals(status="active"):
            print(f"{goal.id}  {goal.name}  progress={goal.progress:.2f}")
        return 0
    if args.action == "relationships":
        for r in g.list_relationships():
            print(f"{r.person_name}  valence={r.valence:.2f} closeness={r.closeness:.2f}")
        return 0
    return 1


def cmd_runtime(args: argparse.Namespace) -> int:
    app = _app(args)
    _json(app.runtime_report())
    app.shutdown()
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    from companion.interfaces.api import serve

    cfg = Config.load()
    host = args.host or cfg.api_host
    port = args.port or cfg.api_port

    async def run() -> int:
        return await serve(_app(args), host=host, port=port)

    return asyncio.run(run())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="companion", description="local-first cognitive companion")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="interactive session")
    r.set_defaults(func=cmd_run)

    it = sub.add_parser("interact", help="interactive session (text/voice/multimodal)")
    mode = it.add_mutually_exclusive_group()
    mode.add_argument("--text", action="store_true", help="text REPL (default)")
    mode.add_argument("--voice", action="store_true", help="voice input via mic (VAD+STT)")
    mode.add_argument("--multimodal", action="store_true",
                      help="voice + camera face perception")
    it.add_argument("--record", metavar="PATH", default=None,
                    help="dev-only: record turns to JSONL (default: off)")
    it.set_defaults(func=cmd_interact)

    d = sub.add_parser("doctor", help="diagnostics")
    d.set_defaults(func=cmd_doctor)

    b = sub.add_parser("benchmark", help="offline model benchmarks")
    b.add_argument("--full", action="store_true", help="also run end-to-end text turns")
    b.add_argument("--e2e", action="store_true",
                   help="pipeline latency breakdown with real models (silent)")
    b.add_argument("--audio", metavar="WAV", default=None,
                   help="feed a 16-bit WAV through VAD->STT->respond->TTS and time each stage")
    b.add_argument("--turns", type=int, default=3)
    b.add_argument("--max-tokens", type=int, default=64)
    b.set_defaults(func=cmd_benchmark)

    m = sub.add_parser("models", help="model cache management")
    ma = m.add_subparsers(dest="action", required=True)
    ma.add_parser("list").set_defaults(action="list", func=cmd_models)
    ma.add_parser("status").set_defaults(action="status", func=cmd_models)
    mi = ma.add_parser("install")
    mi.add_argument("model")
    mi.set_defaults(action="install", func=cmd_models)
    mr = ma.add_parser("remove")
    mr.add_argument("model")
    mr.set_defaults(action="remove", func=cmd_models)

    mem = sub.add_parser("memory", help="inspect and manage memories")
    mma = mem.add_subparsers(dest="action", required=True)
    mma.add_parser("stats").set_defaults(action="stats", func=cmd_memory)
    ml = mma.add_parser("list")
    ml.add_argument("--status", default="")
    ml.add_argument("--limit", type=int, default=100)
    ml.set_defaults(action="list", func=cmd_memory)
    for act in ("forget", "lock", "unlock"):
        x = mma.add_parser(act)
        x.add_argument("memory_id")
        x.set_defaults(action=act, func=cmd_memory)
    mc = mma.add_parser("correct")
    mc.add_argument("memory_id")
    mc.add_argument("text")
    mc.set_defaults(action="correct", func=cmd_memory)

    sk = sub.add_parser("skills", help="inspect skills, permissions and tools")
    ska = sk.add_subparsers(dest="action", required=True)
    ska.add_parser("list").set_defaults(action="list", func=cmd_skills)
    ska.add_parser("permissions").set_defaults(action="permissions", func=cmd_skills)
    ska.add_parser("tools").set_defaults(action="tools", func=cmd_skills)
    skd = ska.add_parser("describe")
    skd.add_argument("skill_id")
    skd.set_defaults(action="describe", func=cmd_skills)

    ident = sub.add_parser("identity", help="inspect or set the agent's identity")
    ia = ident.add_subparsers(dest="action", required=True)
    ia.add_parser("show").set_defaults(action="show", func=cmd_identity)
    ia.add_parser("history").set_defaults(action="history", func=cmd_identity)
    iset = ia.add_parser("set")
    iset.add_argument("name")
    iset.set_defaults(action="set", func=cmd_identity)

    why = sub.add_parser("why", help="explain why a fact or memory is believed")
    why.add_argument("target", help="fact id or memory id")
    why.set_defaults(func=cmd_why)

    data = sub.add_parser("data", help="export / import the cognitive state")
    da = data.add_subparsers(dest="action", required=True)
    dex = da.add_parser("export")
    dex.add_argument("path")
    dex.set_defaults(action="export", func=cmd_data)
    dim = da.add_parser("import")
    dim.add_argument("path")
    dim.add_argument("--replace", action="store_true",
                     help="clear existing state before importing")
    dim.set_defaults(action="import", func=cmd_data)

    pers = sub.add_parser("personality", help="inspect the personality model")
    pers.add_argument("action", choices=["inspect"], nargs="?", default="inspect")
    pers.set_defaults(func=cmd_personality)

    g = sub.add_parser("graph", help="inspect the knowledge graph")
    ga = g.add_subparsers(dest="action", required=True)
    for act in ("entities", "facts", "goals", "relationships"):
        x = ga.add_parser(act)
        x.add_argument("--limit", type=int, default=100)
        x.set_defaults(action=act, func=cmd_graph)

    rt = sub.add_parser("runtime", help="runtime health + metrics")
    rt.set_defaults(func=cmd_runtime)

    api = sub.add_parser("api", help="run the HTTP API server")
    api.add_argument("--host", default=None, help="bind host (default: config api.host)")
    api.add_argument("--port", type=int, default=None, help="bind port (default: config api.port)")
    api.set_defaults(func=cmd_api)

    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}")
        if getattr(args, "verbose", False):
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
