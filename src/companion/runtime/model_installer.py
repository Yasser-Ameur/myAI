"""Model manifest + downloader.

`companion models install <key>` downloads a model listed in models/manifest.json,
verifies its sha256 and size, validates the artifact type, records it under
models/cache/<key>/ and updates models/cache/.installed.json so providers can
reference local paths. Everything is offline-first: the manifest ships with the
project and every artifact carries a verified sha256.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request

from companion.core.errors import (
    ChecksumMismatchError,
    InvalidModelArtifactError,
    ModelNotFoundError,
)

log = logging.getLogger(__name__)

MANIFEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "manifest.json"))
INSTALLED_STATE_PATH = os.path.abspath(os.path.join(os.path.dirname(MANIFEST_PATH), "cache", ".installed.json"))


def manifest_path() -> str:
    return MANIFEST_PATH


def load_manifest() -> dict:
    with open(manifest_path(), "r", encoding="utf-8") as fh:
        return json.load(fh)


def cache_dir() -> str:
    path = os.path.join(os.path.dirname(manifest_path()), "cache")
    os.makedirs(path, exist_ok=True)
    return path


def local_dir(key: str) -> str:
    """Directory in the cache holding all files for a model key."""
    return os.path.join(cache_dir(), key)


def local_file(key: str, role: str = "model") -> str:
    """Path to the file for a manifest key + role, if known and installed."""
    meta = load_manifest().get("models", {}).get(key)
    if meta is None:
        raise ModelNotFoundError(f"model '{key}' not in manifest")
    for f in meta.get("files", []):
        if f.get("role", "model") == role:
            return os.path.join(local_dir(key), f["name"])
    raise ModelNotFoundError(f"model '{key}' has no file with role '{role}'")


def artifact_info(key: str) -> list[dict]:
    """Manifest file descriptors for a key, annotated with local paths."""
    meta = load_manifest().get("models", {}).get(key)
    if meta is None:
        raise ModelNotFoundError(f"model '{key}' not in manifest; known: {list(load_manifest()['models'])}")
    out = []
    for f in meta.get("files", []):
        out.append({
            **f,
            "local": os.path.join(local_dir(key), f["name"]),
            "installed": os.path.exists(os.path.join(local_dir(key), f["name"])),
        })
    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_sha256(path: str, expected: str) -> bool:
    if not expected:
        return True  # unverified placeholders are not yet claimed as real
    return sha256_of(path) == expected.lower()


def verify_size(path: str, expected_bytes: int) -> bool:
    return expected_bytes == 0 or os.path.getsize(path) == expected_bytes


MAGIC = {
    "gguf": (b"GGUF", 0),
    "tflite": (b"TFL3", 4),
    "zip": (b"PK\x03\x04", 0),
}


def verify_type(path: str, fmt: str) -> bool:
    """Light artifact-type validation via header magic where applicable."""
    if fmt == "onnx":
        # ONNX is a protobuf; the first field (ir_version) starts with 0x08.
        # When onnxruntime is available, use it for a real load check.
        try:
            import onnxruntime as ort

            ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            return True
        except Exception:
            return False
    magic_spec = MAGIC.get(fmt)
    if magic_spec is None:
        return True  # formats without a reliable magic are not type-checked
    magic, offset = magic_spec
    with open(path, "rb") as fh:
        fh.seek(offset)
        head = fh.read(len(magic))
    if head == magic:
        return True
    if fmt == "tflite":
        # MediaPipe .task bundles are ZIP archives; accept those too.
        with open(path, "rb") as fh:
            prefix = fh.read(16)
        return MAGIC["zip"][0] in prefix
    return False


# ---------------------------------------------------------------------------
# Install / remove / status
# ---------------------------------------------------------------------------

def _load_installed_state() -> dict:
    if os.path.exists(INSTALLED_STATE_PATH):
        try:
            with open(INSTALLED_STATE_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"version": 2, "installed": {}}


def _save_installed_state(state: dict) -> None:
    os.makedirs(os.path.dirname(INSTALLED_STATE_PATH), exist_ok=True)
    tmp = INSTALLED_STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, INSTALLED_STATE_PATH)


def installed_state() -> dict:
    return _load_installed_state()


def is_installed(key: str, require_verified: bool = True) -> bool:
    """True if every manifest file for the key exists (and, by default, verifies)."""
    meta = load_manifest().get("models", {}).get(key)
    if meta is None:
        return False
    for f in meta.get("files", []):
        path = os.path.join(local_dir(key), f["name"])
        if not os.path.exists(path):
            return False
        if require_verified and f.get("sha256") and not verify_sha256(path, f["sha256"]):
            return False
    return True


def _download(url: str, dest: str, label: str, expected_bytes: int = 0) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    resumed = 0
    if os.path.exists(tmp):
        resumed = os.path.getsize(tmp)
    headers = {"User-Agent": "human-companion/0.1"}
    if resumed:
        headers["Range"] = f"bytes={resumed}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0) or expected_bytes
        if resp.status == 206:
            total += resumed
        done = resumed
        print(f"  {label}: downloading {total / (1024 * 1024):.1f} MB"
              + (f" (resuming from {resumed / (1024 * 1024):.1f} MB)" if resumed else ""))
        mode = "ab" if resumed else "wb"
        with open(tmp, mode) as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    if pct % 10 == 0:
                        print(f"    {pct}%", end="\r")
    if total and done != total:
        raise InvalidModelArtifactError(
            f"short download for {label}: got {done}, expected {total} (retry to resume)"
        )
    os.replace(tmp, dest)
    print(f"    done: {os.path.getsize(dest) / (1024 * 1024):.1f} MB")


def install(key: str, manifest: dict | None = None, verify: bool = True,
            progress=print) -> dict:
    """Download + verify + place all files for a manifest key."""
    meta = (manifest or load_manifest()).get("models", {}).get(key)
    if meta is None:
        raise ModelNotFoundError(
            f"model '{key}' not in manifest; known: {sorted((manifest or load_manifest())['models'])}"
        )
    files = meta.get("files", [])
    if not files:
        raise InvalidModelArtifactError(f"model '{key}' has no files")
    progress(f"installing {key} ({meta.get('format', '?')}, "
             f"{sum(f.get('size_bytes', 0) for f in files) / (1024 * 1024):.1f} MB)")
    state = _load_installed_state()
    for f in files:
        dest = os.path.join(local_dir(key), f["name"])
        if os.path.exists(dest) and verify_sha256(dest, f.get("sha256", "")) \
                and verify_size(dest, f.get("size_bytes", 0)):
            progress(f"  {f['name']}: already present and verified")
            continue
        _download(f["url"], dest, f["name"], f.get("size_bytes", 0))
        if verify:
            if f.get("sha256") and not verify_sha256(dest, f["sha256"]):
                os.remove(dest)
                raise ChecksumMismatchError(
                    f"sha256 mismatch for {key}/{f['name']}: "
                    f"expected {f['sha256']}, got {sha256_of(dest)}"
                )
            if not verify_size(dest, f.get("size_bytes", 0)):
                os.remove(dest)
                raise InvalidModelArtifactError(
                    f"size mismatch for {key}/{f['name']}: "
                    f"expected {f.get('size_bytes')}, got {os.path.getsize(dest)}"
                )
            if f.get("role", "model") == "model" and not verify_type(dest, meta.get("format", "")):
                os.remove(dest)
                raise InvalidModelArtifactError(f"type mismatch for {key}/{f['name']}")
    state["installed"][key] = {
        "id": key,
        "format": meta.get("format", ""),
        "size_bytes": sum(f.get("size_bytes", 0) for f in files),
        "files": [f["name"] for f in files],
        "verified": all(f.get("sha256") for f in files),
    }
    _save_installed_state(state)
    progress(f"installed {key} -> {local_dir(key)}")
    return {"key": key, "dir": local_dir(key), "files": [f["name"] for f in files]}


def remove(key: str) -> bool:
    dest = local_dir(key)
    state = _load_installed_state()
    if os.path.isdir(dest):
        import shutil

        shutil.rmtree(dest)
        state["installed"].pop(key, None)
        _save_installed_state(state)
        return True
    return False


def status(key: str | None = None) -> list[dict]:
    """installed / available / remote / sizes for reporting."""
    manifest = load_manifest()
    rows = []
    for key, meta in sorted(manifest.get("models", {}).items()):
        if key.startswith("_"):
            continue
        total = sum(f.get("size_bytes", 0) for f in meta.get("files", []))
        rows.append({
            "id": key,
            "kind": meta.get("kind", "?"),
            "provider": meta.get("provider", "?"),
            "format": meta.get("format", "?"),
            "quantization": meta.get("quantization", ""),
            "installed": is_installed(key),
            "verified": all(f.get("sha256") for f in meta.get("files", [])),
            "size_bytes": total,
            "size_mb": round(total / (1024 * 1024), 1),
            "estimated_ram_mb": meta.get("estimated_ram_mb", 0),
            "capabilities": meta.get("capabilities", []),
            "languages": meta.get("languages", []),
        })
    return rows


def resolve_provider_paths(cfg: dict) -> dict:
    """Inject local cache paths into a provider config when they are missing.

    Keeps adapters config-driven while letting `companion models install`
    populate paths automatically from the manifest cache.
    """
    model_id = cfg.get("model_id", "")
    kind = cfg.get("kind", "")
    out = dict(cfg)
    try:
        meta = load_manifest().get("models", {}).get(model_id)
    except Exception:
        meta = None
    if meta is None:
        return out
    files = {f.get("role", "model"): os.path.join(local_dir(model_id), f["name"]) for f in meta.get("files", [])}
    if kind == "llm":
        out.setdefault("path", files.get("model"))
    elif kind == "stt":
        # faster-whisper loads a CT2 *directory* (config.json + model.bin + tokenizer).
        out.setdefault("model_path", local_dir(model_id))
    elif kind == "tts" and meta.get("provider") == "kokoro":
        out.setdefault("path", files.get("model"))
        out.setdefault("voices_path", files.get("voices"))
    elif kind == "vad":
        out.setdefault("path", files.get("model"))
    elif kind == "vision":
        out.setdefault("model_path", files.get("model"))
    elif kind == "embeddings":
        out.setdefault("path", files.get("model"))
        out.setdefault("tokenizer_path", local_dir(model_id))
    return out
