# Models

All model selection lives in `config/companion.yaml`. Application code never
hard-codes a model name. The offline catalog lives in `models/manifest.json`.

## Slots

| Slot               | Kind        | Default provider | Default model     | Approx RAM |
|--------------------|-------------|------------------|-------------------|------------|
| `llm.default`      | LLM         | `llama_cpp`      | qwen3-1.7b (Q4_K_M) | ~1.2 GB  |
| `llm.fast`         | LLM         | `llama_cpp`      | qwen3-0.6b (Q4_K_M) | ~0.5 GB  |
| `llm.reasoning`    | LLM         | `llama_cpp`      | qwen3-1.7b        | ~1.2 GB  |
| `stt.default`      | STT         | `whisper_cpp`    | whisper-base      | ~0.6 GB  |
| `tts.default`      | TTS         | `kokoro`         | kokoro-82m        | ~0.4 GB  |
| `vision.face`      | Vision      | `mediapipe`      | face-landmarker   | ~0.25 GB |
| `embeddings.default` | Embeddings | `onnx`          | bge-small-en      | ~0.1 GB  |
| `vad.default`      | VAD         | `silero`         | silero-v4         | ~0.05 GB |

## Fallbacks

Every optional dependency is checked at provider construction via
`BaseAdapter.require(module_name, hint)`, which raises
`ProviderNotAvailableError` with install instructions. `factory.build_provider`
catches it and installs a deterministic fallback for that slot:

| Kind      | Fallback                                            |
|-----------|-----------------------------------------------------|
| LLM       | `MockLLMProvider` (schema-aware, rule responses)    |
| STT       | `MockSTTProvider` (echoes a canned transcript)      |
| TTS       | `MockTTSProvider` (text-only, no audio)             |
| VAD       | `NullVADProvider` (energy-based when `energy`)      |
| Vision    | `MockFaceProvider` (neutral expression)             |
| Embeddings | `MockEmbeddingProvider` (deterministic vector)     |

The registry records `fallback: true` + reason in `meta(slot)`, the CLI warns
once at startup, and `doctor` reports it. The system runs with zero weights
installed; it just won't be very smart or speak audio.

## Offline install

```powershell
companion models list                # manifest catalog
companion models install qwen3-1.7b  # download + sha256 verify into cache
companion models remove qwen3-1.7b
```

`runtime/model_installer.py` verifies SHA-256 when the manifest provides one,
skips existing files, and reports size/state.

## Dependency extras

```powershell
pip install "myai[llm]"        # llama-cpp-python
pip install "myai[stt]"        # faster-whisper
pip install "myai[tts]"        # kokoro-onnx, soundfile, piper-tts
pip install "myai[vad]"        # onnxruntime (+torch for some paths)
pip install "myai[embeddings]" # onnxruntime
pip install "myai[vision]"     # mediapipe
pip install "myai[obs]"        # psutil for RAM probes
```

## Tuning knobs (config `providers:`)

- `llama_cpp.n_ctx` (default 4096), `n_batch` (256), `gpu_layers: auto`
- `whisper_cpp.n_threads: auto`, `beam_size: 1`
- `kokoro.device: cpu`, `voice: ff_siwis`
- `silero.onnx: true`, `threshold: 0.5`
- `onnx_embeddings.device: cpu`

## Manifest (v1)

| Key                | Format | Size  |
|--------------------|--------|-------|
| `qwen3-0.6b`       | GGUF   | 471 MB |
| `qwen3-1.7b`       | GGUF   | 1.1 GB |
| `qwen3-4b`         | GGUF   | 2.4 GB |
| `whisper-base`     | CT2    | 151 MB |
| `whisper-small`    | CT2    | 484 MB |
| `kokoro-82m`       | ONNX   | 326 MB |
| `silero-vad`       | ONNX   | 2 MB   |
| `face-landmarker`  | TFLite | 7 MB   |
| `bge-small-en`     | ONNX   | 24 MB  |
