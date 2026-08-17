# Replacing Models

Every slot is replaceable through configuration alone — no code changes.

## LLM (default `llm.default`)

1. Get a GGUF file (any model llama.cpp can load, e.g. Qwen, Llama, Gemma,
   Mistral).
2. Add it to `models/manifest.json`:
   ```json
   "my-model-7b": { "kind": "llm", "format": "gguf",
     "url": "https://.../my-model-7b-Q4_K_M.gguf",
     "sha256": "...", "size_mb": 4300 }
   ```
   then `companion models install my-model-7b`.
3. Point the slot at it in `config/companion.yaml`:
   ```yaml
   models:
     llm:
       default:
         provider: llama_cpp
         model_id: my-model-7b
         quantization: q4_k_m
   ```
   `model_id` resolves against the local model cache; alternatively set
   `path: C:/models/my-model.gguf` directly.

`llm.fast` and `llm.reasoning` work the same way — choose a small model for
classification/extraction and a larger one for reasoning.

## STT (`stt.default`)

The `whisper_cpp` provider expects a CTranslate2 whisper model
(`model.bin` + `config.json` + `tokenizer.json`):

```yaml
models:
  stt:
    default:
      provider: whisper_cpp
      model_id: whisper-small
```

Any faster-whisper-compatible model works (whisper tiny/base/small/medium,
multilingual). Point `model_path` at a local CTranslate2 dir to use a custom one.

## TTS (`tts.default`)

Kokoro provider:

```yaml
models:
  tts:
    default:
      provider: kokoro
      model_id: kokoro-82m
      voice: ff_siwis
```

With `path`/`voices_path` you can use any Kokoro `.onnx` + voices bin pair. An
alternative `piper` provider exists (`pip install "human-companion[tts]"`):
set `provider: piper` with `model_path: <piper .onnx>`. Both return 16-bit PCM
mono audio consumed by `application/speech_output.py`.

## Embeddings (`embeddings.default`)

ONNX embedding provider (default `bge-small-en`, 384-dim):

```yaml
models:
  embeddings:
    default:
      provider: onnx
      model_id: bge-small-en
      dimension: 384
```

`dimension` must match the model. Swap in any sentence-embedding ONNX export
(e.g. `all-MiniLM-L6-v2`); the vector index is dimension-agnostic.

## VAD (`vad.default`)

- `silero` — requires onnxruntime; `model_id: silero-v4`, `threshold: 0.5`.
- `energy` — dependency-free energy-based VAD; set `provider: energy`.

## Vision (`vision.face`)

`mediapipe` provider loads `face_landmarker.task` and produces
`FaceObservation` (expression, gaze, mood estimate). Use any mediapipe
face-landmarker task file via `path`.

## Validating a swap

```powershell
companion doctor          # slots show your provider/model, deps resolved
companion benchmark       # real load/first-token/tps numbers (no SIMULATED label)
companion runtime         # live resident RAM vs the AI budget
```
