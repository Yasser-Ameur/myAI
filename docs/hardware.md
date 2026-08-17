# Hardware

The runtime detects hardware at startup and derives a memory budget that the
memory guard enforces while running. Everything targets a modest CPU laptop
with shared-memory iGPU — no discrete GPU required.

## Detection

`runtime/hardware.py` detects:

- CPU core/thread count
- total RAM (via psutil if installed, else a ctypes `GlobalMemoryStatusEx`
  probe)
- GPU presence and shared vs dedicated memory

`build_hardware_profile(profile_name, total_ram_mb=None,
reserve_system_mb=4096, max_ai_mb=None)` returns a `HardwareProfile` with
`max_ai_ram_mb` (AI budget) computed as `total_ram - reserve`.

## Profiles

| Profile      | Use                                                        |
|--------------|------------------------------------------------------------|
| `ultra_low`  | 8 GB class machines; smaller n_ctx, fewer threads, greedy   |
| `balanced`   | default; targets ~16 GB RAM laptops                        |
| `performance`| faster tokens-per-sec at higher memory cost                 |
| `gpu`        | machine with a dedicated (non-shared) GPU; bigger budget + GPU execution |
| `custom`     | use explicit `runtime.memory.max_ai_ram_mb`                |
| `auto`       | pick from detected RAM; upgrades to `gpu` on a dedicated GPU |

On the reference machine (Intel i7-1185G7, 4c/8t, 16 GB, no discrete GPU)
`auto` produces: budget 8192 MB, `gpu_memory_shared: True`,
`preferred_execution_mode: cpu_first`, `profile_name: balanced`.

## Budget math

AI budget is shared across resident models. `memory_guard.py` computes pressure:

- **normal** — total estimated resident RAM below ~75% of budget
- **elevated** — between 75% and ~90%; background tasks are deprioritized
- **critical** — above ~90%; models are unloaded in this order: STT, TTS,
  embeddings, LLM (kept only if actively generating)

`Scheduler` classifies work as `REALTIME` (conversation turn), `INTERACTIVE`
(extraction, retrieval) or `BACKGROUND` (reflection, consolidation) and refuses
to launch background work while the machine is at critical pressure.

## GPU / Vulkan

`gpu_acceleration: auto` probes for a Vulkan-capable iGPU and for a dedicated
NVIDIA GPU (`nvidia-smi`). Actual offload is gated by
`should_use_gpu(profile, benchmark_result)` which requires a measured GPU
tokens-per-sec strictly greater than `cpu_tps * 1.15` — a GPU that is not
meaningfully faster than CPU is not used. A dedicated (non-shared) `cuda`
backend is only adopted when `gpu_memory_shared` is false. Set
`hardware.gpu_acceleration: vulkan|cuda` to force the attempt. On a machine
with a discrete GPU, `profile: auto` selects the `gpu` profile
(`preferred_execution_mode: gpu`, larger AI budget) automatically.

## Observing

```powershell
companion doctor       # prints the detected profile, budget, and slot RAM
companion benchmark    # per-slot load / first-token / tokens-per-sec
companion runtime      # live health + metrics
```
