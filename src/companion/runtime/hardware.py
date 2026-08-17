"""Hardware detection and profiles.

The runtime detects the machine, estimates usable memory, picks a profile and
logs the decision. GPU acceleration is optional and never required for
correctness. On the target laptop the Iris Xe's "8 GB" is shared system RAM,
so CPU-first execution is the default. `auto` upgrades to the `gpu` profile
only when a dedicated (non-shared) GPU is detected, and even then a benchmark
gate must prove it is clearly faster before GPU layers are used.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from companion.core.errors import HardwareError

log = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    cpu_threads: int = 4
    ram_total_mb: int = 16384
    ram_budget_mb: int = 8192
    gpu_available: bool = False
    gpu_backend: str = "none"          # none | vulkan | cuda | metal
    gpu_memory_shared: bool = True
    gpu_upgrade: bool = False          # a dedicated (non-shared) GPU was detected
    preferred_execution_mode: str = "cpu"  # cpu | gpu | cpu_first
    profile_name: str = "balanced"

    def to_dict(self) -> dict:
        return {
            "cpu_threads": self.cpu_threads,
            "ram_total_mb": self.ram_total_mb,
            "ram_budget_mb": self.ram_budget_mb,
            "gpu_available": self.gpu_available,
            "gpu_backend": self.gpu_backend,
            "gpu_memory_shared": self.gpu_memory_shared,
            "gpu_upgrade": self.gpu_upgrade,
            "preferred_execution_mode": self.preferred_execution_mode,
            "profile_name": self.profile_name,
        }


def detect_cpu_threads() -> int:
    try:
        return os.cpu_count() or 4
    except Exception:  # pragma: no cover
        return 4


def detect_ram_mb() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total / (1024 * 1024))
    except ImportError:
        # Windows fallback
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            ms = MEMORYSTATUSEX()
            ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return int(ms.ullTotalPhys / (1024 * 1024))
        except Exception:
            return 16384


def detect_gpu_backend() -> tuple[bool, str, bool]:
    """Return (available, backend, shared_memory). Conservative on Windows.

    A dedicated NVIDIA GPU (nvidia-smi present) is adopted as `cuda` with its
    own VRAM. Everything else — including a Vulkan loader — is treated as
    shared-memory and NOT adopted until benchmarked (see should_use_vulkan).
    """
    try:
        # Dedicated NVIDIA GPU: nvidia-smi is installed with the driver.
        nvidia_smi = None
        if os.name == "nt":
            for candidate in ("C:/Windows/System32/nvidia-smi.exe",
                              "C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe"):
                if os.path.exists(candidate):
                    nvidia_smi = candidate
                    break
        else:
            import shutil
            nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            return True, "cuda", False
    except Exception:
        pass
    try:
        # Vulkan loader present: could be integrated (shared) or discrete. We
        # only report the loader, never adopt it, until it is benchmarked.
        if os.name == "nt" and os.path.exists("C:/Windows/System32/vulkan-1.dll"):
            return False, "vulkan", True  # present but not adopted until benchmarked
    except Exception:
        pass
    return False, "none", True


def build_hardware_profile(profile_name: str = "balanced",
                           total_ram_mb: int | None = None,
                           reserve_system_mb: int = 4096,
                           max_ai_mb: int | None = None) -> HardwareProfile:
    threads = detect_cpu_threads()
    ram = total_ram_mb or detect_ram_mb()
    gpu_ok, backend, shared = detect_gpu_backend()

    base = {
        "ultra_low": dict(ram_budget_mb=max_ai_mb or 4096, threads=max(2, threads // 2),
                          execution="cpu", profile_name="ultra_low"),
        "balanced": dict(ram_budget_mb=max_ai_mb or 8192, threads=max(4, threads),
                         execution="cpu_first", profile_name="balanced"),
        "performance": dict(ram_budget_mb=max_ai_mb or 12288, threads=threads,
                            execution="cpu_first", profile_name="performance"),
        "gpu": dict(ram_budget_mb=max_ai_mb or 16384, threads=threads,
                    execution="gpu", profile_name="gpu"),
        "custom": dict(ram_budget_mb=max_ai_mb or 8192, threads=threads,
                       execution="cpu_first", profile_name="custom"),
    }.get(profile_name)

    dedicated_gpu = gpu_ok and not shared

    if profile_name == "auto":
        if dedicated_gpu:
            base = dict(ram_budget_mb=max_ai_mb or 16384, threads=threads,
                        execution="gpu", profile_name="gpu")
        elif ram >= 32768:
            base = dict(ram_budget_mb=max_ai_mb or 12288, threads=threads,
                        execution="cpu_first", profile_name="performance")
        elif ram >= 15360:  # a real 16 GB machine reports ~16.0-15.7 GB usable
            base = dict(ram_budget_mb=max_ai_mb or 8192, threads=threads,
                        execution="cpu_first", profile_name="balanced")
        else:
            base = dict(ram_budget_mb=max_ai_mb or 4096, threads=max(2, threads // 2),
                        execution="cpu", profile_name="ultra_low")

    if base is None:
        raise HardwareError(f"unknown hardware profile: {profile_name}")

    budget = min(base["ram_budget_mb"], max(2048, ram - reserve_system_mb))
    profile = HardwareProfile(
        cpu_threads=base["threads"],
        ram_total_mb=ram,
        ram_budget_mb=budget,
        gpu_available=gpu_ok and backend != "none",
        gpu_backend=backend if gpu_ok else "none",
        gpu_memory_shared=shared,
        gpu_upgrade=dedicated_gpu,
        preferred_execution_mode=base["execution"],
        profile_name=base["profile_name"],
    )
    log.info(
        "hardware: profile=%s cpu=%d ram=%dMB ai_budget=%dMB gpu=%s(%s,shared=%s,upgrade=%s)",
        profile.profile_name, profile.cpu_threads, profile.ram_total_mb,
        profile.ram_budget_mb, profile.gpu_backend, profile.gpu_available,
        profile.gpu_memory_shared, profile.gpu_upgrade,
    )
    return profile


def should_use_gpu(profile: HardwareProfile, benchmark_result: dict | None = None) -> bool:
    """GPU layers only if capability AND benchmark gate both pass.

    A dedicated GPU (cuda) is adopted when its tokens/s beat CPU by the same
    headroom; shared-memory Vulkan must additionally clear the conservative
    adoption rule (it never adopts unless clearly faster).
    """
    if not profile.gpu_available or profile.gpu_backend not in ("vulkan", "cuda"):
        return False
    if profile.gpu_backend == "cuda" and profile.gpu_memory_shared:
        return False  # shared-memory cuda is never adopted
    if benchmark_result is None:
        return False  # benchmark gate required
    cpu_tps = benchmark_result.get("cpu_tokens_per_s", 0.0)
    gpu_tps = benchmark_result.get("gpu_tokens_per_s", 0.0)
    return gpu_tps > cpu_tps * 1.15  # only if clearly faster


def should_use_vulkan(profile: HardwareProfile, benchmark_result: dict | None = None) -> bool:
    """Backward-compatible alias for the Vulkan-only gate."""
    return should_use_gpu(profile, benchmark_result) and profile.gpu_backend == "vulkan"
