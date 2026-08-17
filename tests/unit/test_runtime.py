from companion.runtime.hardware import (
    HardwareProfile,
    build_hardware_profile,
    should_use_gpu,
    should_use_vulkan,
)
from companion.runtime.memory_guard import MemoryGuard
from companion.runtime.scheduler import Scheduler, WorkloadClass


def test_balanced_profile_budget():
    profile = build_hardware_profile("balanced", total_ram_mb=16106, max_ai_mb=8192)
    assert profile.profile_name == "balanced"
    assert profile.ram_budget_mb == 8192
    assert profile.cpu_threads >= 4
    assert profile.preferred_execution_mode == "cpu_first"


def test_auto_profile_scales_with_ram():
    small = build_hardware_profile("auto", total_ram_mb=6144)
    laptop16gb = build_hardware_profile("auto", total_ram_mb=16106)
    big = build_hardware_profile("auto", total_ram_mb=65536)
    assert small.profile_name == "ultra_low"
    assert laptop16gb.profile_name == "balanced"  # a real 16 GB laptop
    assert big.profile_name == "performance"
    assert small.ram_budget_mb < laptop16gb.ram_budget_mb < big.ram_budget_mb


def test_gpu_profile_upgrade():
    profile = build_hardware_profile("gpu", total_ram_mb=32768)
    assert profile.profile_name == "gpu"
    assert profile.preferred_execution_mode == "gpu"
    assert profile.ram_budget_mb > build_hardware_profile("balanced", total_ram_mb=32768).ram_budget_mb


def test_vulkan_gate_requires_benchmark_headroom():
    profile = HardwareProfile(gpu_available=True, gpu_backend="vulkan")
    assert not should_use_vulkan(profile, {"cpu_tokens_per_s": 120, "gpu_tokens_per_s": 100})
    assert not should_use_vulkan(profile, {"cpu_tokens_per_s": 120, "gpu_tokens_per_s": 130})
    assert should_use_vulkan(profile, {"cpu_tokens_per_s": 120, "gpu_tokens_per_s": 200})
    assert not should_use_vulkan(profile, None)  # benchmark gate required


def test_cuda_gate_is_dedicated_only():
    cuda = HardwareProfile(gpu_available=True, gpu_backend="cuda", gpu_memory_shared=False)
    assert not should_use_gpu(cuda, {"cpu_tokens_per_s": 120, "gpu_tokens_per_s": 100})
    assert should_use_gpu(cuda, {"cpu_tokens_per_s": 120, "gpu_tokens_per_s": 200})
    shared = HardwareProfile(gpu_available=True, gpu_backend="cuda", gpu_memory_shared=True)
    assert not should_use_gpu(shared, {"cpu_tokens_per_s": 120, "gpu_tokens_per_s": 300})


def test_scheduler_gates_background_on_interactive():
    sched = Scheduler(max_background_workers=1)
    assert sched.can_run_background()
    sched.begin_interactive()
    assert not sched.can_run_background()
    sched.end_interactive()
    assert sched.can_run_background()


def test_scheduler_worker_limit():
    sched = Scheduler(max_background_workers=1)
    sched.begin_background()
    assert not sched.can_run_background()
    sched.end_background()
    assert sched.can_run_background()


def test_scheduler_workload_classification():
    sched = Scheduler()
    assert sched.class_of("vad") == WorkloadClass.REALTIME
    assert sched.class_of("stt") == WorkloadClass.INTERACTIVE
    assert sched.class_of("embedding") == WorkloadClass.BACKGROUND


def test_memory_guard_escalation_levels():
    guard = MemoryGuard(threshold_elevated_mb=100, threshold_critical_mb=200,
                        process_ram_provider=lambda: 300)
    assert guard.check() == "critical"
    guard2 = MemoryGuard(threshold_elevated_mb=100, threshold_critical_mb=200,
                         process_ram_provider=lambda: 120)
    assert guard2.check() == "elevated"
    guard3 = MemoryGuard(threshold_elevated_mb=100, threshold_critical_mb=200,
                         process_ram_provider=lambda: 50)
    assert guard3.check() == "normal"
