"""Hardware report: CPU, RAM and CUDA/MPS availability. Author: Sandeep Belgavi.

Never raises: anything that cannot be detected is reported as absent. Same JSON keys as the Java
``dev.belgavi.llm.HardwareProbe``.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Report:
    os: str
    arch: str
    cpuCores: int  # noqa: N815 - JSON keys are shared with the Java record
    ramBytes: int  # noqa: N815
    cudaAvailable: bool  # noqa: N815
    cudaDevices: list[str]  # noqa: N815
    mpsAvailable: bool  # noqa: N815
    tier: str


def is_apple_silicon(system: str, machine: str) -> bool:
    return system.lower() == "darwin" and machine.lower() in ("arm64", "aarch64")


def tier(cuda_devices: list[str], mps: bool) -> str:
    if cuda_devices:
        return "GPU"
    return "CPU+" if mps else "CPU"


def total_ram_bytes() -> int:
    """Total physical RAM, or 0 if unknown."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        pass
    try:  # Windows has no sysconf
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = _Mem()
        m.dwLength = ctypes.sizeof(_Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))  # type: ignore[attr-defined]
        return int(m.ullTotalPhys)
    except Exception:  # noqa: BLE001 - probe must not crash
        return 0


def cuda_devices() -> list[str]:
    """CUDA devices from the fixed command ``nvidia-smi -L``; empty if absent or failing."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line.startswith("GPU ")]


def probe() -> Report:
    system, machine = platform.system(), platform.machine()
    cuda = cuda_devices()
    mps = is_apple_silicon(system, machine)
    return Report(
        os=system, arch=machine, cpuCores=os.cpu_count() or 1, ramBytes=total_ram_bytes(),
        cudaAvailable=bool(cuda), cudaDevices=cuda, mpsAvailable=mps, tier=tier(cuda, mps),
    )


def main() -> None:
    print(json.dumps(asdict(probe()), indent=2))


if __name__ == "__main__":
    main()
