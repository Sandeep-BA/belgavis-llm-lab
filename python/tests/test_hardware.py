"""Author: Sandeep Belgavi."""

import json
from dataclasses import asdict

from belgavi_llm import hardware

KEYS = {
    "os", "arch", "cpuCores", "ramBytes", "cudaAvailable", "cudaDevices", "mpsAvailable", "tier",
}


def test_probe_reports_cpu_without_crashing():
    r = hardware.probe()
    assert r.cpuCores >= 1
    assert r.ramBytes >= 0
    assert r.cudaAvailable == bool(r.cudaDevices)
    assert r.tier in {"GPU", "CPU+", "CPU"}


def test_json_keys_match_java():
    assert set(json.loads(json.dumps(asdict(hardware.probe())))) == KEYS


def test_apple_silicon():
    assert hardware.is_apple_silicon("Darwin", "arm64")
    assert not hardware.is_apple_silicon("Darwin", "x86_64")
    assert not hardware.is_apple_silicon("Linux", "aarch64")


def test_tier_rules():
    assert hardware.tier(["GPU 0: x"], True) == "GPU"
    assert hardware.tier([], True) == "CPU+"
    assert hardware.tier([], False) == "CPU"


def test_missing_nvidia_smi_is_empty(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
    assert hardware.cuda_devices() == []
