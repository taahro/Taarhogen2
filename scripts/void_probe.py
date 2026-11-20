"""Read-only hardware capability probe for the TAARHOGEN2 / VOID worker path.

This script does not download model weights, allocate GPU tensors, or execute inference.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

MODEL_VRAM_PLANNING_GB = 13
GIB = 1024**3


def _memory_snapshot() -> dict[str, int | None]:
    total = available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(":")
            parts = raw.strip().split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) * 1024
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
    if total is None:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            total = os.sysconf("SC_PHYS_PAGES") * page_size
            available = os.sysconf("SC_AVPHYS_PAGES") * page_size
        except (AttributeError, OSError, ValueError):
            pass
    return {"totalBytes": total, "availableBytes": available}


def _nvidia_smi_devices() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    devices = []
    for line in completed.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            devices.append({
                "name": fields[0],
                "totalVramBytes": int(fields[1]) * 1024**2,
                "freeVramBytes": int(fields[2]) * 1024**2,
                "source": "nvidia-smi",
            })
        except ValueError:
            continue
    return devices


def _gpu_snapshot() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "torchAvailable": False,
            "cudaAvailable": False,
            "cudaVersion": None,
            "hipVersion": None,
            "devices": _nvidia_smi_devices(),
        }

    devices: list[dict[str, Any]] = []
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            total_bytes = int(properties.total_memory)
            free_bytes = None
            try:
                free_bytes, _ = torch.cuda.mem_get_info(index)
                free_bytes = int(free_bytes)
            except (AttributeError, RuntimeError, TypeError):
                pass
            devices.append({
                "index": index,
                "name": str(properties.name),
                "totalVramBytes": total_bytes,
                "freeVramBytes": free_bytes,
                "source": "pytorch",
            })

    if not devices:
        devices = _nvidia_smi_devices()

    return {
        "torchAvailable": True,
        "torchVersion": str(torch.__version__),
        "cudaAvailable": cuda_available,
        "cudaVersion": getattr(torch.version, "cuda", None),
        "hipVersion": getattr(torch.version, "hip", None),
        "devices": devices,
    }


def assess_readiness(gpu: dict[str, Any]) -> dict[str, Any]:
    devices = gpu.get("devices") or []
    if not gpu.get("torchAvailable"):
        return {
            "status": "probe_incomplete",
            "reason": "PyTorch is not installed in this runtime. Install or select a compatible runtime before model testing.",
        }
    if not gpu.get("cudaAvailable"):
        return {
            "status": "cpu_only",
            "reason": "PyTorch cannot access a CUDA/ROCm device. CPU generation is not recommended for this model.",
        }
    if not devices:
        return {
            "status": "gpu_details_unavailable",
            "reason": "A GPU runtime is visible, but device memory details could not be measured.",
        }

    planning_bytes = MODEL_VRAM_PLANNING_GB * GIB
    max_total = max(int(device.get("totalVramBytes") or 0) for device in devices)
    max_free = max(int(device.get("freeVramBytes") or 0) for device in devices)
    if max_total < planning_bytes:
        return {
            "status": "below_planning_vram_estimate",
            "reason": "Reported total VRAM is below the approximate planning estimate. Do not download model weights yet.",
        }
    if max_free and max_free < planning_bytes:
        return {
            "status": "low_free_vram",
            "reason": "Reported free VRAM is below the approximate planning estimate. Close other workloads or evaluate an explicitly tested offload strategy.",
        }
    return {
        "status": "candidate_for_load_test",
        "reason": "Reported GPU memory meets the approximate planning estimate. This is not proof of successful model loading or generation.",
    }


def collect_report() -> dict[str, Any]:
    gpu = _gpu_snapshot()
    memory = _memory_snapshot()
    disk = shutil.disk_usage(Path.cwd())
    return {
        "schemaVersion": "1.0.0",
        "probe": "TAARHOGEN2_VOID_CAPABILITY",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "operatingSystem": platform.system(),
            "architecture": platform.machine(),
            "pythonVersion": platform.python_version(),
        },
        "memory": memory,
        "disk": {
            "path": str(Path.cwd()),
            "totalBytes": disk.total,
            "freeBytes": disk.free,
        },
        "gpu": gpu,
        "software": {
            "diffusersAvailable": importlib.util.find_spec("diffusers") is not None,
            "transformersAvailable": importlib.util.find_spec("transformers") is not None,
            "accelerateAvailable": importlib.util.find_spec("accelerate") is not None,
        },
        "model": {
            "id": os.getenv("TAARHOGEN2_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B"),
            "approximateVramPlanningGb": MODEL_VRAM_PLANNING_GB,
        },
        "assessment": assess_readiness(gpu),
        "safety": {
            "modelWeightsDownloaded": False,
            "gpuTensorAllocated": False,
            "inferenceExecuted": False,
            "networkRequestMade": False,
        },
        "limitations": [
            "This is a hardware inventory only; it does not load the model.",
            "The VRAM figure is approximate and must be validated against the current model card and actual runtime.",
            "A candidate result is not a generation-success guarantee.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe hardware for TAARHOGEN2 without downloading model weights.")
    parser.add_argument("--output", help="Optional JSON output path. If omitted, print the report to stdout.")
    args = parser.parse_args()

    report = collect_report()
    rendered = json.dumps(report, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print("CAPABILITY_REPORT=" + str(output_path))
    else:
        print(rendered)


if __name__ == "__main__":
    main()
