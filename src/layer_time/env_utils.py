"""Environment capture and reproducibility utilities."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import psutil


def _safe_run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""


def capture_environment(run_dir: Path) -> None:
    """Write environment metadata into run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)

    env: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }

    # Torch is optional at import time here; record if available.
    try:
        import torch  # noqa: F401

        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            env["cuda_device_name_0"] = torch.cuda.get_device_name(0)
    except Exception:
        env["torch_version"] = None

    # Git commit (if repo)
    env["git_commit"] = _safe_run(["git", "rev-parse", "HEAD"])

    (run_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    # Full package freeze
    freeze = _safe_run([sys.executable, "-m", "pip", "freeze"])
    (run_dir / "pip_freeze.txt").write_text(freeze + "\n", encoding="utf-8")
