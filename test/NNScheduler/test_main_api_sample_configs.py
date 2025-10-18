#!/usr/bin/env python3
"""Smoke test main_api.py against bundled sample configs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "test" / "sample_config"


def _collect_sample_configs() -> list[Path]:
    configs = sorted(SAMPLE_DIR.glob("*.json"), key=lambda p: p.name)
    if not configs:
        raise RuntimeError(f"No sample configs found in {SAMPLE_DIR}")
    return configs


SAMPLE_CONFIGS = _collect_sample_configs()


@pytest.mark.parametrize("config_path", SAMPLE_CONFIGS, ids=lambda p: p.stem)
def test_main_api_runs_sample_config(tmp_path: Path, config_path: Path):
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["AI_SCHEDULER_ARTIFACTS_DIR"] = str(output_dir)

    result = subprocess.run(
        [
            sys.executable,
            "main_api.py",
            str(config_path),
            "--output",
            str(output_dir),
            "--no-banner",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    assert result.returncode == 0, (
        f"main_api.py failed for {config_path}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    produced_files = list(output_dir.glob("**/*"))
    assert produced_files, f"Expected artifacts for {config_path}, got none"
