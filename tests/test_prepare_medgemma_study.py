from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_medgemma_study.py"
SPEC = importlib.util.spec_from_file_location("prepare_medgemma_study", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def promoted_amendment() -> dict:
    return {
        "status": "promoted_after_result_blind_benchmark",
        "promoted_runtime_profile": {
            "runtime_profile_id": "metal-profile",
            "parameters": {
                "n_ctx": 4096,
                "n_gpu_layers": -1,
                "n_batch": 1024,
                "n_ubatch": 512,
                "n_threads": 4,
                "n_threads_batch": 10,
                "flash_attn": True,
            },
        },
    }


def test_runtime_amendment_is_injected_into_every_inference_command(tmp_path: Path) -> None:
    cohorts = [
        {
            "cohort_id": "development",
            "records": 12,
            "role": "development",
            "database": "inputs/development.db",
        }
    ]
    commands = MODULE.command_plan(tmp_path, cohorts, promoted_amendment())
    inference = commands[0]["command"]

    assert inference[inference.index("--runtime-profile-id") + 1] == "metal-profile"
    assert inference[inference.index("--n-gpu-layers") + 1] == "-1"
    assert "--flash-attn" in inference


def test_unpromoted_runtime_amendment_is_rejected() -> None:
    amendment = promoted_amendment()
    amendment["status"] = "candidate"

    with pytest.raises(ValueError, match="has not passed"):
        MODULE.runtime_arguments(amendment)
