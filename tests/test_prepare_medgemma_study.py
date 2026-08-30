from __future__ import annotations

import importlib.util
import sqlite3
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


def test_native_interface_is_explicit_in_inference_and_comparison_commands(
    tmp_path: Path,
) -> None:
    cohorts = [
        {
            "cohort_id": "zoe_evaluation_1395",
            "records": 1395,
            "role": "evaluation",
            "database": "inputs/zoe_evaluation_1395.db",
        }
    ]
    commands = MODULE.command_plan(
        tmp_path, cohorts, promoted_amendment(), "native_chat"
    )
    inference = commands[0]["command"]
    comparison = next(
        item["command"] for item in commands if item["stage"].endswith("compare_submitted")
    )

    assert inference[inference.index("--classification-interface") + 1] == "native_chat"
    assert "--classification-only" in inference
    assert "--local-model-only" in inference
    assert "--capture-classification-logprobs" not in inference
    assert comparison[comparison.index("--model-a-id") + 1] == (
        "medgemma-independent-native-interface-q2-v1"
    )


def test_source_population_makes_incomplete_exclusions_explicit(tmp_path: Path) -> None:
    database = tmp_path / "cohort.db"
    labels = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]
    columns = ", ".join(f'"{label}" INTEGER' for label in labels)
    with sqlite3.connect(database) as connection:
        connection.execute(
            f'CREATE TABLE reports ("Hashed_ReportURN" TEXT, {columns})'
        )
        connection.execute(
            f'INSERT INTO reports VALUES ({", ".join("?" for _ in range(6))})',
            ["complete", 1, 2, 3, 4, 1],
        )
        connection.execute(
            f'INSERT INTO reports VALUES ({", ".join("?" for _ in range(6))})',
            ["incomplete", 1, 2, 3, 4, None],
        )

    assert MODULE.source_population(database) == {
        "candidate_records": 2,
        "complete_records": 1,
        "excluded_incomplete_records": 1,
        "execute_records": 1,
    }
