from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/medgemma_mission_control.py"
SPEC = importlib.util.spec_from_file_location("medgemma_mission_control", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_prefix_receipt_is_stable_when_later_rows_are_appended(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    fieldnames = [
        "Hashed_ReportURN",
        "classifications",
        "classify_elapsed_seconds",
        "classify_prompt_tokens",
        "classify_completion_tokens",
    ]
    rows = [
        {
            "Hashed_ReportURN": f"key-{index}",
            "classifications": json.dumps(
                {label: 1 for label in MODULE.EXPECTED_LABELS}
            ),
            "classify_elapsed_seconds": "2.0",
            "classify_prompt_tokens": "100",
            "classify_completion_tokens": "20",
        }
        for index in range(3)
    ]

    def write(count: int) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows[:count])

    write(2)
    before = MODULE.prefix_receipt(path, 2)
    write(3)
    after = MODULE.prefix_receipt(path, 2)
    assert before == after
    assert before["prefix_records"] == 2
    assert before["valid_structured_outputs"] == 2


def test_health_only_recovers_an_orphaned_running_state() -> None:
    assert MODULE.classify_health("running", False, 10, 300) == "orphaned_recoverable"
    assert MODULE.classify_health("failed", False, 10, 300) == "terminal_failed"
    assert MODULE.classify_health("stopped", False, 10, 300) == "terminal_stopped"
    assert MODULE.classify_health("running", True, 301, 300) == "running_stalled"
    assert MODULE.classify_health("running", True, 10, 300) == "running_healthy"


def test_supervisor_identity_is_bound_to_runner_and_run_directory(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        MODULE,
        "process_command",
        lambda _pid: f"python scripts/run_tiered_medgemma_study.py run --run-dir {tmp_path}",
    )
    assert MODULE.supervisor_identity_matches(123, tmp_path)
    assert not MODULE.supervisor_identity_matches(123, tmp_path / "other")
