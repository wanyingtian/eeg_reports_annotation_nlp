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


def test_public_status_refresh_is_owned_by_mission_control(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    controller = MODULE.MissionControl(
        run_dir=tmp_path / "run",
        compute_repo=tmp_path / "repo",
        tier_plan=tmp_path / "plan.json",
        public_status=tmp_path / "progress.json",
        public_heartbeat=tmp_path / "heartbeat.json",
        poll_seconds=15,
        stall_seconds=300,
        max_orphan_resumes=1,
    )
    controller.refresh_public_status()
    assert "status" in captured["command"]
    assert "--tier-plan" in captured["command"]
    assert captured["kwargs"]["check"] is True


def test_maintenance_checkpoint_hashes_atomic_products(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    product = run_dir / "products/cohort/raw.csv"
    product.parent.mkdir(parents=True)
    product.write_text("Hashed_ReportURN,classifications\ncase-1,{}\n", encoding="utf-8")
    stage = run_dir / "stages/T0.done.json"
    stage.parent.mkdir(parents=True)
    stage.write_text("{}\n", encoding="utf-8")
    (run_dir / "state.json").write_text(
        json.dumps({"status": "stopped"}) + "\n", encoding="utf-8"
    )
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    public_status = tmp_path / "progress.json"
    public_status.write_text(
        json.dumps(
            {
                "study_id": "study-1",
                "completed_records": 1,
                "target_records": 10,
                "completed_tiers": ["T0"],
                "current_tier": "T1",
                "current_stage": "cohort__to_10",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE.MissionControl, "refresh_public_status", lambda _self: None)
    monkeypatch.setattr(
        MODULE,
        "git_revision",
        lambda path: {"revision": path.name, "worktree_dirty": False},
    )
    controller = MODULE.MissionControl(
        run_dir=run_dir,
        compute_repo=tmp_path / "compute",
        tier_plan=plan,
        public_status=public_status,
        public_heartbeat=tmp_path / "heartbeat.json",
        poll_seconds=15,
        stall_seconds=300,
        max_orphan_resumes=1,
    )

    receipt = controller.maintenance_checkpoint("test maintenance")

    assert receipt["execution_state"] == "stopped"
    assert receipt["products"][0]["records"] == 1
    assert receipt["products"][0]["sha256"] == MODULE.sha256_file(product)
    assert receipt["resume_policy"]["automatic_restart"] is False
    assert (run_dir / receipt["receipt_path"]).exists()
