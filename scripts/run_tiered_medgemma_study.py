#!/usr/bin/env python3
"""Run the frozen MedGemma comparator in result-blind, resumable tiers."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
from collections import Counter
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_LABELS = [
    "focal_epileptiform_activity",
    "generalized_epileptiform_activity",
    "focal_non_epileptiform_activity",
    "generalized_non_epileptiform_activity",
    "abnormality",
]
TERMINAL_STATES = {"completed", "failed", "stopped"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    path.chmod(0o600)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_revision() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"revision": revision, "worktree_dirty": dirty}


def replace_flag(command: list[str], flag: str, value: str) -> list[str]:
    output = list(command)
    index = output.index(flag)
    output[index + 1] = value
    return output


def normalize_command(command: list[str]) -> list[str]:
    output = list(command)
    if output and output[0] == "python":
        output[0] = sys.executable
    return output


def validate_plan(plan: dict[str, Any], job: dict[str, Any], study_plan: Path) -> None:
    if plan.get("status") != "preregistered_before_inference":
        raise ValueError("Tier plan must be preregistered before inference")
    if plan.get("study_id") != job.get("study_id"):
        raise ValueError("Tier plan and prepared job study IDs differ")
    if plan.get("configuration_id") != job.get("configuration_id"):
        raise ValueError("Tier plan and prepared job configuration IDs differ")
    if plan.get("source_study_plan_sha256") != sha256_file(study_plan):
        raise ValueError("Frozen study-plan hash does not match tier plan")
    cohorts = {item["cohort_id"]: int(item["records"]) for item in job["cohorts"]}
    previous = {cohort: 0 for cohort in cohorts}
    for tier in plan["tiers"]:
        targets = tier["targets"]
        if set(targets) != set(cohorts):
            raise ValueError(f"{tier['tier_id']}: target cohorts do not match prepared job")
        for cohort, value in targets.items():
            value = int(value)
            if value < previous[cohort] or value > cohorts[cohort]:
                raise ValueError(f"{tier['tier_id']}: non-monotonic or excessive {cohort} target")
            previous[cohort] = value
    if previous != cohorts:
        raise ValueError("Final tier does not cover every prepared report")
    if plan["post_inference"].get("partial_reference_metrics_allowed") is not False:
        raise ValueError("Partial reference metrics must remain prohibited")
    planned_amendment = plan.get("runtime_amendment")
    prepared_amendment = job.get("runtime_amendment")
    if planned_amendment != prepared_amendment:
        raise ValueError("Tier plan and prepared job runtime amendments differ")


def validate_run_inputs(run_dir: Path, job: dict[str, Any]) -> None:
    for cohort in job["cohorts"]:
        for key, hash_key in [
            ("database", "database_sha256"),
            ("manifest", "manifest_sha256"),
        ]:
            path = run_dir / cohort[key]
            if sha256_file(path) != cohort[hash_key]:
                raise ValueError(f"Prepared {key} hash mismatch for {cohort['cohort_id']}")


def classification_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in {1, 2, 3, 4}:
        return value
    if isinstance(value, str) and value.strip() in {"1", "2", "3", "4"}:
        return int(value.strip())
    return None


def inspect_output(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "completed_records": 0,
        "valid_structured_outputs": 0,
        "invalid_structured_outputs": 0,
        "duplicate_report_keys": 0,
        "full_pattern_cardinality": 0,
        "predicted_level_counts": {label: {} for label in JSON_LABELS},
        "classification_seconds_total": 0.0,
        "classification_seconds_mean": None,
        "prompt_tokens_total": 0,
        "completion_tokens_total": 0,
    }
    if not path.exists():
        return summary
    keys: list[str] = []
    patterns: Counter[tuple[int, ...]] = Counter()
    levels = {label: Counter() for label in JSON_LABELS}
    timing: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            summary["completed_records"] += 1
            keys.append(str(row.get("Hashed_ReportURN", "")))
            try:
                parsed = json.loads(row.get("classifications", ""))
                pattern = tuple(classification_value(parsed.get(label)) for label in JSON_LABELS)
                if any(value is None for value in pattern):
                    raise ValueError("classification contains a non-level value")
            except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
                summary["invalid_structured_outputs"] += 1
            else:
                typed_pattern = tuple(int(value) for value in pattern if value is not None)
                patterns[typed_pattern] += 1
                summary["valid_structured_outputs"] += 1
                for label, value in zip(JSON_LABELS, typed_pattern, strict=True):
                    levels[label][str(value)] += 1
            with suppress(KeyError, TypeError, ValueError):
                timing.append(float(row["classify_elapsed_seconds"]))
            for source, destination in [
                ("classify_prompt_tokens", "prompt_tokens_total"),
                ("classify_completion_tokens", "completion_tokens_total"),
            ]:
                with suppress(KeyError, TypeError, ValueError):
                    summary[destination] += int(float(row[source]))
    summary["duplicate_report_keys"] = len(keys) - len(set(keys))
    summary["full_pattern_cardinality"] = len(patterns)
    summary["predicted_level_counts"] = {
        label: dict(sorted(counter.items())) for label, counter in levels.items()
    }
    if timing:
        summary["classification_seconds_total"] = sum(timing)
        summary["classification_seconds_mean"] = sum(timing) / len(timing)
    return summary


def current_status(
    run_dir: Path,
    plan_path: Path,
    plan: dict[str, Any],
    job: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    cohorts: dict[str, Any] = {}
    completed_total = 0
    target_total = 0
    inference_seconds = 0.0
    observed_records = 0
    for cohort in job["cohorts"]:
        cohort_id = cohort["cohort_id"]
        output = inspect_output(run_dir / f"products/{cohort_id}/raw.csv")
        completed = output["completed_records"]
        target = int(cohort["records"])
        completed_total += completed
        target_total += target
        inference_seconds += float(output["classification_seconds_total"])
        observed_records += completed
        cohorts[cohort_id] = {
            "role": cohort["role"],
            "completed_records": completed,
            "target_records": target,
            "remaining_records": max(target - completed, 0),
            "coverage_fraction": completed / target if target else 1.0,
            **{key: value for key, value in output.items() if key != "completed_records"},
        }
    planning_rate = float(plan["planning_seconds_per_report"])
    observed_rate = inference_seconds / observed_records if observed_records else None
    eta_rate = observed_rate or planning_rate
    remaining = max(target_total - completed_total, 0)
    early_targets = plan["early_view"]["minimum_completed_by_cohort"]
    early_view_ready = all(
        cohorts[cohort]["completed_records"] >= int(target)
        for cohort, target in early_targets.items()
    )
    execution_state = state.get("status", "prepared_no_inference")
    if execution_state == "completed":
        validation_state = "validated_complete"
        result_state = "completed_governed"
    elif completed_total:
        validation_state = "operational_partial"
        result_state = "no_partial_performance_result"
    else:
        validation_state = "preregistered"
        result_state = "no_result"
    return {
        "schema_version": 1,
        "updated_at_utc": utc_now(),
        "study_id": job["study_id"],
        "configuration_id": job["configuration_id"],
        "execution_plan_id": plan["execution_plan_id"],
        "execution_plan_sha256": sha256_file(plan_path),
        "state_axes": {
            "execution": execution_state,
            "validation": validation_state,
            "result": result_state,
            "manuscript_admission": "proposed_not_admitted",
        },
        "current_tier": state.get("current_tier"),
        "current_stage": state.get("current_stage"),
        "completed_tiers": state.get("completed_tiers", []),
        "early_cross_cohort_view_ready": early_view_ready,
        "completed_records": completed_total,
        "target_records": target_total,
        "remaining_records": remaining,
        "coverage_fraction": completed_total / target_total,
        "observed_seconds_per_report": observed_rate,
        "planning_seconds_per_report": planning_rate,
        "estimated_remaining_seconds": eta_rate * remaining,
        "cohorts": cohorts,
        "reporting_boundary": {
            "reference_outcomes_accessed_for_progress": False,
            "partial_performance_metrics_computed": False,
            "performance_claim_available": execution_state == "completed",
            "manuscript_claim_available": False,
            "permitted_interim_use": (
                "coverage, validity, degeneracy, timing, ETA, and methods drafting only"
            ),
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "runtime_profile_id": (
                job.get("runtime_amendment") or {}
            ).get("runtime_profile_id", "llama-cpp-python-default"),
        },
        "repository": git_revision(),
        "privacy_boundary": (
            "Aggregate operational status only; no report text, report or patient keys, "
            "reference labels, keyed predictions, or partial performance metrics."
        ),
    }


def write_status(
    run_dir: Path,
    plan_path: Path,
    plan: dict[str, Any],
    job: dict[str, Any],
    state: dict[str, Any],
    public_output: Path | None,
) -> dict[str, Any]:
    payload = current_status(run_dir, plan_path, plan, job, state)
    atomic_json(run_dir / "receipts/progress/current.json", payload)
    if public_output:
        atomic_json(public_output, payload)
    return payload


def read_state(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "state.json"
    state = read_json(state_path)
    state.setdefault("completed_tiers", [])
    state.setdefault("completed_post_inference_stages", [])
    state.setdefault("current_tier", None)
    state.setdefault("current_stage", None)
    return state


def write_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_json(run_dir / "state.json", state)


def run_command(run_dir: Path, stage: str, command: list[str]) -> None:
    log_path = run_dir / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] {' '.join(command)}\n")
        log.flush()
        subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def inference_command(job: dict[str, Any], cohort_id: str, target: int) -> list[str]:
    stage = f"{cohort_id}_inference"
    command = next(item["command"] for item in job["commands"] if item["stage"] == stage)
    return replace_flag(normalize_command(command), "--num-reports", str(target))


def validate_transport_gate(status: dict[str, Any], plan: dict[str, Any]) -> None:
    gate = plan["transport_gate"]
    cohort = status["cohorts"][gate["cohort_id"]]
    if cohort["completed_records"] != int(gate["target_records"]):
        raise RuntimeError("Transport cohort did not reach its exact target")
    if gate["stop_on_any_invalid_structured_output"] and cohort[
        "invalid_structured_outputs"
    ]:
        raise RuntimeError("Transport cohort contains invalid structured output")
    if cohort["duplicate_report_keys"]:
        raise RuntimeError("Transport cohort contains duplicate report keys")
    if gate["stop_on_single_full_classification_pattern"] and cohort[
        "full_pattern_cardinality"
    ] <= 1:
        raise RuntimeError("Transport cohort produced a single classification pattern")


def run_post_inference(
    run_dir: Path,
    job: dict[str, Any],
    state: dict[str, Any],
    status_callback: Any,
) -> None:
    completed = state["completed_post_inference_stages"]
    for item in job["commands"]:
        stage = item["stage"]
        if stage.endswith("_inference") or stage in completed:
            continue
        state["current_stage"] = stage
        write_state(run_dir, state)
        status_callback()
        run_command(run_dir, stage, normalize_command(item["command"]))
        completed.append(stage)
        write_state(run_dir, state)
        status_callback()


def run_study(
    run_dir: Path,
    plan_path: Path,
    plan: dict[str, Any],
    job: dict[str, Any],
    public_output: Path | None,
    stop_after_tier: str | None,
) -> None:
    lock_path = run_dir / ".tiered-run.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_state(run_dir)
        if state.get("status") == "completed":
            write_status(run_dir, plan_path, plan, job, state, public_output)
            return
        state.update(
            {
                "schema_version": 2,
                "status": "running",
                "execution_plan_id": plan["execution_plan_id"],
                "execution_plan_sha256": sha256_file(plan_path),
                "started_at_utc": state.get("started_at_utc") or utc_now(),
                "supervisor_pid": os.getpid(),
            }
        )

        def checkpoint() -> dict[str, Any]:
            write_state(run_dir, state)
            return write_status(run_dir, plan_path, plan, job, state, public_output)

        def stop_handler(signum: int, _frame: Any) -> None:
            state["status"] = "stopped"
            state["stop_signal"] = signum
            checkpoint()
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)

        try:
            checkpoint()
            for tier in plan["tiers"]:
                tier_id = tier["tier_id"]
                if tier_id in state["completed_tiers"]:
                    continue
                state["current_tier"] = tier_id
                for cohort in job["cohorts"]:
                    cohort_id = cohort["cohort_id"]
                    target = int(tier["targets"][cohort_id])
                    current = inspect_output(
                        run_dir / f"products/{cohort_id}/raw.csv"
                    )["completed_records"]
                    if current >= target:
                        continue
                    stage = f"{tier_id}__{cohort_id}__to_{target}"
                    state["current_stage"] = stage
                    checkpoint()
                    run_command(run_dir, stage, inference_command(job, cohort_id, target))
                    checkpoint()
                if tier_id == plan["tiers"][0]["tier_id"]:
                    validate_transport_gate(checkpoint(), plan)
                state["completed_tiers"].append(tier_id)
                state["current_stage"] = None
                checkpoint()
                if stop_after_tier == tier_id:
                    state["status"] = "stopped"
                    state["stop_reason"] = "requested_stop_after_tier"
                    checkpoint()
                    return
            run_post_inference(run_dir, job, state, checkpoint)
            state["status"] = "completed"
            state["current_tier"] = None
            state["current_stage"] = None
            state["completed_at_utc"] = utc_now()
            checkpoint()
        except BaseException as error:
            if state.get("status") not in TERMINAL_STATES:
                state["status"] = "failed"
                state["failure_type"] = type(error).__name__
                state["failure_message"] = str(error)
                checkpoint()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["dry-run", "run", "status"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--tier-plan",
        type=Path,
        default=REPO_ROOT
        / "review/model-receipts/medgemma-independent-tiered-execution.preregistered.json",
    )
    parser.add_argument("--public-status-output", type=Path)
    parser.add_argument("--stop-after-tier")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve(strict=True)
    plan_path = args.tier_plan.expanduser().resolve(strict=True)
    study_plan = run_dir / "study-plan.json"
    job = read_json(run_dir / "job.json")
    plan = read_json(plan_path)
    validate_plan(plan, job, study_plan)
    validate_run_inputs(run_dir, job)
    state = read_state(run_dir)
    public_output = (
        args.public_status_output.expanduser().resolve()
        if args.public_status_output
        else None
    )
    status = write_status(run_dir, plan_path, plan, job, state, public_output)
    if args.command == "dry-run":
        if git_revision()["worktree_dirty"]:
            raise RuntimeError("Commit the tier runner and plan before governed execution")
        print(json.dumps({"valid": True, "status": status}, indent=2))
    elif args.command == "status":
        print(json.dumps(status, indent=2))
    else:
        if git_revision()["worktree_dirty"]:
            raise RuntimeError("Governed execution requires a clean repository revision")
        run_study(
            run_dir,
            plan_path,
            plan,
            job,
            public_output,
            args.stop_after_tier,
        )


if __name__ == "__main__":
    main()
