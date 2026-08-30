#!/usr/bin/env python3
"""Adopt and supervise a running tiered MedGemma study without changing its science."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eeg_review.protected_execution import (
    AuthorizationValidation,
    ProtectedExecutionLocked,
    assert_governed_run_active,
    authorize_plan_before_governed_access,
)

MISSION_SCHEMA_VERSION = 1
TERMINAL_STATES = {"completed", "failed", "stopped"}
TRANSIENT_NAMES = {".tiered-run.lock", "run.lock", "heartbeat.json"}
TRANSIENT_RELATIVE_PATHS = {"receipts/mission-control/state.json"}
EXPECTED_LABELS = {
    "focal_epileptiform_activity",
    "generalized_epileptiform_activity",
    "focal_non_epileptiform_activity",
    "generalized_non_epileptiform_activity",
    "abnormality",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


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


def process_alive(pid: int | None) -> bool:
    if not pid or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def process_command(pid: int) -> str | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value or None


def supervisor_identity_matches(pid: int, run_dir: Path) -> bool:
    command = process_command(pid)
    return bool(
        command
        and "run_tiered_medgemma_study.py" in command
        and str(run_dir) in command
    )


def git_revision(repository: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"revision": revision, "worktree_dirty": dirty}


def prefix_receipt(path: Path, target: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    valid = 0
    invalid = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if rows >= target:
                break
            try:
                parsed = json.loads(row.get("classifications", ""))
                if not isinstance(parsed, dict) or set(parsed) != EXPECTED_LABELS:
                    raise ValueError("unexpected classification object")
                if any(str(value).strip() not in {"1", "2", "3", "4"} for value in parsed.values()):
                    raise ValueError("classification contains an invalid level")
            except (json.JSONDecodeError, TypeError, ValueError):
                invalid += 1
                canonical_classification: dict[str, int] | str = row.get(
                    "classifications", ""
                )
            else:
                valid += 1
                canonical_classification = {
                    key: int(str(parsed[key]).strip()) for key in sorted(parsed)
                }
            canonical = {
                "Hashed_ReportURN": row.get("Hashed_ReportURN"),
                "runtime_profile_id": row.get("runtime_profile_id"),
                "classifications": canonical_classification,
            }
            digest.update(
                (json.dumps(canonical, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
            rows += 1
    if rows != target:
        raise ValueError(f"{path.name}: expected {target} prefix rows, found {rows}")
    return {
        "prefix_records": rows,
        "prefix_sha256": digest.hexdigest(),
        "valid_structured_outputs": valid,
        "invalid_structured_outputs": invalid,
    }


def csv_record_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def classify_health(
    execution_state: str,
    supervisor_alive: bool,
    seconds_since_progress: float,
    stall_seconds: float,
) -> str:
    if execution_state in TERMINAL_STATES:
        return f"terminal_{execution_state}"
    if not supervisor_alive:
        return "orphaned_recoverable" if execution_state == "running" else "supervisor_missing"
    if execution_state == "running" and seconds_since_progress > stall_seconds:
        return "running_stalled"
    return "running_healthy"


class MissionControl:
    def __init__(
        self,
        run_dir: Path,
        compute_repo: Path,
        tier_plan: Path,
        public_status: Path,
        public_heartbeat: Path,
        poll_seconds: float,
        stall_seconds: float,
        max_orphan_resumes: int,
        python_executable: Path | None = None,
        authorization_path: Path | None = None,
        authorization: AuthorizationValidation | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.compute_repo = compute_repo
        self.tier_plan_path = tier_plan
        self.public_status_path = public_status
        self.public_heartbeat_path = public_heartbeat
        self.poll_seconds = poll_seconds
        self.stall_seconds = stall_seconds
        self.max_orphan_resumes = max_orphan_resumes
        self.python_executable = python_executable or compute_repo / ".venv/bin/python"
        self.authorization_path = authorization_path
        self.authorization = authorization
        self.receipt_dir = run_dir / "receipts/mission-control"
        self.control_state_path = self.receipt_dir / "state.json"

    def runner_command(self, action: str) -> list[str]:
        command = [
            str(self.python_executable),
            str(self.compute_repo / "scripts/run_tiered_medgemma_study.py"),
            action,
            "--run-dir",
            str(self.run_dir),
            "--tier-plan",
            str(self.tier_plan_path),
        ]
        if self.authorization_path:
            command.extend(["--authorization", str(self.authorization_path)])
        return command

    def control_state(self) -> dict[str, Any]:
        if self.control_state_path.exists():
            return read_json(self.control_state_path)
        status = read_json(self.public_status_path)
        return {
            "schema_version": MISSION_SCHEMA_VERSION,
            "adopted_at_utc": utc_now(),
            "last_completed_records": int(status["completed_records"]),
            "last_progress_at_utc": status["updated_at_utc"],
            "receipted_tiers": [],
            "orphan_resume_attempts": 0,
            "finalized": False,
        }

    def save_control_state(self, state: dict[str, Any]) -> None:
        state["updated_at_utc"] = utc_now()
        atomic_json(self.control_state_path, state)

    def refresh_public_status(self) -> None:
        command = self.runner_command("status")
        command.extend(["--public-status-output", str(self.public_status_path)])
        subprocess.run(
            command,
            cwd=self.compute_repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    def adopt(self) -> dict[str, Any]:
        state = read_json(self.run_dir / "state.json")
        supervisor_pid = int(state.get("supervisor_pid") or 0)
        if not process_alive(supervisor_pid) or not supervisor_identity_matches(
            supervisor_pid, self.run_dir
        ):
            raise RuntimeError("Cannot adopt a live matching tiered-study supervisor")
        compute_revision = git_revision(self.compute_repo)
        mission_revision = git_revision(Path(__file__).resolve().parents[1])
        receipt = {
            "schema_version": MISSION_SCHEMA_VERSION,
            "study_id": read_json(self.run_dir / "job.json")["study_id"],
            "adopted_at_utc": utc_now(),
            "supervisor": {
                "pid": supervisor_pid,
                "alive_at_adoption": True,
                "hostname": socket.gethostname(),
                "command": process_command(supervisor_pid),
                "started_at_utc": state.get("started_at_utc"),
            },
            "compute_repository": compute_revision,
            "mission_control_repository": mission_revision,
            "execution_plan_sha256": sha256_file(self.tier_plan_path),
            "authorization_receipt_sha256": (
                self.authorization.receipt_sha256 if self.authorization else None
            ),
            "policy": {
                "python_executable": str(self.python_executable),
                "poll_seconds": self.poll_seconds,
                "stall_seconds": self.stall_seconds,
                "max_orphan_resumes": self.max_orphan_resumes,
                "restart_failed_or_stopped_run": False,
                "restart_only_orphaned_running_state": True,
                "refresh_public_status_each_tick": True,
            },
        }
        initial_adoption = self.receipt_dir / "adoption.json"
        if initial_adoption.exists():
            prior_adoption = read_json(initial_adoption)
            if (
                prior_adoption["mission_control_repository"]["revision"]
                == mission_revision["revision"]
                and prior_adoption["execution_plan_sha256"]
                == receipt["execution_plan_sha256"]
            ):
                return prior_adoption
            receipt["prior_adoption_sha256"] = sha256_file(initial_adoption)
            adoption_path = self.receipt_dir / (
                f"adoption-upgrade-{mission_revision['revision'][:12]}.json"
            )
        else:
            adoption_path = initial_adoption
        atomic_json(adoption_path, receipt)
        atomic_json(
            self.run_dir / "supervisor.json",
            {
                "pid": supervisor_pid,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "repository_revision_at_start": compute_revision["revision"],
                "authorization_receipt_sha256": (
                    self.authorization.receipt_sha256 if self.authorization else None
                ),
                "started_at_utc": state.get("started_at_utc"),
                "adopted_by_mission_control": True,
            },
        )
        atomic_json(
            self.run_dir / "launcher.json",
            {
                "launcher_pid": os.getppid(),
                "hostname": socket.gethostname(),
                "command": process_command(supervisor_pid),
                "launched_at_utc": state.get("started_at_utc"),
                "adopted_at_utc": utc_now(),
                "authorization_receipt_sha256": (
                    self.authorization.receipt_sha256 if self.authorization else None
                ),
                "log": "tier-specific logs under logs/",
            },
        )
        control = self.control_state()
        self.save_control_state(control)
        return receipt

    def tier_receipts(self, control: dict[str, Any]) -> list[str]:
        state = read_json(self.run_dir / "state.json")
        plan = read_json(self.tier_plan_path)
        newly_receipted = []
        for tier in plan["tiers"]:
            tier_id = tier["tier_id"]
            if tier_id not in state.get("completed_tiers", []):
                continue
            outputs = {}
            for cohort_id, target_value in tier["targets"].items():
                target = int(target_value)
                if target == 0:
                    continue
                outputs[cohort_id] = prefix_receipt(
                    self.run_dir / f"products/{cohort_id}/raw.csv", target
                )
            receipt = {
                "schema_version": MISSION_SCHEMA_VERSION,
                "prefix_hash_schema": "stable-key-runtime-classification-v2",
                "stage": tier_id,
                "completed_at_utc": utc_now(),
                "execution_plan_sha256": sha256_file(self.tier_plan_path),
                "targets": tier["targets"],
                "prefix_outputs": outputs,
                "interpretation": (
                    "Immutable prefix hashes and operational validity only; no reference "
                    "outcomes or partial performance metrics."
                ),
            }
            receipt_path = self.run_dir / f"stages/{tier_id}.done.json"
            if receipt_path.exists():
                existing = read_json(receipt_path)
                for key in ("stage", "execution_plan_sha256", "targets"):
                    if existing.get(key) != receipt[key]:
                        raise RuntimeError(f"{tier_id}: immutable tier metadata no longer matches")
                if (
                    existing.get("prefix_hash_schema")
                    == "stable-key-runtime-classification-v2"
                    and existing.get("prefix_outputs") != receipt["prefix_outputs"]
                ):
                    raise RuntimeError(f"{tier_id}: immutable tier prefix no longer matches")
            else:
                atomic_json(receipt_path, receipt)
                newly_receipted.append(tier_id)
            if tier_id not in control["receipted_tiers"]:
                control["receipted_tiers"].append(tier_id)
        return newly_receipted

    def resume_orphan(self, control: dict[str, Any]) -> int:
        attempt = int(control["orphan_resume_attempts"]) + 1
        log_path = self.run_dir / f"logs/mission-control-resume-{attempt}.log"
        command = self.runner_command("run")
        command.extend(["--public-status-output", str(self.public_status_path)])
        if platform.system() == "Darwin" and Path("/usr/bin/caffeinate").exists():
            command = ["/usr/bin/caffeinate", "-dimsu", *command]
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=self.compute_repo,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        control["orphan_resume_attempts"] = attempt
        atomic_json(
            self.receipt_dir / f"orphan-resume-{attempt}.json",
            {
                "schema_version": MISSION_SCHEMA_VERSION,
                "created_at_utc": utc_now(),
                "launcher_pid": process.pid,
                "command": command,
                "policy_reason": "supervisor missing while governed state remained running",
                "log": str(log_path.relative_to(self.run_dir)),
            },
        )
        return process.pid

    def final_transfer_manifest(self) -> dict[str, Any]:
        destination = self.run_dir / "final-transfer-manifest.json"
        files = []
        for path in sorted(self.run_dir.rglob("*")):
            if (
                not path.is_file()
                or path == destination
                or path.name in TRANSIENT_NAMES
                or str(path.relative_to(self.run_dir)) in TRANSIENT_RELATIVE_PATHS
            ):
                continue
            relative = path.relative_to(self.run_dir)
            first = relative.parts[0]
            sensitivity = (
                "governed_case_level_or_derived_product"
                if first in {"inputs", "manifests", "comparators", "products", "analysis"}
                else "operational_metadata_or_aggregate_receipt"
            )
            files.append(
                {
                    "path": str(relative),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "sensitivity": sensitivity,
                }
            )
        payload = {
            "schema_version": MISSION_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "study_id": read_json(self.run_dir / "job.json")["study_id"],
            "authorization_receipt_sha256": (
                self.authorization.receipt_sha256 if self.authorization else None
            ),
            "compute_repository": git_revision(self.compute_repo),
            "mission_control_repository": git_revision(Path(__file__).resolve().parents[1]),
            "transfer_rule": (
                "Transfer only through an approved governed channel; preserve relative paths "
                "and verify every SHA-256 before validation or resume."
            ),
            "files": files,
        }
        atomic_json(destination, payload)
        return payload

    def maintenance_checkpoint(self, reason: str) -> dict[str, Any]:
        self.refresh_public_status()
        compute_state = read_json(self.run_dir / "state.json")
        if compute_state["status"] not in {"stopped", "completed"}:
            raise RuntimeError("Maintenance checkpoints require a stopped or completed run")
        status = read_json(self.public_status_path)
        products = []
        for raw_path in sorted((self.run_dir / "products").glob("*/raw.csv")):
            products.append(
                {
                    "path": str(raw_path.relative_to(self.run_dir)),
                    "records": csv_record_count(raw_path),
                    "size_bytes": raw_path.stat().st_size,
                    "sha256": sha256_file(raw_path),
                }
            )
        stage_receipts = []
        for stage_path in sorted((self.run_dir / "stages").glob("*.done.json")):
            stage_receipts.append(
                {
                    "path": str(stage_path.relative_to(self.run_dir)),
                    "sha256": sha256_file(stage_path),
                }
            )
        created_at = utc_now()
        payload = {
            "schema_version": MISSION_SCHEMA_VERSION,
            "checkpoint_type": "governed_live_maintenance",
            "created_at_utc": created_at,
            "reason": reason,
            "study_id": status["study_id"],
            "execution_state": compute_state["status"],
            "state_sha256": sha256_file(self.run_dir / "state.json"),
            "execution_plan_sha256": sha256_file(self.tier_plan_path),
            "compute_repository": git_revision(self.compute_repo),
            "mission_control_repository": git_revision(
                Path(__file__).resolve().parents[1]
            ),
            "completed_records": int(status["completed_records"]),
            "target_records": int(status["target_records"]),
            "completed_tiers": status["completed_tiers"],
            "current_tier": status.get("current_tier"),
            "current_stage": status.get("current_stage"),
            "products": products,
            "stage_receipts": stage_receipts,
            "resume_policy": {
                "resumable": compute_state["status"] == "stopped",
                "exact_key_resume": True,
                "automatic_restart": False,
                "rule": (
                    "Resume only after the maintenance change has its own committed revision, "
                    "runtime receipt, and result-blind validation gate."
                ),
            },
            "reporting_boundary": (
                "Operational maintenance receipt only; no report text, report keys, reference "
                "outcomes, partial performance metrics, or manuscript admission."
            ),
        }
        timestamp = created_at.replace(":", "").replace("+00:00", "Z").replace("-", "")
        destination = self.receipt_dir / f"maintenance-checkpoint-{timestamp}.json"
        atomic_json(destination, payload)
        return {**payload, "receipt_path": str(destination.relative_to(self.run_dir))}

    def tick(self) -> dict[str, Any]:
        self.refresh_public_status()
        control = self.control_state()
        compute_state = read_json(self.run_dir / "state.json")
        status = read_json(self.public_status_path)
        completed = int(status["completed_records"])
        if completed > int(control["last_completed_records"]):
            control["last_completed_records"] = completed
            control["last_progress_at_utc"] = status["updated_at_utc"]
        elapsed = (datetime.now(UTC) - parse_time(control["last_progress_at_utc"])).total_seconds()
        supervisor_pid = int(compute_state.get("supervisor_pid") or 0)
        alive = process_alive(supervisor_pid) and supervisor_identity_matches(
            supervisor_pid, self.run_dir
        )
        health = classify_health(
            compute_state["status"], alive, elapsed, self.stall_seconds
        )
        resumed = False
        if (
            health == "orphaned_recoverable"
            and int(control["orphan_resume_attempts"]) < self.max_orphan_resumes
        ):
            self.resume_orphan(control)
            resumed = True
            health = "orphan_resume_launched"
        newly_receipted = self.tier_receipts(control)
        if compute_state["status"] == "completed" and not control["finalized"]:
            manifest = self.final_transfer_manifest()
            control["finalized"] = True
            control["final_transfer_manifest_sha256"] = sha256_file(
                self.run_dir / "final-transfer-manifest.json"
            )
            control["final_transfer_file_count"] = len(manifest["files"])
        heartbeat = {
            "schema_version": MISSION_SCHEMA_VERSION,
            "updated_at_utc": utc_now(),
            "study_id": status["study_id"],
            "mission_control_revision": git_revision(
                Path(__file__).resolve().parents[1]
            )["revision"],
            "mission_control_health": health,
            "supervisor_alive": alive,
            "supervisor_pid": supervisor_pid,
            "execution_state": compute_state["status"],
            "current_tier": compute_state.get("current_tier"),
            "current_stage": compute_state.get("current_stage"),
            "completed_records": completed,
            "target_records": int(status["target_records"]),
            "last_progress_at_utc": control["last_progress_at_utc"],
            "seconds_since_progress": elapsed,
            "stall_threshold_seconds": self.stall_seconds,
            "newly_receipted_tiers": newly_receipted,
            "receipted_tiers": control["receipted_tiers"],
            "orphan_resume_attempts": control["orphan_resume_attempts"],
            "orphan_resume_launched_this_tick": resumed,
            "finalized": control["finalized"],
            "reporting_boundary": (
                "Operational supervision only; no report text, keys, reference outcomes, "
                "partial performance metrics, or manuscript admission."
            ),
        }
        atomic_json(self.receipt_dir / "heartbeat.json", heartbeat)
        public = {key: value for key, value in heartbeat.items() if key != "supervisor_pid"}
        atomic_json(self.public_heartbeat_path, public)
        self.save_control_state(control)
        return heartbeat

    def watch(self) -> None:
        while True:
            heartbeat = self.tick()
            if heartbeat["execution_state"] in TERMINAL_STATES:
                return
            time.sleep(self.poll_seconds)

    def launch_watch(self) -> dict[str, Any]:
        watcher_path = self.receipt_dir / "watcher.json"
        if watcher_path.exists():
            prior = read_json(watcher_path)
            if process_alive(int(prior.get("pid") or 0)):
                raise RuntimeError(f"Mission-control watcher {prior['pid']} is already running")
        command = [
            str(self.python_executable),
            str(Path(__file__).resolve()),
            "watch",
            "--run-dir",
            str(self.run_dir),
            "--compute-repo",
            str(self.compute_repo),
            "--tier-plan",
            str(self.tier_plan_path),
            "--public-status",
            str(self.public_status_path),
            "--public-heartbeat",
            str(self.public_heartbeat_path),
            "--python-executable",
            str(self.python_executable),
            "--poll-seconds",
            str(self.poll_seconds),
            "--stall-seconds",
            str(self.stall_seconds),
            "--max-orphan-resumes",
            str(self.max_orphan_resumes),
        ]
        if self.authorization_path:
            command.extend(["--authorization", str(self.authorization_path)])
        log_path = self.run_dir / "logs/mission-control-watch.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        receipt = {
            "schema_version": MISSION_SCHEMA_VERSION,
            "pid": process.pid,
            "created_at_utc": utc_now(),
            "command": command,
            "log": str(log_path.relative_to(self.run_dir)),
        }
        atomic_json(watcher_path, receipt)
        time.sleep(1)
        if process.poll() is not None:
            raise RuntimeError(f"Mission-control watcher exited; inspect {log_path}")
        return receipt


def controller_from_args(args: argparse.Namespace) -> MissionControl:
    # Read only the public control plan and documentary receipt before resolving any
    # governed run or output path.
    tier_plan = args.tier_plan.expanduser().resolve(strict=True)
    plan = read_json(tier_plan)
    authorization_path = (
        args.authorization.expanduser().resolve(strict=True) if args.authorization else None
    )
    authorization = authorize_plan_before_governed_access(plan, authorization_path)
    python_executable = (
        args.python_executable.expanduser().absolute()
        if args.python_executable
        else None
    )
    if python_executable is not None and not python_executable.exists():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    assert_governed_run_active(run_dir)
    return MissionControl(
        run_dir=run_dir,
        compute_repo=args.compute_repo.expanduser().resolve(strict=True),
        tier_plan=tier_plan,
        public_status=args.public_status.expanduser().resolve(strict=True),
        public_heartbeat=args.public_heartbeat.expanduser().resolve(),
        poll_seconds=args.poll_seconds,
        stall_seconds=args.stall_seconds,
        max_orphan_resumes=args.max_orphan_resumes,
        python_executable=python_executable,
        authorization_path=authorization_path,
        authorization=authorization,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["adopt", "status", "watch", "launch-watch", "checkpoint"]
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compute-repo", type=Path, required=True)
    parser.add_argument("--tier-plan", type=Path, required=True)
    parser.add_argument("--public-status", type=Path, required=True)
    parser.add_argument("--public-heartbeat", type=Path, required=True)
    parser.add_argument(
        "--authorization",
        type=Path,
        help=(
            "Documentary authorization receipt required by protected tier plans. It is "
            "validated before any governed run path is resolved or opened."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--stall-seconds", type=float, default=300.0)
    parser.add_argument("--max-orphan-resumes", type=int, default=1)
    parser.add_argument(
        "--python-executable",
        type=Path,
        help="Pinned Python environment used for status refresh and orphan-only recovery.",
    )
    parser.add_argument(
        "--reason",
        default="bounded live maintenance",
        help="Reason recorded by the governed maintenance checkpoint command.",
    )
    args = parser.parse_args()
    try:
        controller = controller_from_args(args)
    except ProtectedExecutionLocked as error:
        print(
            json.dumps(
                {
                    "mission_control_started": False,
                    "protected_evaluation_unlocked": False,
                    "blockers": list(error.blockers),
                },
                indent=2,
            )
        )
        raise SystemExit(2) from error
    if args.command == "adopt":
        print(json.dumps(controller.adopt(), indent=2))
    elif args.command == "status":
        print(json.dumps(controller.tick(), indent=2))
    elif args.command == "checkpoint":
        print(json.dumps(controller.maintenance_checkpoint(args.reason), indent=2))
    elif args.command == "launch-watch":
        print(json.dumps(controller.launch_watch(), indent=2))
    else:
        controller.watch()


if __name__ == "__main__":
    main()
