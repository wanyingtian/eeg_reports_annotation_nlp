#!/usr/bin/env python3
"""Dependency-queued, capped Mistral interface follow-up using the study supervisor."""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import json
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_job import (  # noqa: E402
    REPO_ROOT,
    Stage,
    Supervisor,
    atomic_csv,
    atomic_json,
    git_revision,
    sha256_file,
    utc_now,
    write_transfer_manifest,
)

from eeg_review.compare import compare_predictions  # noqa: E402
from eeg_review.evidence_extraction import (  # noqa: E402
    aggregate_inspections,
    classification_levels,
    load_fixed_evidence_inputs,
)
from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL  # noqa: E402
from eeg_review.metrics import evaluate_predictions  # noqa: E402
from eeg_review.protected_execution import assert_governed_run_active  # noqa: E402

PLAN = REPO_ROOT / "review/model-receipts/mistral-native-interface-sensitivity.preregistered.json"
POLICY = REPO_ROOT / "review/model-receipts/mistral-native-interface-small-followup.execution.json"
STUDY_ID = "jbhi-02463-mistral-native-interface-small-followup-v1"
DEPENDENCY_ID = "jbhi-02463-post-submission-medgemma-native-interface-sensitivity-v1"
SCRIPT = Path(__file__).resolve()
LAUNCH_LABEL = "ca.sbergner.eeg.mistral-native-small-followup"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ["llama-cpp-python", "huggingface-hub", "pandas"]
    }


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("classification_records") != 100 or policy.get("evidence_records") != 20:
        raise ValueError("follow-up is capped at 100 classification and 20 evidence cases")
    if policy.get("evidence_positions") != [0, 20]:
        raise ValueError("evidence sample must be the prespecified first 20 development cases")
    if policy.get("max_new_model_calls") != 140:
        raise ValueError("only 100 classification plus two 20-case evidence calls are planned")
    if policy.get("protected_evaluation_allowed") is not False:
        raise ValueError("this small follow-up cannot run protected evaluation")
    if policy.get("automatic_expansion_allowed") is not False:
        raise ValueError("automatic scope expansion is forbidden")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    plan, policy = read_json(PLAN), read_json(POLICY)
    validate_policy(policy)
    root = args.run_dir.expanduser().resolve()
    if root.exists():
        raise FileExistsError("refusing to overwrite an existing follow-up directory")
    source = args.source_run.expanduser().resolve(strict=True)
    manifest = args.manifest.expanduser().resolve(strict=True)
    sources = {
        "development.db": (
            source / "inputs/zoe_development_100.db",
            plan["development_surface"]["source_database_sha256"],
        ),
        "development.manifest.csv": (manifest, plan["development_surface"]["manifest_sha256"]),
        "raw-parent.run.json": (
            source / "products/llm/zoe/raw.run.json",
            plan["paired_raw_comparator"]["full_run_receipt_sha256"],
        ),
    }
    raw = source / "products/llm/zoe/raw.csv"
    if sha256_file(raw) != plan["paired_raw_comparator"]["full_output_sha256"]:
        raise ValueError("completed Mistral raw parent changed")
    for name, (path, expected) in sources.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input mismatch: {name}")
    parent_runtime = read_json(sources["raw-parent.run.json"][0])["environment"]["packages"]
    if runtime_versions() != parent_runtime:
        raise ValueError("runtime packages differ from the completed raw Mistral parent")
    fixed = load_fixed_evidence_inputs(
        dataset=sources["development.db"][0],
        predictions=raw,
        manifest=manifest,
    )
    if len(fixed) != 100:
        raise ValueError("development manifest must contain exactly 100 cases")
    dependency = args.dependency_run.expanduser().resolve(strict=True)
    dependency_job = read_json(dependency / "job.json")
    if dependency_job["study_id"] != DEPENDENCY_ID:
        raise ValueError("wrong dependency study")
    os.umask(0o077)
    for name in ["inputs", "products", "analysis", "receipts", "logs", "stages"]:
        (root / name).mkdir(parents=True, mode=0o700)
    for name, (path, _expected) in sources.items():
        shutil.copyfile(path, root / "inputs" / name)
    shutil.copyfile(PLAN, root / "inputs/scientific-plan.json")
    shutil.copyfile(POLICY, root / "inputs/small-workload.json")
    telemetry_columns = [
        "classify_elapsed_seconds",
        "classify_prompt_tokens",
        "classify_completion_tokens",
    ]
    raw_source = pd.read_csv(
        raw, usecols=["Hashed_ReportURN", "classifications", *telemetry_columns]
    )
    raw_subset = (
        raw_source.set_index("Hashed_ReportURN")
        .loc[fixed["Hashed_ReportURN"].tolist()]
        .reset_index()
    )
    atomic_csv(root / "inputs/raw-development.csv", raw_subset)
    atomic_csv(root / "inputs/evidence.manifest.csv", fixed[["Hashed_ReportURN"]].iloc[:20])
    job = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "repository_revision": git_revision(),
        "repository": str(REPO_ROOT),
        "python_executable": sys.executable,
        "runtime_versions": runtime_versions(),
        "created_at_utc": utc_now(),
        "parent_plan_sha256": sha256_file(PLAN),
        "policy_sha256": sha256_file(POLICY),
        "inputs": [
            {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            for path in sorted((root / "inputs").iterdir())
        ],
        "dependency": {
            "run_dir": str(dependency),
            "study_id": DEPENDENCY_ID,
            "job_sha256": sha256_file(dependency / "job.json"),
            "configuration_id": dependency_job["configuration_id"],
            "public_status": str(args.dependency_status.expanduser().resolve()),
            "public_heartbeat": str(args.dependency_heartbeat.expanduser().resolve()),
        },
        "public_status": str(args.public_status.expanduser().resolve()),
        "new_model_call_cap": 140,
        "protected_evaluation": False,
        "scope": "100 development classifications and raw/native evidence on first 20 cases",
        "authorization_basis": "Steven requested this bounded local development follow-up in chat",
        "distribution": "all keyed products governed; aggregate summary for author review only",
    }
    atomic_json(root / "job.json", job)
    atomic_json(
        root / "state.json",
        {
            "status": "queued",
            "updated_at_utc": utc_now(),
            "stages": {},
            "current_stage": None,
        },
    )
    return publish_status(root)


def validate_job(root: Path) -> dict[str, Any]:
    job = read_json(root / "job.json")
    if job["study_id"] != STUDY_ID or job.get("new_model_call_cap") != 140:
        raise ValueError("unexpected follow-up identity or scope")
    if job.get("protected_evaluation") is not False:
        raise ValueError("protected cohort execution is not part of this job")
    if git_revision() != job["repository_revision"]:
        raise ValueError("compute revision changed; explicit migration is required")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("compute worktree must remain clean")
    if runtime_versions() != job["runtime_versions"]:
        raise ValueError("runtime packages changed; explicit migration is required")
    for item in job["inputs"]:
        if sha256_file(root / item["path"]) != item["sha256"]:
            raise ValueError(f"input hash changed: {item['path']}")
    validate_policy(read_json(root / "inputs/small-workload.json"))
    if sha256_file(PLAN) != job["parent_plan_sha256"]:
        raise ValueError("scientific plan changed")
    if sha256_file(POLICY) != job["policy_sha256"]:
        raise ValueError("execution policy changed")
    assert_governed_run_active(root)
    return job


def dependency_readiness(dependency: dict[str, Any]) -> tuple[bool, str]:
    root = Path(dependency["run_dir"])
    assert_governed_run_active(root)
    if sha256_file(root / "job.json") != dependency["job_sha256"]:
        raise ValueError("dependency job identity changed")
    status = read_json(Path(dependency["public_status"]))
    heartbeat = read_json(Path(dependency["public_heartbeat"]))
    if (
        status.get("study_id") != dependency["study_id"]
        or status.get("configuration_id") != dependency["configuration_id"]
    ):
        raise ValueError("dependency status identity mismatch")
    state = read_json(root / "state.json")
    if state["status"] != "completed":
        if state["status"] == "failed":
            raise ValueError("MedGemma failed; do not dispatch the follow-up")
        return False, f"waiting for MedGemma ({state['status']})"
    if status["completed_records"] != 1894 or status["target_records"] != 1894:
        return False, "waiting for complete dependency coverage receipt"
    if not heartbeat.get("finalized"):
        return False, "waiting for MedGemma final transfer manifest"
    if heartbeat.get("study_id") != dependency["study_id"]:
        raise ValueError("dependency heartbeat identity mismatch")
    with (root / ".tiered-run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False, "waiting for MedGemma compute lock release"
    return True, "MedGemma completed; compute lock released"


def stages(root: Path) -> list[Stage]:
    job = read_json(root / "job.json")
    inputs = root / "inputs"
    native = root / "products/native-classification.csv"
    python = job["python_executable"]
    internal = [python, str(SCRIPT), "stage", "--run-dir", str(root), "--name"]
    pipeline = [
        python,
        str(REPO_ROOT / "src/LLM_pipeline/pipeline.py"),
        "--num-reports",
        "100",
        "--dataset-path",
        str(inputs / "development.db"),
        "--dataset-id",
        "mistral-native-development-100",
        "--model",
        "mistral",
        "--classification-only",
        "--classification-interface",
        "native_chat",
        "--local-model-only",
        "--output-csv",
        str(native),
        "--resume-output",
        "--outdir",
        str(root / "products"),
        "--flush-every",
        "1",
        "--n-ctx",
        "4096",
        "--n-gpu-layers",
        "30",
        "--max-tokens",
        "3000",
        "--comment",
        "Bounded post-submission development interface sensitivity; no evaluation",
    ]
    output = [
        Stage(
            "native_classification_100",
            pipeline,
            (native, native.with_suffix(".run.json")),
            native,
            100,
        ),
        Stage(
            "structural_freeze",
            [*internal, "structural_freeze"],
            (root / "receipts/structural-freeze.json",),
        ),
    ]
    plan = read_json(inputs / "scientific-plan.json")
    for interface in ["raw_completion", "native_chat"]:
        evidence = root / f"products/evidence-{interface}.csv"
        command = [
            python,
            str(REPO_ROOT / "scripts/run_fixed_classification_explanations.py"),
            "--run-id",
            f"{STUDY_ID}/evidence/{interface}",
            "--dataset",
            str(inputs / "development.db"),
            "--predictions",
            str(inputs / "raw-development.csv"),
            "--manifest",
            str(inputs / "evidence.manifest.csv"),
            "--output-csv",
            str(evidence),
            "--model",
            "mistral",
            "--interface",
            interface,
            "--expected-dataset-sha256",
            sha256_file(inputs / "development.db"),
            "--expected-predictions-sha256",
            sha256_file(inputs / "raw-development.csv"),
            "--expected-manifest-sha256",
            sha256_file(inputs / "evidence.manifest.csv"),
            "--expected-records",
            "20",
            "--flush-every",
            "1",
            "--resume",
        ]
        if interface == "native_chat":
            command.extend(
                [
                    "--expected-chat-template-sha256",
                    plan["classification_interface"]["chat_template_sha256"],
                ]
            )
        output.append(
            Stage(
                f"evidence_{interface}_20",
                command,
                (
                    evidence,
                    evidence.with_suffix(".run.json"),
                    evidence.with_suffix(".execution.json"),
                ),
                evidence,
                20,
            )
        )
    output.append(
        Stage(
            "analysis_and_author_summary",
            [*internal, "analysis_and_author_summary"],
            (root / "analysis/author-summary.json", root / "analysis/author-summary.md"),
        )
    )
    return output


def freeze_classification(root: Path) -> dict[str, Any]:
    job = read_json(root / "job.json")
    plan = read_json(root / "inputs/scientific-plan.json")
    raw = root / "products/native-classification.csv"
    receipt_path = raw.with_suffix(".run.json")
    receipt = read_json(receipt_path)
    frame = pd.read_csv(raw)
    manifest = pd.read_csv(root / "inputs/development.manifest.csv")
    issues = []
    if (
        len(frame) != 100
        or frame["Hashed_ReportURN"].tolist() != manifest["Hashed_ReportURN"].tolist()
    ):
        issues.append("native output must match all 100 frozen keys in order")
    patterns = set()
    invalid = 0
    for raw_json in frame["classifications"]:
        try:
            patterns.add(tuple(classification_levels(raw_json).values()))
        except (ValueError, TypeError):
            invalid += 1
    if invalid or len(patterns) < 2:
        issues.append("invalid or constant structured outputs")
    interface = plan["classification_interface"]
    observed = {
        "model": receipt["model"]["sha256"],
        "prompt": receipt["prompts"]["classify"]["sha256"],
        "grammar": receipt["grammars"]["classify"]["sha256"],
        "template": receipt["input_policy"]["embedded_chat_template"]["sha256"],
        "task": receipt["input_policy"]["task_message_template"]["sha256"],
    }
    expected = {
        "model": plan["model"]["sha256"],
        "prompt": interface["classification_prompt_sha256"],
        "grammar": interface["grammar_sha256"],
        "template": interface["chat_template_sha256"],
        "task": interface["task_message_template_sha256"],
    }
    if observed != expected:
        issues.append("model/prompt/grammar/template/task identity mismatch")
    if receipt["sampling"] != interface["sampling"]:
        issues.append("sampling changed")
    parent_receipt = read_json(root / "inputs/raw-parent.run.json")
    if receipt["environment"]["packages"] != parent_receipt["environment"]["packages"]:
        issues.append("runtime packages differ from the raw comparator")
    if receipt["output"]["sha256"] != sha256_file(raw):
        issues.append("output checksum mismatch")
    if receipt["dataset"]["sha256"] != plan["development_surface"]["source_database_sha256"]:
        issues.append("dataset mismatch")
    load = receipt["model"]["load_parameters"]
    if load.get("n_ctx") != 4096 or load.get("n_gpu_layers") != 30:
        issues.append("historical load settings changed")
    if receipt["environment"]["git"] != {
        "revision": job["repository_revision"],
        "worktree_dirty": False,
    }:
        issues.append("producing revision or worktree state changed")
    if receipt["model"]["artifact_access"]["mode"] != "local_cache_only":
        issues.append("model resolution was not local only")
    if (
        receipt["input_policy"]["classification_interface_mode"] != "native_chat"
        or set(frame["classification_interface_mode"]) != {"native_chat"}
        or receipt["reports_completed"] != 100
    ):
        issues.append("classification interface or completion receipt mismatch")
    if receipt["execution_surface"] != {"classification": True, "explanations": False}:
        issues.append("unexpected execution surface")
    if (root / "products/crash_report.txt").exists():
        issues.append("worker failure requires explicit review")
    result = {
        "study_id": STUDY_ID,
        "created_at_utc": utc_now(),
        "selected_for_freeze": not issues,
        "blockers": issues,
        "reference_metrics_accessed_for_selection": False,
        "records": len(frame),
        "invalid_outputs": invalid,
        "distinct_patterns": len(patterns),
        "output_sha256": sha256_file(raw),
        "run_receipt_sha256": sha256_file(receipt_path),
        "manifest_sha256": sha256_file(root / "inputs/development.manifest.csv"),
        "protected_evaluation_authorized": False,
    }
    destination = root / "receipts/structural-freeze.json"
    if destination.exists():
        existing = read_json(destination)
        for key in ["output_sha256", "run_receipt_sha256", "selected_for_freeze"]:
            if existing[key] != result[key]:
                raise ValueError("immutable structural freeze changed")
        result = existing
    else:
        atomic_json(destination, result)
    if issues:
        raise ValueError("structural freeze rejected: " + "; ".join(issues))
    return result


def prediction_table(raw: Path, destination: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw)
    records = []
    for _, row in frame.iterrows():
        parsed = classification_levels(row["classifications"])
        records.append(
            {
                "Hashed_ReportURN": row["Hashed_ReportURN"],
                **{label: parsed[key] for key, label in JSON_KEY_TO_LABEL.items()},
            }
        )
    output = pd.DataFrame(records)
    atomic_csv(destination, output)
    return output


def analyze(root: Path, iterations: int = 2000) -> dict[str, Any]:
    freeze = read_json(root / "receipts/structural-freeze.json")
    native_path = root / "products/native-classification.csv"
    if not freeze["selected_for_freeze"] or sha256_file(native_path) != freeze["output_sha256"]:
        raise ValueError("analysis requires the unchanged result-blind frozen output")
    native = prediction_table(native_path, root / "analysis/native-predictions.csv")
    raw = prediction_table(
        root / "inputs/raw-development.csv", root / "analysis/raw-predictions.csv"
    )
    options = {
        "require_complete_reference": True,
        "require_exact_key_set": True,
        "bootstrap_iterations": iterations,
        "seed": 20260718,
    }
    evaluations = {}
    for name in ["native", "raw"]:
        evaluations[name] = evaluate_predictions(
            root / "inputs/development.db",
            root / f"analysis/{name}-predictions.csv",
            root / f"analysis/{name}",
            **options,
        )
    paired = compare_predictions(
        root / "inputs/development.db",
        root / "analysis/native-predictions.csv",
        root / "analysis/raw-predictions.csv",
        root / "analysis/paired",
        model_a_id="mistral-native-development",
        model_b_id="mistral-reproduced-raw-development",
        multiplicity="holm",
        **options,
    )
    agreement = {}
    for label in JSON_KEY_TO_LABEL.values():
        a, b = native[label], raw[label]
        agreement[label] = {
            "same_core": int(((a >= 3) == (b >= 3)).sum()),
            "same_four_level": int((a == b).sum()),
            "records": 100,
        }
    classification_operational = {}
    for name, path in [("native", native_path), ("raw", root / "inputs/raw-development.csv")]:
        telemetry = pd.read_csv(path)
        classification_operational[name] = {
            "elapsed_seconds_mean": float(telemetry["classify_elapsed_seconds"].mean()),
            "elapsed_seconds_total": float(telemetry["classify_elapsed_seconds"].sum()),
            "prompt_tokens_total": int(telemetry["classify_prompt_tokens"].sum()),
            "completion_tokens_total": int(telemetry["classify_completion_tokens"].sum()),
            "timing_interpretation": "descriptive; different run dates and machine load",
        }
    evidence = {}
    evidence_operational = {}
    expected_keys = pd.read_csv(root / "inputs/evidence.manifest.csv")["Hashed_ReportURN"].tolist()
    fixed = pd.read_csv(root / "inputs/raw-development.csv").set_index("Hashed_ReportURN")
    for interface in ["raw_completion", "native_chat"]:
        path = root / f"products/evidence-{interface}.csv"
        frame = pd.read_csv(path)
        if frame["Hashed_ReportURN"].tolist() != expected_keys:
            raise ValueError("evidence key coverage differs from the frozen first 20 cases")
        for _, row in frame.iterrows():
            if row["fixed_classifications"] != fixed.at[row["Hashed_ReportURN"], "classifications"]:
                raise ValueError("evidence classification source changed")
        receipt = read_json(path.with_suffix(".run.json"))
        if receipt["output"]["sha256"] != sha256_file(path):
            raise ValueError("evidence output checksum mismatch")
        evidence[interface] = aggregate_inspections(frame)
        evidence_operational[interface] = {
            "elapsed_seconds_mean": float(frame["elapsed_seconds"].mean()),
            "elapsed_seconds_total": float(frame["elapsed_seconds"].sum()),
            "prompt_tokens_total": int(frame["prompt_tokens"].sum()),
            "completion_tokens_total": int(frame["completion_tokens"].sum()),
            "phrase_counts": frame["evidence_phrases"].value_counts().sort_index().to_dict(),
        }
    result = {
        "study_id": STUDY_ID,
        "created_at_utc": utc_now(),
        "status": "completed_development_followup_not_manuscript_admitted",
        "scope": {"classification_reports": 100, "evidence_reports_per_interface": 20},
        "structural_freeze_sha256": sha256_file(root / "receipts/structural-freeze.json"),
        "classification_agreement": agreement,
        "classification_operational": classification_operational,
        "native_evaluation": evaluations["native"]["labels"],
        "raw_evaluation": evaluations["raw"]["labels"],
        "paired_comparison": paired["labels"],
        "evidence_quality": evidence,
        "evidence_operational": evidence_operational,
        "boundaries": [
            "Development-only: no protected evaluation was run for this Mistral follow-up.",
            "All favorable and unfavorable differences are retained; no equivalence claim is made.",
            "Report-level intervals do not establish patient independence.",
            "Twenty evidence cases are descriptive, not a validated clinical sample.",
            "Exact substring checks test traceability, not semantic or causal faithfulness.",
            "Fuzzy/semantic factuality and learned polarity checks are deferred.",
            "Author review is required before manuscript placement or onward aggregate release.",
        ],
        "repository_revision": git_revision(),
    }
    atomic_json(root / "analysis/author-summary.json", result)
    lines = [
        "# Mistral interface follow-up — author review only",
        "",
        result["status"],
        "",
        "100 development classifications; 20 evidence cases per interface. No protected rerun.",
        "",
        "| Category | Same core / 100 | Same four-level / 100 | Native − raw core accuracy |",
        "|---|---:|---:|---:|",
    ]
    for label, values in agreement.items():
        delta = paired["labels"][label]["effects_a_minus_b"]["core_accuracy_difference"]
        lines.append(
            f"| {label} | {values['same_core']} | {values['same_four_level']} | {delta:+.1%} |"
        )
    lines += ["", "## Fixed-classification evidence", ""]
    for interface, values in evidence.items():
        lines.append(
            f"- {interface}: {values['valid_structured_outputs']}/20 valid; "
            f"{values['decision_copy_mismatches']} copied-decision mismatches; "
            f"{values['exact_traceable_phrases']}/{values['evidence_phrases']} "
            f"non-fallback phrases found verbatim; {values['fallback_phrases']} fallback phrases."
        )
    lines += ["", "## Boundaries", "", *[f"- {value}" for value in result["boundaries"]], ""]
    (root / "analysis/author-summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def csv_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def publish_status(root: Path, reason: str | None = None) -> dict[str, Any]:
    job, state = read_json(root / "job.json"), read_json(root / "state.json")
    counts = {
        "native_classification": csv_count(root / "products/native-classification.csv"),
        "evidence_raw": csv_count(root / "products/evidence-raw_completion.csv"),
        "evidence_native": csv_count(root / "products/evidence-native_chat.csv"),
    }
    status = {
        "study_id": STUDY_ID,
        "updated_at_utc": utc_now(),
        "state": state["status"],
        "current_stage": state.get("current_stage"),
        "progress": counts,
        "targets": {"native_classification": 100, "evidence_raw": 20, "evidence_native": 20},
        "dependency": "MedGemma native protected evaluation and final transfer receipt",
        "note": reason or (state.get("note") if state["status"] in {"queued", "failed"} else None),
        "protected_evaluation": False,
        "automatic_expansion": False,
        "privacy_boundary": "counts and execution state only; no case data or partial performance",
    }
    atomic_json(Path(job["public_status"]), status)
    return status


def finalize(root: Path, job: dict[str, Any]) -> None:
    """Idempotent completion publication, including after a last-moment interruption."""
    prefix = Path(job["public_status"]).with_suffix("")
    atomic_json(
        Path(str(prefix) + "-result.json"), read_json(root / "analysis/author-summary.json")
    )
    target = Path(str(prefix) + "-result.md")
    temporary = target.with_suffix(".md.tmp")
    shutil.copyfile(root / "analysis/author-summary.md", temporary)
    temporary.replace(target)
    write_transfer_manifest(root)
    publish_status(root)


def run_lock_released(root: Path) -> bool:
    with (root / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def watch(root: Path) -> None:
    os.umask(0o077)
    with (root / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        job = validate_job(root)
        state = read_json(root / "state.json")
        if state["status"] == "completed":
            finalize(root, job)
            return
        if state["status"] == "failed":
            publish_status(root)
            return
        while True:
            if (root / "PAUSED").exists():
                state = read_json(root / "state.json")
                state.update(status="paused", updated_at_utc=utc_now())
                atomic_json(root / "state.json", state)
                publish_status(root)
                return
            ready, reason = dependency_readiness(job["dependency"])
            if ready and run_lock_released(root):
                break
            if ready:
                reason = "waiting for previous follow-up worker to release its checkpoint lock"
            state = read_json(root / "state.json")
            state.update(status="queued", note=reason, updated_at_utc=utc_now())
            atomic_json(root / "state.json", state)
            publish_status(root, reason)
            time.sleep(20)
        validate_job(root)
        last_progress = 0.0
        stage_started = time.monotonic()
        previous_stage = None

        def progress(supervisor: Supervisor, stage: Stage) -> None:
            nonlocal last_progress, stage_started, previous_stage
            if stage.name != previous_stage:
                previous_stage = stage.name
                stage_started = time.monotonic()
            if time.monotonic() - stage_started > 7200:
                raise TimeoutError(
                    "follow-up stage exceeded two-hour safety limit; retained for review"
                )
            if (root / "PAUSED").exists() or (root / "ECLIPSED.json").exists():
                supervisor.handle_signal(signal.SIGTERM, None)
            if time.monotonic() - last_progress > 10:
                publish_status(root)
                last_progress = time.monotonic()

        supervisor = Supervisor(root, stages=stages(root), progress_callback=progress)
        supervisor.run()
        state = read_json(root / "state.json")
        if state["status"] == "completed":
            finalize(root, job)
        publish_status(root)


def launch(root: Path) -> dict[str, Any]:
    job = validate_job(root)
    agents = Path.home() / "Library/LaunchAgents"
    path = agents / f"{LAUNCH_LABEL}.plist"
    if path.exists():
        raise FileExistsError(
            "follow-up LaunchAgent already exists; use launchctl kickstart to resume"
        )
    agents.mkdir(parents=True, exist_ok=True)
    command = [job["python_executable"], str(SCRIPT), "watch", "--run-dir", str(root)]
    if sys.platform == "darwin":
        command = ["/usr/bin/caffeinate", "-i", *command]
    plist = {
        "Label": LAUNCH_LABEL,
        "ProgramArguments": command,
        "WorkingDirectory": str(REPO_ROOT),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 60,
        "EnvironmentVariables": {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        },
        "StandardOutPath": str(root / "logs/queue.log"),
        "StandardErrorPath": str(root / "logs/queue.log"),
        "ProcessType": "Background",
        "ExitTimeOut": 30,
        "AbandonProcessGroup": False,
    }
    path.write_bytes(plistlib.dumps(plist))
    path.chmod(0o600)
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
    receipt = {
        "created_at_utc": utc_now(),
        "launch_label": LAUNCH_LABEL,
        "plist": str(path),
        "program": command,
        "resume_policy": "restart after unexpected exit/login; scientific failures stop for review",
    }
    atomic_json(root / "receipts/queue-launch.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "launch",
            "watch",
            "status",
            "stage",
            "pause",
            "resume",
        ],
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dependency-run", type=Path)
    parser.add_argument("--dependency-status", type=Path)
    parser.add_argument("--dependency-heartbeat", type=Path)
    parser.add_argument("--public-status", type=Path)
    parser.add_argument("--name", choices=["structural_freeze", "analysis_and_author_summary"])
    args = parser.parse_args()
    root = args.run_dir.expanduser().resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.umask(0o077)
    try:
        if args.action == "prepare":
            result = prepare(args)
        elif args.action == "launch":
            result = launch(root)
        elif args.action == "watch":
            watch(root)
            return
        elif args.action == "status":
            result = publish_status(root)
        elif args.action == "pause":
            (root / "PAUSED").touch(mode=0o600)
            result = {"pause_requested": True, "in_flight_report_may_be_recomputed_on_resume": True}
        elif args.action == "resume":
            validate_job(root)
            if read_json(root / "state.json")["status"] == "failed":
                raise ValueError("failed scientific stage requires review, not blind resume")
            (root / "PAUSED").unlink(missing_ok=True)
            subprocess.run(
                ["launchctl", "kickstart", f"gui/{os.getuid()}/{LAUNCH_LABEL}"], check=True
            )
            result = {"resume_requested": True}
        else:
            validate_job(root)
            result = (
                freeze_classification(root) if args.name == "structural_freeze" else analyze(root)
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        if args.action == "watch":
            state = read_json(root / "state.json")
            state.update(status="failed", note=str(exc), updated_at_utc=utc_now())
            atomic_json(root / "state.json", state)
            publish_status(root)
            print(f"Follow-up stopped for review: {exc}", file=sys.stderr)
            return  # Successful exit prevents an unattended retry loop on a scientific failure.
        raise


if __name__ == "__main__":
    main()
