#!/usr/bin/env python3
"""One frozen prompt refinement, using the existing checkpointed study supervisor."""

from __future__ import annotations

import argparse
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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_job import (  # noqa: E402
    REPO_ROOT,
    Stage,
    Supervisor,
    atomic_csv,
    atomic_json,
    git_revision,
    marker_path,
    sha256_file,
    stage_is_complete,
    utc_now,
)

from eeg_review.compare import compare_predictions  # noqa: E402
from eeg_review.evidence_extraction import (  # noqa: E402
    aggregate_inspections,
    classification_levels,
    load_fixed_evidence_inputs,
)
from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL  # noqa: E402
from eeg_review.metrics import evaluate_predictions  # noqa: E402
from eeg_review.native_interface import native_task_message_template, sha256_text  # noqa: E402
from eeg_review.prompt_versions import (  # noqa: E402
    MEDGEMMA_FOCAL_V2,
    classification_prompt,
    development_verdict,
    validate_prompt_resume,
)
from eeg_review.protected_execution import assert_governed_run_active  # noqa: E402

PLAN = REPO_ROOT / "review/model-receipts/medgemma-native-focal-v2.development-plan.json"
SCRIPT = Path(__file__).resolve()
LABEL = "ca.sbergner.eeg.medgemma-prompt-v2-development"
STUDY_ID = "jbhi-02463-medgemma-native-focal-v2-development"
KEY = "Hashed_ReportURN"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def versions():
    return {
        name: importlib.metadata.version(name)
        for name in ["llama-cpp-python", "huggingface-hub", "pandas"]
    }


def prompt_text():
    sys.path.insert(0, str(REPO_ROOT / "src/LLM_pipeline"))
    import pipeline

    return classification_prompt(pipeline.PROMPT_CLASSIFY, MEDGEMMA_FOCAL_V2)


def validate_plan(plan):
    expected = {
        "study_id": STUDY_ID,
        "candidate_count": 1,
        "classification_records": 100,
        "evidence_records": 20,
        "evidence_positions": [0, 20],
        "planned_committed_model_calls": 120,
        "protected_evaluation_allowed": False,
        "automatic_expansion_allowed": False,
        "independent_test_claim_allowed": False,
        "prior_protected_results_informed_hypothesis": True,
    }
    if any(
        type(plan.get(key)) is not type(value) or plan.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("v2 scope or exploratory evidence boundary changed")
    if plan["interface"]["prompt_version"] != MEDGEMMA_FOCAL_V2:
        raise ValueError("wrong prompt version")
    if plan["interface"]["mode"] != "native_chat" or plan["execution"]["local_only"] is not True:
        raise ValueError("native/local interface required")
    if plan["model"]["weights_or_training_changed"] or plan["model"]["weights_redistributed"]:
        raise ValueError("model/training/distribution expansion forbidden")
    if plan["evidence"]["use_for_prompt_selection"]:
        raise ValueError("evidence cannot select this prompt")
    if not plan["analysis"]["retain_all_outcomes"]:
        raise ValueError("all outcomes must be retained")


def validate_code(plan):
    prompt = prompt_text()
    if sha256_text(prompt) != plan["interface"]["prompt_sha256"]:
        raise ValueError("v2 prompt bytes differ from frozen plan")
    if (
        sha256_text(native_task_message_template(prompt))
        != plan["interface"]["task_message_sha256"]
    ):
        raise ValueError("v2 task layout differs from frozen plan")
    for name, expected in [
        ("result_grammar.gbnf", plan["interface"]["grammar_sha256"]),
        ("result_grammar_exp.gbnf", plan["evidence"]["grammar_sha256"]),
    ]:
        if sha256_file(REPO_ROOT / "src/LLM_pipeline" / name) != expected:
            raise ValueError("frozen grammar changed")
    return prompt


def prepare(args):
    plan = read(PLAN)
    validate_plan(plan)
    prompt = validate_code(plan)
    root = args.run_dir.resolve()
    if root.exists():
        raise FileExistsError("existing run is never overwritten")
    source = args.source_run.resolve(strict=True)
    for item in plan["source_files"].values():
        if sha256_file(source / item["path"]) != item["sha256"]:
            raise ValueError("frozen development parent hash mismatch")
    parent = read(source / plan["source_files"]["v1.run.json"]["path"])
    if parent["environment"]["packages"] != versions():
        raise ValueError("runtime differs from native v1")
    if parent["model"]["sha256"] != plan["model"]["sha256"]:
        raise ValueError("v1 model differs")
    for key, value in plan["runtime"]["parameters"].items():
        if parent["model"]["load_parameters"][key] != value:
            raise ValueError("runtime parameters differ from native v1")
    if parent["sampling"] != plan["interface"]["sampling"]:
        raise ValueError("sampling differs from native v1")
    fixed = load_fixed_evidence_inputs(
        dataset=source / plan["source_files"]["development.db"]["path"],
        predictions=source / plan["source_files"]["v1.csv"]["path"],
        manifest=source / plan["source_files"]["development.manifest.csv"]["path"],
    )
    if len(fixed) != 100:
        raise ValueError("only the frozen 100 development records are allowed")
    for name in ["inputs", "products", "analysis", "receipts", "logs", "stages"]:
        (root / name).mkdir(parents=True, mode=0o700)
    for name, item in plan["source_files"].items():
        shutil.copyfile(source / item["path"], root / "inputs" / name)
    shutil.copyfile(PLAN, root / "inputs/plan.json")
    (root / "inputs/prompt-v2.txt").write_text(prompt, encoding="utf-8")
    atomic_csv(root / "inputs/evidence.manifest.csv", fixed[[KEY]].iloc[:20])
    job = {
        "study_id": STUDY_ID,
        "configuration_id": plan["configuration_id"],
        "created_at_utc": utc_now(),
        "repository_revision": git_revision(),
        "python_executable": sys.executable,
        "runtime_versions": versions(),
        "plan_sha256": sha256_file(PLAN),
        "public_status": str(args.public_status.resolve()),
        "inputs": [
            {"path": str(p.relative_to(root)), "sha256": sha256_file(p)}
            for p in sorted((root / "inputs").iterdir())
        ],
        "authorization": "Steven authorized one local development prompt version in chat.",
        "prior_protected_results_informed_hypothesis": True,
        "protected_evaluation": False,
        "planned_committed_model_calls": 120,
    }
    atomic_json(root / "job.json", job)
    atomic_json(
        root / "receipts/pre-inference-freeze.json",
        {
            "created_at_utc": utc_now(),
            "job_sha256": sha256_file(root / "job.json"),
            "prompt_sha256": sha256_text(prompt),
            "plan_sha256": sha256_file(PLAN),
            "v2_outputs_exist": False,
            "prior_protected_results_informed_hypothesis": True,
            "independent_confirmation": False,
        },
    )
    atomic_json(root / "state.json", {"status": "prepared", "stages": {}, "current_stage": None})
    return publish_status(root)


def validate_job(root):
    job, plan = read(root / "job.json"), read(root / "inputs/plan.json")
    validate_plan(plan)
    if job["study_id"] != STUDY_ID or job["protected_evaluation"] is not False:
        raise ValueError("wrong job identity or scope")
    freeze = read(root / "receipts/pre-inference-freeze.json")
    if sha256_file(root / "job.json") != freeze["job_sha256"]:
        raise ValueError("job changed after pre-inference freeze")
    if git_revision() != job["repository_revision"]:
        raise ValueError("repository revision changed; explicit migration required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).strip():
        raise ValueError("producing worktree must be clean")
    if versions() != job["runtime_versions"]:
        raise ValueError("runtime changed; explicit migration required")
    if sha256_file(PLAN) != job["plan_sha256"]:
        raise ValueError("plan changed")
    for item in job["inputs"]:
        if sha256_file(root / item["path"]) != item["sha256"]:
            raise ValueError("frozen input changed: " + item["path"])
    validate_code(plan)
    assert_governed_run_active(root)
    return job, plan


def runtime_arguments(plan):
    args = []
    for key, value in plan["runtime"]["parameters"].items():
        args += (
            ["--" + key.replace("_", "-")]
            if value is True
            else ["--" + key.replace("_", "-"), str(value)]
        )
    return args


def classification_command(root, count=100):
    if count not in {3, 100}:
        raise ValueError("only the three-case checkpoint or 100-case development run is allowed")
    job, plan = read(root / "job.json"), read(root / "inputs/plan.json")
    return [
        job["python_executable"],
        str(REPO_ROOT / "src/LLM_pipeline/pipeline.py"),
        "--num-reports",
        str(count),
        "--dataset-path",
        str(root / "inputs/development.db"),
        "--dataset-id",
        "medgemma-native-focal-v2-development",
        "--model",
        plan["model"]["registry_name"],
        "--classification-only",
        "--classification-interface",
        "native_chat",
        "--classification-prompt-version",
        MEDGEMMA_FOCAL_V2,
        "--local-model-only",
        "--output-csv",
        str(root / "products/v2.csv"),
        "--resume-output",
        "--outdir",
        str(root / "products"),
        "--flush-every",
        "1",
        "--max-tokens",
        "256",
        "--runtime-profile-id",
        plan["runtime"]["profile_id"],
        *runtime_arguments(plan),
    ]


def evidence_command(root):
    job, plan = read(root / "job.json"), read(root / "inputs/plan.json")
    return [
        job["python_executable"],
        str(REPO_ROOT / "scripts/run_fixed_classification_explanations.py"),
        "--run-id",
        STUDY_ID + "/evidence-first20",
        "--dataset",
        str(root / "inputs/development.db"),
        "--predictions",
        str(root / "products/v2.csv"),
        "--manifest",
        str(root / "inputs/evidence.manifest.csv"),
        "--output-csv",
        str(root / "products/evidence-v2.csv"),
        "--model",
        plan["model"]["registry_name"],
        "--interface",
        "native_chat",
        "--expected-dataset-sha256",
        sha256_file(root / "inputs/development.db"),
        "--expected-predictions-sha256",
        sha256_file(root / "products/v2.csv"),
        "--expected-manifest-sha256",
        sha256_file(root / "inputs/evidence.manifest.csv"),
        "--expected-records",
        "20",
        "--expected-chat-template-sha256",
        plan["interface"]["chat_template_sha256"],
        "--expected-prompt-sha256",
        plan["evidence"]["prompt_sha256"],
        "--expected-grammar-sha256",
        plan["evidence"]["grammar_sha256"],
        "--flush-every",
        "1",
        "--max-tokens",
        str(plan["evidence"]["max_tokens"]),
        "--resume",
        *runtime_arguments(plan),
    ]


def inspect_classification(root, count=100):
    job, plan = read(root / "job.json"), read(root / "inputs/plan.json")
    path = root / "products/v2.csv"
    frame, manifest = pd.read_csv(path), pd.read_csv(root / "inputs/development.manifest.csv")
    if frame[KEY].tolist() != manifest[KEY].tolist()[:count] or len(frame) != count:
        raise ValueError("classification output is not the exact frozen manifest prefix")
    validate_prompt_resume(frame, MEDGEMMA_FOCAL_V2, prompt_text())
    for value in frame["classifications"]:
        classification_levels(value)
    receipt = read(path.with_suffix(".run.json"))
    if receipt["reports_completed"] != count:
        raise ValueError("run receipt count differs from checkpoint")
    if receipt["prompts"]["classify"]["version"] != MEDGEMMA_FOCAL_V2:
        raise ValueError("run receipt prompt version differs")
    for field in ["hf_hub_offline", "hf_hub_telemetry_disabled"]:
        if receipt["environment"].get(field) is not True:
            raise ValueError("offline execution settings were not receipted")
    observed = [
        receipt["model"]["sha256"],
        receipt["prompts"]["classify"]["sha256"],
        receipt["grammars"]["classify"]["sha256"],
        receipt["input_policy"]["embedded_chat_template"]["sha256"],
        receipt["input_policy"]["task_message_template"]["sha256"],
    ]
    expected = [
        plan["model"]["sha256"],
        plan["interface"]["prompt_sha256"],
        plan["interface"]["grammar_sha256"],
        plan["interface"]["chat_template_sha256"],
        plan["interface"]["task_message_sha256"],
    ]
    if observed != expected or receipt["sampling"] != plan["interface"]["sampling"]:
        raise ValueError("classification interface/model/sampling receipt mismatch")
    if receipt["environment"]["git"] != {
        "revision": job["repository_revision"],
        "worktree_dirty": False,
    }:
        raise ValueError("producing source identity changed")
    if receipt["environment"]["packages"] != job["runtime_versions"]:
        raise ValueError("producing packages changed")
    if receipt["model"]["artifact_access"]["mode"] != "local_cache_only":
        raise ValueError("model was not resolved offline")
    if receipt["dataset"]["sha256"] != sha256_file(root / "inputs/development.db"):
        raise ValueError("wrong classification data")
    if receipt["execution_surface"] != {"classification": True, "explanations": False}:
        raise ValueError("wrong execution surface")
    for key, value in plan["runtime"]["parameters"].items():
        if receipt["model"]["load_parameters"][key] != value:
            raise ValueError("runtime parameter drift")
    if receipt["output"]["sha256"] != sha256_file(path):
        raise ValueError("classification output hash changed")
    return {
        "records": count,
        "output_sha256": sha256_file(path),
        "run_receipt_sha256": sha256_file(path.with_suffix(".run.json")),
        "prompt_sha256": observed[1],
        "reference_metrics_used": False,
    }


def prediction_table(source, destination):
    frame = pd.read_csv(source)
    rows = [
        {
            KEY: row[KEY],
            **{
                label: classification_levels(row["classifications"])[key]
                for key, label in JSON_KEY_TO_LABEL.items()
            },
        }
        for _, row in frame.iterrows()
    ]
    result = pd.DataFrame(rows)
    atomic_csv(destination, result)
    return result


def analyze(root):
    job, plan = validate_job(root)
    frozen = read(root / "receipts/classification-complete.json")
    if frozen != inspect_classification(root):
        raise ValueError("frozen v2 classification changed")
    evaluations, predictions = {}, {}
    for name, path in [("v1", root / "inputs/v1.csv"), ("v2", root / "products/v2.csv")]:
        out = root / f"analysis/{name}-predictions.csv"
        predictions[name] = prediction_table(path, out)
        evaluations[name] = evaluate_predictions(
            root / "inputs/development.db",
            out,
            root / f"analysis/{name}",
            require_complete_reference=True,
            require_exact_key_set=True,
            bootstrap_iterations=2000,
            seed=20260718,
        )
    paired = compare_predictions(
        root / "inputs/development.db",
        root / "analysis/v2-predictions.csv",
        root / "analysis/v1-predictions.csv",
        root / "analysis/paired",
        model_a_id=plan["configuration_id"],
        model_b_id=plan["parent_configuration_id"],
        require_complete_reference=True,
        require_exact_key_set=True,
        bootstrap_iterations=2000,
        seed=20260718,
        multiplicity="holm",
    )
    points = {
        name: {label: value["point_estimates"] for label, value in result["labels"].items()}
        for name, result in evaluations.items()
    }
    evidence_path = root / "products/evidence-v2.csv"
    evidence = pd.read_csv(evidence_path)
    manifest = pd.read_csv(root / "inputs/evidence.manifest.csv")
    if evidence[KEY].tolist() != manifest[KEY].tolist():
        raise ValueError("evidence coverage changed")
    fixed = pd.read_csv(root / "products/v2.csv").set_index(KEY)
    for _, row in evidence.iterrows():
        if row["fixed_classifications"] != fixed.at[row[KEY], "classifications"]:
            raise ValueError("evidence did not use frozen v2 decisions")
    receipt = read(evidence_path.with_suffix(".run.json"))
    if receipt["output"]["sha256"] != sha256_file(evidence_path):
        raise ValueError("evidence output hash changed")
    # Existing nonempty policy remains unchanged. Explicitly label its conditional denominator.
    evidence_summary = aggregate_inspections(evidence)
    evidence_summary["phrase_denominator"] = "phrases in outputs passing existing nonempty policy"
    evidence_summary["invalid_outputs_not_dropped"] = True
    result = {
        "study_id": STUDY_ID,
        "configuration_id": plan["configuration_id"],
        "status": "completed_exploratory_development_not_manuscript_admitted",
        "created_at_utc": utc_now(),
        "repository_revision": job["repository_revision"],
        "classification_records": 100,
        "evidence_records": 20,
        "prior_protected_results_informed_hypothesis": True,
        "protected_evaluation_run": False,
        "independent_confirmation": False,
        "development_verdict": development_verdict(points["v1"], points["v2"]),
        "v1": evaluations["v1"]["labels"],
        "v2": evaluations["v2"]["labels"],
        "paired_comparison": paired["labels"],
        "evidence_quality": evidence_summary,
        "support": {
            label: {"positive": p["tp"] + p["fn"], "negative": p["tn"] + p["fp"]}
            for label, p in points["v1"].items()
        },
        "same_core": {
            label: int(((predictions["v1"][label] >= 3) == (predictions["v2"][label] >= 3)).sum())
            for label in JSON_KEY_TO_LABEL.values()
        },
        "source_hashes": {
            "v1": sha256_file(root / "inputs/v1.csv"),
            "v2": sha256_file(root / "products/v2.csv"),
            "evidence": sha256_file(evidence_path),
        },
        "boundaries": [
            "Development only; prior evaluation-informed hypothesis, not independent testing.",
            "No automatic prompt search or protected-cohort rerun.",
            "All five labels and negative/null outcomes retained.",
            "Report-level uncertainty; rare-category support is limited.",
            "Evidence exact matching is not clinical or causal validation.",
            "No paired v1 evidence-quality claim; this evidence stage is v2-only.",
            "Author review is required for manuscript placement and onward release.",
        ],
    }
    atomic_json(root / "analysis/author-summary.json", result)
    lines = [
        "# MedGemma prompt v2 development result",
        "",
        result["status"],
        "",
        "| Category | V1 FP/FN | V2 FP/FN | V1 F1 | V2 F1 | Same core / 100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in JSON_KEY_TO_LABEL.values():
        a, b = points["v1"][label], points["v2"][label]
        lines.append(
            f"| {label} | {a['fp']}/{a['fn']} | {b['fp']}/{b['fn']} | "
            f"{a['f1']:.1%} | {b['f1']:.1%} | {result['same_core'][label]} |"
        )
    lines += [
        "",
        "Development rule: " + result["development_verdict"]["status"],
        "",
        "Evidence: " + json.dumps(evidence_summary, sort_keys=True),
        "",
        *["- " + line for line in result["boundaries"]],
        "",
    ]
    (root / "analysis/author-summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def stages(root):
    job = read(root / "job.json")
    internal = [job["python_executable"], str(SCRIPT), "stage", "--run-dir", str(root), "--name"]
    raw, evidence = root / "products/v2.csv", root / "products/evidence-v2.csv"
    return [
        Stage(
            "classification_100",
            classification_command(root),
            (raw, raw.with_suffix(".run.json")),
            raw,
            100,
        ),
        Stage(
            "validate_classification",
            [*internal, "validate_classification"],
            (root / "receipts/classification-complete.json",),
        ),
        Stage(
            "evidence_20",
            [*internal, "evidence_20"],
            (evidence, evidence.with_suffix(".run.json")),
            evidence,
            20,
        ),
        Stage(
            "analyze",
            [*internal, "analyze"],
            (root / "analysis/author-summary.json", root / "analysis/author-summary.md"),
        ),
    ]


def publish_status(root):
    job, state = read(root / "job.json"), read(root / "state.json")
    progress, remaining = {}, 0.0
    for name, path, target, estimate in [
        ("classification", root / "products/v2.csv", 100, 20.0),
        ("evidence", root / "products/evidence-v2.csv", 20, 60.0),
    ]:
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        count = len(frame)
        field = "classify_elapsed_seconds" if name == "classification" else "elapsed_seconds"
        mean = float(frame[field].mean()) if count and field in frame else None
        progress[name] = {"completed": count, "target": target, "mean_seconds": mean}
        remaining += max(target - count, 0) * (mean or estimate)
    result = {
        "study_id": STUDY_ID,
        "updated_at_utc": utc_now(),
        "state": state["status"],
        "current_stage": state.get("current_stage"),
        "progress": progress,
        "estimated_remaining_seconds": round(remaining) if state["status"] != "completed" else 0,
        "eta_caveat": "unobserved stages use planning estimates; excludes interrupted downtime",
        "protected_evaluation": False,
        "automatic_expansion": False,
        "performance_during_generation": "not inspected",
        "note": state.get("note"),
    }
    atomic_json(Path(job["public_status"]), result)
    return result


def finalize(root):
    if read(root / "state.json")["status"] != "completed":
        raise ValueError("cannot finalize an incomplete run")
    for stage in stages(root):
        if not stage_is_complete(root, stage):
            raise ValueError("completed output changed; refusing to re-seal it")
    job = read(root / "job.json")
    # Scientific receipt excludes changing logs/locks/state; operational files remain in the run.
    paths = [root / "job.json"]
    for folder in ["inputs", "products", "analysis", "receipts"]:
        paths += [p for p in (root / folder).rglob("*") if p.is_file()]
    atomic_json(
        root / "final-scientific-manifest.json",
        {
            "study_id": STUDY_ID,
            "repository_revision": job["repository_revision"],
            "files": [
                {"path": str(p.relative_to(root)), "sha256": sha256_file(p)} for p in sorted(paths)
            ],
            "excluded": (
                "mutable logs, control files, launch state and operational transfer manifest"
            ),
            "distribution": "governed channel only; keyed products and report text remain private",
        },
    )
    prefix = Path(job["public_status"]).with_suffix("")
    atomic_json(Path(str(prefix) + "-result.json"), read(root / "analysis/author-summary.json"))
    shutil.copyfile(root / "analysis/author-summary.md", Path(str(prefix) + "-result.md"))
    publish_status(root)


def watch(root):
    with (root / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        validate_job(root)
        state = read(root / "state.json")
        if state["status"] == "failed" or (root / "PAUSED").exists():
            publish_status(root)
            return
        if state["status"] == "completed":
            finalize(root)
            return
        for stage in stages(root):
            if marker_path(root, stage).exists() and not stage_is_complete(root, stage):
                raise ValueError("completed stage changed; refusing silent recomputation")
        last = 0.0
        started, previous = time.monotonic(), None

        def progress(supervisor, stage):
            nonlocal last, started, previous
            if stage.name != previous:
                previous, started = stage.name, time.monotonic()
            if time.monotonic() - started > 7200:
                raise TimeoutError("stage exceeded bounded two-hour time limit")
            if (root / "PAUSED").exists() or (root / "ECLIPSED.json").exists():
                supervisor.handle_signal(signal.SIGTERM, None)
            if time.monotonic() - last > 10:
                publish_status(root)
                last = time.monotonic()

        Supervisor(root, stages=stages(root), progress_callback=progress).run()
        if read(root / "state.json")["status"] == "completed":
            finalize(root)
        publish_status(root)


def launch(root):
    job, _ = validate_job(root)
    path = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    if path.exists():
        raise FileExistsError("LaunchAgent exists; use resume rather than replacing it")
    command = [
        "/usr/bin/caffeinate",
        "-i",
        job["python_executable"],
        str(SCRIPT),
        "watch",
        "--run-dir",
        str(root),
    ]
    config = {
        "Label": LABEL,
        "ProgramArguments": command,
        "WorkingDirectory": str(REPO_ROOT),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 60,
        "EnvironmentVariables": {
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        },
        "StandardOutPath": str(root / "logs/launcher.log"),
        "StandardErrorPath": str(root / "logs/launcher.log"),
        "ExitTimeOut": 30,
        "AbandonProcessGroup": False,
        "ProcessType": "Background",
    }
    path.write_bytes(plistlib.dumps(config))
    path.chmod(0o600)
    atomic_json(root / "receipts/launch.json", {"label": LABEL, "created_at_utc": utc_now()})
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
    return {"launched": True, "label": LABEL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "dry-run",
            "smoke",
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
    parser.add_argument("--public-status", type=Path)
    parser.add_argument("--name", choices=["validate_classification", "evidence_20", "analyze"])
    args = parser.parse_args()
    root = args.run_dir.resolve()
    os.umask(0o077)
    os.environ.update(
        HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", PYTHONPATH=str(REPO_ROOT / "src")
    )
    try:
        if args.action == "prepare":
            result = prepare(args)
        elif args.action == "status":
            result = publish_status(root)
        elif args.action == "pause":
            (root / "PAUSED").touch(mode=0o600)
            result = {"pause_requested": True}
        else:
            _job, plan = validate_job(root)
            if args.action == "dry-run":
                from llm_models import resolve_model_artifact

                _, receipt = resolve_model_artifact(
                    plan["model"]["registry_name"],
                    load_overrides=plan["runtime"]["parameters"],
                    local_files_only=True,
                )
                result = {
                    "ready": receipt["sha256"] == plan["model"]["sha256"],
                    "inference_performed": False,
                    "planned_committed_calls": 120,
                    "protected_evaluation": False,
                }
            elif args.action == "smoke":
                with (root / "run.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with (root / "logs/smoke.log").open("a") as log:
                        subprocess.run(
                            classification_command(root, 3),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            cwd=REPO_ROOT,
                            pass_fds=(lock.fileno(),),
                        )
                result = inspect_classification(root, 3)
                atomic_json(root / "receipts/smoke.json", result)
                shutil.copyfile(root / "products/v2.run.json", root / "receipts/smoke.run.json")
                publish_status(root)
            elif args.action == "launch":
                result = launch(root)
            elif args.action == "watch":
                watch(root)
                return
            elif args.action == "resume":
                if read(root / "state.json")["status"] == "failed":
                    raise ValueError("scientific failure requires review, not automatic resume")
                (root / "PAUSED").unlink(missing_ok=True)
                subprocess.run(["launchctl", "kickstart", f"gui/{os.getuid()}/{LABEL}"], check=True)
                result = {"resume_requested": True}
            elif args.name == "validate_classification":
                result = inspect_classification(root)
                atomic_json(root / "receipts/classification-complete.json", result)
            elif args.name == "evidence_20":
                if read(root / "receipts/classification-complete.json") != inspect_classification(
                    root
                ):
                    raise ValueError("fixed classifications changed before evidence")
                subprocess.run(evidence_command(root), check=True, cwd=REPO_ROOT)
                result = {"evidence_stage_finished": True}
            elif args.name == "analyze":
                result = analyze(root)
            else:
                raise ValueError("stage name required")
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        if args.action == "watch":
            state = read(root / "state.json")
            state.update(status="failed", note=str(exc), updated_at_utc=utc_now())
            atomic_json(root / "state.json", state)
            publish_status(root)
            print("Prompt v2 stopped for review: " + str(exc), file=sys.stderr)
            return
        raise


if __name__ == "__main__":
    main()
