#!/usr/bin/env python3
"""Audit the completed MedGemma v1 split, freeze, and local-execution lineage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from eeg_review.io import atomic_write_json, load_table
from eeg_review.manifest import sha256_file
from eeg_review.study_integrity import Partition, audit_partitions

ROOT = Path(__file__).resolve().parents[1]
KEY = "Hashed_ReportURN"
REPORT = "Report"
PLAN_PATH = "review/model-receipts/medgemma-native-interface-sensitivity.preregistered.json"
PROTECTED_PLAN_PATH = (
    "review/model-receipts/medgemma-native-protected-tiered-execution.preregistered.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--protected-run", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--development-plan-commit", required=True)
    parser.add_argument("--protected-plan-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def git_file(commit: str, path: str) -> tuple[dict, str]:
    resolved = subprocess.run(
        ["git", "rev-parse", f"{commit}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    content = subprocess.run(
        ["git", "show", f"{resolved}:{path}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return json.loads(content), resolved


def git_commit_time(commit: str) -> datetime:
    value = subprocess.run(
        ["git", "show", "-s", "--format=%cI", commit],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return parse_time(value)


def validate_manifest(frame: pd.DataFrame, manifest_path: Path) -> None:
    manifest = pd.read_csv(manifest_path, usecols=[KEY])
    keys = frame[KEY].map(str).tolist()
    manifest_keys = manifest[KEY].map(str).tolist()
    if keys != manifest_keys:
        raise ValueError(f"dataset order or keys differ from manifest: {manifest_path.name}")
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate report key in dataset: {manifest_path.name}")


def validate_transfer_manifest(run_root: Path, transfer_path: Path) -> int:
    payload = read_json(transfer_path)
    seen: set[str] = set()
    for item in payload["files"]:
        raw_path = item["path"]
        candidate = Path(raw_path)
        if candidate.is_absolute() or ".." in candidate.parts or raw_path in seen:
            raise ValueError("unsafe or duplicate final-transfer path")
        seen.add(raw_path)
        resolved = (run_root / candidate).resolve(strict=True)
        if not resolved.is_relative_to(run_root):
            raise ValueError("final-transfer path escapes run root")
        if sha256_file(resolved) != item["sha256"]:
            raise ValueError(f"final-transfer hash mismatch: {raw_path}")
    return len(seen)


def validate_raw_receipt(receipt: dict, configuration: dict, records: int) -> None:
    interface = configuration["interface"]
    if receipt["reports_completed"] != records:
        raise ValueError("protected inference did not complete its declared population")
    if receipt["model"]["sha256"] != configuration["model"]["sha256"]:
        raise ValueError("protected model differs from the frozen configuration")
    expected = {
        "prompt": (receipt["prompts"]["classify"]["sha256"], interface["historical_prompt_sha256"]),
        "grammar": (receipt["grammars"]["classify"]["sha256"], interface["grammar_sha256"]),
        "chat template": (
            receipt["input_policy"]["embedded_chat_template"]["sha256"],
            interface["chat_template_sha256"],
        ),
        "task message": (
            receipt["input_policy"]["task_message_template"]["sha256"],
            interface["task_message_template_sha256"],
        ),
    }
    for label, (actual, frozen) in expected.items():
        if actual != frozen:
            raise ValueError(f"protected {label} differs from the frozen configuration")
    policy = receipt["input_policy"]
    if policy["report_field"] != REPORT or policy["classification_interface_mode"] != "native_chat":
        raise ValueError("protected input interface changed")
    if receipt["execution_surface"] != {"classification": True, "explanations": False}:
        raise ValueError("protected execution surface changed")
    access = receipt["model"]["artifact_access"]
    if access != {"mode": "local_cache_only", "network_lookup_allowed": False}:
        raise ValueError("protected model execution was not local-cache-only")


def run(args: argparse.Namespace) -> dict:
    os.umask(0o077)
    development = args.development_run.expanduser().resolve(strict=True)
    protected = args.protected_run.expanduser().resolve(strict=True)
    authorization_path = args.authorization.expanduser().resolve(strict=True)
    output_root = args.output_dir.expanduser().resolve()
    governed_root = (ROOT / "data/governed/analysis-runs").resolve()
    if not output_root.is_relative_to(governed_root) or output_root == governed_root:
        raise ValueError("output must be a dedicated governed analysis-run directory")

    dev_plan, dev_commit = git_file(args.development_plan_commit, PLAN_PATH)
    protected_plan, protected_commit = git_file(
        args.protected_plan_commit, PROTECTED_PLAN_PATH
    )
    development_receipt = read_json(
        development / "products/zoe_development_native_100/raw.run.json"
    )
    authorization = read_json(authorization_path)
    job = read_json(protected / "job.json")
    state = read_json(protected / "state.json")
    source_plan_path = protected / "study-plan.json"
    source_plan = read_json(source_plan_path)
    final_transfer = protected / "final-transfer-manifest.json"

    if dev_plan["status"] != "frozen_for_development_execution":
        raise ValueError("development plan was not frozen at the supplied commit")
    selection = dev_plan["development_stage"]["selection_rule"]
    if (
        selection["candidate_count"] != 1
        or selection["reference_metric_used_for_selection"] is not False
    ):
        raise ValueError("development selection was not a singleton result-blind rule")
    if dev_plan["sensitivity_configuration"]["weights_or_training_change_allowed"] is not False:
        raise ValueError("development plan permits a weights or training change")
    if protected_plan["status"] != "preregistered_before_inference":
        raise ValueError("protected plan was not frozen before inference")
    if sha256_file(source_plan_path) != protected_plan["source_study_plan_sha256"]:
        raise ValueError("tiered plan does not bind the source study plan")
    policy = source_plan["execution_policy"]
    if (
        policy["configuration_search_allowed"] is not False
        or policy["partial_reference_metrics_allowed"] is not False
        or protected_plan["post_inference"]["partial_reference_metrics_allowed"]
        is not False
    ):
        raise ValueError("protected plan permits outcome-responsive configuration changes")

    study_id = dev_plan["study_id"]
    configuration = protected_plan["configuration_id"]
    documents = [authorization, job, protected_plan, source_plan]
    if any(item["study_id"] != study_id for item in documents):
        raise ValueError("study identity changed across the lineage")
    configuration_ids = [
        authorization["configuration_id"],
        job["configuration_id"],
        protected_plan["configuration_id"],
        source_plan["independent_configuration"]["configuration_id"],
        dev_plan["sensitivity_configuration"]["configuration_id"],
    ]
    if any(value != configuration for value in configuration_ids):
        raise ValueError("configuration identity changed across the lineage")
    if authorization["status"] != "confirmed":
        raise ValueError("protected authorization is not confirmed")
    source = authorization_path.parent / authorization["authority"]["confirmation_source"]
    if sha256_file(source) != authorization["authority"]["confirmation_source_sha256"]:
        raise ValueError("authorization source changed")
    if sha256_file(authorization_path) != job["protected_authorization"]["receipt_sha256"]:
        raise ValueError("job does not bind the supplied authorization")
    if read_json(final_transfer)["authorization_receipt_sha256"] != sha256_file(authorization_path):
        raise ValueError("final transfer does not bind the supplied authorization")
    if state["status"] != "completed" or (protected / "ECLIPSED.json").exists():
        raise ValueError("protected run is incomplete or eclipsed")

    dev_started = parse_time(development_receipt["execution_started_at_utc"])
    protected_started = parse_time(state["started_at_utc"])
    authorized_at = parse_time(authorization["authority"]["confirmed_at_utc"])
    if not git_commit_time(dev_commit) < dev_started:
        raise ValueError("development configuration was not committed before execution")
    if not authorized_at < git_commit_time(protected_commit) < protected_started:
        raise ValueError("protected authorization/plan/run chronology is invalid")

    partitions: dict[str, Partition] = {}
    dev_db = development / "inputs/zoe_development_native_100.db"
    dev_manifest = development / "manifests/zoe_development_native_100.csv"
    dev_frame = load_table(dev_db, [KEY, REPORT])
    validate_manifest(dev_frame, dev_manifest)
    development_selection = read_json(
        development / "receipts/native-development-selection.json"
    )
    if sha256_file(dev_manifest) != development_selection["identity"]["manifest_sha256"]:
        raise ValueError("development manifest differs from the frozen receipt")
    partitions["zoe_development_100"] = Partition("development", dev_frame)

    configuration_payload = source_plan["independent_configuration"]
    raw_receipts: dict[str, dict] = {}
    for cohort in job["cohorts"]:
        cohort_id = cohort["cohort_id"]
        if cohort["role"] != "evaluation":
            raise ValueError("protected job contains a non-evaluation cohort")
        database = protected / cohort["database"]
        manifest = protected / cohort["manifest"]
        if sha256_file(database) != cohort["database_sha256"]:
            raise ValueError(f"database changed for {cohort_id}")
        if sha256_file(manifest) != cohort["manifest_sha256"]:
            raise ValueError(f"manifest changed for {cohort_id}")
        frame = load_table(database, [KEY, REPORT])
        validate_manifest(frame, manifest)
        if len(frame) != cohort["records"]:
            raise ValueError(f"population changed for {cohort_id}")
        partitions[cohort_id] = Partition("held_out_evaluation", frame)
        receipt = read_json(protected / f"products/{cohort_id}/raw.run.json")
        validate_raw_receipt(receipt, configuration_payload, cohort["records"])
        if receipt["dataset"]["sha256"] != cohort["database_sha256"]:
            raise ValueError(f"run receipt does not bind the dataset for {cohort_id}")
        if parse_time(receipt["execution_started_at_utc"]) < protected_started:
            raise ValueError("cohort inference predates the protected run")
        raw_receipts[cohort_id] = {
            "records_completed": receipt["reports_completed"],
            "dataset_sha256": receipt["dataset"]["sha256"],
            "output_sha256": receipt["output"]["sha256"],
        }

    partition_audit = audit_partitions(partitions)
    verified_transfer_files = validate_transfer_manifest(protected, final_transfer)
    result = {
        "schema_version": 1,
        "analysis_id": "jbhi-medgemma-v1-study-integrity-20260902",
        "status": "completed_public_safe_integrity_audit",
        "study_id": study_id,
        "configuration_id": configuration,
        "development_evaluation_separation": partition_audit,
        "lineage": {
            "development_plan_commit": dev_commit,
            "development_plan_committed_before_execution": True,
            "singleton_selection_without_reference_metric": True,
            "weights_or_training_change_allowed": False,
            "protected_authorization_sha256": sha256_file(authorization_path),
            "protected_plan_commit": protected_commit,
            "protected_plan_committed_before_inference": True,
            "source_study_plan_sha256": sha256_file(source_plan_path),
            "configuration_search_on_evaluation_allowed": False,
            "partial_reference_metrics_allowed": False,
            "final_transfer_manifest_sha256": sha256_file(final_transfer),
            "final_transfer_files_verified": verified_transfer_files,
        },
        "execution": {
            "inference_location": "local llama.cpp process",
            "model_artifact_access": "local_cache_only",
            "remote_inference_or_model_lookup": False,
            "report_or_prediction_egress": False,
            "classification_only": True,
            "cohorts": raw_receipts,
        },
        "interpretation": [
            "The MedGemma v1 configuration was fixed before its 100-report "
            "development transport run.",
            "The protected evaluation plan was fixed after authorization and "
            "before evaluation inference.",
            "No report key or normalized report text crosses the 100-report "
            "development and 1,894-report evaluation boundary.",
            "Later v2 and v2.1 development experiments were not applied to the "
            "protected evaluation cohorts.",
            "This establishes execution and split integrity, not patient "
            "independence or clinical validity.",
        ],
        "contains_report_keys_text_labels_or_predictions": False,
    }
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    result_path = output_root / "aggregate-study-integrity.json"
    completion_path = output_root / "COMPLETE.json"
    if completion_path.exists():
        complete = read_json(completion_path)
        if sha256_file(result_path) != complete["aggregate_sha256"]:
            raise ValueError("completed study-integrity output changed")
        if read_json(result_path) != result:
            raise ValueError("resume refused: study-integrity inputs changed")
        print("Completed study-integrity audit verified; no recomputation.")
        return result
    atomic_write_json(result_path, result)
    atomic_write_json(
        completion_path,
        {
            "schema_version": 1,
            "analysis_id": result["analysis_id"],
            "aggregate_sha256": sha256_file(result_path),
            "inference_performed": False,
            "distribution": "public_safe_after_author_review",
        },
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
