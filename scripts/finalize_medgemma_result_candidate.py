#!/usr/bin/env python3
"""Create a public-safe aggregate receipt for a completed governed MedGemma study."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

LABELS = ["Abnormality", "Focal Epi", "Focal Non-epi", "Gen Epi", "Gen Non-epi"]
COHORTS = ["zoe_evaluation_1395", "maria_evaluation_499"]
COMPARATORS = ["submitted_mistral", "reproduced_mistral", "second_annotator"]
FORBIDDEN_KEYS = {"Hashed_ReportURN", "report_text", "patient_key", "keyed_predictions"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in all_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in all_keys(child)}
    return set()


def verify_transfer_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "final-transfer-manifest.json"
    manifest = read_json(path)
    mismatches = []
    for item in manifest["files"]:
        source = run_dir / item["path"]
        observed = sha256_file(source) if source.exists() else None
        if observed != item["sha256"]:
            mismatches.append(item["path"])
    if mismatches:
        raise ValueError(f"Final transfer manifest has {len(mismatches)} mismatches")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "files": len(manifest["files"]),
        "hash_mismatches": 0,
    }


def comparison_record(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary["key_alignment"]["exact_three_way_key_set"]:
        raise ValueError("Comparison lacks exact three-way key alignment")
    labels = {}
    for label in LABELS:
        item = summary["labels"][label]
        effects = item["effects_a_minus_b"]
        intervals = item["paired_confidence_intervals_95"]
        labels[label] = {
            "n": item["n"],
            "model_a_core_accuracy": item["model_a_point_estimates"]["core_accuracy"],
            "model_b_core_accuracy": item["model_b_point_estimates"]["core_accuracy"],
            "core_accuracy_difference": effects["core_accuracy_difference"],
            "core_accuracy_difference_ci_95": intervals["core_accuracy_difference"],
            "certainty_adjusted_accuracy_difference": effects[
                "certainty_adjusted_accuracy_difference"
            ],
            "certainty_adjusted_accuracy_difference_ci_95": intervals[
                "certainty_adjusted_accuracy_difference"
            ],
            "false_negative_rate_difference": effects["false_negative_rate_difference"],
            "false_negative_rate_difference_ci_95": intervals[
                "false_negative_rate_difference"
            ],
            "core_mcnemar_holm_p_value": item["discordant_correctness"]["core_accuracy"][
                "multiplicity_adjusted_p_value"
            ],
        }
    return {
        "models": summary["models"],
        "matched_records": summary["matched_records"],
        "exact_three_way_key_set": True,
        "bootstrap": summary["bootstrap"],
        "multiplicity": summary["multiplicity"],
        "labels": labels,
    }


def build_receipt(run_dir: Path) -> dict[str, Any]:
    state = read_json(run_dir / "state.json")
    progress = read_json(run_dir / "receipts/progress/current.json")
    if state["status"] != "completed" or progress["state_axes"]["execution"] != "completed":
        raise ValueError("Study execution is not complete")
    if int(progress["completed_records"]) != int(progress["target_records"]):
        raise ValueError("Study does not have exact final population coverage")
    if any(
        cohort["invalid_structured_outputs"] or cohort["duplicate_report_keys"]
        for cohort in progress["cohorts"].values()
    ):
        raise ValueError("Completed study contains invalid outputs or duplicate keys")

    sources = {}
    comparisons = {}
    for cohort in COHORTS:
        evaluation_path = run_dir / f"analysis/{cohort}/medgemma/evaluation_summary.json"
        evaluation = read_json(evaluation_path)
        if not evaluation["key_alignment"]["exact_key_set"]:
            raise ValueError(f"{cohort}: evaluation key set is not exact")
        sources[str(evaluation_path.relative_to(run_dir))] = sha256_file(evaluation_path)
        cohort_comparisons = {}
        for comparator in COMPARATORS:
            path = run_dir / (
                f"analysis/{cohort}/vs_{comparator}/paired_comparison_summary.json"
            )
            summary = read_json(path)
            sources[str(path.relative_to(run_dir))] = sha256_file(path)
            cohort_comparisons[comparator] = comparison_record(summary)
        comparisons[cohort] = cohort_comparisons

    all_primary_core_intervals_below_zero = all(
        comparison["labels"][label]["core_accuracy_difference_ci_95"]["high"] < 0
        for cohort in comparisons.values()
        for comparator, comparison in cohort.items()
        if comparator in {"submitted_mistral", "reproduced_mistral"}
        for label in LABELS
    )
    started = datetime.fromisoformat(state["started_at_utc"])
    completed = datetime.fromisoformat(state["completed_at_utc"])
    receipt = {
        "schema_version": 1,
        "evidence_id": "JBHI-02463-2026-MEDGEMMA-RESULT-CANDIDATE-2026-08-29",
        "status": "completed_validated_author_review_candidate",
        "manuscript_admission": "proposed_not_admitted",
        "study_id": progress["study_id"],
        "configuration_id": progress["configuration_id"],
        "execution_plan_id": progress["execution_plan_id"],
        "runtime": progress["runtime"],
        "repository": progress["repository"],
        "execution": {
            "started_at_utc": state["started_at_utc"],
            "completed_at_utc": state["completed_at_utc"],
            "wall_seconds": (completed - started).total_seconds(),
            "completed_records": progress["completed_records"],
            "target_records": progress["target_records"],
            "valid_structured_outputs": sum(
                item["valid_structured_outputs"] for item in progress["cohorts"].values()
            ),
            "invalid_structured_outputs": sum(
                item["invalid_structured_outputs"] for item in progress["cohorts"].values()
            ),
            "duplicate_report_keys": sum(
                item["duplicate_report_keys"] for item in progress["cohorts"].values()
            ),
            "mean_inference_seconds_per_report": progress["observed_seconds_per_report"],
        },
        "final_transfer_manifest": verify_transfer_manifest(run_dir),
        "source_aggregate_sha256": sources,
        "comparisons": comparisons,
        "cross_comparison_summary": {
            "primary_comparisons": ["submitted_mistral", "reproduced_mistral"],
            "all_20_primary_core_accuracy_intervals_below_zero": (
                all_primary_core_intervals_below_zero
            ),
            "interpretation": (
                "In the independently specified Q2 matched-historical-interface configuration, "
                "MedGemma had lower core accuracy than both submitted and reproduced Mistral "
                "for all five labels in both cohorts. Some false-negative-rate differences "
                "favored MedGemma, alongside materially lower specificity and overcalling."
            ),
        },
        "limitations": [
            (
                "This independently specified Q2 matched-interface configuration is not "
                "Vasily's v5g producing configuration."
            ),
            (
                "No patient key was available; intervals and McNemar tests operate at report "
                "level and do not establish patient independence."
            ),
            (
                "The historical raw-completion prompt was preserved for matched-interface "
                "comparison and may not be MedGemma's optimal native chat interface."
            ),
            (
                "Q2 quantization and configuration-specific results do not characterize every "
                "MedGemma deployment."
            ),
            (
                "Author and clinical review are required before any aggregate is admitted to "
                "the manuscript."
            ),
        ],
        "privacy": {
            "public_safe_aggregate": True,
            "contains_report_text": False,
            "contains_report_or_patient_keys": False,
            "contains_keyed_predictions": False,
        },
    }
    forbidden = sorted(FORBIDDEN_KEYS & all_keys(receipt))
    if forbidden:
        raise ValueError(f"Public-safe result receipt contains forbidden keys: {forbidden}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    output = args.output.expanduser().absolute()
    payload = build_receipt(run_dir)
    atomic_json(output, payload)
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
