#!/usr/bin/env python3
"""Create a public-safe author-review candidate from a completed protected native run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from eeg_review.protected_execution import (
    AuthorizationValidation,
    ProtectedExecutionLocked,
    assert_governed_run_active,
    authorize_plan_before_governed_access,
    validate_frozen_parent_receipts,
    validate_protected_job_binding,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "jbhi-02463-post-submission-medgemma-native-interface-sensitivity-v1"
CONFIGURATION_ID = (
    "jbhi-02463/comparator/medgemma-27b-text-it/configuration/"
    "independent-native-interface-q2-v1"
)
MODEL_A_ID = "medgemma-independent-native-interface-q2-v1"
COHORTS = {"zoe_evaluation_1395": 1395, "maria_evaluation_499": 499}
LABELS = ["Abnormality", "Focal Epi", "Focal Non-epi", "Gen Epi", "Gen Non-epi"]
COMPARATORS = {
    "submitted_mistral": "submitted-mistral",
    "reproduced_mistral": "reproduced-mistral",
    "second_annotator": "second-annotator",
}
PRIMARY_COMPARATORS = {"submitted_mistral", "reproduced_mistral"}
FORBIDDEN_FIELDS = {"Hashed_ReportURN", "Report", "report_text", "patient_key"}


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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in nested_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in nested_keys(child)}
    return set()


def validate_bootstrap(summary: dict[str, Any], source: str) -> None:
    bootstrap = summary.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError(f"{source}: bootstrap receipt is missing")
    if bootstrap.get("iterations") != 2000:
        raise ValueError(f"{source}: bootstrap iterations differ from the frozen plan")
    if bootstrap.get("seed") != 20260718:
        raise ValueError(f"{source}: bootstrap seed differs from the frozen plan")
    if bootstrap.get("unit") != "report" or bootstrap.get("cluster_column_supplied") is not False:
        raise ValueError(f"{source}: expected bounded report-level inference")


def verify_final_transfer_manifest(
    run_dir: Path,
    authorization: AuthorizationValidation,
    required_source_hashes: dict[str, str],
) -> dict[str, Any]:
    path = run_dir / "final-transfer-manifest.json"
    manifest = read_json(path)
    if manifest.get("study_id") != STUDY_ID:
        raise ValueError("Final transfer manifest study_id mismatch")
    if manifest.get("authorization_receipt_sha256") != authorization.receipt_sha256:
        raise ValueError("Final transfer manifest authorization binding mismatch")
    mismatches: list[str] = []
    manifested: dict[str, str] = {}
    for item in manifest.get("files", []):
        source = run_dir / item["path"]
        manifested[item["path"]] = item.get("sha256")
        if not source.is_file() or sha256_file(source) != item.get("sha256"):
            mismatches.append(item["path"])
    if mismatches:
        raise ValueError(f"Final transfer manifest has {len(mismatches)} mismatches")
    missing_sources = sorted(
        path
        for path, digest in required_source_hashes.items()
        if manifested.get(path) != digest
    )
    if missing_sources:
        raise ValueError(
            f"Final transfer manifest omits {len(missing_sources)} aggregate sources"
        )
    return {
        "sha256": sha256_file(path),
        "files": len(manifest.get("files", [])),
        "hash_mismatches": 0,
        "authorization_receipt_sha256": authorization.receipt_sha256,
    }


def effect_direction(interval: dict[str, Any]) -> str:
    low = float(interval["low"])
    high = float(interval["high"])
    if low > 0:
        return "higher"
    if high < 0:
        return "lower"
    return "interval_includes_zero"


def evaluation_record(summary: dict[str, Any], expected: int, source: str) -> dict[str, Any]:
    if summary.get("key_alignment", {}).get("exact_key_set") is not True:
        raise ValueError(f"{source}: evaluation lacks exact key alignment")
    if summary.get("reference_records") != expected or summary.get("matched_records") != expected:
        raise ValueError(f"{source}: evaluation population mismatch")
    validate_bootstrap(summary, source)
    labels: dict[str, Any] = {}
    for label in LABELS:
        item = summary.get("labels", {}).get(label)
        if not isinstance(item, dict):
            raise ValueError(f"{source}: label {label} is missing")
        if item.get("excluded_invalid_or_missing_pairs") != 0:
            raise ValueError(f"{source}: label {label} has excluded pairs")
        point = item["point_estimates"]
        if point.get("n") != expected:
            raise ValueError(f"{source}: label {label} has incorrect n")
        labels[label] = {
            "point_estimates": point,
            "confidence_intervals_95": item["confidence_intervals_95"],
        }
    return {
        "records": expected,
        "exact_key_set": True,
        "bootstrap": summary["bootstrap"],
        "labels": labels,
        "interpretation_limits": summary.get("interpretation_limits", []),
    }


def comparison_record(
    summary: dict[str, Any], expected: int, comparator_id: str, source: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if summary.get("key_alignment", {}).get("exact_three_way_key_set") is not True:
        raise ValueError(f"{source}: comparison lacks exact three-way key alignment")
    if summary.get("matched_records") != expected:
        raise ValueError(f"{source}: matched population mismatch")
    if summary.get("models") != {"a": MODEL_A_ID, "b": comparator_id}:
        raise ValueError(f"{source}: model identity or direction mismatch")
    validate_bootstrap(summary, source)
    multiplicity = summary.get("multiplicity", {})
    if multiplicity.get("method") != "holm":
        raise ValueError(f"{source}: multiplicity method differs from the frozen plan")

    labels: dict[str, Any] = {}
    claims: list[dict[str, Any]] = []
    for label in LABELS:
        item = summary.get("labels", {}).get(label)
        if not isinstance(item, dict) or item.get("n") != expected:
            raise ValueError(f"{source}: label {label} is missing or has incorrect n")
        if item.get("excluded_invalid_or_missing_three_way_pairs") != 0:
            raise ValueError(f"{source}: label {label} has excluded three-way pairs")
        effects = item["effects_a_minus_b"]
        intervals = item["paired_confidence_intervals_95"]
        labels[label] = {
            "n": expected,
            "model_a_point_estimates": item["model_a_point_estimates"],
            "model_b_point_estimates": item["model_b_point_estimates"],
            "effects_a_minus_b": effects,
            "paired_confidence_intervals_95": intervals,
            "discordant_correctness": item["discordant_correctness"],
        }
        for effect in [
            "core_accuracy_difference",
            "certainty_adjusted_accuracy_difference",
            "false_negative_rate_difference",
        ]:
            interval = intervals[effect]
            claims.append(
                {
                    "label": label,
                    "effect": effect,
                    "estimate_a_minus_b": effects[effect],
                    "ci_95": interval,
                    "direction_by_interval": effect_direction(interval),
                    "interval_unit": "report",
                    "bootstrap_iterations": 2000,
                    "status": "candidate_author_review_not_admitted",
                }
            )
    return (
        {
            "models": summary["models"],
            "matched_records": expected,
            "exact_three_way_key_set": True,
            "bootstrap": summary["bootstrap"],
            "multiplicity": multiplicity,
            "labels": labels,
            "interpretation_limits": summary.get("interpretation_limits", []),
        },
        claims,
    )


def build_receipt(
    run_dir: Path,
    study_plan_path: Path,
    tier_plan_path: Path,
    authorization: AuthorizationValidation,
) -> dict[str, Any]:
    study_plan = read_json(study_plan_path)
    tier_plan = read_json(tier_plan_path)
    validate_frozen_parent_receipts(study_plan, REPOSITORY_ROOT)
    if sha256_file(run_dir / "study-plan.json") != sha256_file(study_plan_path):
        raise ValueError("Governed run study-plan hash mismatch")

    job = read_json(run_dir / "job.json")
    validate_protected_job_binding(tier_plan, job, authorization)
    state = read_json(run_dir / "state.json")
    progress = read_json(run_dir / "receipts/progress/current.json")
    if state.get("status") != "completed":
        raise ValueError("Protected study execution is not complete")
    if progress.get("state_axes", {}).get("execution") != "completed":
        raise ValueError("Public-safe progress receipt is not complete")
    if progress.get("study_id") != STUDY_ID or progress.get("configuration_id") != CONFIGURATION_ID:
        raise ValueError("Completed progress receipt has the wrong scientific identity")
    if progress.get("execution_plan_sha256") != sha256_file(tier_plan_path):
        raise ValueError("Completed progress receipt tier-plan hash mismatch")
    if progress.get("completed_records") != sum(COHORTS.values()):
        raise ValueError("Completed progress receipt population mismatch")
    for cohort_id, expected in COHORTS.items():
        cohort = progress.get("cohorts", {}).get(cohort_id, {})
        if cohort.get("completed_records") != expected or cohort.get("target_records") != expected:
            raise ValueError(f"{cohort_id}: completed population mismatch")
        if cohort.get("invalid_structured_outputs") or cohort.get("duplicate_report_keys"):
            raise ValueError(f"{cohort_id}: invalid outputs or duplicate keys remain")

    evaluations: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    claims: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for cohort_id, expected in COHORTS.items():
        evaluation_path = run_dir / f"analysis/{cohort_id}/medgemma/evaluation_summary.json"
        evaluation = read_json(evaluation_path)
        evaluations[cohort_id] = evaluation_record(
            evaluation, expected, str(evaluation_path.relative_to(run_dir))
        )
        source_hashes[str(evaluation_path.relative_to(run_dir))] = sha256_file(evaluation_path)
        comparisons[cohort_id] = {}
        for comparator_name, comparator_id in COMPARATORS.items():
            comparison_path = run_dir / (
                f"analysis/{cohort_id}/vs_{comparator_name}/paired_comparison_summary.json"
            )
            comparison, comparison_claims = comparison_record(
                read_json(comparison_path),
                expected,
                comparator_id,
                str(comparison_path.relative_to(run_dir)),
            )
            comparisons[cohort_id][comparator_name] = comparison
            source_hashes[str(comparison_path.relative_to(run_dir))] = sha256_file(
                comparison_path
            )
            for claim in comparison_claims:
                claims.append(
                    {
                        "claim_id": (
                            f"native-protected/{cohort_id}/{comparator_name}/"
                            f"{claim['label']}/{claim['effect']}"
                        ),
                        "cohort_id": cohort_id,
                        "comparator": comparator_name,
                        "source_sha256": sha256_file(comparison_path),
                        **claim,
                    }
                )

    primary_core = [
        claim
        for claim in claims
        if claim["comparator"] in PRIMARY_COMPARATORS
        and claim["effect"] == "core_accuracy_difference"
    ]
    direction_counts = {
        direction: sum(claim["direction_by_interval"] == direction for claim in primary_core)
        for direction in ["higher", "lower", "interval_includes_zero"]
    }
    started = datetime.fromisoformat(state["started_at_utc"])
    completed = datetime.fromisoformat(state["completed_at_utc"])
    final_transfer_manifest = verify_final_transfer_manifest(
        run_dir, authorization, source_hashes
    )
    receipt = {
        "schema_version": 1,
        "evidence_id": "JBHI-02463-2026-MEDGEMMA-NATIVE-PROTECTED-RESULT-CANDIDATE",
        "status": "completed_validated_author_review_candidate",
        "manuscript_admission": "proposed_not_admitted",
        "study_id": STUDY_ID,
        "configuration_id": CONFIGURATION_ID,
        "evidence_layer": "post_submission_medgemma_native_interface_sensitivity",
        "lineage_boundaries": {
            "completed_matched_historical_q2": "immutable_primary_comparator_evidence",
            "this_native_interface_result": "separate_post_submission_sensitivity",
            "external_v5g": "separate_configuration_pending_exact_intake",
        },
        "authorization": {
            "receipt_sha256": authorization.receipt_sha256,
            "documentary_gate_validated": True,
            "legal_or_ethics_determination_made_by_software": False,
        },
        "execution": {
            "started_at_utc": state["started_at_utc"],
            "completed_at_utc": state["completed_at_utc"],
            "wall_seconds": (completed - started).total_seconds(),
            "completed_records": progress["completed_records"],
            "valid_structured_outputs": sum(
                item["valid_structured_outputs"] for item in progress["cohorts"].values()
            ),
            "invalid_structured_outputs": 0,
            "duplicate_report_keys": 0,
            "mean_inference_seconds_per_report": progress["observed_seconds_per_report"],
            "runtime": progress["runtime"],
            "repository": progress["repository"],
            "execution_plan_sha256": sha256_file(tier_plan_path),
            "study_plan_sha256": sha256_file(study_plan_path),
        },
        "final_transfer_manifest": final_transfer_manifest,
        "source_aggregate_sha256": source_hashes,
        "evaluations": evaluations,
        "comparisons": comparisons,
        "claim_candidates": claims,
        "cross_comparison_summary": {
            "primary_core_accuracy_claims": len(primary_core),
            "direction_counts_by_95_ci": direction_counts,
            "interpretation": (
                "Direction counts summarize prespecified paired report-level intervals; they "
                "are not a model ranking and require label-level and clinical interpretation."
            ),
        },
        "authoring_candidates": {
            "methods": (
                "We evaluated the frozen MedGemma-27B Q2_K native-chat sensitivity once on "
                "the 1,395-report Zoe and 499-report Maria evaluation cohorts. Comparisons "
                "used identical report keys, 2,000 paired report-bootstrap replicates with "
                "seed 20260718, exact report-level McNemar sensitivity tests, and Holm "
                "adjustment across the five-label family."
            ),
            "results": (
                f"Across the {len(primary_core)} prespecified core-accuracy comparisons with "
                "submitted and reproduced Mistral, the 95% paired report-level interval was "
                f"above zero for {direction_counts['higher']}, below zero for "
                f"{direction_counts['lower']}, and included zero for "
                f"{direction_counts['interval_includes_zero']}. Label-level effects, "
                "false-negative tradeoffs, and all unfavorable outcomes remain in the "
                "source-hashed claim table."
            ),
            "reviewer_response": (
                "We retained the completed matched-historical Q2 result and added the "
                "native-interface evaluation as a separately frozen post-submission "
                "sensitivity. No protected evaluation outcome selected or altered the "
                "configuration, and every favorable, null, and unfavorable result was retained."
            ),
            "status": "candidate_language_requires_author_and_clinical_review",
        },
        "limitations": [
            (
                "No confirmed patient key was available; intervals resample reports and "
                "do not establish patient independence."
            ),
            (
                "Exact McNemar tests are report-level sensitivity analyses and do not "
                "account for repeated reports within patients."
            ),
            (
                "This Q2_K native-interface configuration does not reproduce or "
                "characterize the external v5g configuration."
            ),
            (
                "Development evidence selected only structural transport; its observed "
                "accuracy is not independent manuscript evidence."
            ),
            "Aggregate interpretation, manuscript placement, and release require author review.",
        ],
        "privacy": {
            "public_safe_aggregate": True,
            "case_level_content_included": False,
            "case_identifiers_included": False,
        },
    }
    forbidden = sorted(FORBIDDEN_FIELDS & nested_keys(receipt))
    if forbidden:
        raise ValueError(f"Public-safe candidate contains forbidden fields: {forbidden}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--study-plan", type=Path, required=True)
    parser.add_argument("--tier-plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Public plans and the documentary receipt are validated before the governed run path.
    study_plan_path = args.study_plan.expanduser().resolve(strict=True)
    tier_plan_path = args.tier_plan.expanduser().resolve(strict=True)
    tier_plan = read_json(tier_plan_path)
    authorization_path = args.authorization.expanduser().resolve(strict=True)
    try:
        authorization = authorize_plan_before_governed_access(
            tier_plan, authorization_path
        )
        if authorization is None:
            raise ProtectedExecutionLocked(["protected authorization gate is absent"])
        validate_frozen_parent_receipts(read_json(study_plan_path), REPOSITORY_ROOT)
    except ProtectedExecutionLocked as error:
        print(
            json.dumps(
                {
                    "finalized": False,
                    "protected_evaluation_unlocked": False,
                    "blockers": list(error.blockers),
                },
                indent=2,
            )
        )
        raise SystemExit(2) from error

    run_dir = args.run_dir.expanduser().resolve(strict=True)
    assert_governed_run_active(run_dir)
    output = args.output.expanduser().absolute()
    receipt = build_receipt(
        run_dir, study_plan_path, tier_plan_path, authorization
    )
    atomic_json(output, receipt)
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
