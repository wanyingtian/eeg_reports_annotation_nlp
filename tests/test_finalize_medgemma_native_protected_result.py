from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eeg_review.protected_execution import authorize_plan_before_governed_access

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/finalize_medgemma_native_protected_result.py"
STUDY_PLAN = (
    ROOT
    / "review/model-receipts/medgemma-native-protected-comparator.preregistered.json"
)
TIER_PLAN = ROOT / (
    "review/model-receipts/"
    "medgemma-native-protected-tiered-execution.preregistered.json"
)
AUTHORIZATION_TEMPLATE = ROOT / (
    "review/model-receipts/medgemma-native-protected-authorization.template.json"
)
SPEC = importlib.util.spec_from_file_location(
    "finalize_medgemma_native_protected_result", SCRIPT
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def confirmed_authorization(tmp_path: Path) -> tuple[Path, object]:
    payload = json.loads(AUTHORIZATION_TEMPLATE.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "confirmed",
            "authorization_id": "synthetic-test-authorization",
            "coverage_statement": "Synthetic test fixture only.",
        }
    )
    payload["authority"] = {
        "role": "approved_study_record",
        "name_or_record": "synthetic test record",
        "confirmation_source": "synthetic fixture",
        "confirmation_source_sha256": "a" * 64,
        "confirmed_at_utc": "2026-08-30T00:00:00Z",
    }
    payload["scope"]["already_transferred_deidentified_reports"] = True
    payload["scope"]["secondary_use_covered"] = True
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    tier = json.loads(TIER_PLAN.read_text(encoding="utf-8"))
    validation = authorize_plan_before_governed_access(tier, path)
    assert validation is not None
    return path, validation


def comparison_summary(records: int, comparator: str) -> dict:
    item = {
        "n": records,
        "excluded_invalid_or_missing_three_way_pairs": 0,
        "model_a_point_estimates": {"core_accuracy": 0.8},
        "model_b_point_estimates": {"core_accuracy": 0.7},
        "effects_a_minus_b": {
            "core_accuracy_difference": 0.1,
            "certainty_adjusted_accuracy_difference": 0.05,
            "false_negative_rate_difference": -0.1,
        },
        "paired_confidence_intervals_95": {
            "core_accuracy_difference": {"low": 0.02, "high": 0.18},
            "certainty_adjusted_accuracy_difference": {"low": -0.01, "high": 0.11},
            "false_negative_rate_difference": {"low": -0.2, "high": -0.01},
        },
        "discordant_correctness": {
            "core_accuracy": {"multiplicity_adjusted_p_value": 0.04},
            "certainty_adjusted_accuracy": {
                "multiplicity_adjusted_p_value": 0.2
            },
        },
    }
    return {
        "schema_version": 1,
        "models": {"a": MODULE.MODEL_A_ID, "b": comparator},
        "matched_records": records,
        "key_alignment": {"exact_three_way_key_set": True},
        "bootstrap": {
            "iterations": 2000,
            "seed": 20260718,
            "unit": "report",
            "cluster_column_supplied": False,
        },
        "multiplicity": {"method": "holm"},
        "labels": {label: item for label in MODULE.LABELS},
        "interpretation_limits": ["report-level only"],
    }


def evaluation_summary(records: int) -> dict:
    item = {
        "excluded_invalid_or_missing_pairs": 0,
        "point_estimates": {"n": records, "core_accuracy": 0.8},
        "confidence_intervals_95": {
            "core_accuracy": {"low": 0.75, "high": 0.85}
        },
    }
    return {
        "schema_version": 1,
        "reference_records": records,
        "matched_records": records,
        "key_alignment": {"exact_key_set": True},
        "bootstrap": {
            "iterations": 2000,
            "seed": 20260718,
            "unit": "report",
            "cluster_column_supplied": False,
        },
        "labels": {label: item for label in MODULE.LABELS},
        "interpretation_limits": ["report-level only"],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepared_completed_run(tmp_path: Path, authorization) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "study-plan.json").write_bytes(STUDY_PLAN.read_bytes())
    tier = json.loads(TIER_PLAN.read_text(encoding="utf-8"))
    commands = []
    for cohort_id in MODULE.COHORTS:
        commands.append(
            {
                "stage": f"{cohort_id}_inference",
                "command": [
                    "python",
                    "pipeline.py",
                    "--classification-only",
                    "--local-model-only",
                    "--classification-interface",
                    "native_chat",
                ],
            }
        )
    write_json(
        run_dir / "job.json",
        {
            "study_id": MODULE.STUDY_ID,
            "configuration_id": MODULE.CONFIGURATION_ID,
            "cohorts": [
                {"cohort_id": cohort_id, "records": records}
                for cohort_id, records in MODULE.COHORTS.items()
            ],
            "commands": commands,
            "protected_authorization": {
                "receipt_sha256": authorization.receipt_sha256,
                "study_id": MODULE.STUDY_ID,
                "configuration_id": MODULE.CONFIGURATION_ID,
            },
            "frozen_parent_receipts": tier["frozen_parent_receipts"],
        },
    )
    write_json(
        run_dir / "state.json",
        {
            "status": "completed",
            "started_at_utc": "2026-08-30T01:00:00+00:00",
            "completed_at_utc": "2026-08-30T10:00:00+00:00",
        },
    )
    write_json(
        run_dir / "receipts/progress/current.json",
        {
            "study_id": MODULE.STUDY_ID,
            "configuration_id": MODULE.CONFIGURATION_ID,
            "execution_plan_sha256": MODULE.sha256_file(TIER_PLAN),
            "state_axes": {"execution": "completed"},
            "completed_records": sum(MODULE.COHORTS.values()),
            "observed_seconds_per_report": 17.5,
            "runtime": {"runtime_profile_id": "metal-profile"},
            "repository": {"revision": "a" * 40, "worktree_dirty": False},
            "cohorts": {
                cohort_id: {
                    "completed_records": records,
                    "target_records": records,
                    "valid_structured_outputs": records,
                    "invalid_structured_outputs": 0,
                    "duplicate_report_keys": 0,
                }
                for cohort_id, records in MODULE.COHORTS.items()
            },
        },
    )
    for cohort_id, records in MODULE.COHORTS.items():
        write_json(
            run_dir / f"analysis/{cohort_id}/medgemma/evaluation_summary.json",
            evaluation_summary(records),
        )
        for name, comparator in MODULE.COMPARATORS.items():
            write_json(
                run_dir
                / f"analysis/{cohort_id}/vs_{name}/paired_comparison_summary.json",
                comparison_summary(records, comparator),
            )
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "sha256": MODULE.sha256_file(path),
                }
            )
    write_json(
        run_dir / "final-transfer-manifest.json",
        {
            "study_id": MODULE.STUDY_ID,
            "authorization_receipt_sha256": authorization.receipt_sha256,
            "files": files,
        },
    )
    return run_dir


def test_interval_direction_is_neutral() -> None:
    assert MODULE.effect_direction({"low": 0.01, "high": 0.2}) == "higher"
    assert MODULE.effect_direction({"low": -0.2, "high": -0.01}) == "lower"
    assert MODULE.effect_direction({"low": -0.1, "high": 0.1}) == "interval_includes_zero"


def test_comparison_rejects_analysis_seed_drift() -> None:
    summary = comparison_summary(10, "submitted-mistral")
    summary["bootstrap"]["seed"] = 7
    with pytest.raises(ValueError, match="seed differs"):
        MODULE.comparison_record(summary, 10, "submitted-mistral", "fixture")


def test_completed_run_yields_source_hashed_author_review_candidate(
    tmp_path: Path,
) -> None:
    _authorization_path, authorization = confirmed_authorization(tmp_path)
    run_dir = prepared_completed_run(tmp_path, authorization)

    receipt = MODULE.build_receipt(
        run_dir, STUDY_PLAN, TIER_PLAN, authorization
    )

    assert receipt["status"] == "completed_validated_author_review_candidate"
    assert receipt["manuscript_admission"] == "proposed_not_admitted"
    assert len(receipt["claim_candidates"]) == 90
    assert receipt["cross_comparison_summary"]["primary_core_accuracy_claims"] == 20
    assert receipt["cross_comparison_summary"]["direction_counts_by_95_ci"] == {
        "higher": 20,
        "lower": 0,
        "interval_includes_zero": 0,
    }
    assert receipt["final_transfer_manifest"]["hash_mismatches"] == 0
    assert not (MODULE.FORBIDDEN_FIELDS & MODULE.nested_keys(receipt))


def test_finalizer_rejects_manifest_that_omits_an_aggregate_source(
    tmp_path: Path,
) -> None:
    _authorization_path, authorization = confirmed_authorization(tmp_path)
    run_dir = prepared_completed_run(tmp_path, authorization)
    manifest_path = run_dir / "final-transfer-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        item
        for item in manifest["files"]
        if item["path"]
        != "analysis/zoe_evaluation_1395/medgemma/evaluation_summary.json"
    ]
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="omits 1 aggregate sources"):
        MODULE.build_receipt(run_dir, STUDY_PLAN, TIER_PLAN, authorization)


def test_finalizer_fails_on_authorization_before_nonexistent_run_path(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            "/definitely/not/a/governed/run",
            "--study-plan",
            str(STUDY_PLAN),
            "--tier-plan",
            str(TIER_PLAN),
            "--authorization",
            str(AUTHORIZATION_TEMPLATE),
            "--output",
            str(tmp_path / "candidate.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "status must be confirmed" in result.stdout
    assert "FileNotFoundError" not in result.stderr
