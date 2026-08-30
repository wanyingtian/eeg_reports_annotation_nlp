from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_tiered_medgemma_study.py"
SPEC = importlib.util.spec_from_file_location("run_tiered_medgemma_study", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
inspect_output = MODULE.inspect_output
sha256_file = MODULE.sha256_file
validate_plan = MODULE.validate_plan
ROOT = Path(__file__).resolve().parents[1]


def test_tier_plan_reaches_every_cohort_and_has_result_blind_partial_reporting(
    tmp_path: Path,
) -> None:
    study_plan = tmp_path / "study-plan.json"
    study_plan.write_text("{}\n", encoding="utf-8")

    cohorts = {"development": 100, "zoe": 1395, "maria": 499}
    job = {
        "study_id": "study",
        "configuration_id": "configuration",
        "cohorts": [
            {"cohort_id": name, "records": count} for name, count in cohorts.items()
        ],
    }
    plan = {
        "status": "preregistered_before_inference",
        "study_id": "study",
        "configuration_id": "configuration",
        "source_study_plan_sha256": sha256_file(study_plan),
        "tiers": [
            {"tier_id": "T0", "targets": {"development": 100, "zoe": 0, "maria": 0}},
            {
                "tier_id": "T1",
                "targets": {"development": 100, "zoe": 50, "maria": 25},
            },
            {
                "tier_id": "T2",
                "targets": {"development": 100, "zoe": 1395, "maria": 499},
            },
        ],
        "post_inference": {"partial_reference_metrics_allowed": False},
    }
    validate_plan(plan, job, study_plan)


def test_tier_plan_rejects_non_monotonic_targets(tmp_path: Path) -> None:
    study_plan = tmp_path / "study-plan.json"
    study_plan.write_text("{}\n", encoding="utf-8")

    job = {
        "study_id": "study",
        "configuration_id": "configuration",
        "cohorts": [{"cohort_id": "zoe", "records": 10}],
    }
    plan = {
        "status": "preregistered_before_inference",
        "study_id": "study",
        "configuration_id": "configuration",
        "source_study_plan_sha256": sha256_file(study_plan),
        "tiers": [
            {"tier_id": "T0", "targets": {"zoe": 8}},
            {"tier_id": "T1", "targets": {"zoe": 7}},
            {"tier_id": "T2", "targets": {"zoe": 10}},
        ],
        "post_inference": {"partial_reference_metrics_allowed": False},
    }
    with pytest.raises(ValueError, match="non-monotonic"):
        validate_plan(plan, job, study_plan)


def test_tier_plan_rejects_runtime_amendment_mismatch(tmp_path: Path) -> None:
    study_plan = tmp_path / "study-plan.json"
    study_plan.write_text("{}\n", encoding="utf-8")
    job = {
        "study_id": "study",
        "configuration_id": "configuration",
        "cohorts": [{"cohort_id": "zoe", "records": 10}],
        "runtime_amendment": {
            "amendment_id": "amendment",
            "sha256": "prepared",
            "runtime_profile_id": "profile",
        },
    }
    plan = {
        "status": "preregistered_before_inference",
        "study_id": "study",
        "configuration_id": "configuration",
        "source_study_plan_sha256": sha256_file(study_plan),
        "runtime_amendment": {
            "amendment_id": "amendment",
            "sha256": "different",
            "runtime_profile_id": "profile",
        },
        "tiers": [{"tier_id": "T0", "targets": {"zoe": 10}}],
        "post_inference": {"partial_reference_metrics_allowed": False},
    }

    with pytest.raises(ValueError, match="runtime amendments differ"):
        validate_plan(plan, job, study_plan)


def test_operational_output_summary_excludes_keys_and_reference_metrics(tmp_path: Path) -> None:
    output = tmp_path / "raw.csv"
    rows = [
        {
            "Hashed_ReportURN": "secret-key-a",
            "classifications": json.dumps(
                {
                    "focal_epileptiform_activity": 1,
                    "generalized_epileptiform_activity": 1,
                    "focal_non_epileptiform_activity": 4,
                    "generalized_non_epileptiform_activity": 1,
                    "abnormality": 4,
                }
            ),
            "classify_elapsed_seconds": "4.0",
            "classify_prompt_tokens": "100",
            "classify_completion_tokens": "20",
        },
        {
            "Hashed_ReportURN": "secret-key-b",
            "classifications": "not-json",
            "classify_elapsed_seconds": "6.0",
            "classify_prompt_tokens": "110",
            "classify_completion_tokens": "21",
        },
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = inspect_output(output)
    serialized = json.dumps(summary)
    assert summary["completed_records"] == 2
    assert summary["valid_structured_outputs"] == 1
    assert summary["invalid_structured_outputs"] == 1
    assert summary["classification_seconds_mean"] == 5.0
    assert "secret-key" not in serialized
    assert "accuracy" not in serialized


def test_native_protected_tier_plan_binds_exact_evaluation_populations() -> None:
    study_plan = (
        ROOT
        / "review/model-receipts/medgemma-native-protected-comparator.preregistered.json"
    )
    tier_plan = json.loads(
        (
            ROOT
            / "review/model-receipts/medgemma-native-protected-tiered-execution.preregistered.json"
        ).read_text(encoding="utf-8")
    )
    job = {
        "study_id": tier_plan["study_id"],
        "configuration_id": tier_plan["configuration_id"],
        "cohorts": [
            {"cohort_id": "zoe_evaluation_1395", "records": 1395},
            {"cohort_id": "maria_evaluation_499", "records": 499},
        ],
        "runtime_amendment": tier_plan["runtime_amendment"],
    }

    validate_plan(tier_plan, job, study_plan)
    assert tier_plan["authorization_gate"]["required"] is True
    assert tier_plan["post_inference"]["partial_reference_metrics_allowed"] is False
    assert set(tier_plan["tiers"][-1]["targets"]) == {
        "zoe_evaluation_1395",
        "maria_evaluation_499",
    }
