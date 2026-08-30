from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from eeg_review.manuscript_admission import validate_manuscript_admission

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_medgemma_native_author_bundle.py"
TEMPLATE = ROOT / (
    "review/model-receipts/medgemma-native-manuscript-admission.template.json"
)
SPEC = importlib.util.spec_from_file_location("render_medgemma_native_author_bundle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate() -> dict:
    claims = []
    comparisons = {}
    source_hashes = {}
    all_comparators = [*MODULE.COMPARATORS, ("second_annotator", "Second Annotator")]
    for cohort_id, _cohort_label, records in MODULE.COHORTS:
        comparisons[cohort_id] = {}
        for comparator, _comparator_label in all_comparators:
            source_hash = (str(len(source_hashes) % 10) * 64)[:64]
            source_hashes[f"analysis/{cohort_id}/{comparator}.json"] = source_hash
            label_records = {}
            for label in MODULE.LABELS:
                label_records[label] = {
                    "model_a_point_estimates": {"core_accuracy": 0.8},
                    "model_b_point_estimates": {"core_accuracy": 0.7},
                    "effects_a_minus_b": {
                        "core_accuracy_difference": 0.1,
                        "certainty_adjusted_accuracy_difference": 0.05,
                        "false_negative_rate_difference": -0.1,
                    },
                    "paired_confidence_intervals_95": {
                        "core_accuracy_difference": {"low": 0.02, "high": 0.18},
                        "certainty_adjusted_accuracy_difference": {
                            "low": -0.01,
                            "high": 0.11,
                        },
                        "false_negative_rate_difference": {
                            "low": -0.2,
                            "high": -0.01,
                        },
                    },
                    "discordant_correctness": {
                        "core_accuracy": {"multiplicity_adjusted_p_value": 0.04}
                    },
                }
                for effect, interval in label_records[label][
                    "paired_confidence_intervals_95"
                ].items():
                    claims.append(
                        {
                            "claim_id": (
                                f"native-protected/{cohort_id}/{comparator}/{label}/{effect}"
                            ),
                            "cohort_id": cohort_id,
                            "comparator": comparator,
                            "label": label,
                            "effect": effect,
                            "estimate_a_minus_b": label_records[label][
                                "effects_a_minus_b"
                            ][effect],
                            "ci_95": interval,
                            "direction_by_interval": MODULE.interval_direction(interval),
                            "interval_unit": "report",
                            "bootstrap_iterations": 2000,
                            "source_sha256": source_hash,
                            "status": "candidate_author_review_not_admitted",
                        }
                    )
            comparisons[cohort_id][comparator] = {
                "matched_records": records,
                "labels": label_records,
            }
    return {
        "evidence_id": MODULE.EVIDENCE_ID,
        "configuration_id": MODULE.CONFIGURATION_ID,
        "status": "completed_validated_author_review_candidate",
        "manuscript_admission": "proposed_not_admitted",
        "privacy": {
            "public_safe_aggregate": True,
            "case_level_content_included": False,
            "case_identifiers_included": False,
        },
        "comparisons": comparisons,
        "claim_candidates": claims,
        "source_aggregate_sha256": source_hashes,
        "authoring_candidates": {
            "methods": "Methods used MedGemma_Q2 and 95% intervals.",
            "results": "Results retained all outcomes.",
            "reviewer_response": "We retained every prespecified result.",
        },
    }


def approved_admission(
    candidate_sha256: str, approved_claim_ids: set[str]
) -> dict:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "approved",
            "candidate_receipt_sha256": candidate_sha256,
        }
    )
    payload["confirmation"] = {
        "role": "corresponding_author",
        "name_or_record": "author-group decision record",
        "source": "governed correspondence",
        "source_sha256": "a" * 64,
        "confirmed_at_utc": "2026-08-30T00:00:00Z",
    }
    payload["decisions"].update(
        {
            "aggregate_release_approved": True,
            "approved_destinations": ["supplement", "reviewer_response"],
            "approved_claim_ids": sorted(approved_claim_ids),
            "methods_language_approved": True,
            "results_language_approved": True,
            "reviewer_response_language_approved": True,
        }
    )
    payload["distribution"][
        "aggregate_candidate"
    ] = "author_approved_for_named_destinations"
    return payload


def test_author_working_bundle_is_complete_and_deterministic() -> None:
    payload = candidate()
    outputs_a, receipt_a = MODULE.build_bundle(payload, "b" * 64, None)
    outputs_b, receipt_b = MODULE.build_bundle(payload, "b" * 64, None)

    assert outputs_a == outputs_b
    assert receipt_a == receipt_b
    assert receipt_a["mode"] == "author_working"
    assert receipt_a["primary_claims_rendered"] == 20
    assert receipt_a["full_claim_ledger_rows"] == 90
    assert "Author-working; not admitted" in outputs_a[MODULE.FILES["table"]]
    assert len(outputs_a[MODULE.FILES["ledger"]].splitlines()) == 91
    assert r"MedGemma\_Q2" in outputs_a[MODULE.FILES["methods"]]


def test_admission_is_hash_bound_and_exactly_covers_primary_claims(
    tmp_path: Path,
) -> None:
    payload = candidate()
    candidate_hash = "b" * 64
    primary = MODULE.primary_claim_ids(payload)
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(
        json.dumps(approved_admission(candidate_hash, primary)) + "\n",
        encoding="utf-8",
    )

    outputs, receipt = MODULE.build_bundle(payload, candidate_hash, admission_path)

    assert receipt["mode"] == "admitted_for_named_destinations"
    assert receipt["approved_destinations"] == ["reviewer_response", "supplement"]
    assert "Author-approved aggregate" in outputs[MODULE.FILES["table"]]


def test_admission_rejects_partial_primary_claim_selection(tmp_path: Path) -> None:
    payload = candidate()
    primary = MODULE.primary_claim_ids(payload)
    primary.pop()
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(
        json.dumps(approved_admission("b" * 64, primary)) + "\n",
        encoding="utf-8",
    )

    result = validate_manuscript_admission(
        admission_path,
        candidate_sha256="b" * 64,
        required_claim_ids=MODULE.primary_claim_ids(payload),
    )

    assert result.valid is False
    assert any("exactly cover" in blocker for blocker in result.blockers)


def test_candidate_cannot_drop_an_unfavorable_or_null_claim() -> None:
    payload = candidate()
    payload["claim_candidates"].pop()
    with pytest.raises(ValueError, match="all 90"):
        MODULE.build_bundle(payload, "b" * 64, None)


def test_candidate_cannot_relabel_an_interval_direction() -> None:
    payload = candidate()
    payload["claim_candidates"][0]["direction_by_interval"] = "lower"
    with pytest.raises(ValueError, match="interval direction mismatch"):
        MODULE.build_bundle(payload, "b" * 64, None)


def test_pending_admission_template_fails_closed() -> None:
    result = validate_manuscript_admission(
        TEMPLATE,
        candidate_sha256="b" * 64,
        required_claim_ids=MODULE.primary_claim_ids(candidate()),
    )
    assert result.valid is False
    assert "status must be approved" in result.blockers
