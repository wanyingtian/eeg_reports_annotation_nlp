from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from eeg_review.evidence_extraction import JSON_KEYS
from eeg_review.reason_traceability import (
    EvidenceUnit,
    audit_traceability,
    split_declared_sentences,
    split_semicolon_segments,
    structured_evidence_units,
    summarize_traceability,
)


def test_public_and_declared_segmentation_remain_distinct() -> None:
    text = "first clause; second clause. Third sentence."
    assert split_semicolon_segments(text) == ("first clause", "second clause. Third sentence.")
    assert split_declared_sentences(text) == ("first clause; second clause.", "Third sentence.")


def test_traceability_separates_verified_quotes_from_review_candidates() -> None:
    units = [
        EvidenceUnit(
            report_key="synthetic-1",
            category="Abnormality",
            report="Normal\nbackground. Focal slowing is present.",
            segments=("Focal slowing is present.", "Normal background.", "slowing focal"),
            segment_roles=("reason", "reason", "reason"),
            source_kind="synthetic",
        )
    ]

    def fuzzy(a: str, b: str) -> int:
        return 90 if a == "slowing focal" and "focal slowing" in b else 0

    rows = audit_traceability(units, fuzzy_ratio=fuzzy)
    assert [row["stage"] for row in rows] == [
        "verified_exact_substring",
        "candidate_whitespace_only",
        "candidate_fuzzy_sentence",
    ]
    summary = summarize_traceability(units, rows)
    assert summary["verified_exact_segments"] == 1
    assert summary["substantive_segments"] == 3
    assert summary["units_with_any_verified_segment"] == 1
    assert summary["units_with_all_segments_verified"] == 0
    assert summary["units_with_all_segments_located"] == 1


def test_semantic_candidates_never_become_verified_quotes() -> None:
    units = [
        EvidenceUnit(
            report_key="synthetic-1",
            category="Abnormality",
            report="Diffuse slowing is seen.",
            segments=("generalized slow activity",),
            segment_roles=("reason",),
            source_kind="synthetic",
        )
    ]

    def encoder(texts):
        return np.array([[1.0, 0.0] for _ in texts])

    rows = audit_traceability(units, encoder=encoder)
    assert rows[0]["stage"] == "candidate_semantic_sentence"
    assert rows[0]["verified_quote"] is False


def _fixed() -> str:
    return json.dumps(dict.fromkeys(JSON_KEYS, 1))


def _conditioned_response() -> str:
    return json.dumps(
        {key: {"decision": 1, "reasons": ["normal"]} for key in JSON_KEYS}
    )


def _independent_response() -> str:
    return json.dumps(
        {
            key: {
                "present_evidence": [],
                "absent_evidence": ["normal"],
                "qualification_evidence": [],
            }
            for key in JSON_KEYS
        }
    )


@pytest.mark.parametrize(
    "raw,expected_role",
    [
        (_conditioned_response(), "decision_conditioned_reason"),
        (_independent_response(), "absent_evidence"),
    ],
)
def test_adapter_supports_both_saved_evidence_contracts(raw: str, expected_role: str) -> None:
    evidence = pd.DataFrame(
        {
            "Hashed_ReportURN": ["synthetic-1"],
            "explanations": [raw],
            "fixed_classifications": [_fixed()],
        }
    )
    reports = pd.DataFrame({"Hashed_ReportURN": ["synthetic-1"], "Report": ["normal"]})
    units = structured_evidence_units(evidence, reports, source_kind="synthetic")
    assert len(units) == 5
    assert all(unit.segment_roles == (expected_role,) for unit in units)


@pytest.mark.parametrize(
    "mutation", ["duplicate_key", "missing_key", "decision_drift", "duplicate_json"]
)
def test_adapter_rejects_ambiguous_or_changed_inputs(mutation: str) -> None:
    raw = _conditioned_response()
    evidence = pd.DataFrame(
        {
            "Hashed_ReportURN": ["synthetic-1"],
            "explanations": [raw],
            "fixed_classifications": [_fixed()],
        }
    )
    reports = pd.DataFrame({"Hashed_ReportURN": ["synthetic-1"], "Report": ["normal"]})
    if mutation == "duplicate_key":
        evidence = pd.concat([evidence, evidence], ignore_index=True)
    elif mutation == "missing_key":
        evidence.loc[0, "Hashed_ReportURN"] = "missing"
    elif mutation == "decision_drift":
        changed = json.loads(raw)
        changed[JSON_KEYS[0]]["decision"] = 4
        evidence.loc[0, "explanations"] = json.dumps(changed)
    else:
        evidence.loc[0, "explanations"] = raw.replace(
            '"decision": 1', '"decision": 1, "decision": 1', 1
        )
    with pytest.raises(ValueError):
        structured_evidence_units(evidence, reports, source_kind="synthetic")


def test_summary_contains_no_keys_or_text() -> None:
    units = [
        EvidenceUnit(
            report_key="secret-key",
            category="Abnormality",
            report="secret report",
            segments=("secret",),
            segment_roles=("reason",),
            source_kind="synthetic",
        )
    ]
    summary = summarize_traceability(units, audit_traceability(units))
    serialized = json.dumps(summary)
    assert "secret-key" not in serialized
    assert "secret report" not in serialized


def test_adapter_retains_blank_and_fallback_outputs_as_explicit_exclusions() -> None:
    value = json.loads(_conditioned_response())
    value[JSON_KEYS[0]]["reasons"] = ["", "No specific mention in the report."]
    evidence = pd.DataFrame(
        {
            "Hashed_ReportURN": ["synthetic-1"],
            "explanations": [json.dumps(value)],
            "fixed_classifications": [_fixed()],
        }
    )
    reports = pd.DataFrame({"Hashed_ReportURN": ["synthetic-1"], "Report": ["normal"]})
    units = structured_evidence_units(evidence, reports, source_kind="synthetic")
    rows = audit_traceability(units)
    assert rows[0]["stage"] == "excluded_blank"
    assert rows[1]["stage"] == "excluded_declared_no_evidence"
    summary = summarize_traceability(units, rows)
    assert summary["excluded_blank_or_declared_absence_segments"] == 2
