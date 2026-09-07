from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from eeg_review.focal_signatures import (
    CUE_TERMS,
    audit_cohort,
    cue_segments,
    transition_group,
)


def write_db(path: Path, table: str, frame: pd.DataFrame) -> None:
    with sqlite3.connect(path) as connection:
        frame.to_sql(table, connection, index=False)


def test_registered_cues_require_terms_in_the_same_segment() -> None:
    matches = cue_segments(
        "History of seizures. No epileptiform discharges. "
        "Focal slowing is present; benign sharp transients are also seen."
    )
    assert matches["history_or_indication_with_seizure_segment"] == [
        "History of seizures."
    ]
    assert matches["negated_epileptiform_segment"] == [
        "No epileptiform discharges."
    ]
    assert matches["focal_slowing_or_attenuation_segment"] == [
        "Focal slowing is present;"
    ]
    assert matches["benign_or_artifact_with_sharp_segment"] == [
        "benign sharp transients are also seen."
    ]
    assert not matches["focal_epileptiform_wording_segment"]


@pytest.mark.parametrize(
    ("reference", "mistral", "medgemma", "expected"),
    [
        (1, 1, 1, "both_correct_negative"),
        (1, 1, 3, "medgemma_only_false_positive"),
        (1, 3, 1, "mistral_only_false_positive"),
        (1, 3, 3, "both_false_positive"),
        (3, 3, 3, "both_true_positive"),
        (3, 1, 3, "medgemma_only_true_positive"),
        (3, 3, 1, "mistral_only_true_positive"),
        (3, 1, 1, "both_false_negative"),
    ],
)
def test_transition_group(
    reference: int, mistral: int, medgemma: int, expected: str
) -> None:
    assert transition_group(reference, mistral, medgemma) == expected


def test_audit_keeps_source_segments_governed_and_aggregate_key_free(tmp_path: Path) -> None:
    reference = pd.DataFrame(
        {
            "Hashed_ReportURN": ["case-a", "case-b"],
            "Report": ["No epileptiform discharges.", "Focal spikes are present."],
            "Focal Epi": [1, 3],
        }
    )
    mistral = pd.DataFrame(
        {"Hashed_ReportURN": ["case-a", "case-b"], "Focal Epi": [1, 1]}
    )
    medgemma = pd.DataFrame(
        {"Hashed_ReportURN": ["case-a", "case-b"], "Focal Epi": [3, 3]}
    )
    reference_path = tmp_path / "reference.db"
    mistral_path = tmp_path / "mistral.db"
    medgemma_path = tmp_path / "medgemma.db"
    write_db(reference_path, "reports", reference)
    write_db(mistral_path, "classifications", mistral)
    write_db(medgemma_path, "classifications", medgemma)

    ledger, aggregate = audit_cohort(
        cohort="synthetic",
        expected_records=2,
        reference_path=reference_path,
        mistral_path=mistral_path,
        medgemma_path=medgemma_path,
    )
    assert ledger.loc[0, "transition_group"] == "medgemma_only_false_positive"
    assert ledger.loc[0, "negated_epileptiform_segment"]
    assert "No epileptiform" in ledger.loc[
        0, "negated_epileptiform_segment_source_segments"
    ]
    assert aggregate["transition_groups"]["medgemma_only_false_positive"][
        "cue_counts"
    ]["negated_epileptiform_segment"] == 1
    serialized = json.dumps(aggregate)
    assert "case-a" not in serialized
    assert "No epileptiform" not in serialized
    assert set(CUE_TERMS) == set(
        aggregate["transition_groups"]["both_correct_negative"]["cue_counts"]
    )


def test_audit_rejects_missing_reference_key(tmp_path: Path) -> None:
    reference = pd.DataFrame(
        {"Hashed_ReportURN": ["a"], "Report": ["Normal."], "Focal Epi": [1]}
    )
    mistral = pd.DataFrame({"Hashed_ReportURN": ["a"], "Focal Epi": [1]})
    medgemma = pd.DataFrame({"Hashed_ReportURN": ["b"], "Focal Epi": [1]})
    reference_path = tmp_path / "reference.db"
    mistral_path = tmp_path / "mistral.db"
    medgemma_path = tmp_path / "medgemma.db"
    write_db(reference_path, "reports", reference)
    write_db(mistral_path, "classifications", mistral)
    write_db(medgemma_path, "classifications", medgemma)
    with pytest.raises(ValueError, match="omit frozen reference keys"):
        audit_cohort(
            cohort="synthetic",
            expected_records=1,
            reference_path=reference_path,
            mistral_path=mistral_path,
            medgemma_path=medgemma_path,
        )


def test_audit_records_and_excludes_prediction_only_extra(tmp_path: Path) -> None:
    reference = pd.DataFrame(
        {"Hashed_ReportURN": ["a"], "Report": ["Normal."], "Focal Epi": [1]}
    )
    mistral = pd.DataFrame(
        {"Hashed_ReportURN": ["a", "excluded"], "Focal Epi": [1, 3]}
    )
    medgemma = pd.DataFrame({"Hashed_ReportURN": ["a"], "Focal Epi": [1]})
    reference_path = tmp_path / "reference.db"
    mistral_path = tmp_path / "mistral.db"
    medgemma_path = tmp_path / "medgemma.db"
    write_db(reference_path, "reports", reference)
    write_db(mistral_path, "classifications", mistral)
    write_db(medgemma_path, "classifications", medgemma)
    ledger, aggregate = audit_cohort(
        cohort="synthetic",
        expected_records=1,
        reference_path=reference_path,
        mistral_path=mistral_path,
        medgemma_path=medgemma_path,
    )
    assert len(ledger) == 1
    assert aggregate["input_key_reconciliation"]["mistral"]["extra_vs_reference"] == 1
