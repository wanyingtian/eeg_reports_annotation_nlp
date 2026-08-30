from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from eeg_review.evidence_extraction import (
    JSON_KEYS,
    aggregate_inspections,
    classification_levels,
    inspect_explanation,
    load_fixed_evidence_inputs,
)


def classification(level: int = 1) -> str:
    return json.dumps({key: level for key in JSON_KEYS})


def explanation(level: int = 1, reason: str = "normal background") -> str:
    return json.dumps(
        {key: {"decision": level, "reasons": [reason]} for key in JSON_KEYS}
    )


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "reports.db"
    with sqlite3.connect(dataset) as connection:
        connection.execute(
            'CREATE TABLE reports ("Hashed_ReportURN" TEXT, "Report" TEXT)'
        )
        connection.executemany(
            "INSERT INTO reports VALUES (?, ?)",
            [("a", "normal background"), ("b", "focal slowing")],
        )
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "Hashed_ReportURN": ["b", "a"],
            "classifications": [classification(4), classification(1)],
        }
    ).to_csv(predictions, index=False)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"Hashed_ReportURN": ["a", "b"]}).to_csv(manifest, index=False)
    return dataset, predictions, manifest


def test_fixed_inputs_follow_manifest_and_validate_classifications(tmp_path: Path) -> None:
    dataset, predictions, manifest = write_inputs(tmp_path)
    frame = load_fixed_evidence_inputs(
        dataset=dataset,
        predictions=predictions,
        manifest=manifest,
    )

    assert frame["Hashed_ReportURN"].tolist() == ["a", "b"]
    assert classification_levels(frame.iloc[0]["classifications"])["abnormality"] == 1


def test_fixed_inputs_reject_missing_and_duplicate_keys(tmp_path: Path) -> None:
    dataset, predictions, manifest = write_inputs(tmp_path)
    pd.DataFrame({"Hashed_ReportURN": ["a", "a"]}).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_fixed_evidence_inputs(
            dataset=dataset,
            predictions=predictions,
            manifest=manifest,
        )


def test_explanation_inspection_is_not_a_causality_measure() -> None:
    fixed = classification(1)
    checked = inspect_explanation(
        explanation(1),
        report="The EEG has normal background.",
        fixed_classification=fixed,
    )
    mismatch = inspect_explanation(
        explanation(4, "invented text"),
        report="The EEG has normal background.",
        fixed_classification=fixed,
    )

    assert checked.structured_output_valid
    assert checked.decision_copy_mismatches == 0
    assert checked.exact_traceable_phrases == 5
    assert mismatch.decision_copy_mismatches == 5
    assert mismatch.exact_traceable_phrases == 0


def test_aggregate_inspections_reports_conservative_rates() -> None:
    frame = pd.DataFrame(
        [
            {
                "structured_output_valid": True,
                "decision_copy_mismatches": 0,
                "evidence_phrases": 5,
                "fallback_phrases": 0,
                "exact_traceable_phrases": 4,
                "casefold_traceable_phrases": 5,
            },
            {
                "structured_output_valid": False,
                "decision_copy_mismatches": 0,
                "evidence_phrases": 0,
                "fallback_phrases": 0,
                "exact_traceable_phrases": 0,
                "casefold_traceable_phrases": 0,
            },
        ]
    )
    summary = aggregate_inspections(frame)

    assert summary["records"] == 2
    assert summary["invalid_structured_outputs"] == 1
    assert summary["exact_traceability_fraction"] == 0.8
