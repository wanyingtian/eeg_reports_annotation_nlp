from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_review.explanation_reconciliation import (
    CATEGORIES,
    ID_COLUMN,
    artifact_census,
    correctness_by_alignment,
    deterministic_traceability,
    join_reference,
    load_explanation_artifact,
    polarity_classifier_alignment,
    positive_traceability_units,
    reconcile_source_snapshot,
    semantic_complete,
    summarize_stages,
)


def artifact_frame(rows: int = 3) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Hashed ID": [f"k{i}" for i in range(rows)],
            "Report": ["normal background", "focal slowing present", "generalized slowing"][:rows],
        }
    )
    for category in CATEGORIES:
        frame[category] = [1, 4, 3][:rows]
        frame[f"{category} Reasons"] = [
            "normal background",
            "focal slowing present",
            "paraphrased generalized disturbance",
        ][:rows]
        frame[f"{category} Reason Polarity"] = [-1, 1, 1][:rows]
    return frame


def test_artifact_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    frame = artifact_frame()
    frame.loc[2, "Hashed ID"] = "k1"
    path = tmp_path / "artifact.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_explanation_artifact(path)


@pytest.mark.parametrize("missing_key", [pd.NA, ""])
def test_artifact_loader_rejects_missing_keys(
    tmp_path: Path, missing_key: object
) -> None:
    frame = artifact_frame()
    frame.loc[1, "Hashed ID"] = missing_key
    path = tmp_path / "artifact.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing report keys"):
        load_explanation_artifact(path)


def test_artifact_loader_retains_missing_model_labels_as_explicit_absence(
    tmp_path: Path,
) -> None:
    frame = artifact_frame()
    frame.loc[2, list(CATEGORIES)] = pd.NA
    path = tmp_path / "artifact.csv"
    frame.to_csv(path, index=False)
    loaded = load_explanation_artifact(path)
    assert artifact_census(loaded)["categories"]["Abnormality"]["missing_model_label"] == 1


def test_positive_polarity_not_model_label_controls_traceability() -> None:
    frame = artifact_frame()
    frame.loc[0, "Focal Epi"] = 4
    frame.loc[0, "Focal Epi Reason Polarity"] = -1
    frame = frame.rename(columns={"Hashed ID": ID_COLUMN})
    units = positive_traceability_units(frame)
    assert len(units) == 10
    assert artifact_census(frame)["positive_polarity_total"] == 10


def test_traceability_stages_are_explicit_and_nonoptimizing() -> None:
    units = positive_traceability_units(artifact_frame().rename(columns={"Hashed ID": ID_COLUMN}))
    stages = deterministic_traceability(units)
    assert summarize_stages(stages)["matched"] == 5

    def encoder(texts):
        return np.array([[1.0, 0.0] if "general" in text else [0.0, 1.0] for text in texts])

    completed, _ = semantic_complete(units, stages, encoder=encoder, threshold=0.70)
    assert summarize_stages(completed)["matched"] == 10


def test_alignment_excludes_fixed_training_rows() -> None:
    frame = pd.concat([artifact_frame()] * 67, ignore_index=True).iloc[:201].copy()
    result = polarity_classifier_alignment(frame, test_start=200)
    assert result["candidate_test_rows"] == 1
    assert result["categories"]["Abnormality"]["n"] == 1


def test_reference_join_requires_complete_manifest() -> None:
    artifact = artifact_frame().rename(columns={"Hashed ID": ID_COLUMN})
    reference = artifact[[ID_COLUMN, *CATEGORIES]].copy()
    manifest = pd.DataFrame({ID_COLUMN: ["k0", "missing"]})
    with pytest.raises(ValueError, match="incomplete"):
        join_reference(artifact, reference, manifest=manifest)


def test_source_snapshot_reconciliation_verifies_order_and_text(tmp_path: Path) -> None:
    artifact = artifact_frame().rename(columns={"Hashed ID": ID_COLUMN})
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as connection:
        artifact[[ID_COLUMN, "Report"]].to_sql("reports", connection, index=False)
    result = reconcile_source_snapshot(artifact, source)
    assert result["artifact_order_matches_source_prefix"]
    assert result["report_text_exact_matches"] == 3


def test_correctness_association_reports_counts_and_not_causality() -> None:
    artifact = artifact_frame().rename(columns={"Hashed ID": ID_COLUMN})
    reference = artifact[[ID_COLUMN, *CATEGORIES]].copy()
    for category in CATEGORIES:
        artifact.loc[2, f"{category} Reason Polarity"] = -1
        reference.loc[2, category] = 1
    joined = join_reference(artifact, reference)
    result = correctness_by_alignment(joined, surface="test")
    abnormality = result["categories"]["Abnormality"]
    assert abnormality["aligned"]["correct"] == 2
    assert abnormality["misaligned"]["total"] == 1
    assert "not causal" in result["interpretation"]
