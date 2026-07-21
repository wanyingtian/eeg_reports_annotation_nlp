from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from eeg_review.metrics import evaluate_predictions


def test_evaluation_reports_core_and_certainty_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference.db"
    with sqlite3.connect(reference) as connection:
        connection.execute(
            'CREATE TABLE reports ("Hashed_ReportURN" TEXT, "Patient" TEXT, "Abnormality" INTEGER)'
        )
        connection.executemany(
            "INSERT INTO reports VALUES (?, ?, ?)",
            [("a", "p1", 1), ("b", "p2", 2), ("c", "p3", 3), ("d", "p4", 4)],
        )
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Hashed_ReportURN", "Abnormality prediction", "Fold"])
        writer.writerows([("a", 1, 1), ("b", 3, 1), ("c", 3, 2), ("d", 4, 2)])

    result = evaluate_predictions(
        reference,
        predictions,
        tmp_path / "evaluation",
        labels=["Abnormality"],
        prediction_columns={"Abnormality": "Abnormality prediction"},
        cluster_column="Patient",
        fold_column="Fold",
        bootstrap_iterations=50,
        seed=7,
    )
    point = result["labels"]["Abnormality"]["point_estimates"]
    assert point["tn"] == 1
    assert point["fp"] == 1
    assert point["fn"] == 0
    assert point["tp"] == 2
    assert point["core_accuracy"] == 0.75
    assert point["certainty_adjusted_accuracy"] == 0.75
    assert result["bootstrap"]["unit"] == "cluster"
    assert result["labels"]["Abnormality"]["fold_variability"]["core_accuracy"]["folds"] == 2
    assert (tmp_path / "evaluation" / "fold_metrics.csv").exists()


def test_evaluation_applies_historical_ranges_and_complete_case_rule(tmp_path: Path) -> None:
    reference = tmp_path / "reference.db"
    with sqlite3.connect(reference) as connection:
        connection.execute(
            'CREATE TABLE reports ("Hashed ID" TEXT, "A" INTEGER, "B" INTEGER)'
        )
        connection.executemany(
            "INSERT INTO reports VALUES (?, ?, ?)",
            [
                ("outside-0", 1, 1),
                ("selected-1", 1, 1),
                ("incomplete", 4, None),
                ("outside-3", 4, 4),
                ("selected-4", 4, 4),
                ("selected-5", 3, 3),
            ],
        )
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Hashed ID", "A", "B"])
        writer.writerows(
            [
                ("outside-0", 4, 4),
                ("selected-1", 1, 1),
                ("incomplete", 4, 4),
                ("outside-3", 1, 1),
                ("selected-4", 4, 4),
                ("selected-5", 3, 3),
            ]
        )

    result = evaluate_predictions(
        reference,
        predictions,
        tmp_path / "evaluation",
        id_column="Hashed ID",
        labels=["A", "B"],
        reference_row_ranges=[(1, 3), (4, 6)],
        require_complete_reference=True,
        bootstrap_iterations=20,
        seed=11,
    )

    assert result["reference_selection"] == {
        "method": "half_open_positional_ranges",
        "source_records": 6,
        "candidate_records": 4,
        "row_ranges": [{"start": 1, "end": 3}, {"start": 4, "end": 6}],
        "complete_reference_required": True,
        "excluded_incomplete_reference_records": 1,
    }
    assert result["reference_records"] == 3
    assert result["matched_records"] == 3
    assert result["labels"]["A"]["point_estimates"]["core_accuracy"] == 1.0
    assert result["labels"]["B"]["point_estimates"]["core_accuracy"] == 1.0
