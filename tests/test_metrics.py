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
