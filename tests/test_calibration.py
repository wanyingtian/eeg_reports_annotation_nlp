from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from eeg_review.calibration import calibrate_predictions, calibration_values


def test_calibration_values_have_declared_fixed_bins() -> None:
    outcome = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    values, bins = calibration_values(outcome, probability, bins=2)
    assert values["brier_score"] == pytest.approx(0.025)
    assert values["expected_calibration_error"] == pytest.approx(0.15)
    assert values["prevalence"] == 0.5
    assert bins[0]["count"] == 2
    assert bins[0]["mean_probability"] == pytest.approx(0.15)
    assert bins[0]["event_rate"] == 0.0
    assert bins[1]["event_rate"] == 1.0


def test_calibrate_predictions_emits_aggregate_receipts(tmp_path: Path) -> None:
    reference = tmp_path / "reference.db"
    with sqlite3.connect(reference) as connection:
        connection.execute(
            'CREATE TABLE reports ("Hashed_ReportURN" TEXT, "Patient" TEXT, "Abnormality" INTEGER)'
        )
        connection.executemany(
            "INSERT INTO reports VALUES (?, ?, ?)",
            [("a", "p1", 1), ("b", "p1", 2), ("c", "p2", 3), ("d", "p3", 4)],
        )
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Hashed_ReportURN", "Prob_Abnormality"])
        writer.writerows([("a", 0.1), ("b", 0.2), ("c", 0.8), ("d", 0.9)])

    output = tmp_path / "calibration"
    result = calibrate_predictions(
        reference,
        predictions,
        output,
        model_id="baseline",
        labels=["Abnormality"],
        cluster_column="Patient",
        bins=2,
        bootstrap_iterations=30,
        seed=4,
    )
    assert result["labels"]["Abnormality"]["point_estimates"]["brier_score"] == (
        pytest.approx(0.025)
    )
    assert result["bootstrap"]["unit"] == "cluster"
    assert (output / "calibration_metrics.csv").exists()
    assert (output / "calibration_bins.csv").exists()
    rendered = (output / "calibration_summary.json").read_text(encoding="utf-8")
    assert "Hashed_ReportURN" not in rendered
    assert "Prob_Abnormality" not in rendered
