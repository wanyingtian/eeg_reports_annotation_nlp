from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from eeg_review.baseline import run_baseline_cv


def test_bow_cv_exports_oof_receipt_and_refitted_model(tmp_path: Path) -> None:
    database = tmp_path / "development.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            'CREATE TABLE reports ("Hashed_ReportURN" TEXT, "Report" TEXT, '
            '"Patient" TEXT, "Abnormality" INTEGER)'
        )
        rows = []
        for index in range(12):
            present = index >= 6
            rows.append(
                (
                    f"r{index}",
                    f"{'abnormal spikes' if present else 'normal background'} report {index}",
                    f"p{index}",
                    4 if present else 1,
                )
            )
        connection.executemany("INSERT INTO reports VALUES (?, ?, ?, ?)", rows)

    output = tmp_path / "baseline"
    result = run_baseline_cv(
        database,
        output,
        model_name="bag_of_words",
        labels=["Abnormality"],
        patient_column="Patient",
        folds=2,
        seed=7,
    )
    assert result["labels"]["Abnormality"]["status"] == "completed"
    receipt = pd.read_csv(output / "oof_predictions.csv")
    assert "Report" not in receipt.columns
    assert receipt["Abnormality probability"].notna().all()
    assert set(receipt["Abnormality fold"]) == {1, 2}
    assert (output / "models" / "abnormality_bag_of_words.joblib").exists()
