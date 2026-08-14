from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from eeg_review.baseline import run_baseline_cv, run_baseline_predict


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

    external = tmp_path / "external.db"
    with sqlite3.connect(external) as connection:
        connection.execute(
            'CREATE TABLE reports ("Hashed_ReportURN" TEXT, "Report" TEXT)'
        )
        connection.executemany(
            "INSERT INTO reports VALUES (?, ?)",
            [("e1", "normal background"), ("e2", "abnormal spikes")],
        )
    prediction_output = tmp_path / "prediction"
    prediction_result = run_baseline_predict(
        external,
        output,
        prediction_output,
        model_name="bag_of_words",
        labels=["Abnormality"],
    )
    assert prediction_result["records"] == 2
    predictions = pd.read_csv(prediction_output / "predictions.csv")
    assert list(predictions["Hashed_ReportURN"]) == ["e1", "e2"]
    assert predictions["Abnormality probability"].between(0, 1).all()
    assert predictions["Abnormality prediction"].isin([1, 2, 3, 4]).all()


def test_low_support_label_keeps_external_fit_without_inventing_folds(
    tmp_path: Path,
) -> None:
    database = tmp_path / "development.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            'CREATE TABLE reports ("Hashed_ReportURN" TEXT, "Report" TEXT, '
            '"Gen Epi" INTEGER)'
        )
        connection.executemany(
            "INSERT INTO reports VALUES (?, ?, ?)",
            [
                (
                    f"r{index}",
                    "generalized spike" if index < 3 else "normal background",
                    4 if index < 3 else 1,
                )
                for index in range(12)
            ],
        )

    output = tmp_path / "baseline"
    result = run_baseline_cv(
        database,
        output,
        model_name="bag_of_words",
        labels=["Gen Epi"],
        folds=5,
    )
    label = result["labels"]["Gen Epi"]
    assert label["status"] == "external_fit_only"
    assert label["oof_records"] == 0
    assert label["final_fit_records"] == 12
    oof = pd.read_csv(output / "oof_predictions.csv")
    assert oof["Gen Epi probability"].isna().all()
    assert oof["Gen Epi fold"].isna().all()

    prediction_output = tmp_path / "prediction"
    prediction = run_baseline_predict(
        database,
        output,
        prediction_output,
        model_name="bag_of_words",
        labels=["Gen Epi"],
    )
    assert prediction["labels"]["Gen Epi"]["status"] == "completed"
    external = pd.read_csv(prediction_output / "predictions.csv")
    assert external["Gen Epi probability"].between(0, 1).all()
