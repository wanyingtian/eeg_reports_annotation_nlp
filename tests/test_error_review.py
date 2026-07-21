from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from eeg_review.error_review import build_error_review_packet


def write_predictions(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Hashed_ReportURN", "Abnormality"])
        writer.writerows([("report-a", 1), ("report-b", 3), ("report-c", 1), ("report-d", 4)])


def test_error_review_requires_governed_acknowledgement(tmp_path: Path) -> None:
    reference = tmp_path / "reference.db"
    with sqlite3.connect(reference) as connection:
        connection.execute('CREATE TABLE reports ("Hashed_ReportURN" TEXT, "Abnormality" INTEGER)')
    predictions = tmp_path / "predictions.csv"
    write_predictions(predictions)
    with pytest.raises(ValueError, match="governed"):
        build_error_review_packet(
            reference,
            predictions,
            tmp_path / "output",
            model_id="model",
            acknowledge_governed_output=False,
            labels=["Abnormality"],
            handle_salt="test-salt",
        )


def test_error_review_packet_omits_source_ids_and_text(tmp_path: Path) -> None:
    reference = tmp_path / "reference.db"
    with sqlite3.connect(reference) as connection:
        connection.execute(
            'CREATE TABLE reports ("Hashed_ReportURN" TEXT, "Patient" TEXT, "Abnormality" INTEGER)'
        )
        connection.executemany(
            "INSERT INTO reports VALUES (?, ?, ?)",
            [
                ("report-a", "patient-1", 3),
                ("report-b", "patient-1", 1),
                ("report-c", "patient-2", 4),
                ("report-d", "patient-3", 1),
            ],
        )
    predictions = tmp_path / "predictions.csv"
    write_predictions(predictions)

    output = tmp_path / "output"
    result = build_error_review_packet(
        reference,
        predictions,
        output,
        model_id="model",
        acknowledge_governed_output=True,
        labels=["Abnormality"],
        cluster_column="Patient",
        max_per_stratum=1,
        seed=4,
        handle_salt="test-salt",
    )

    packet = pd.read_csv(output / "clinical_error_review_packet.csv")
    assert result["selected_case_rows"] == 2
    assert set(packet["error_type"]) == {"false_negative", "false_positive"}
    assert packet["case_handle"].str.startswith("case-").all()
    rendered = (output / "clinical_error_review_packet.csv").read_text(encoding="utf-8")
    for sensitive_value in ("report-a", "report-b", "report-c", "report-d", "patient-"):
        assert sensitive_value not in rendered
    assert "Report" not in packet.columns
