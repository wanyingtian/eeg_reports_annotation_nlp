from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from eeg_review.audit import audit_dataset, audit_overlap

LABELS = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]


def make_database(path: Path, rows: list[tuple]) -> None:
    with sqlite3.connect(path) as connection:
        columns = ", ".join(f'"{label}" INTEGER' for label in LABELS)
        create_table = (
            'CREATE TABLE reports ("Hashed_ReportURN" TEXT, "Report" TEXT, '
            f'"Patient" TEXT, {columns})'
        )
        connection.execute(create_table)
        placeholders = ", ".join("?" for _ in range(3 + len(LABELS)))
        connection.executemany(f"INSERT INTO reports VALUES ({placeholders})", rows)


def test_audit_is_aggregate_only(tmp_path: Path) -> None:
    database = tmp_path / "cohort.db"
    make_database(
        database,
        [
            ("id-a", "Normal EEG", "patient-1", 1, 1, 1, 1, 1),
            ("id-b", "Abnormal EEG", "patient-1", 4, 1, 3, 1, 4),
            ("id-c", " abnormal   eeg ", "patient-2", 4, 1, 3, 1, 4),
        ],
    )
    output = tmp_path / "audit"
    result = audit_dataset(database, "test", output, patient_column="Patient")
    assert result["records"] == 3
    assert result["reports"]["rows_in_exact_duplicate_groups"] == 2
    assert result["patient_independence"]["unique_patients"] == 2
    rendered = (output / "cohort_audit.json").read_text(encoding="utf-8")
    assert "Abnormal EEG" not in rendered
    assert "id-a" not in rendered
    assert json.loads(rendered)["labels"]["Abnormality"]["core_present"] == 2


def test_overlap_emits_counts_not_identifiers(tmp_path: Path) -> None:
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    make_database(left, [("shared", "Same report", "p1", 1, 1, 1, 1, 1)])
    make_database(right, [("shared", "Same report", "p2", 1, 1, 1, 1, 1)])
    result = audit_overlap({"left": left, "right": right}, tmp_path / "overlap")
    comparison = result["comparisons"][0]
    assert comparison["shared_report_identifiers"] == 1
    assert comparison["shared_exact_normalized_reports"] == 1
    rendered = json.dumps(result)
    assert "Same report" not in rendered
    assert 'shared"' not in rendered
