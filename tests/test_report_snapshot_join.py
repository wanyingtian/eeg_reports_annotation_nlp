import importlib.util
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_snapshot_join", ROOT / "scripts/audit_report_snapshot_join.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_same_report_keys_and_text_join_regardless_of_order():
    a = module.report_index([("r1", "One"), ("r2", "Two")])
    b = module.report_index([("r3", "Three"), ("r2", "Two"), ("r1", "One")])
    result = module.compare(a, b)
    assert result["complete_exact_report_join"]
    assert result["exact_text_matches_on_same_key"] == 2
    assert not result["patient_membership_established"]


def test_identical_text_different_key_is_not_a_join():
    result = module.compare({"r1": "Same text"}, {"r2": "Same text"})
    assert result["keys_missing"] == 1
    assert not result["complete_exact_report_join"]


def test_empty_query_cannot_claim_complete_reconciliation():
    assert not module.compare({}, {"r1": "One"})["complete_exact_report_join"]


def test_whitespace_change_is_retained_as_text_difference():
    result = module.compare({"r1": "Same text"}, {"r1": "Same  text"})
    assert result["text_mismatches_on_same_key"] == 1
    assert not result["complete_exact_report_join"]


@pytest.mark.parametrize(
    "rows",
    [
        [("r1", "One"), ("r1", "One")],
        [(None, "One")],
        [("r1", None)],
        [(" ", "One")],
        [(123, "One")],
    ],
)
def test_bad_keys_or_text_cannot_be_silently_joined(rows):
    with pytest.raises(ValueError):
        module.report_index(rows)


@pytest.mark.parametrize("key", ["Hashed ID", "Hashed_ReportURN"])
def test_explicit_key_alias_and_read_only_db(tmp_path, key):
    path = tmp_path / "source.db"
    with sqlite3.connect(path) as conn:
        conn.execute(f'CREATE TABLE reports ("{key}" TEXT, Report TEXT, Abnormality TEXT)')
        conn.execute("INSERT INTO reports VALUES (?, ?, ?)", ("r1", "Synthetic report", "unused"))
    before = path.read_bytes()
    records, metadata = module.load_reports(path)
    assert records == {"r1": "Synthetic report"}
    assert metadata["report_key_column"] == key
    assert path.read_bytes() == before


def test_two_key_columns_are_ambiguous_not_silently_preferred(tmp_path):
    path = tmp_path / "source.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            'CREATE TABLE reports ("Hashed ID" TEXT, "Hashed_ReportURN" TEXT, Report TEXT)'
        )
    with pytest.raises(ValueError, match="one explicit"):
        module.load_reports(path)


def test_missing_source_never_creates_empty_database(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        module.load_reports(path)
    assert not path.exists()
