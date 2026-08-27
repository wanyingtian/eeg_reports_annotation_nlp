from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from eeg_review.adaptation_plan import validate_adaptation_plan
from eeg_review.audit import DEFAULT_LABELS
from eeg_review.development_manifest import (
    MANIFEST_FILENAME,
    RECEIPT_FILENAME,
    create_development_manifest,
    prepare_adaptation_execution_plan,
)
from eeg_review.manifest import sha256_file

PLAN = (
    Path(__file__).parents[1] / "review/model-receipts/mistral-task-adaptation.preregistered.json"
)


def _write_reference(path: Path, *, records: int = 100) -> Path:
    levels = [1, 2, 3, 4]
    frame = pd.DataFrame(
        {
            "Hashed_ReportURN": [f"report-{index:03d}" for index in range(records)],
            **{
                label: [levels[index % len(levels)] for index in range(records)]
                for label in DEFAULT_LABELS
            },
        }
    )
    with sqlite3.connect(path) as connection:
        frame.to_sql("reports", connection, index=False)
    return path


def test_manifest_creation_freezes_keys_without_emitting_them_in_receipt(
    tmp_path: Path,
) -> None:
    reference = _write_reference(tmp_path / "reference.db")
    output = tmp_path / "governed"

    result = create_development_manifest(
        reference,
        output,
        acknowledge_governed_output=True,
    )

    manifest_path = output / MANIFEST_FILENAME
    receipt_path = output / RECEIPT_FILENAME
    assert result["records"] == 100
    assert result["complete_labels"] is True
    assert result["manifest"]["sha256"] == sha256_file(manifest_path)
    assert manifest_path.stat().st_mode & 0o777 == 0o600
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    rendered_receipt = receipt_path.read_text(encoding="utf-8")
    assert "report-000" not in rendered_receipt
    assert "Hashed_ReportURN" in rendered_receipt


def test_manifest_creation_rejects_wrong_population_and_duplicate_keys(tmp_path: Path) -> None:
    short_reference = _write_reference(tmp_path / "short.db", records=99)
    with pytest.raises(ValueError, match="exactly 100"):
        create_development_manifest(
            short_reference,
            tmp_path / "short-output",
            acknowledge_governed_output=True,
        )

    duplicate_reference = _write_reference(tmp_path / "duplicate.db")
    with sqlite3.connect(duplicate_reference) as connection:
        connection.execute(
            'UPDATE reports SET "Hashed_ReportURN" = ? WHERE rowid = 2',
            ("report-000",),
        )
    with pytest.raises(ValueError, match="duplicate"):
        create_development_manifest(
            duplicate_reference,
            tmp_path / "duplicate-output",
            acknowledge_governed_output=True,
        )


def test_manifest_creation_rejects_incomplete_reference_labels(tmp_path: Path) -> None:
    reference = _write_reference(tmp_path / "reference.db")
    with sqlite3.connect(reference) as connection:
        connection.execute('UPDATE reports SET "Gen Epi" = NULL WHERE rowid = 1')

    with pytest.raises(ValueError, match="invalid or missing Gen Epi"):
        create_development_manifest(
            reference,
            tmp_path / "output",
            acknowledge_governed_output=True,
        )


def test_execution_preparation_binds_exact_reference_and_manifest(tmp_path: Path) -> None:
    reference = _write_reference(tmp_path / "reference.db")
    governed = tmp_path / "governed"
    create_development_manifest(
        reference,
        governed,
        acknowledge_governed_output=True,
    )
    manifest = governed / MANIFEST_FILENAME
    manifest_receipt = governed / RECEIPT_FILENAME
    output_plan = governed / "mistral-task-adaptation.execution.json"

    result = prepare_adaptation_execution_plan(
        PLAN,
        reference,
        manifest,
        manifest_receipt,
        output_plan,
        acknowledge_governed_output=True,
    )

    payload = json.loads(output_plan.read_text(encoding="utf-8"))
    development_signal = next(
        signal
        for signal in payload["signals"]
        if signal["signal_id"] == "zoe_development_first_100_ra"
    )
    assert development_signal["artifact"]["sha256"] == sha256_file(reference)
    assert payload["certainty_mapping"]["development_manifest"]["sha256"] == sha256_file(
        manifest
    )
    assert result["status"] == "preregistered_unfrozen"
    assert result["ready_for_evaluation"] is False
    validation = validate_adaptation_plan(
        output_plan,
        bundle_root=output_plan.parent,
        check_files=True,
    )
    assert validation["design_valid"] is True
    assert validation["ready_for_implementation"] is True
    assert validation["ready_for_evaluation"] is False


def test_execution_preparation_rejects_manifest_tampering(tmp_path: Path) -> None:
    reference = _write_reference(tmp_path / "reference.db")
    governed = tmp_path / "governed"
    create_development_manifest(
        reference,
        governed,
        acknowledge_governed_output=True,
    )
    manifest = governed / MANIFEST_FILENAME
    frame = pd.read_csv(manifest)
    frame.loc[0, "Hashed_ReportURN"] = "different-report"
    frame.to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="checksum"):
        prepare_adaptation_execution_plan(
            PLAN,
            reference,
            manifest,
            governed / RECEIPT_FILENAME,
            governed / "execution.json",
            acknowledge_governed_output=True,
        )
