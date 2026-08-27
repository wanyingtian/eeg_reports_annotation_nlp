from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from eeg_review.compare import compare_predictions, exact_mcnemar_p_value, holm_adjust


def write_predictions(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Hashed_ReportURN", "Abnormality"])
        writer.writerows(rows)


def test_exact_mcnemar_and_holm_values() -> None:
    assert exact_mcnemar_p_value(4, 0) == pytest.approx(0.125)
    assert exact_mcnemar_p_value(6, 0) == pytest.approx(0.03125)
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert holm_adjust([0.125, 0.03125]) == pytest.approx([0.125, 0.0625])


def test_compare_emits_aggregate_paired_receipt(tmp_path: Path) -> None:
    reference = tmp_path / "reference.db"
    with sqlite3.connect(reference) as connection:
        connection.execute(
            'CREATE TABLE reports ("Hashed_ReportURN" TEXT, "Patient" TEXT, "Abnormality" INTEGER)'
        )
        connection.executemany(
            "INSERT INTO reports VALUES (?, ?, ?)",
            [
                ("a", "p1", 1),
                ("b", "p1", 2),
                ("c", "p2", 3),
                ("d", "p3", 3),
                ("e", "p4", 4),
                ("f", "p5", 4),
            ],
        )
    model_a = tmp_path / "model-a.csv"
    model_b = tmp_path / "model-b.csv"
    write_predictions(
        model_a,
        [(key, level) for key, level in zip("abcdef", [1, 2, 3, 3, 4, 4], strict=True)],
    )
    write_predictions(
        model_b,
        [(key, level) for key, level in zip("abcdef", [3, 3, 1, 1, 1, 1], strict=True)],
    )

    output = tmp_path / "comparison"
    result = compare_predictions(
        reference,
        model_a,
        model_b,
        output,
        model_a_id="model-a",
        model_b_id="model-b",
        labels=["Abnormality"],
        cluster_column="Patient",
        bootstrap_iterations=50,
        seed=9,
    )

    label = result["labels"]["Abnormality"]
    assert label["effects_a_minus_b"]["core_accuracy_difference"] == pytest.approx(1.0)
    assert label["effects_a_minus_b"]["certainty_adjusted_accuracy_difference"] == pytest.approx(
        1.0
    )
    assert label["discordant_correctness"]["core_accuracy"]["a_correct_b_wrong"] == 6
    assert label["discordant_correctness"]["core_accuracy"]["a_wrong_b_correct"] == 0
    assert label["discordant_correctness"]["core_accuracy"]["mcnemar_exact_p_value"] == (
        pytest.approx(0.03125)
    )
    assert result["bootstrap"]["unit"] == "cluster"
    assert any("McNemar" in limit for limit in result["interpretation_limits"])
    assert (output / "paired_comparisons.csv").exists()
    rendered = (output / "paired_comparison_summary.json").read_text(encoding="utf-8")
    assert "Hashed_ReportURN" not in rendered
    assert '"a"' not in json.dumps(json.loads(rendered)["labels"])


def test_compare_can_require_exact_keys_and_patient_grouping(tmp_path: Path) -> None:
    reference = tmp_path / "reference.db"
    with sqlite3.connect(reference) as connection:
        connection.execute(
            'CREATE TABLE reports ("Hashed_ReportURN" TEXT, "Patient" TEXT, "Abnormality" INTEGER)'
        )
        connection.executemany(
            "INSERT INTO reports VALUES (?, ?, ?)",
            [("a", "p1", 1), ("b", "p2", 4)],
        )
    model_a = tmp_path / "model-a.csv"
    model_b = tmp_path / "model-b.csv"
    write_predictions(model_a, [("a", 1), ("b", 4)])
    write_predictions(model_b, [("a", 1)])

    with pytest.raises(ValueError, match="Exact three-way report-key alignment failed"):
        compare_predictions(
            reference,
            model_a,
            model_b,
            tmp_path / "comparison",
            model_a_id="model-a",
            model_b_id="model-b",
            labels=["Abnormality"],
            cluster_column="Patient",
            require_exact_key_set=True,
        )
    with pytest.raises(ValueError, match="no cluster column"):
        compare_predictions(
            reference,
            model_a,
            model_a,
            tmp_path / "comparison",
            model_a_id="model-a",
            model_b_id="model-b",
            labels=["Abnormality"],
            require_patient_grouping=True,
        )
