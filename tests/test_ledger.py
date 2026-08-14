from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eeg_review.ledger import build_result_ledger


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_result_ledger_consolidates_aggregate_receipts(tmp_path: Path) -> None:
    point = {
        "n": 10,
        "tn": 5,
        "fp": 1,
        "fn": 2,
        "tp": 2,
        "core_accuracy": 0.7,
        "precision": 2 / 3,
        "recall_sensitivity": 0.5,
        "specificity": 5 / 6,
        "f1": 4 / 7,
        "certainty_adjusted_accuracy": 0.6,
        "core_kappa": 0.4,
        "four_level_kappa": 0.3,
    }
    intervals = {
        metric: {"low": value - 0.1, "high": value + 0.1}
        for metric, value in point.items()
        if metric not in {"n", "tn", "fp", "fn", "tp"}
    }
    evaluation = tmp_path / "evaluation/evaluation_summary.json"
    write_summary(
        evaluation,
        {
            "schema_version": 1,
            "bootstrap": {"unit": "report", "iterations": 20},
            "interpretation_limits": ["report-level only"],
            "labels": {
                "Abnormality": {
                    "point_estimates": point,
                    "confidence_intervals_95": intervals,
                }
            },
        },
    )
    calibration = tmp_path / "calibration/calibration_summary.json"
    write_summary(
        calibration,
        {
            "schema_version": 1,
            "bootstrap": {"unit": "report", "iterations": 20},
            "interpretation_limits": ["report-level only"],
            "labels": {
                "Abnormality": {
                    "point_estimates": {
                        "n": 10,
                        "positives": 4,
                        "prevalence": 0.4,
                        "mean_predicted_probability": 0.5,
                        "brier_score": 0.2,
                        "log_loss": 0.6,
                        "expected_calibration_error": 0.1,
                    },
                    "confidence_intervals_95": {
                        "brier_score": {"low": 0.1, "high": 0.3}
                    },
                }
            },
        },
    )
    comparison = tmp_path / "comparison/paired_comparison_summary.json"
    write_summary(
        comparison,
        {
            "schema_version": 1,
            "bootstrap": {"unit": "report", "iterations": 20},
            "interpretation_limits": ["report-level only"],
            "labels": {
                "Abnormality": {
                    "n": 10,
                    "effects_a_minus_b": {"core_accuracy_difference": -0.1},
                    "paired_confidence_intervals_95": {
                        "core_accuracy_difference": {"low": -0.2, "high": 0.0}
                    },
                    "discordant_correctness": {
                        "core_accuracy": {
                            "a_correct_b_wrong": 1,
                            "a_wrong_b_correct": 2,
                            "both_correct": 6,
                            "both_wrong": 1,
                            "mcnemar_exact_p_value": 1.0,
                            "multiplicity_adjusted_p_value": 1.0,
                        }
                    },
                }
            },
        },
    )

    output = tmp_path / "ledger"
    result = build_result_ledger(
        output,
        evaluations={"eval": evaluation},
        calibrations={"cal": calibration},
        comparisons={"pair": comparison},
    )
    assert result["privacy_boundary"].startswith("aggregate receipts only")
    assert result["row_counts"]["evaluation_ledger.csv"] == len(point)
    evaluation_rows = pd.read_csv(output / "evaluation_ledger.csv")
    accuracy = evaluation_rows.loc[evaluation_rows["metric"] == "core_accuracy"].iloc[0]
    assert accuracy["numerator"] == 7
    assert accuracy["denominator"] == 10
    prevalence_rows = pd.read_csv(output / "calibration_ledger.csv")
    prevalence = prevalence_rows.loc[prevalence_rows["metric"] == "prevalence"].iloc[0]
    assert prevalence["numerator"] == 4
    assert prevalence["denominator"] == 10
    assert (output / "paired_effect_ledger.csv").exists()
    assert (output / "discordance_ledger.csv").exists()
    assert (output / "result_ledger.json").exists()
    assert (output / "run_manifest.json").exists()
