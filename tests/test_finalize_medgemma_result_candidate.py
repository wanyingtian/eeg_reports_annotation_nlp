from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/finalize_medgemma_result_candidate.py"
)
SPEC = importlib.util.spec_from_file_location("finalize_medgemma_result_candidate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_comparison_record_preserves_bounded_paired_effects() -> None:
    label = {
        "n": 10,
        "effects_a_minus_b": {
            "core_accuracy_difference": -0.2,
            "certainty_adjusted_accuracy_difference": -0.1,
            "false_negative_rate_difference": -0.3,
        },
        "paired_confidence_intervals_95": {
            "core_accuracy_difference": {"low": -0.3, "high": -0.1},
            "certainty_adjusted_accuracy_difference": {"low": -0.2, "high": 0.0},
            "false_negative_rate_difference": {"low": -0.5, "high": -0.1},
        },
        "model_a_point_estimates": {"core_accuracy": 0.7},
        "model_b_point_estimates": {"core_accuracy": 0.9},
        "discordant_correctness": {
            "core_accuracy": {"multiplicity_adjusted_p_value": 0.01}
        },
    }
    summary = {
        "key_alignment": {"exact_three_way_key_set": True},
        "models": {"a": "a", "b": "b"},
        "matched_records": 10,
        "bootstrap": {"unit": "report"},
        "multiplicity": {"method": "holm"},
        "labels": {name: label for name in MODULE.LABELS},
    }

    receipt = MODULE.comparison_record(summary)

    assert receipt["exact_three_way_key_set"] is True
    assert receipt["labels"]["Abnormality"]["core_accuracy_difference"] == -0.2
    assert "Hashed_ReportURN" not in MODULE.all_keys(receipt)
