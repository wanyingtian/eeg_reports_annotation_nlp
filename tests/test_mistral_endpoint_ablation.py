from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mistral_endpoint_ablation",
    ROOT / "scripts/analyze_mistral_endpoint_guidance_ablation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def frame(rows):
    return pd.DataFrame(rows, columns=[MODULE.KEY, *MODULE.JSON_KEY_TO_LABEL.values()])


def test_consistency_checks_keep_both_declared_directions_distinct():
    data = frame(
        [
            ["a", 1, 1, 1, 1, 4],
            ["b", 4, 1, 1, 1, 1],
            ["c", 1, 1, 1, 1, 1],
        ]
    )
    assert MODULE.consistency_violations(data) == {
        "all_subtypes_absent_but_abnormality_present": 1,
        "any_subtype_present_but_abnormality_absent": 1,
    }


def test_metrics_separate_binary_core_from_exact_four_level_agreement():
    reference = frame(
        [
            ["a", 1, 1, 1, 1, 1],
            ["b", 4, 4, 4, 4, 4],
        ]
    )
    predictions = frame(
        [
            ["a", 2, 1, 1, 1, 1],
            ["b", 3, 4, 4, 4, 4],
        ]
    )
    result = MODULE.metrics(reference, predictions)
    assert result["Focal Epi"]["core_agreement"] == 1.0
    assert result["Focal Epi"]["exact_four_level_agreement"] == 0.0
    assert result["Focal Epi"]["level_counts"] == {"1": 0, "2": 1, "3": 1, "4": 0}
