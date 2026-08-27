from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_review.adaptation_plan import parse_adaptation_plan
from eeg_review.audit import DEFAULT_LABELS
from eeg_review.certainty_adapter import (
    fit_certainty_adapter,
    map_probability_to_certainty,
    select_certainty_margin,
)
from eeg_review.manifest import sha256_file

PLAN = (
    Path(__file__).parents[1] / "review/model-receipts/mistral-task-adaptation.preregistered.json"
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    keys = [f"report-{index:03d}" for index in range(100)]
    levels = np.asarray([1] * 20 + [2] * 30 + [3] * 30 + [4] * 20)
    probabilities = np.asarray([0.1] * 20 + [0.35] * 30 + [0.65] * 30 + [0.9] * 20)
    manifest = tmp_path / "development-manifest.csv"
    reference = tmp_path / "reference.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"Hashed_ReportURN": keys}).to_csv(manifest, index=False)
    pd.DataFrame(
        {
            "Hashed_ReportURN": keys,
            **{label: levels for label in DEFAULT_LABELS},
        }
    ).to_csv(reference, index=False)
    pd.DataFrame(
        {
            "Hashed_ReportURN": keys,
            "adaptation_classification_mode": ["binary_core_certainty_adapter"] * 100,
            **{f"Prob_{label}": probabilities for label in DEFAULT_LABELS},
        }
    ).to_csv(predictions, index=False)
    prediction_receipt = tmp_path / "predictions.run.json"
    prediction_receipt.write_text(
        json.dumps(
            {
                "calibration_instrumentation": {
                    "enabled": True,
                    "classification_mode": "binary_core_certainty_adapter",
                },
                "output": {"sha256": sha256_file(predictions)},
                "model": {"sha256": "1" * 64},
                "prompts": {"classify": {"sha256": "2" * 64}},
                "grammars": {"classify": {"sha256": "3" * 64}},
            }
        ),
        encoding="utf-8",
    )

    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["certainty_mapping"]["resampling_iterations"] = 100
    payload["certainty_mapping"]["development_manifest"] = {
        "path": manifest.name,
        "sha256": sha256_file(manifest),
    }
    contract = tmp_path / "adaptation-plan.json"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    return contract, reference, predictions, prediction_receipt, manifest


def test_probability_mapping_preserves_fixed_core_boundary() -> None:
    result = map_probability_to_certainty(
        np.asarray([0.39, 0.4, 0.499, 0.5, 0.599, 0.6]), margin=0.1
    )

    assert result.tolist() == [1, 2, 2, 3, 3, 4]
    assert np.isin(result[:3], [1, 2]).all()
    assert np.isin(result[3:], [3, 4]).all()


def test_margin_selection_retains_all_candidates_and_uses_declared_tie_break() -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    specification = parse_adaptation_plan(payload).certainty_mapping
    reference = np.asarray([1] * 20 + [2] * 30 + [3] * 30 + [4] * 20)
    probability = np.asarray([0.1] * 20 + [0.35] * 30 + [0.65] * 30 + [0.9] * 20)

    result = select_certainty_margin(reference, probability, specification)

    assert result.fitted is True
    assert result.margin == pytest.approx(0.2)
    assert [row["margin"] for row in result.candidate_scores] == [0.1, 0.2, 0.3]
    assert result.candidate_scores[0]["exact_four_level_agreement"] == pytest.approx(0.4)
    assert result.candidate_scores[1]["exact_four_level_agreement"] == pytest.approx(1.0)
    assert result.candidate_scores[2]["exact_four_level_agreement"] == pytest.approx(1.0)


def test_sparse_core_side_uses_historical_margin_without_claiming_fit() -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    specification = parse_adaptation_plan(payload).certainty_mapping
    reference = np.asarray([1] * 96 + [4] * 4)
    probability = np.asarray([0.1] * 96 + [0.9] * 4)

    result = select_certainty_margin(reference, probability, specification)

    assert result.fitted is False
    assert result.margin == pytest.approx(0.1)
    assert result.core_positive_pairs == 4
    assert "insufficient_development_support" in result.reason


def test_fit_emits_aggregate_adapter_and_stability_receipts(tmp_path: Path) -> None:
    contract, reference, predictions, prediction_receipt, manifest = _write_fixture(tmp_path)
    output = tmp_path / "output"

    result = fit_certainty_adapter(
        contract,
        reference,
        predictions,
        prediction_receipt,
        manifest,
        output,
        acknowledge_governed_inputs=True,
    )

    assert result["ready_for_freeze_review"] is True
    assert result["ready_for_evaluation"] is False
    assert result["labels_using_preregistered_fallback"] == []
    abnormality = result["labels"]["Abnormality"]
    assert abnormality["selected_margin"] == pytest.approx(0.2)
    assert abnormality["leave_one_out_diagnostic"]["exact_four_level_agreement"] == (
        pytest.approx(1.0)
    )
    assert sum(
        abnormality["bootstrap_selection_stability"]["selection_counts"].values()
    ) == 100
    assert (output / "certainty_adapter.json").exists()
    assert (output / "certainty_margin_candidates.csv").exists()
    rendered = (output / "certainty_adapter_fit_receipt.json").read_text(encoding="utf-8")
    assert "report-000" not in rendered
    assert "Hashed_ReportURN" not in rendered


def test_fit_blocks_prediction_keys_outside_development_manifest(tmp_path: Path) -> None:
    contract, reference, predictions, prediction_receipt, manifest = _write_fixture(tmp_path)
    frame = pd.read_csv(predictions)
    frame.loc[len(frame)] = [
        "evaluation-report",
        "binary_core_certainty_adapter",
        *([0.5] * len(DEFAULT_LABELS)),
    ]
    frame.to_csv(predictions, index=False)
    receipt = json.loads(prediction_receipt.read_text(encoding="utf-8"))
    receipt["output"]["sha256"] = sha256_file(predictions)
    prediction_receipt.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="predictions_extra=1"):
        fit_certainty_adapter(
            contract,
            reference,
            predictions,
            prediction_receipt,
            manifest,
            tmp_path / "output",
            acknowledge_governed_inputs=True,
        )


def test_fit_requires_preregistered_manifest_checksum(tmp_path: Path) -> None:
    contract, reference, predictions, prediction_receipt, manifest = _write_fixture(tmp_path)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["certainty_mapping"]["development_manifest"]["sha256"] = "0" * 64
    contract.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        fit_certainty_adapter(
            contract,
            reference,
            predictions,
            prediction_receipt,
            manifest,
            tmp_path / "output",
            acknowledge_governed_inputs=True,
        )
