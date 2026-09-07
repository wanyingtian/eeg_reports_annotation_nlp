from __future__ import annotations

import json

import pandas as pd
import pytest

from eeg_review.input_fairness import (
    eligible_reference_rows,
    evaluate_exclusion_sensitivity,
    public_exposure_summary,
    tokenizer_diagnostics,
)

LABELS = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]


class FakeTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        return_offsets_mapping: bool,
        max_length: int | None = None,
    ) -> dict:
        del add_special_tokens, return_offsets_mapping
        offsets = [(0, 0)] + [(index, index + 1) for index in range(len(text))] + [(0, 0)]
        if truncation:
            assert max_length is not None
            offsets = offsets[: max_length - 1] + [(0, 0)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def reference_frame() -> pd.DataFrame:
    rows = [
        {"Hashed_ReportURN": "a", "Report": "short", **{label: 1 for label in LABELS}},
        {
            "Hashed_ReportURN": "b",
            "Report": "1234567890\nIMPRESSION:\nAbnormal",
            **{label: 4 for label in LABELS},
        },
        {"Hashed_ReportURN": "c", "Report": "invalid", **{label: 2 for label in LABELS}},
    ]
    rows[-1]["Gen Epi"] = None
    return pd.DataFrame(rows)


def test_eligibility_and_tokenizer_diagnostic_do_not_emit_report_text() -> None:
    eligible, excluded = eligible_reference_rows(reference_frame())
    assert excluded == 1
    diagnostic = tokenizer_diagnostics(eligible, FakeTokenizer(), maximum_sequence_length=10)
    assert diagnostic["truncation_exposed"].tolist() == [False, True]
    long_row = diagnostic.iloc[1]
    assert long_row["first_recognized_heading"] == "impression"
    assert bool(long_row["first_recognized_heading_retained"]) is False
    assert "Abnormal" not in diagnostic.to_csv(index=False)


def test_exclusion_sensitivity_uses_same_rows_for_every_model() -> None:
    eligible, _ = eligible_reference_rows(reference_frame())
    diagnostic = tokenizer_diagnostics(eligible, FakeTokenizer(), maximum_sequence_length=10)
    predictions = eligible[["Hashed_ReportURN", *LABELS]].copy()
    predictions.loc[predictions["Hashed_ReportURN"] == "b", "Abnormality"] = 1
    result = evaluate_exclusion_sensitivity(
        eligible,
        diagnostic,
        {"model": (predictions, {label: label for label in LABELS})},
        cohort="test",
    )
    abnormality = result[result["category"] == "Abnormality"].iloc[0]
    assert abnormality["eligible_reports"] == 2
    assert abnormality["content_complete_reports"] == 1
    assert abnormality["all_core_accuracy"] == 0.5
    assert abnormality["content_complete_core_accuracy"] == 1.0
    assert abnormality["change_core_accuracy"] == 0.5


def test_public_summary_contains_no_keys_or_text() -> None:
    eligible, _ = eligible_reference_rows(reference_frame())
    diagnostic = tokenizer_diagnostics(eligible, FakeTokenizer(), maximum_sequence_length=10)
    summary = public_exposure_summary({"test": diagnostic})
    rendered = json.dumps(summary)
    assert summary["cohorts"]["test"]["truncation_exposed_reports"] == 1
    assert summary["cohorts"]["test"]["exposed_with_first_recognized_heading_beyond_boundary"] == 1
    assert '"a"' not in rendered
    assert "Abnormal" not in rendered


def test_missing_prediction_key_is_rejected() -> None:
    eligible, _ = eligible_reference_rows(reference_frame())
    diagnostic = tokenizer_diagnostics(eligible, FakeTokenizer(), maximum_sequence_length=10)
    predictions = eligible.iloc[:1][["Hashed_ReportURN", *LABELS]]
    with pytest.raises(ValueError, match="missing 1 eligible report keys"):
        evaluate_exclusion_sensitivity(
            eligible,
            diagnostic,
            {"model": (predictions, {label: label for label in LABELS})},
            cohort="test",
        )
