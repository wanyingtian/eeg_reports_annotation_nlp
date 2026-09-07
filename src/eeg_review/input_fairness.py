"""Read-only input-length sensitivity for saved EEG model predictions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .audit import DEFAULT_LABELS
from .metrics import metric_values

SECTION_HEADING = re.compile(
    r"(?im)^[ \t]*(impression|interpretation|conclusion|clinical[ \t]+correlation)"
    r"[ \t]*:?"
)


def _validated_keys(frame: pd.DataFrame, *, key_column: str) -> pd.Series:
    if key_column not in frame:
        raise ValueError(f"missing report-key column: {key_column}")
    keys = frame[key_column].map(lambda value: str(value).strip())
    if (keys == "").any() or keys.duplicated().any():
        raise ValueError("report keys must be nonblank and unique")
    return keys


def eligible_reference_rows(
    frame: pd.DataFrame,
    *,
    key_column: str = "Hashed_ReportURN",
    report_column: str = "Report",
    labels: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Return complete five-label reference rows without changing their order."""
    labels = labels or DEFAULT_LABELS
    missing = sorted({key_column, report_column, *labels} - set(frame.columns))
    if missing:
        raise ValueError(f"reference is missing columns: {missing}")
    result = frame.copy()
    result[key_column] = _validated_keys(result, key_column=key_column)
    invalid_report = result[report_column].map(
        lambda value: not isinstance(value, str) or not value.strip()
    )
    if invalid_report.any():
        raise ValueError("report text must be a nonblank string")
    valid = pd.DataFrame(
        {
            label: pd.to_numeric(result[label], errors="coerce").isin([1, 2, 3, 4])
            for label in labels
        }
    ).all(axis=1)
    excluded = int((~valid).sum())
    result = result.loc[valid].reset_index(drop=True)
    for label in labels:
        result[label] = pd.to_numeric(result[label]).astype(int)
    return result, excluded


def tokenizer_diagnostics(
    frame: pd.DataFrame,
    tokenizer: Any,
    *,
    maximum_sequence_length: int = 512,
    key_column: str = "Hashed_ReportURN",
    report_column: str = "Report",
) -> pd.DataFrame:
    """Locate reports exposed to right truncation without returning report text."""
    if maximum_sequence_length < 3:
        raise ValueError("maximum_sequence_length must leave room for content and special tokens")
    _validated_keys(frame, key_column=key_column)
    rows: list[dict[str, Any]] = []
    for key, report in zip(frame[key_column], frame[report_column], strict=True):
        full = tokenizer(
            report,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        clipped = tokenizer(
            report,
            add_special_tokens=True,
            truncation=True,
            max_length=maximum_sequence_length,
            return_offsets_mapping=True,
        )
        token_count = len(full["input_ids"])
        offsets = clipped["offset_mapping"]
        retained_character_end = max((int(end) for start, end in offsets if end > start), default=0)
        headings = [
            {
                "heading": " ".join(match.group(1).lower().split()),
                "start": int(match.start()),
                "retained": bool(match.start() < retained_character_end),
            }
            for match in SECTION_HEADING.finditer(report)
        ]
        rows.append(
            {
                key_column: str(key).strip(),
                "token_count_with_special_tokens": token_count,
                "maximum_sequence_length": maximum_sequence_length,
                "truncation_exposed": token_count > maximum_sequence_length,
                "retained_character_end": retained_character_end,
                "recognized_section_headings": len(headings),
                "recognized_section_headings_beyond_boundary": sum(
                    not item["retained"] for item in headings
                ),
                "first_recognized_heading": headings[0]["heading"] if headings else None,
                "first_recognized_heading_retained": headings[0]["retained"] if headings else None,
            }
        )
    return pd.DataFrame(rows)


def _standardize_predictions(
    frame: pd.DataFrame,
    *,
    prediction_columns: Mapping[str, str],
    key_column: str,
    labels: list[str],
) -> pd.DataFrame:
    missing_mappings = sorted(set(labels) - set(prediction_columns))
    if missing_mappings:
        raise ValueError(f"prediction mappings are missing labels: {missing_mappings}")
    required = {key_column, *[prediction_columns[label] for label in labels]}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prediction frame is missing columns: {missing}")
    result = frame[[key_column, *[prediction_columns[label] for label in labels]]].copy()
    result[key_column] = _validated_keys(result, key_column=key_column)
    result = result.rename(columns={prediction_columns[label]: label for label in labels})
    for label in labels:
        numeric = pd.to_numeric(result[label], errors="coerce")
        if not numeric.isin([1, 2, 3, 4]).all():
            raise ValueError(f"predictions for {label} must all be in 1..4")
        result[label] = numeric.astype(int)
    return result


def evaluate_exclusion_sensitivity(
    reference: pd.DataFrame,
    diagnostics: pd.DataFrame,
    predictions: Mapping[str, tuple[pd.DataFrame, Mapping[str, str]]],
    *,
    cohort: str,
    key_column: str = "Hashed_ReportURN",
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compare full eligible rows with a common truncation-excluded population."""
    labels = labels or DEFAULT_LABELS
    reference_keys = set(_validated_keys(reference, key_column=key_column))
    diagnostics_keys = set(_validated_keys(diagnostics, key_column=key_column))
    if diagnostics_keys != reference_keys:
        raise ValueError("tokenizer diagnostics must exactly match eligible reference keys")
    exposed_keys = set(
        diagnostics.loc[diagnostics["truncation_exposed"].astype(bool), key_column].map(str)
    )
    populations = {
        "all_eligible": reference_keys,
        "bert_content_complete": reference_keys - exposed_keys,
    }
    rows: list[dict[str, Any]] = []
    for model, (raw_predictions, mappings) in sorted(predictions.items()):
        model_predictions = _standardize_predictions(
            raw_predictions,
            prediction_columns=mappings,
            key_column=key_column,
            labels=labels,
        )
        prediction_keys = set(model_predictions[key_column])
        missing = reference_keys - prediction_keys
        if missing:
            raise ValueError(f"{model} is missing {len(missing)} eligible report keys")
        joined = reference[[key_column, *labels]].merge(
            model_predictions,
            on=key_column,
            how="left",
            validate="one_to_one",
            suffixes=("__reference", "__prediction"),
        )
        values_by_population: dict[str, dict[str, dict[str, float | int]]] = {}
        for population, keys in populations.items():
            selected = joined[joined[key_column].isin(keys)]
            values_by_population[population] = {}
            for label in labels:
                values_by_population[population][label] = metric_values(
                    selected[f"{label}__reference"].to_numpy(dtype=int),
                    selected[f"{label}__prediction"].to_numpy(dtype=int),
                )
        for label in labels:
            full = values_by_population["all_eligible"][label]
            retained = values_by_population["bert_content_complete"][label]
            row: dict[str, Any] = {
                "cohort": cohort,
                "model": model,
                "category": label,
                "eligible_reports": len(reference_keys),
                "truncation_exposed_reports": len(exposed_keys),
                "content_complete_reports": len(reference_keys - exposed_keys),
            }
            for metric, full_value in full.items():
                retained_value = retained[metric]
                row[f"all_{metric}"] = full_value
                row[f"content_complete_{metric}"] = retained_value
                if isinstance(full_value, (int, np.integer)):
                    row[f"change_{metric}"] = int(retained_value) - int(full_value)
                else:
                    full_float = float(full_value)
                    retained_float = float(retained_value)
                    row[f"change_{metric}"] = (
                        retained_float - full_float
                        if np.isfinite(full_float) and np.isfinite(retained_float)
                        else float("nan")
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def public_exposure_summary(
    diagnostics_by_cohort: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    """Aggregate tokenizer exposure without emitting report keys or text."""
    cohorts: dict[str, dict[str, object]] = {}
    for cohort, frame in sorted(diagnostics_by_cohort.items()):
        exposed = frame[frame["truncation_exposed"].astype(bool)]
        with_headings = exposed[exposed["recognized_section_headings"] > 0]
        cohorts[cohort] = {
            "eligible_reports": len(frame),
            "truncation_exposed_reports": len(exposed),
            "truncation_exposed_percent": 100.0 * len(exposed) / len(frame),
            "exposed_with_recognized_downstream_heading": len(with_headings),
            "exposed_with_any_recognized_heading_beyond_boundary": int(
                (exposed["recognized_section_headings_beyond_boundary"] > 0).sum()
            ),
            "exposed_with_first_recognized_heading_beyond_boundary": int(
                (with_headings["first_recognized_heading_retained"] == False).sum()  # noqa: E712
            ),
        }
    return {
        "cohorts": cohorts,
        "contains_report_keys_or_text": False,
        "recognized_headings": [
            "impression",
            "interpretation",
            "conclusion",
            "clinical correlation",
        ],
    }
