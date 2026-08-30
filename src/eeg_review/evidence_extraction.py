"""Validated inputs and conservative quality measures for fixed-label evidence extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .io import load_table

JSON_KEYS = (
    "focal_epileptiform_activity",
    "generalized_epileptiform_activity",
    "focal_non_epileptiform_activity",
    "generalized_non_epileptiform_activity",
    "abnormality",
)
FALLBACK_EVIDENCE = "No specific mention in the report."


@dataclass(frozen=True)
class ExplanationInspection:
    structured_output_valid: bool
    decision_copy_mismatches: int
    evidence_phrases: int
    fallback_phrases: int
    exact_traceable_phrases: int
    casefold_traceable_phrases: int
    error: str | None = None


def classification_levels(raw: str) -> dict[str, int]:
    """Parse the exact fixed classification JSON and require five four-level decisions."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("classification is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != set(JSON_KEYS):
        raise ValueError("classification must contain exactly the five EEG keys")
    output: dict[str, int] = {}
    for key in JSON_KEYS:
        item = value[key]
        if isinstance(item, bool) or str(item).strip() not in {"1", "2", "3", "4"}:
            raise ValueError(f"classification value is not a four-level decision: {key}")
        output[key] = int(item)
    return output


def inspect_explanation(
    raw: str,
    *,
    report: str,
    fixed_classification: str,
) -> ExplanationInspection:
    """Check structure, copied decisions, and conservative verbatim traceability."""
    fixed = classification_levels(fixed_classification)
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != set(JSON_KEYS):
            raise ValueError("explanation must contain exactly the five EEG keys")
        mismatches = 0
        evidence = 0
        fallbacks = 0
        exact = 0
        casefold = 0
        for key in JSON_KEYS:
            item = value[key]
            if not isinstance(item, dict) or set(item) != {"decision", "reasons"}:
                raise ValueError(f"invalid explanation object: {key}")
            decision = item["decision"]
            if isinstance(decision, bool) or str(decision).strip() not in {"1", "2", "3", "4"}:
                raise ValueError(f"invalid explanation decision: {key}")
            mismatches += int(int(decision) != fixed[key])
            reasons = item["reasons"]
            if not isinstance(reasons, list) or not reasons:
                raise ValueError(f"explanation reasons must be a non-empty list: {key}")
            for reason in reasons:
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(f"explanation reason must be non-empty text: {key}")
                if reason == FALLBACK_EVIDENCE:
                    fallbacks += 1
                    continue
                evidence += 1
                exact += int(reason in report)
                casefold += int(reason.casefold() in report.casefold())
        return ExplanationInspection(
            structured_output_valid=True,
            decision_copy_mismatches=mismatches,
            evidence_phrases=evidence,
            fallback_phrases=fallbacks,
            exact_traceable_phrases=exact,
            casefold_traceable_phrases=casefold,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return ExplanationInspection(
            structured_output_valid=False,
            decision_copy_mismatches=0,
            evidence_phrases=0,
            fallback_phrases=0,
            exact_traceable_phrases=0,
            casefold_traceable_phrases=0,
            error=str(exc),
        )


def load_fixed_evidence_inputs(
    *,
    dataset: Path,
    predictions: Path,
    manifest: Path,
    table: str = "reports",
    id_column: str = "Hashed_ReportURN",
    report_column: str = "Report",
    classification_column: str = "classifications",
) -> pd.DataFrame:
    """Join reports and fixed predictions in the exact governed manifest order."""
    source = load_table(dataset, [id_column, report_column], table)
    prediction_frame = pd.read_csv(predictions, usecols=[id_column, classification_column])
    manifest_frame = pd.read_csv(manifest, usecols=[id_column])

    for name, frame in [
        ("dataset", source),
        ("predictions", prediction_frame),
        ("manifest", manifest_frame),
    ]:
        keys = frame[id_column].astype("string")
        if keys.isna().any() or (keys.str.len() == 0).any():
            raise ValueError(f"{name} contains a missing report key")
        if keys.duplicated().any():
            raise ValueError(f"{name} contains duplicate report keys")
        frame[id_column] = keys.astype(str)

    manifest_keys = manifest_frame[id_column].tolist()
    source_index = source.set_index(id_column)
    prediction_index = prediction_frame.set_index(id_column)
    missing_source = [key for key in manifest_keys if key not in source_index.index]
    missing_prediction = [key for key in manifest_keys if key not in prediction_index.index]
    if missing_source:
        raise ValueError(f"dataset is missing {len(missing_source)} manifest keys")
    if missing_prediction:
        raise ValueError(f"predictions are missing {len(missing_prediction)} manifest keys")

    records: list[dict[str, Any]] = []
    for key in manifest_keys:
        report = source_index.at[key, report_column]
        if not isinstance(report, str) or not report.strip():
            raise ValueError("dataset contains a missing or empty report")
        classification = str(prediction_index.at[key, classification_column])
        classification_levels(classification)
        records.append(
            {
                id_column: key,
                report_column: report,
                classification_column: classification,
            }
        )
    return pd.DataFrame(records)


def aggregate_inspections(frame: pd.DataFrame) -> dict[str, Any]:
    """Aggregate only non-identifying evidence-quality counters."""
    records = int(len(frame))
    valid = int(pd.to_numeric(frame["structured_output_valid"], errors="coerce").fillna(0).sum())
    evidence = int(pd.to_numeric(frame["evidence_phrases"], errors="coerce").fillna(0).sum())
    exact = int(
        pd.to_numeric(frame["exact_traceable_phrases"], errors="coerce").fillna(0).sum()
    )
    casefold = int(
        pd.to_numeric(frame["casefold_traceable_phrases"], errors="coerce").fillna(0).sum()
    )
    return {
        "records": records,
        "valid_structured_outputs": valid,
        "invalid_structured_outputs": records - valid,
        "decision_copy_mismatches": int(
            pd.to_numeric(frame["decision_copy_mismatches"], errors="coerce").fillna(0).sum()
        ),
        "evidence_phrases": evidence,
        "fallback_phrases": int(
            pd.to_numeric(frame["fallback_phrases"], errors="coerce").fillna(0).sum()
        ),
        "exact_traceable_phrases": exact,
        "casefold_traceable_phrases": casefold,
        "exact_traceability_fraction": exact / evidence if evidence else None,
        "casefold_traceability_fraction": casefold / evidence if evidence else None,
    }
