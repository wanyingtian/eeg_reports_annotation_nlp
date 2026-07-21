from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import DEFAULT_LABELS
from .io import atomic_write_csv, atomic_write_json, load_table
from .manifest import build_manifest
from .metrics import metric_values, select_reference_rows


def exact_mcnemar_p_value(a_correct_b_wrong: int, a_wrong_b_correct: int) -> float:
    """Two-sided exact McNemar p-value conditional on discordant pairs."""
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        return 1.0
    tail = min(a_correct_b_wrong, a_wrong_b_correct)
    log_probabilities = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2)
        for k in range(tail + 1)
    ]
    maximum = max(log_probabilities)
    lower_tail = math.exp(maximum) * sum(math.exp(value - maximum) for value in log_probabilities)
    return min(1.0, 2 * lower_tail)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise error correction, returned in original order."""
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def comparison_effects(
    reference: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
) -> dict[str, float]:
    values_a = metric_values(reference, prediction_a)
    values_b = metric_values(reference, prediction_b)
    recall_a = float(values_a["recall_sensitivity"])
    recall_b = float(values_b["recall_sensitivity"])
    return {
        "core_accuracy_difference": float(values_a["core_accuracy"])
        - float(values_b["core_accuracy"]),
        "certainty_adjusted_accuracy_difference": float(values_a["certainty_adjusted_accuracy"])
        - float(values_b["certainty_adjusted_accuracy"]),
        "false_negative_rate_difference": (1 - recall_a) - (1 - recall_b),
    }


def paired_bootstrap_intervals(
    reference: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    clusters: np.ndarray | None,
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    rng = np.random.default_rng(seed)
    effect_names = [
        "core_accuracy_difference",
        "certainty_adjusted_accuracy_difference",
        "false_negative_rate_difference",
    ]
    samples = {name: [] for name in effect_names}
    if clusters is None:
        groups = [np.array([index]) for index in range(reference.size)]
    else:
        groups = [np.flatnonzero(clusters == value) for value in pd.unique(clusters)]
    for _ in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        effects = comparison_effects(
            reference[indices], prediction_a[indices], prediction_b[indices]
        )
        for name, value in effects.items():
            if np.isfinite(value):
                samples[name].append(value)

    intervals: dict[str, dict[str, float | None]] = {}
    for name, values in samples.items():
        if values:
            low, high = np.percentile(values, [2.5, 97.5])
            intervals[name] = {"low": float(low), "high": float(high)}
        else:
            intervals[name] = {"low": None, "high": None}
    return intervals


def discordant_correctness(
    reference: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    *,
    core: bool,
) -> dict[str, int | float]:
    if core:
        reference = np.isin(reference, [3, 4]).astype(int)
        prediction_a = np.isin(prediction_a, [3, 4]).astype(int)
        prediction_b = np.isin(prediction_b, [3, 4]).astype(int)
    correct_a = prediction_a == reference
    correct_b = prediction_b == reference
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    return {
        "both_correct": int(np.sum(correct_a & correct_b)),
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": int(np.sum(~correct_a & ~correct_b)),
        "mcnemar_exact_p_value": exact_mcnemar_p_value(a_correct_b_wrong, a_wrong_b_correct),
    }


def compare_predictions(
    reference_path: Path,
    predictions_a_path: Path,
    predictions_b_path: Path,
    output_dir: Path,
    *,
    model_a_id: str,
    model_b_id: str,
    reference_table: str = "reports",
    prediction_a_table: str = "classifications",
    prediction_b_table: str = "classifications",
    id_column: str = "Hashed_ReportURN",
    labels: list[str] | None = None,
    prediction_a_columns: dict[str, str] | None = None,
    prediction_b_columns: dict[str, str] | None = None,
    cluster_column: str | None = None,
    reference_row_ranges: list[tuple[int, int]] | None = None,
    require_complete_reference: bool = False,
    bootstrap_iterations: int = 2000,
    seed: int = 20260718,
    multiplicity: str = "holm",
) -> dict[str, Any]:
    labels = labels or DEFAULT_LABELS
    prediction_a_columns = prediction_a_columns or {label: label for label in labels}
    prediction_b_columns = prediction_b_columns or {label: label for label in labels}
    for model_id, mappings in (
        (model_a_id, prediction_a_columns),
        (model_b_id, prediction_b_columns),
    ):
        missing = sorted(set(labels) - set(mappings))
        if missing:
            raise ValueError(f"Missing prediction-column mappings for {model_id}: {missing}")
    if model_a_id == model_b_id:
        raise ValueError("Model IDs must be distinct")
    if multiplicity not in {"holm", "none"}:
        raise ValueError("Multiplicity must be 'holm' or 'none'")

    reference_columns = [id_column, *labels]
    if cluster_column:
        reference_columns.append(cluster_column)
    reference = load_table(reference_path, reference_columns, reference_table)
    reference, selection = select_reference_rows(reference, reference_row_ranges)
    if require_complete_reference:
        complete = pd.DataFrame(
            {
                label: pd.to_numeric(reference[label], errors="coerce").isin([1, 2, 3, 4])
                for label in labels
            }
        ).all(axis=1)
        selection["complete_reference_required"] = True
        selection["excluded_incomplete_reference_records"] = int((~complete).sum())
        reference = reference.loc[complete].reset_index(drop=True)
    else:
        selection["complete_reference_required"] = False
        selection["excluded_incomplete_reference_records"] = 0

    def load_predictions(
        path: Path, table: str, mappings: dict[str, str], suffix: str
    ) -> pd.DataFrame:
        frame = load_table(path, [id_column, *[mappings[label] for label in labels]], table)
        if frame[id_column].duplicated().any():
            raise ValueError(f"Prediction identifiers are not unique for model {suffix}")
        return frame.rename(
            columns={mappings[label]: f"{label}__prediction_{suffix}" for label in labels}
        )

    if reference[id_column].duplicated().any():
        raise ValueError("Reference identifiers are not unique")
    predictions_a = load_predictions(
        predictions_a_path, prediction_a_table, prediction_a_columns, "a"
    )
    predictions_b = load_predictions(
        predictions_b_path, prediction_b_table, prediction_b_columns, "b"
    )
    merged = reference.merge(predictions_a, on=id_column, how="inner", validate="one_to_one")
    merged = merged.merge(predictions_b, on=id_column, how="inner", validate="one_to_one")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "models": {"a": model_a_id, "b": model_b_id},
        "difference_direction": "model_a_minus_model_b",
        "reference_records": int(len(reference)),
        "prediction_a_records": int(len(predictions_a)),
        "prediction_b_records": int(len(predictions_b)),
        "matched_records": int(len(merged)),
        "unmatched_reference_records": int(len(reference) - len(merged)),
        "reference_selection": selection,
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": seed,
            "unit": "cluster" if cluster_column else "report",
            "cluster_column_supplied": bool(cluster_column),
        },
        "multiplicity": {
            "method": multiplicity,
            "family": "all labels x core/four-level exact McNemar tests in this run",
        },
        "labels": {},
        "interpretation_limits": [],
    }
    if not cluster_column:
        summary["interpretation_limits"].append(
            "No patient/cluster column supplied; paired intervals resample reports and do not "
            "establish patient independence."
        )
    summary["interpretation_limits"].append(
        "Exact McNemar tests operate on report-level discordant correctness pairs. They do not "
        "account for repeated reports within patients; when patient clusters exist, treat the "
        "cluster-bootstrap interval as primary and McNemar as a sensitivity analysis."
    )

    test_locations: list[tuple[str, str]] = []
    raw_p_values: list[float] = []
    rows: list[dict[str, Any]] = []
    for offset, label in enumerate(labels):
        reference_series = pd.to_numeric(merged[label], errors="coerce")
        prediction_a_series = pd.to_numeric(merged[f"{label}__prediction_a"], errors="coerce")
        prediction_b_series = pd.to_numeric(merged[f"{label}__prediction_b"], errors="coerce")
        valid = (
            reference_series.isin([1, 2, 3, 4])
            & prediction_a_series.isin([1, 2, 3, 4])
            & prediction_b_series.isin([1, 2, 3, 4])
        )
        reference_values = reference_series[valid].to_numpy(dtype=int)
        prediction_a_values = prediction_a_series[valid].to_numpy(dtype=int)
        prediction_b_values = prediction_b_series[valid].to_numpy(dtype=int)
        if reference_values.size == 0:
            raise ValueError(f"No valid three-way paired labels for {label}")
        clusters = (
            merged.loc[valid, cluster_column].astype(str).to_numpy() if cluster_column else None
        )
        if cluster_column and merged.loc[valid, cluster_column].isna().any():
            raise ValueError(f"Missing cluster identifiers among valid pairs for {label}")

        point_a = metric_values(reference_values, prediction_a_values)
        point_b = metric_values(reference_values, prediction_b_values)
        effects = comparison_effects(reference_values, prediction_a_values, prediction_b_values)
        intervals = paired_bootstrap_intervals(
            reference_values,
            prediction_a_values,
            prediction_b_values,
            clusters,
            iterations=bootstrap_iterations,
            seed=seed + offset,
        )
        discordance = {
            "core_accuracy": discordant_correctness(
                reference_values, prediction_a_values, prediction_b_values, core=True
            ),
            "certainty_adjusted_accuracy": discordant_correctness(
                reference_values, prediction_a_values, prediction_b_values, core=False
            ),
        }
        label_summary = {
            "n": int(reference_values.size),
            "reference_core_positive": int(np.isin(reference_values, [3, 4]).sum()),
            "excluded_invalid_or_missing_three_way_pairs": int((~valid).sum()),
            "model_a_point_estimates": point_a,
            "model_b_point_estimates": point_b,
            "effects_a_minus_b": effects,
            "paired_confidence_intervals_95": intervals,
            "discordant_correctness": discordance,
        }
        summary["labels"][label] = label_summary
        for outcome, details in discordance.items():
            test_locations.append((label, outcome))
            raw_p_values.append(float(details["mcnemar_exact_p_value"]))

        for effect_name, estimate in effects.items():
            interval = intervals[effect_name]
            rows.append(
                {
                    "label": label,
                    "outcome": effect_name,
                    "model_a": model_a_id,
                    "model_b": model_b_id,
                    "estimate_a_minus_b": estimate,
                    "ci_low": interval["low"],
                    "ci_high": interval["high"],
                    "a_correct_b_wrong": "",
                    "a_wrong_b_correct": "",
                    "mcnemar_exact_p_value": "",
                    "multiplicity_adjusted_p_value": "",
                }
            )

    adjusted = holm_adjust(raw_p_values) if multiplicity == "holm" else raw_p_values
    for (label, outcome), raw, corrected in zip(
        test_locations, raw_p_values, adjusted, strict=True
    ):
        details = summary["labels"][label]["discordant_correctness"][outcome]
        details["multiplicity_adjusted_p_value"] = corrected
        rows.append(
            {
                "label": label,
                "outcome": outcome,
                "model_a": model_a_id,
                "model_b": model_b_id,
                "estimate_a_minus_b": "",
                "ci_low": "",
                "ci_high": "",
                "a_correct_b_wrong": details["a_correct_b_wrong"],
                "a_wrong_b_correct": details["a_wrong_b_correct"],
                "mcnemar_exact_p_value": raw,
                "multiplicity_adjusted_p_value": corrected,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "paired_comparison_summary.json", summary)
    atomic_write_csv(output_dir / "paired_comparisons.csv", pd.DataFrame(rows))
    atomic_write_json(
        output_dir / "run_manifest.json",
        build_manifest(
            "compare",
            [reference_path, predictions_a_path, predictions_b_path],
            {
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
                "reference_table": reference_table,
                "prediction_a_table": prediction_a_table,
                "prediction_b_table": prediction_b_table,
                "id_column": id_column,
                "labels": labels,
                "prediction_a_columns": prediction_a_columns,
                "prediction_b_columns": prediction_b_columns,
                "cluster_column": cluster_column,
                "reference_row_ranges": reference_row_ranges,
                "require_complete_reference": require_complete_reference,
                "bootstrap_iterations": bootstrap_iterations,
                "seed": seed,
                "multiplicity": multiplicity,
            },
        ),
    )
    return summary
