from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import DEFAULT_LABELS
from .io import atomic_write_csv, atomic_write_json, load_table
from .manifest import build_manifest


def select_reference_rows(
    reference: pd.DataFrame,
    row_ranges: list[tuple[int, int]] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply explicit half-open positional ranges without modifying the source."""
    source_records = len(reference)
    if not row_ranges:
        return reference, {
            "method": "all_rows",
            "source_records": source_records,
            "candidate_records": source_records,
            "row_ranges": None,
        }

    previous_end = 0
    for index, (start, end) in enumerate(row_ranges):
        if start < 0 or end <= start:
            raise ValueError(f"Invalid reference range {start}:{end}")
        if end > source_records:
            raise ValueError(
                f"Reference range {start}:{end} exceeds source length {source_records}"
            )
        if index and start < previous_end:
            raise ValueError("Reference ranges must be ordered and non-overlapping")
        previous_end = end

    selected = pd.concat(
        [reference.iloc[start:end] for start, end in row_ranges],
        ignore_index=True,
    )
    return selected, {
        "method": "half_open_positional_ranges",
        "source_records": source_records,
        "candidate_records": len(selected),
        "row_ranges": [{"start": start, "end": end} for start, end in row_ranges],
    }


def cohen_kappa(reference: np.ndarray, prediction: np.ndarray, levels: list[int]) -> float:
    if reference.size == 0:
        return float("nan")
    observed = float(np.mean(reference == prediction))
    ref_prob = np.array([np.mean(reference == level) for level in levels])
    pred_prob = np.array([np.mean(prediction == level) for level in levels])
    expected = float(np.dot(ref_prob, pred_prob))
    return float("nan") if np.isclose(expected, 1.0) else (observed - expected) / (1.0 - expected)


def metric_values(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    ref_core = np.isin(reference, [3, 4]).astype(int)
    pred_core = np.isin(prediction, [3, 4]).astype(int)
    tn = int(np.sum((ref_core == 0) & (pred_core == 0)))
    fp = int(np.sum((ref_core == 0) & (pred_core == 1)))
    fn = int(np.sum((ref_core == 1) & (pred_core == 0)))
    tp = int(np.sum((ref_core == 1) & (pred_core == 1)))

    def divide(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    return {
        "n": int(reference.size),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "core_accuracy": float(np.mean(ref_core == pred_core)),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": divide(tn, tn + fp),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "certainty_adjusted_accuracy": float(np.mean(reference == prediction)),
        "core_kappa": cohen_kappa(ref_core, pred_core, [0, 1]),
        "four_level_kappa": cohen_kappa(reference, prediction, [1, 2, 3, 4]),
    }


def bootstrap_intervals(
    reference: np.ndarray,
    prediction: np.ndarray,
    clusters: np.ndarray | None,
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    rng = np.random.default_rng(seed)
    metric_names = [
        "core_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "certainty_adjusted_accuracy",
        "core_kappa",
        "four_level_kappa",
    ]
    samples = {name: [] for name in metric_names}
    if clusters is None:
        groups = [np.array([index]) for index in range(reference.size)]
    else:
        groups = [np.flatnonzero(clusters == value) for value in pd.unique(clusters)]
    for _ in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        values = metric_values(reference[indices], prediction[indices])
        for name in metric_names:
            value = float(values[name])
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


def confusion_matrix(
    reference: np.ndarray, prediction: np.ndarray, levels: list[int]
) -> list[list[int]]:
    return [
        [int(np.sum((reference == actual) & (prediction == predicted))) for predicted in levels]
        for actual in levels
    ]


def evaluate_predictions(
    reference_path: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    reference_table: str = "reports",
    prediction_table: str = "classifications",
    id_column: str = "Hashed_ReportURN",
    labels: list[str] | None = None,
    prediction_columns: dict[str, str] | None = None,
    cluster_column: str | None = None,
    fold_column: str | None = None,
    reference_row_ranges: list[tuple[int, int]] | None = None,
    require_complete_reference: bool = False,
    require_exact_key_set: bool = False,
    require_patient_grouping: bool = False,
    bootstrap_iterations: int = 2000,
    seed: int = 20260718,
) -> dict[str, Any]:
    labels = labels or DEFAULT_LABELS
    prediction_columns = prediction_columns or {label: label for label in labels}
    missing_mappings = sorted(set(labels) - set(prediction_columns))
    if missing_mappings:
        raise ValueError(f"Missing prediction-column mappings for: {missing_mappings}")
    if require_patient_grouping and not cluster_column:
        raise ValueError("Patient grouping is required but no cluster column was supplied")

    reference_columns = [id_column, *labels]
    if cluster_column:
        reference_columns.append(cluster_column)
    prediction_input_columns = [id_column, *[prediction_columns[label] for label in labels]]
    if fold_column:
        prediction_input_columns.append(fold_column)
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
    predictions = load_table(predictions_path, prediction_input_columns, prediction_table)
    if reference[id_column].isna().any():
        raise ValueError("Reference identifiers are missing")
    if predictions[id_column].isna().any():
        raise ValueError("Prediction identifiers are missing")
    if reference[id_column].duplicated().any():
        raise ValueError("Reference identifiers are not unique")
    if predictions[id_column].duplicated().any():
        raise ValueError("Prediction identifiers are not unique")

    reference_keys = set(reference[id_column])
    prediction_keys = set(predictions[id_column])
    key_alignment = {
        "reference_missing_from_predictions": len(reference_keys - prediction_keys),
        "predictions_extra_vs_reference": len(prediction_keys - reference_keys),
    }
    key_alignment["exact_key_set"] = not any(key_alignment.values())
    key_alignment["exact_key_set_required"] = require_exact_key_set
    if require_exact_key_set and not key_alignment["exact_key_set"]:
        counts = ", ".join(
            f"{name}={value}"
            for name, value in key_alignment.items()
            if name not in {"exact_key_set", "exact_key_set_required"}
        )
        raise ValueError(f"Exact report-key alignment failed: {counts}")

    predictions = predictions.rename(
        columns={prediction_columns[label]: f"{label}__prediction" for label in labels}
    )
    merged = reference.merge(predictions, on=id_column, how="inner", validate="one_to_one")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "reference_records": int(len(reference)),
        "prediction_records": int(len(predictions)),
        "matched_records": int(len(merged)),
        "unmatched_reference_records": int(len(reference) - len(merged)),
        "unmatched_prediction_records": int(len(predictions) - len(merged)),
        "key_alignment": key_alignment,
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": seed,
            "unit": "cluster" if cluster_column else "report",
            "cluster_column_supplied": bool(cluster_column),
        },
        "labels": {},
        "fold_column": fold_column,
        "reference_selection": selection,
        "interpretation_limits": [],
    }
    if not cluster_column:
        summary["interpretation_limits"].append(
            "No patient/cluster column supplied; confidence intervals resample reports and "
            "do not establish patient independence."
        )

    metric_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    matrices: dict[str, Any] = {}
    for offset, label in enumerate(labels):
        ref = pd.to_numeric(merged[label], errors="coerce")
        pred = pd.to_numeric(merged[f"{label}__prediction"], errors="coerce")
        valid = ref.isin([1, 2, 3, 4]) & pred.isin([1, 2, 3, 4])
        reference_values = ref[valid].to_numpy(dtype=int)
        prediction_values = pred[valid].to_numpy(dtype=int)
        if reference_values.size == 0:
            raise ValueError(f"No valid paired four-level labels for {label}")
        clusters = (
            merged.loc[valid, cluster_column].astype(str).to_numpy() if cluster_column else None
        )
        if cluster_column and merged.loc[valid, cluster_column].isna().any():
            raise ValueError(f"Missing cluster identifiers among valid pairs for {label}")
        point = metric_values(reference_values, prediction_values)
        intervals = bootstrap_intervals(
            reference_values,
            prediction_values,
            clusters,
            iterations=bootstrap_iterations,
            seed=seed + offset,
        )
        label_summary = {
            "excluded_invalid_or_missing_pairs": int((~valid).sum()),
            "point_estimates": point,
            "confidence_intervals_95": intervals,
        }
        if fold_column:
            fold_values: dict[str, list[float]] = {}
            for fold, fold_frame in merged.loc[valid].groupby(fold_column, dropna=False):
                fold_reference = pd.to_numeric(fold_frame[label], errors="raise").to_numpy(
                    dtype=int
                )
                fold_prediction = pd.to_numeric(
                    fold_frame[f"{label}__prediction"], errors="raise"
                ).to_numpy(dtype=int)
                values = metric_values(fold_reference, fold_prediction)
                for metric, value in values.items():
                    fold_metric_rows.append(
                        {"label": label, "fold": str(fold), "metric": metric, "value": value}
                    )
                    if metric not in {"tn", "fp", "fn", "tp", "n"} and np.isfinite(value):
                        fold_values.setdefault(metric, []).append(float(value))
            label_summary["fold_variability"] = {
                metric: {
                    "folds": len(values),
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
                for metric, values in fold_values.items()
            }
        summary["labels"][label] = label_summary
        for metric, value in point.items():
            if metric in {"tn", "fp", "fn", "tp", "n"}:
                metric_rows.append(
                    {
                        "label": label,
                        "metric": metric,
                        "estimate": value,
                        "ci_low": "",
                        "ci_high": "",
                    }
                )
            else:
                interval = intervals[metric]
                metric_rows.append(
                    {
                        "label": label,
                        "metric": metric,
                        "estimate": value,
                        "ci_low": interval["low"],
                        "ci_high": interval["high"],
                    }
                )
        ref_core = np.isin(reference_values, [3, 4]).astype(int)
        pred_core = np.isin(prediction_values, [3, 4]).astype(int)
        matrices[label] = {
            "core_levels": ["absent", "present"],
            "core": confusion_matrix(ref_core, pred_core, [0, 1]),
            "four_level_levels": [1, 2, 3, 4],
            "four_level": confusion_matrix(reference_values, prediction_values, [1, 2, 3, 4]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "evaluation_summary.json", summary)
    atomic_write_csv(output_dir / "metrics.csv", pd.DataFrame(metric_rows))
    if fold_metric_rows:
        atomic_write_csv(output_dir / "fold_metrics.csv", pd.DataFrame(fold_metric_rows))
    atomic_write_json(output_dir / "confusion_matrices.json", matrices)
    atomic_write_json(
        output_dir / "run_manifest.json",
        build_manifest(
            "evaluate",
            [reference_path, predictions_path],
            {
                "reference_table": reference_table,
                "prediction_table": prediction_table,
                "id_column": id_column,
                "labels": labels,
                "prediction_columns": prediction_columns,
                "cluster_column": cluster_column,
                "fold_column": fold_column,
                "reference_row_ranges": reference_row_ranges,
                "require_complete_reference": require_complete_reference,
                "require_exact_key_set": require_exact_key_set,
                "require_patient_grouping": require_patient_grouping,
                "bootstrap_iterations": bootstrap_iterations,
                "seed": seed,
            },
        ),
    )
    return summary
