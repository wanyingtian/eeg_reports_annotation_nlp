from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import DEFAULT_LABELS
from .io import atomic_write_csv, atomic_write_json, load_table
from .manifest import build_manifest
from .metrics import select_reference_rows


def calibration_values(
    outcome: np.ndarray, probability: np.ndarray, *, bins: int
) -> tuple[dict[str, float | int], list[dict[str, float | int | None]]]:
    if outcome.size == 0:
        raise ValueError("Calibration requires at least one valid pair")
    if bins < 2:
        raise ValueError("Calibration requires at least two bins")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be within [0, 1]")
    clipped = np.clip(probability, 1e-15, 1 - 1e-15)
    brier = float(np.mean((probability - outcome) ** 2))
    log_loss = float(-np.mean(outcome * np.log(clipped) + (1 - outcome) * np.log1p(-clipped)))

    bin_index = np.minimum((probability * bins).astype(int), bins - 1)
    bin_rows: list[dict[str, float | int | None]] = []
    weighted_gap = 0.0
    for index in range(bins):
        selected = bin_index == index
        count = int(selected.sum())
        mean_probability = float(np.mean(probability[selected])) if count else None
        event_rate = float(np.mean(outcome[selected])) if count else None
        absolute_gap = abs(mean_probability - event_rate) if count else None
        if count:
            assert absolute_gap is not None
            weighted_gap += count / outcome.size * absolute_gap
        bin_rows.append(
            {
                "bin": index + 1,
                "lower_inclusive": index / bins,
                "upper_inclusive": (index + 1) / bins if index == bins - 1 else None,
                "upper_exclusive": (index + 1) / bins if index < bins - 1 else None,
                "count": count,
                "mean_probability": mean_probability,
                "event_rate": event_rate,
                "absolute_gap": absolute_gap,
            }
        )
    return (
        {
            "n": int(outcome.size),
            "positives": int(outcome.sum()),
            "prevalence": float(np.mean(outcome)),
            "mean_predicted_probability": float(np.mean(probability)),
            "brier_score": brier,
            "log_loss": log_loss,
            "expected_calibration_error": float(weighted_gap),
        },
        bin_rows,
    )


def calibration_bootstrap_intervals(
    outcome: np.ndarray,
    probability: np.ndarray,
    clusters: np.ndarray | None,
    *,
    bins: int,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    rng = np.random.default_rng(seed)
    metric_names = ["brier_score", "log_loss", "expected_calibration_error"]
    samples = {name: [] for name in metric_names}
    if clusters is None:
        groups = [np.array([index]) for index in range(outcome.size)]
    else:
        groups = [np.flatnonzero(clusters == value) for value in pd.unique(clusters)]
    for _ in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        values, _ = calibration_values(outcome[indices], probability[indices], bins=bins)
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


def calibrate_predictions(
    reference_path: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    model_id: str,
    reference_table: str = "reports",
    prediction_table: str = "classifications",
    id_column: str = "Hashed_ReportURN",
    labels: list[str] | None = None,
    probability_columns: dict[str, str] | None = None,
    cluster_column: str | None = None,
    reference_row_ranges: list[tuple[int, int]] | None = None,
    require_complete_reference: bool = False,
    bins: int = 10,
    bootstrap_iterations: int = 2000,
    seed: int = 20260718,
) -> dict[str, Any]:
    labels = labels or DEFAULT_LABELS
    probability_columns = probability_columns or {label: f"Prob_{label}" for label in labels}
    missing = sorted(set(labels) - set(probability_columns))
    if missing:
        raise ValueError(f"Missing probability-column mappings for: {missing}")
    if bins < 2:
        raise ValueError("Calibration requires at least two bins")

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

    prediction_columns = [id_column, *[probability_columns[label] for label in labels]]
    predictions = load_table(predictions_path, prediction_columns, prediction_table)
    if reference[id_column].duplicated().any():
        raise ValueError("Reference identifiers are not unique")
    if predictions[id_column].duplicated().any():
        raise ValueError("Prediction identifiers are not unique")
    predictions = predictions.rename(
        columns={probability_columns[label]: f"{label}__probability" for label in labels}
    )
    merged = reference.merge(predictions, on=id_column, how="inner", validate="one_to_one")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "model_id": model_id,
        "probability_interpretation": "estimated probability of core-positive levels 3 or 4",
        "reference_records": int(len(reference)),
        "prediction_records": int(len(predictions)),
        "matched_records": int(len(merged)),
        "unmatched_reference_records": int(len(reference) - len(merged)),
        "reference_selection": selection,
        "binning": {"strategy": "fixed_width", "bins": bins, "range": [0.0, 1.0]},
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": seed,
            "unit": "cluster" if cluster_column else "report",
            "cluster_column_supplied": bool(cluster_column),
        },
        "labels": {},
        "interpretation_limits": [
            "ECE depends on the declared bin count and boundaries and is not a model-invariant "
            "property.",
            "These binary probability metrics do not calibrate a generated four-level LLM label.",
        ],
    }
    if not cluster_column:
        summary["interpretation_limits"].append(
            "No patient/cluster column supplied; intervals resample reports and do not establish "
            "patient independence."
        )

    metric_rows: list[dict[str, Any]] = []
    all_bin_rows: list[dict[str, Any]] = []
    for offset, label in enumerate(labels):
        reference_level = pd.to_numeric(merged[label], errors="coerce")
        probability = pd.to_numeric(merged[f"{label}__probability"], errors="coerce")
        valid = reference_level.isin([1, 2, 3, 4]) & probability.between(0, 1, inclusive="both")
        outcome_values = reference_level[valid].isin([3, 4]).to_numpy(dtype=int)
        probability_values = probability[valid].to_numpy(dtype=float)
        clusters = (
            merged.loc[valid, cluster_column].astype(str).to_numpy() if cluster_column else None
        )
        if cluster_column and merged.loc[valid, cluster_column].isna().any():
            raise ValueError(f"Missing cluster identifiers among valid pairs for {label}")
        point, bin_rows = calibration_values(outcome_values, probability_values, bins=bins)
        intervals = calibration_bootstrap_intervals(
            outcome_values,
            probability_values,
            clusters,
            bins=bins,
            iterations=bootstrap_iterations,
            seed=seed + offset,
        )
        summary["labels"][label] = {
            "excluded_invalid_or_missing_pairs": int((~valid).sum()),
            "point_estimates": point,
            "confidence_intervals_95": intervals,
            "bins": bin_rows,
        }
        for metric, value in point.items():
            interval = intervals.get(metric, {"low": None, "high": None})
            metric_rows.append(
                {
                    "label": label,
                    "metric": metric,
                    "estimate": value,
                    "ci_low": interval["low"],
                    "ci_high": interval["high"],
                }
            )
        for row in bin_rows:
            all_bin_rows.append({"label": label, **row})

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "calibration_summary.json", summary)
    atomic_write_csv(output_dir / "calibration_metrics.csv", pd.DataFrame(metric_rows))
    atomic_write_csv(output_dir / "calibration_bins.csv", pd.DataFrame(all_bin_rows))
    atomic_write_json(
        output_dir / "run_manifest.json",
        build_manifest(
            "calibrate",
            [reference_path, predictions_path],
            {
                "model_id": model_id,
                "reference_table": reference_table,
                "prediction_table": prediction_table,
                "id_column": id_column,
                "labels": labels,
                "probability_columns": probability_columns,
                "cluster_column": cluster_column,
                "reference_row_ranges": reference_row_ranges,
                "require_complete_reference": require_complete_reference,
                "bins": bins,
                "bootstrap_iterations": bootstrap_iterations,
                "seed": seed,
            },
        ),
    )
    return summary
