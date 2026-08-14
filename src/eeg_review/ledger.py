from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .io import atomic_write_csv, atomic_write_json
from .manifest import build_manifest, sha256_file


def _load_summary(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("labels"), dict):
        raise ValueError(f"Unsupported or malformed analysis summary: {resolved}")
    return payload


def _optional_number(value: Any) -> int | float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _interval(intervals: dict[str, Any], metric: str) -> tuple[Any, Any]:
    interval = intervals.get(metric, {})
    return _optional_number(interval.get("low")), _optional_number(interval.get("high"))


def _fraction(metric: str, point: dict[str, Any]) -> tuple[int | None, int | None]:
    n = int(point["n"])
    tn = int(point["tn"])
    fp = int(point["fp"])
    fn = int(point["fn"])
    tp = int(point["tp"])
    fractions = {
        "core_accuracy": (tn + tp, n),
        "precision": (tp, tp + fp),
        "recall_sensitivity": (tp, tp + fn),
        "specificity": (tn, tn + fp),
        "f1": (2 * tp, 2 * tp + fp + fn),
        "certainty_adjusted_accuracy": (
            int(round(float(point["certainty_adjusted_accuracy"]) * n)),
            n,
        ),
    }
    return fractions.get(metric, (None, None))


def _source_receipt(analysis_id: str, kind: str, path: Path) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "analysis_id": analysis_id,
        "kind": kind,
        "filename": path.name,
        "sha256": sha256_file(path),
    }
    run_manifest = path.parent / "run_manifest.json"
    if run_manifest.exists():
        payload = json.loads(run_manifest.read_text(encoding="utf-8"))
        receipt["run_manifest_sha256"] = sha256_file(run_manifest)
        receipt["command"] = payload.get("command")
        receipt["parameters"] = payload.get("parameters")
    return receipt


def build_result_ledger(
    output_dir: Path,
    *,
    evaluations: dict[str, Path] | None = None,
    calibrations: dict[str, Path] | None = None,
    comparisons: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Consolidate aggregate receipts without reading case-level inputs."""
    evaluations = evaluations or {}
    calibrations = calibrations or {}
    comparisons = comparisons or {}
    all_inputs = [*evaluations.values(), *calibrations.values(), *comparisons.values()]
    if not all_inputs:
        raise ValueError("At least one aggregate summary is required")
    identifiers = [*evaluations, *calibrations, *comparisons]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Analysis identifiers must be unique across all receipt kinds")

    receipts: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    discordance_rows: list[dict[str, Any]] = []
    interpretation_limits: list[str] = []

    for analysis_id, path in sorted(evaluations.items()):
        path = path.expanduser().resolve(strict=True)
        summary = _load_summary(path)
        receipts.append(_source_receipt(analysis_id, "evaluation", path))
        bootstrap = summary.get("bootstrap", {})
        interpretation_limits.extend(summary.get("interpretation_limits", []))
        for label, label_summary in summary["labels"].items():
            point = label_summary["point_estimates"]
            intervals = label_summary.get("confidence_intervals_95", {})
            for metric, estimate in point.items():
                low, high = _interval(intervals, metric)
                numerator, denominator = _fraction(metric, point)
                evaluation_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "label": label,
                        "metric": metric,
                        "estimate": _optional_number(estimate),
                        "ci_low": low,
                        "ci_high": high,
                        "numerator": numerator,
                        "denominator": denominator,
                        "n": int(point["n"]),
                        "interval_unit": bootstrap.get("unit"),
                        "bootstrap_iterations": bootstrap.get("iterations"),
                        "source_sha256": sha256_file(path),
                    }
                )

    for analysis_id, path in sorted(calibrations.items()):
        path = path.expanduser().resolve(strict=True)
        summary = _load_summary(path)
        receipts.append(_source_receipt(analysis_id, "calibration", path))
        bootstrap = summary.get("bootstrap", {})
        interpretation_limits.extend(summary.get("interpretation_limits", []))
        for label, label_summary in summary["labels"].items():
            point = label_summary["point_estimates"]
            intervals = label_summary.get("confidence_intervals_95", {})
            for metric, estimate in point.items():
                low, high = _interval(intervals, metric)
                numerator = denominator = None
                if metric == "prevalence":
                    numerator, denominator = int(point["positives"]), int(point["n"])
                calibration_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "label": label,
                        "metric": metric,
                        "estimate": _optional_number(estimate),
                        "ci_low": low,
                        "ci_high": high,
                        "numerator": numerator,
                        "denominator": denominator,
                        "n": int(point["n"]),
                        "interval_unit": bootstrap.get("unit"),
                        "bootstrap_iterations": bootstrap.get("iterations"),
                        "source_sha256": sha256_file(path),
                    }
                )

    for analysis_id, path in sorted(comparisons.items()):
        path = path.expanduser().resolve(strict=True)
        summary = _load_summary(path)
        receipt = _source_receipt(analysis_id, "comparison", path)
        receipts.append(receipt)
        bootstrap = summary.get("bootstrap", {})
        parameters = receipt.get("parameters") or {}
        interpretation_limits.extend(summary.get("interpretation_limits", []))
        for label, label_summary in summary["labels"].items():
            effects = label_summary["effects_a_minus_b"]
            intervals = label_summary.get("paired_confidence_intervals_95", {})
            for metric, estimate in effects.items():
                low, high = _interval(intervals, metric)
                effect_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "label": label,
                        "effect": metric,
                        "model_a": parameters.get("model_a_id"),
                        "model_b": parameters.get("model_b_id"),
                        "estimate_a_minus_b": _optional_number(estimate),
                        "ci_low": low,
                        "ci_high": high,
                        "n": int(label_summary["n"]),
                        "interval_unit": bootstrap.get("unit"),
                        "bootstrap_iterations": bootstrap.get("iterations"),
                        "source_sha256": sha256_file(path),
                    }
                )
            for outcome, details in label_summary["discordant_correctness"].items():
                discordance_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "label": label,
                        "outcome": outcome,
                        "model_a": parameters.get("model_a_id"),
                        "model_b": parameters.get("model_b_id"),
                        "a_correct_b_wrong": int(details["a_correct_b_wrong"]),
                        "a_wrong_b_correct": int(details["a_wrong_b_correct"]),
                        "both_correct": int(details["both_correct"]),
                        "both_wrong": int(details["both_wrong"]),
                        "mcnemar_exact_p_value": _optional_number(
                            details["mcnemar_exact_p_value"]
                        ),
                        "multiplicity_adjusted_p_value": _optional_number(
                            details["multiplicity_adjusted_p_value"]
                        ),
                        "source_sha256": sha256_file(path),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "evaluation_ledger.csv": evaluation_rows,
        "calibration_ledger.csv": calibration_rows,
        "paired_effect_ledger.csv": effect_rows,
        "discordance_ledger.csv": discordance_rows,
    }
    for filename, rows in tables.items():
        atomic_write_csv(output_dir / filename, pd.DataFrame(rows))

    result = {
        "schema_version": 1,
        "privacy_boundary": "aggregate receipts only; no report text or row identifiers",
        "sources": receipts,
        "row_counts": {filename: len(rows) for filename, rows in tables.items()},
        "interpretation_limits": sorted(set(interpretation_limits)),
        "evaluation": evaluation_rows,
        "calibration": calibration_rows,
        "paired_effects": effect_rows,
        "discordance": discordance_rows,
    }
    atomic_write_json(output_dir / "result_ledger.json", result)
    atomic_write_json(
        output_dir / "run_manifest.json",
        build_manifest(
            "result-ledger",
            all_inputs,
            {
                "evaluations": sorted(evaluations),
                "calibrations": sorted(calibrations),
                "comparisons": sorted(comparisons),
            },
        ),
    )
    return result
