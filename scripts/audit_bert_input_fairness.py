#!/usr/bin/env python3
"""Audit BERT right-truncation exposure and rescore saved prediction surfaces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_review.audit import DEFAULT_LABELS
from eeg_review.input_fairness import (
    eligible_reference_rows,
    evaluate_exclusion_sensitivity,
    public_exposure_summary,
    tokenizer_diagnostics,
)
from eeg_review.io import atomic_write_csv, atomic_write_json, load_table
from eeg_review.manifest import build_manifest, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "review/model-receipts/bert-input-fairness-sensitivity.preregistered.json"
GOVERNED_ROOT = (ROOT / "data/governed/analysis-runs").resolve()
KEY = "Hashed_ReportURN"
REPORT = "Report"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-governed-output", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def load_prediction(
    spec: dict[str, Any], *, base: Path
) -> tuple[pd.DataFrame, dict[str, str], Path]:
    path = resolve_path(spec["path"], base=base)
    table = spec.get("table", "reports")
    mappings = spec["prediction_columns"]
    missing = sorted(set(DEFAULT_LABELS) - set(mappings))
    if missing:
        raise ValueError(f"prediction spec is missing mappings: {missing}")
    columns = [KEY, *[mappings[label] for label in DEFAULT_LABELS]]
    return load_table(path, columns, table), mappings, path


def verify_completed(output: Path, config_sha256: str) -> bool:
    completion = output / "COMPLETE.json"
    if not completion.exists():
        return False
    payload = read_json(completion)
    if payload["config_sha256"] != config_sha256:
        raise ValueError("completed analysis uses a different configuration")
    for name, expected in payload["outputs"].items():
        path = output / name
        if sha256_file(path) != expected:
            raise ValueError(f"completed output changed: {name}")
    print("Completed BERT input-fairness sensitivity verified; no recomputation.")
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.umask(0o077)
    if not args.acknowledge_governed_output:
        raise ValueError("--acknowledge-governed-output is required")
    config_path = args.config.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output == GOVERNED_ROOT or not output.is_relative_to(GOVERNED_ROOT):
        raise ValueError("output must be a dedicated governed analysis-run directory")
    config_sha256 = sha256_file(config_path)
    if output.exists() and verify_completed(output, config_sha256):
        return read_json(output / "aggregate-input-fairness.json")

    plan = read_json(PLAN)
    if plan["status"] != "preregistered_before_case_level_inspection":
        raise ValueError("input-fairness plan is not preregistered")
    config = read_json(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported analysis configuration schema")
    planned_revision = plan["fixed_inputs"]["bert_tokenizer"]["revision"]
    tokenizer_snapshot = resolve_path(config["tokenizer_snapshot"], base=ROOT)
    if tokenizer_snapshot.name != planned_revision:
        raise ValueError("tokenizer snapshot revision differs from the preregistered plan")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot, local_files_only=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer is required for offset mapping")
    if tokenizer.truncation_side != "right" or tokenizer.model_max_length != 512:
        raise ValueError("tokenizer boundary differs from the historical BERT configuration")

    diagnostics_by_cohort: dict[str, pd.DataFrame] = {}
    metric_frames: list[pd.DataFrame] = []
    governed_cases: list[pd.DataFrame] = []
    input_paths: list[Path] = [PLAN, config_path]
    input_roles: list[dict[str, str]] = []
    excluded_incomplete: dict[str, int] = {}
    for cohort, cohort_spec in sorted(config["cohorts"].items()):
        reference_spec = cohort_spec["reference"]
        reference_path = resolve_path(reference_spec["path"], base=ROOT)
        reference = load_table(
            reference_path,
            [KEY, REPORT, *DEFAULT_LABELS],
            reference_spec.get("table", "reports"),
        )
        reference, excluded = eligible_reference_rows(reference)
        excluded_incomplete[cohort] = excluded
        diagnostics = tokenizer_diagnostics(reference, tokenizer)
        diagnostics.insert(0, "cohort", cohort)
        diagnostics_by_cohort[cohort] = diagnostics
        governed_cases.append(diagnostics.loc[diagnostics["truncation_exposed"]].copy())
        input_paths.append(reference_path)
        input_roles.append(
            {
                "role": f"{cohort}:reference",
                "name": reference_path.name,
                "sha256": sha256_file(reference_path),
            }
        )

        model_predictions: dict[str, tuple[pd.DataFrame, dict[str, str]]] = {}
        for model, prediction_spec in sorted(cohort_spec["predictions"].items()):
            frame, mappings, prediction_path = load_prediction(prediction_spec, base=ROOT)
            model_predictions[model] = (frame, mappings)
            input_paths.append(prediction_path)
            input_roles.append(
                {
                    "role": f"{cohort}:{model}",
                    "name": prediction_path.name,
                    "sha256": sha256_file(prediction_path),
                }
            )
        metric_frames.append(
            evaluate_exclusion_sensitivity(
                reference,
                diagnostics,
                model_predictions,
                cohort=cohort,
            )
        )

    metrics = pd.concat(metric_frames, ignore_index=True)
    cases = pd.concat(governed_cases, ignore_index=True)
    exposure = public_exposure_summary(diagnostics_by_cohort)
    exposure["excluded_incomplete_reference_records"] = excluded_incomplete
    measured = [
        "core_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "certainty_adjusted_accuracy",
        "core_kappa",
        "four_level_kappa",
    ]
    maxima: dict[str, dict[str, Any]] = {}
    for metric in measured:
        column = f"change_{metric}"
        finite = metrics[np.isfinite(pd.to_numeric(metrics[column], errors="coerce"))].copy()
        if finite.empty:
            maxima[metric] = {"absolute_change": None}
            continue
        index = finite[column].abs().idxmax()
        row = finite.loc[index]
        maxima[metric] = {
            "absolute_change": abs(float(row[column])),
            "signed_change": float(row[column]),
            "cohort": row["cohort"],
            "model": row["model"],
            "category": row["category"],
        }

    result = {
        "schema_version": 1,
        "study_id": plan["study_id"],
        "status": "completed_read_only_sensitivity",
        "inference_or_model_fitting_performed": False,
        "tokenizer": {
            "model_id": "bert-base-uncased",
            "revision": planned_revision,
            "maximum_sequence_length": 512,
            "truncation_side": "right",
            "local_only": True,
        },
        "exposure": exposure,
        "maximum_absolute_metric_changes": maxima,
        "interpretation_boundaries": [
            "The complete historical and reproduced populations remain unchanged.",
            "The sensitivity removes every report exposed to BERT right truncation "
            "from all models, so the compared report population is identical.",
            "Small changes do not establish equivalence; large changes would identify "
            "a report-length sensitivity for follow-up.",
            "Recognized section headings are conservative text markers, not proof that "
            "an unrecognized section is absent.",
            "This analysis does not establish patient independence, clinical validity, "
            "or a causal model-family effect.",
        ],
        "contains_report_keys_or_text": False,
    }

    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    cases_path = output / "governed-truncation-exposed-cases.csv"
    metrics_path = output / "aggregate-metrics.csv"
    aggregate_path = output / "aggregate-input-fairness.json"
    receipt_path = output / "run-receipt.json"
    atomic_write_csv(cases_path, cases)
    atomic_write_csv(metrics_path, metrics)
    atomic_write_json(aggregate_path, json_ready(result))
    receipt = build_manifest(
        command="audit_bert_input_fairness.py",
        inputs=input_paths,
        parameters={
            "study_id": plan["study_id"],
            "plan_sha256": sha256_file(PLAN),
            "config_sha256": config_sha256,
            "tokenizer_revision": planned_revision,
            "model_surfaces": sorted(
                config["cohorts"][next(iter(config["cohorts"]))]["predictions"]
            ),
            "input_roles": input_roles,
        },
        privacy_boundary=(
            "Report keys and tokenizer boundary diagnostics remain governed; aggregate "
            "metrics contain no report keys or text and require author review before release."
        ),
    )
    atomic_write_json(receipt_path, receipt)
    outputs = {
        path.name: sha256_file(path)
        for path in (cases_path, metrics_path, aggregate_path, receipt_path)
    }
    atomic_write_json(
        output / "COMPLETE.json",
        {
            "schema_version": 1,
            "study_id": plan["study_id"],
            "config_sha256": config_sha256,
            "outputs": outputs,
            "inference_or_model_fitting_performed": False,
        },
    )
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
