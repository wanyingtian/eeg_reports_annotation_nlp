#!/usr/bin/env python3
"""Run the frozen read-only focal-epileptiform error-signature audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from eeg_review.focal_signatures import CUE_TERMS, TRANSITION_GROUPS, audit_cohort
from eeg_review.io import atomic_write_csv, atomic_write_json
from eeg_review.manifest import build_manifest, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "review/model-receipts/medgemma-focal-error-signature.preregistered.json"
GOVERNED_ROOT = (ROOT / "data/governed/analysis-runs").resolve()


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.umask(0o077)
    if not args.acknowledge_governed_output:
        raise ValueError("--acknowledge-governed-output is required")
    config_path = args.config.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output == GOVERNED_ROOT or not output.is_relative_to(GOVERNED_ROOT):
        raise ValueError("output must be a dedicated governed analysis-run directory")
    if output.exists():
        raise ValueError("refusing to overwrite an existing analysis directory")

    plan = read_json(PLAN)
    if plan["status"] != "frozen_before_case_level_cue_results":
        raise ValueError("focal error-signature plan is not frozen")
    if plan["registered_cues"] != list(CUE_TERMS):
        raise ValueError("implemented cue order differs from the frozen plan")
    if plan["paired_transition_groups"] != list(TRANSITION_GROUPS):
        raise ValueError("implemented transition groups differ from the frozen plan")
    if not plan["retention_rule"].startswith("Retain every"):
        raise ValueError("frozen retention rule is missing")

    config = read_json(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported analysis configuration schema")
    planned = plan["fixed_population"]["cohorts"]
    if set(config["cohorts"]) != set(planned):
        raise ValueError("configured cohorts differ from the frozen plan")

    ledgers: list[pd.DataFrame] = []
    aggregates: dict[str, Any] = {}
    input_paths = [PLAN, config_path]
    input_roles: list[dict[str, str]] = []
    for cohort, spec in sorted(config["cohorts"].items()):
        paths = {
            role: resolve_path(spec[role], base=ROOT)
            for role in ("reference", "mistral", "medgemma")
        }
        input_paths.extend(paths.values())
        input_roles.extend(
            {
                "role": f"{cohort}:{role}",
                "name": path.name,
                "sha256": sha256_file(path),
            }
            for role, path in paths.items()
        )
        ledger, aggregate = audit_cohort(
            cohort=cohort,
            expected_records=int(planned[cohort]),
            reference_path=paths["reference"],
            mistral_path=paths["mistral"],
            medgemma_path=paths["medgemma"],
        )
        ledgers.append(ledger)
        aggregates[cohort] = aggregate

    result = {
        "schema_version": 1,
        "study_id": plan["study_id"],
        "status": "completed_read_only_explanatory_audit",
        "study_role": plan["study_role"],
        "inference_or_model_fitting_performed": False,
        "target_label": plan["fixed_population"]["target_label"],
        "segment_rule": plan["segment_rule"],
        "registered_cues": list(CUE_TERMS),
        "cohorts": aggregates,
        "interpretation_boundaries": [
            "The target was selected after the aggregate focal-epileptiform trade-off was known.",
            "Cue definitions were frozen before case-level cue results were inspected.",
            "Lexical co-occurrence is descriptive and does not establish clinical error cause.",
            "The audit does not establish patient independence or clinical validity.",
            "Protected-cohort findings must not be used to tune a prompt and then be "
            "presented as unbiased evaluation on the same reports.",
        ],
        "contains_report_keys_text_or_source_segments": False,
    }

    output.mkdir(mode=0o700, parents=True)
    ledger_path = output / "governed-focal-error-signatures.csv"
    aggregate_path = output / "aggregate-focal-error-signatures.json"
    receipt_path = output / "run-receipt.json"
    atomic_write_csv(ledger_path, pd.concat(ledgers, ignore_index=True))
    atomic_write_json(aggregate_path, result)
    receipt = build_manifest(
        command="audit_focal_error_signatures.py",
        inputs=input_paths,
        parameters={
            "study_id": plan["study_id"],
            "plan_sha256": sha256_file(PLAN),
            "config_sha256": sha256_file(config_path),
            "input_roles": input_roles,
            "registered_cues": list(CUE_TERMS),
            "transition_groups": list(TRANSITION_GROUPS),
        },
        privacy_boundary=(
            "Report keys and matched source segments remain governed; the aggregate "
            "contains only counts and proportions and requires author review before release."
        ),
    )
    atomic_write_json(receipt_path, receipt)
    outputs = {
        path.name: sha256_file(path)
        for path in (ledger_path, aggregate_path, receipt_path)
    }
    atomic_write_json(
        output / "COMPLETE.json",
        {
            "schema_version": 1,
            "study_id": plan["study_id"],
            "config_sha256": sha256_file(config_path),
            "outputs": outputs,
            "inference_or_model_fitting_performed": False,
        },
    )
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
