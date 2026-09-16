#!/usr/bin/env python3
"""Analyze the frozen 2 x 2 model-by-evidence-schema development experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from eeg_review.evidence_extraction import classification_levels
from eeg_review.evidence_schema_factorial import (
    factorial_contrasts,
    factorial_evidence_units,
    paired_schema_transitions,
)
from eeg_review.io import atomic_write_csv, atomic_write_json, load_table
from eeg_review.reason_traceability import audit_traceability
from eeg_review.reference_aligned_traceability import (
    aggregate_reference_aligned_rows,
    build_reference_aligned_rows,
)

ID_COLUMN = "Hashed_ReportURN"
REPORT_COLUMN = "Report"
REPO_ROOT = Path(__file__).resolve().parents[1]
GOVERNED_ROOT = Path(
    "/Users/sbergner/Research/eeg/eeg_reports_annotation_nlp/data/governed/study-runs"
)
MISTRAL_RUN = GOVERNED_ROOT / "jbhi-mistral-native-small-followup-20260830"
MEDGEMMA_DC_RUN = GOVERNED_ROOT / "jbhi-medgemma-native-focal-v2-development-20260830"
MEDGEMMA_IND_RUN = (
    GOVERNED_ROOT / "jbhi-medgemma-native-scope-v21-development-20260831"
)
OUTPUT = GOVERNED_ROOT / "jbhi-evidence-schema-factorial-development-20260916"
DATASET = MISTRAL_RUN / "inputs/development.db"
MANIFEST = MISTRAL_RUN / "inputs/evidence.manifest.csv"
PLAN = REPO_ROOT / "review/model-receipts/evidence-schema-factorial.development-plan.json"
PROTOCOL = REPO_ROOT / "review/EVIDENCE_SCHEMA_FACTORIAL_PLAN_2026-09-16.md"
REFERENCE_COLUMNS = {
    "focal_epileptiform_activity": "Focal Epi",
    "generalized_epileptiform_activity": "Gen Epi",
    "focal_non_epileptiform_activity": "Focal Non-epi",
    "generalized_non_epileptiform_activity": "Gen Non-epi",
    "abnormality": "Abnormality",
}
CELLS = {
    "mistral_decision_conditioned": {
        "model_factor": "mistral",
        "evidence_schema_factor": "decision_conditioned",
        "evidence": MISTRAL_RUN / "products/evidence-native_chat.csv",
        "predictions": MISTRAL_RUN / "inputs/raw-development.csv",
        "planned_hash_key": "mistral_decision_conditioned",
    },
    "mistral_independent_category_evidence": {
        "model_factor": "mistral",
        "evidence_schema_factor": "independent_category_evidence",
        "evidence": OUTPUT / "products/mistral-independent-category-evidence.csv",
        "predictions": MISTRAL_RUN / "inputs/raw-development.csv",
        "planned_hash_key": "mistral_independent_category_evidence",
    },
    "medgemma_decision_conditioned": {
        "model_factor": "medgemma",
        "evidence_schema_factor": "decision_conditioned",
        "evidence": MEDGEMMA_DC_RUN / "products/evidence-v2.csv",
        "predictions": MEDGEMMA_DC_RUN / "products/v2.csv",
        "planned_hash_key": "medgemma_decision_conditioned",
    },
    "medgemma_independent_category_evidence": {
        "model_factor": "medgemma",
        "evidence_schema_factor": "independent_category_evidence",
        "evidence": MEDGEMMA_IND_RUN / "products/evidence-v21.csv",
        "predictions": MEDGEMMA_DC_RUN / "products/v2.csv",
        "planned_hash_key": "medgemma_independent_category_evidence",
    },
}
IMPLEMENTATION = (
    Path(__file__).resolve(),
    REPO_ROOT / "src/eeg_review/evidence_schema_factorial.py",
    REPO_ROOT / "src/eeg_review/reference_aligned_traceability.py",
    REPO_ROOT / "src/eeg_review/reason_traceability.py",
)
OUTPUT_FILES = (
    "factorial_units.csv",
    "factorial_cells.csv",
    "factorial_strata.csv",
    "factorial_contrasts.csv",
    "factorial_paired_transitions.csv",
    "factorial_summary.json",
    "analysis_receipt.json",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protect_tree(path: Path) -> None:
    path.chmod(0o700)
    for child in path.iterdir():
        if child.is_dir():
            _protect_tree(child)
        else:
            child.chmod(0o600)


def _output_hashes(path: Path) -> dict[str, str]:
    missing = [name for name in OUTPUT_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"factorial analysis is missing files: {missing}")
    return {name: sha256_path(path / name) for name in OUTPUT_FILES}


def _load_reference() -> tuple[pd.DataFrame, list[str]]:
    reference = load_table(
        DATASET,
        [ID_COLUMN, REPORT_COLUMN, *REFERENCE_COLUMNS.values()],
    )
    reference[ID_COLUMN] = reference[ID_COLUMN].astype(str)
    manifest_keys = pd.read_csv(MANIFEST, usecols=[ID_COLUMN])[ID_COLUMN].astype(str).tolist()
    if len(manifest_keys) != 20 or len(set(manifest_keys)) != 20:
        raise ValueError("factorial manifest must contain 20 unique keys")
    reference = reference.set_index(ID_COLUMN).loc[manifest_keys].reset_index()
    for column in REFERENCE_COLUMNS.values():
        values = pd.to_numeric(reference[column], errors="raise").astype(int)
        if not values.isin([1, 2, 3, 4]).all():
            raise ValueError("reference label violates four-level contract")
        reference[column] = values
    return reference, manifest_keys


def _cell_rows(
    cell_id: str,
    config: dict[str, Any],
    reference: pd.DataFrame,
    manifest_keys: list[str],
) -> tuple[list[dict[str, Any]], int]:
    evidence = pd.read_csv(Path(config["evidence"]))
    predictions = pd.read_csv(
        Path(config["predictions"]),
        usecols=[ID_COLUMN, "classifications"],
    )
    for frame in (evidence, predictions):
        frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    evidence = evidence.set_index(ID_COLUMN).loc[manifest_keys].reset_index()
    predictions = predictions.set_index(ID_COLUMN).loc[manifest_keys].reset_index()
    if evidence[ID_COLUMN].tolist() != manifest_keys:
        raise ValueError(f"{cell_id}: evidence order differs from frozen manifest")
    units = factorial_evidence_units(
        evidence,
        reference[[ID_COLUMN, REPORT_COLUMN]],
        source_kind=cell_id,
    )
    prediction_map = {
        (str(row[ID_COLUMN]), category): level
        for _, row in predictions.iterrows()
        for category, level in classification_levels(str(row["classifications"])).items()
    }
    reference_index = reference.set_index(ID_COLUMN)
    reference_map = {
        (key, category): int(reference_index.at[key, column])
        for key in manifest_keys
        for category, column in REFERENCE_COLUMNS.items()
    }
    rows = build_reference_aligned_rows(
        units,
        audit_traceability(units),
        predictions=prediction_map,
        references=reference_map,
        configured_system=cell_id,
    )
    for row in rows:
        row["model_factor"] = config["model_factor"]
        row["evidence_schema_factor"] = config["evidence_schema_factor"]
    valid_records = int(evidence["structured_output_valid"].fillna(False).astype(bool).sum())
    return rows, valid_records


def verify_analysis(path: Path = OUTPUT) -> dict[str, Any]:
    completion = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
    actual = _output_hashes(path)
    if actual != completion["output_sha256"]:
        raise ValueError("factorial output hashes do not match completion record")
    units = pd.read_csv(path / "factorial_units.csv")
    cells = pd.read_csv(path / "factorial_cells.csv")
    strata = pd.read_csv(path / "factorial_strata.csv")
    contrasts = pd.read_csv(path / "factorial_contrasts.csv")
    transitions = pd.read_csv(path / "factorial_paired_transitions.csv")
    expected = {
        "units": (len(units), 400),
        "cells": (len(cells), 4),
        "strata": (len(strata), 120),
        "contrasts": (len(contrasts), 12),
        "transitions": (len(transitions), 16),
    }
    wrong = {name: value for name, value in expected.items() if value[0] != value[1]}
    if wrong:
        raise ValueError(f"factorial output shape mismatch: {wrong}")
    if {"report_key", "report_text_sha256"} & set(cells.columns):
        raise ValueError("public-safe cell table contains governed identifiers")
    if any(item.stat().st_mode & 0o077 for item in path.rglob("*")):
        raise ValueError("factorial output contains accessible files")
    return {
        "cells": len(cells),
        "units": len(units),
        "contrasts": len(contrasts),
        "paired_transitions": len(transitions),
        "hashes_verified": len(actual),
    }


def run_analysis() -> dict[str, Any]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan["status"] != "frozen_before_missing_cell_inference":
        raise ValueError("factorial plan is not frozen")
    if sha256_path(PROTOCOL) != plan["protocol"]["sha256"]:
        raise ValueError("factorial protocol hash mismatch")
    if sha256_path(DATASET) != plan["surface"]["dataset_sha256"]:
        raise ValueError("factorial dataset hash mismatch")
    if sha256_path(MANIFEST) != plan["surface"]["manifest_sha256"]:
        raise ValueError("factorial manifest hash mismatch")
    for cell_id, config in CELLS.items():
        planned = plan["cells"][config["planned_hash_key"]]
        if (
            "output_sha256" in planned
            and sha256_path(Path(config["evidence"])) != planned["output_sha256"]
        ):
            raise ValueError(f"{cell_id}: saved evidence hash mismatch")
        if sha256_path(Path(config["predictions"])) != planned[
            "fixed_classifications_for_analysis_sha256"
            if "fixed_classifications_for_analysis_sha256" in planned
            else "fixed_classifications_sha256"
        ]:
            raise ValueError(f"{cell_id}: fixed classification hash mismatch")
    missing_receipt = CELLS["mistral_independent_category_evidence"]["evidence"].with_suffix(
        ".run.json"
    )
    receipt = json.loads(missing_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("classifications_supplied_to_model") is not False
        or receipt.get("evidence_mode") != "independent-category-evidence-v1"
        or receipt["inputs"]["records"] != 20
    ):
        raise ValueError("missing factorial cell violated independent evidence contract")

    reference, manifest_keys = _load_reference()
    unit_rows: list[dict[str, Any]] = []
    valid_records: dict[str, int] = {}
    for cell_id, config in CELLS.items():
        rows, valid = _cell_rows(cell_id, config, reference, manifest_keys)
        unit_rows.extend(rows)
        valid_records[cell_id] = valid
    strata = aggregate_reference_aligned_rows(unit_rows)
    factors = {
        cell_id: (config["model_factor"], config["evidence_schema_factor"])
        for cell_id, config in CELLS.items()
    }
    for row in strata:
        model, schema = factors[str(row["configured_system"])]
        row["model_factor"] = model
        row["evidence_schema_factor"] = schema
    cells = []
    for row in strata:
        if row["stratification"] != "overall":
            continue
        cells.append(
            {
                **row,
                "valid_records": valid_records[str(row["configured_system"])],
                "records": 20,
            }
        )
    contrasts = factorial_contrasts(cells)
    transitions = paired_schema_transitions(unit_rows)

    atomic_write_csv(OUTPUT / "factorial_units.csv", pd.DataFrame(unit_rows))
    atomic_write_csv(OUTPUT / "factorial_cells.csv", pd.DataFrame(cells))
    atomic_write_csv(OUTPUT / "factorial_strata.csv", pd.DataFrame(strata))
    atomic_write_csv(OUTPUT / "factorial_contrasts.csv", pd.DataFrame(contrasts))
    atomic_write_csv(
        OUTPUT / "factorial_paired_transitions.csv",
        pd.DataFrame(transitions),
    )
    summary = {
        "experiment_id": plan["experiment_id"],
        "status": "complete_development_factorial",
        "cells": cells,
        "contrasts": contrasts,
        "interpretation_boundaries": [
            "Purposive 20-report development surface only.",
            "Configured-model and evidence-schema factors are descriptive.",
            "Source location is not clinical relevance, sufficiency or entailment.",
            "No held-out execution, hypothesis test or confidence interval.",
        ],
    }
    atomic_write_json(OUTPUT / "factorial_summary.json", summary)
    analysis_receipt = {
        "experiment_id": plan["experiment_id"],
        "status": "complete",
        "plan_sha256": sha256_path(PLAN),
        "protocol_sha256": sha256_path(PROTOCOL),
        "input_sha256": {
            cell_id: {
                "evidence": sha256_path(Path(config["evidence"])),
                "predictions": sha256_path(Path(config["predictions"])),
            }
            for cell_id, config in CELLS.items()
        },
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_path(path) for path in IMPLEMENTATION
        },
        "new_model_inference": {
            "cells": 1,
            "reports": 20,
            "cell": "mistral_independent_category_evidence",
        },
        "held_out_reports_used": False,
        "blinded_review_unmasked": False,
    }
    atomic_write_json(OUTPUT / "analysis_receipt.json", analysis_receipt)
    _protect_tree(OUTPUT)
    completion = {
        "experiment_id": plan["experiment_id"],
        "status": "complete_and_frozen",
        "output_sha256": _output_hashes(OUTPUT),
    }
    atomic_write_json(OUTPUT / "COMPLETE.json", completion)
    _protect_tree(OUTPUT)
    return verify_analysis(OUTPUT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_analysis() if args.verify_only else run_analysis()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
