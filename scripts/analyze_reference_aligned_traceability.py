#!/usr/bin/env python3
"""Run the frozen reference-aligned traceability analysis without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from eeg_review.evidence_extraction import JSON_KEYS, classification_levels
from eeg_review.io import atomic_write_csv, atomic_write_json, load_table
from eeg_review.reason_traceability import audit_traceability, structured_evidence_units
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
DATASET = (
    GOVERNED_ROOT
    / "jbhi-medgemma-native-chat-development-20260829/inputs/zoe_development_native_100.db"
)
MANIFEST = (
    GOVERNED_ROOT
    / "jbhi-medgemma-native-chat-development-20260829/manifests/zoe_development_native_100.csv"
)
MEDGEMMA_EVIDENCE = (
    GOVERNED_ROOT
    / "jbhi-medgemma-v1-evidence-development-20260916/products/medgemma-v1-evidence-all100.csv"
)
MISTRAL_SAVED = GOVERNED_ROOT / "jbhi-native-20260814/products/llm/zoe/raw.csv"
OUTPUT = (
    GOVERNED_ROOT
    / "jbhi-medgemma-v1-evidence-development-20260916/reference-aligned-traceability-v1"
)
PROTOCOL = REPO_ROOT / "review/REFERENCE_ALIGNED_EVIDENCE_TRACEABILITY_PROTOCOL_2026-09-16.md"
PRE_EXECUTION = (
    REPO_ROOT
    / "review/model-receipts/reference-aligned-evidence-traceability.pre-execution.json"
)
IMPLEMENTATION = (
    Path(__file__).resolve(),
    REPO_ROOT / "src/eeg_review/reference_aligned_traceability.py",
    REPO_ROOT / "src/eeg_review/reason_traceability.py",
)
OUTPUT_FILES = (
    "reference_aligned_units.csv",
    "reference_aligned_aggregate.csv",
    "reference_aligned_summary.json",
    "analysis_receipt.json",
)
REFERENCE_COLUMNS = {
    "focal_epileptiform_activity": "Focal Epi",
    "generalized_epileptiform_activity": "Gen Epi",
    "focal_non_epileptiform_activity": "Focal Non-epi",
    "generalized_non_epileptiform_activity": "Gen Non-epi",
    "abnormality": "Abnormality",
}
SYSTEMS = {
    "mistral_historical_interface_saved": {
        "path": MISTRAL_SAVED,
        "classification_column": "classifications",
    },
    "medgemma_native_v1_fixed_decision": {
        "path": MEDGEMMA_EVIDENCE,
        "classification_column": "fixed_classifications",
    },
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protect_tree(path: Path) -> None:
    path.chmod(0o700)
    for child in path.iterdir():
        child.chmod(0o600)


def _output_hashes(path: Path) -> dict[str, str]:
    missing = [name for name in OUTPUT_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"analysis output is missing files: {missing}")
    return {name: sha256_path(path / name) for name in OUTPUT_FILES}


def _load_reference() -> tuple[pd.DataFrame, list[str]]:
    columns = [ID_COLUMN, REPORT_COLUMN, *REFERENCE_COLUMNS.values()]
    reference = load_table(DATASET, columns)
    reference[ID_COLUMN] = reference[ID_COLUMN].astype(str)
    manifest_keys = pd.read_csv(MANIFEST, usecols=[ID_COLUMN])[ID_COLUMN].astype(str).tolist()
    if len(manifest_keys) != 100 or len(set(manifest_keys)) != 100:
        raise ValueError("development manifest must contain exactly 100 unique keys")
    reference = reference.set_index(ID_COLUMN).loc[manifest_keys].reset_index()
    for column in REFERENCE_COLUMNS.values():
        values = pd.to_numeric(reference[column], errors="raise").astype(int)
        if not values.isin([1, 2, 3, 4]).all():
            raise ValueError(f"reference label is outside four-level contract: {column}")
        reference[column] = values
    return reference, manifest_keys


def _load_system_rows(
    *,
    system: str,
    config: dict[str, Any],
    reference: pd.DataFrame,
    manifest_keys: list[str],
) -> list[dict[str, Any]]:
    classification_column = str(config["classification_column"])
    evidence = pd.read_csv(
        Path(config["path"]),
        usecols=[ID_COLUMN, classification_column, "explanations"],
    )
    evidence[ID_COLUMN] = evidence[ID_COLUMN].astype(str)
    evidence = evidence.set_index(ID_COLUMN).loc[manifest_keys].reset_index()
    reports = reference[[ID_COLUMN, REPORT_COLUMN]]
    units = structured_evidence_units(
        evidence,
        reports,
        source_kind=system,
        classification_column=classification_column,
    )
    predictions = {
        (str(row[ID_COLUMN]), category): level
        for _, row in evidence.iterrows()
        for category, level in classification_levels(str(row[classification_column])).items()
    }
    reference_index = reference.set_index(ID_COLUMN)
    references = {
        (key, category): int(reference_index.at[key, column])
        for key in manifest_keys
        for category, column in REFERENCE_COLUMNS.items()
    }
    return build_reference_aligned_rows(
        units,
        audit_traceability(units),
        predictions=predictions,
        references=references,
        configured_system=system,
    )


def verify_analysis(path: Path = OUTPUT) -> dict[str, Any]:
    completion = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
    actual = _output_hashes(path)
    if actual != completion["output_sha256"]:
        raise ValueError("analysis output hashes do not match completion record")
    units = pd.read_csv(path / "reference_aligned_units.csv")
    aggregate = pd.read_csv(path / "reference_aligned_aggregate.csv")
    if len(units) != 1000:
        raise ValueError("analysis must contain 1,000 configured-system units")
    if not units.groupby("configured_system").size().eq(500).all():
        raise ValueError("each configured system must contribute 500 units")
    if len(aggregate) != 60:
        raise ValueError("analysis must contain all 60 pre-specified aggregate strata")
    if {"report_key", "report_text_sha256"} & set(aggregate.columns):
        raise ValueError("aggregate output contains governed unit identifiers")
    if any(item.stat().st_mode & 0o077 for item in path.rglob("*")):
        raise ValueError("analysis output contains accessible files")
    return {
        "configured_systems": units["configured_system"].nunique(),
        "units": len(units),
        "aggregate_strata": len(aggregate),
        "hashes_verified": len(actual),
    }


def run_analysis() -> dict[str, Any]:
    pre_execution = json.loads(PRE_EXECUTION.read_text(encoding="utf-8"))
    if pre_execution["status"] != "frozen_before_aggregate_calculation":
        raise ValueError("pre-execution record is not frozen")
    for key, path in {
        "development_database": DATASET,
        "development_manifest": MANIFEST,
        "medgemma_saved_evidence": MEDGEMMA_EVIDENCE,
        "mistral_saved_classification_and_evidence": MISTRAL_SAVED,
    }.items():
        if sha256_path(path) != pre_execution["input_sha256"][key]:
            raise ValueError(f"frozen input hash mismatch: {key}")
    if sha256_path(PROTOCOL) != pre_execution["protocol"]["sha256"]:
        raise ValueError("frozen protocol hash mismatch")

    reference, manifest_keys = _load_reference()
    unit_rows = [
        row
        for system, config in SYSTEMS.items()
        for row in _load_system_rows(
            system=system,
            config=config,
            reference=reference,
            manifest_keys=manifest_keys,
        )
    ]
    aggregate_rows = aggregate_reference_aligned_rows(unit_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_csv(OUTPUT / "reference_aligned_units.csv", pd.DataFrame(unit_rows))
    atomic_write_csv(
        OUTPUT / "reference_aligned_aggregate.csv",
        pd.DataFrame(aggregate_rows),
    )
    overall = [row for row in aggregate_rows if row["stratification"] == "overall"]
    summary = {
        "analysis_id": pre_execution["analysis_id"],
        "status": "complete_development_descriptive_analysis",
        "surface": {
            "reports": 100,
            "categories": len(JSON_KEYS),
            "configured_systems": len(SYSTEMS),
            "units": len(unit_rows),
        },
        "overall": overall,
        "interpretation_term": "reference-aligned evidence traceability",
        "boundaries": [
            "Development surface only; no held-out or population inference.",
            "Reference agreement is not independently established clinical correctness.",
            "Source location does not establish relevance, sufficiency or entailment.",
            "Configured-system differences cannot be attributed to model weights alone.",
            "The blinded 20-case review and its unblinding key were not used.",
        ],
    }
    atomic_write_json(OUTPUT / "reference_aligned_summary.json", summary)
    receipt = {
        "analysis_id": pre_execution["analysis_id"],
        "status": "complete",
        "pre_execution_sha256": sha256_path(PRE_EXECUTION),
        "protocol_sha256": sha256_path(PROTOCOL),
        "input_sha256": {
            str(path): sha256_path(path)
            for path in (DATASET, MANIFEST, MEDGEMMA_EVIDENCE, MISTRAL_SAVED)
        },
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_path(path) for path in IMPLEMENTATION
        },
        "new_model_inference": False,
        "unblinded_review_material_used": False,
    }
    atomic_write_json(OUTPUT / "analysis_receipt.json", receipt)
    _protect_tree(OUTPUT)
    completion = {
        "analysis_id": pre_execution["analysis_id"],
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
