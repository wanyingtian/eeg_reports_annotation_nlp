from __future__ import annotations

import hashlib
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from .io import atomic_write_csv, atomic_write_json, load_table
from .manifest import build_manifest

DEFAULT_LABELS = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]


def normalize_report(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text.casefold()).strip()


def private_digest(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def describe_numeric(series: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "minimum": int(clean.min()),
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "maximum": int(clean.max()),
    }


def audit_dataset(
    dataset: Path,
    dataset_id: str,
    output_dir: Path,
    *,
    table: str = "reports",
    id_column: str = "Hashed_ReportURN",
    report_column: str = "Report",
    labels: list[str] | None = None,
    patient_column: str | None = None,
    split_column: str | None = None,
) -> dict[str, Any]:
    labels = labels or DEFAULT_LABELS
    columns = [id_column, report_column, *labels]
    for optional in (patient_column, split_column):
        if optional and optional not in columns:
            columns.append(optional)
    frame = load_table(dataset, columns, table)

    identifiers = frame[id_column].astype("string")
    reports = frame[report_column].map(normalize_report)
    report_hashes = reports.map(private_digest)
    word_counts = reports.str.split().map(len)

    duplicate_id_mask = identifiers.notna() & identifiers.duplicated(keep=False)
    duplicate_report_mask = reports.ne("") & report_hashes.duplicated(keep=False)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "records": int(len(frame)),
        "identifier": {
            "column": id_column,
            "missing": int(identifiers.isna().sum()),
            "unique": int(identifiers.nunique(dropna=True)),
            "rows_in_duplicate_id_groups": int(duplicate_id_mask.sum()),
        },
        "reports": {
            "column": report_column,
            "missing_or_blank": int(reports.eq("").sum()),
            "unique_normalized": int(report_hashes[reports.ne("")].nunique()),
            "rows_in_exact_duplicate_groups": int(duplicate_report_mask.sum()),
            "word_count": describe_numeric(word_counts),
        },
        "patient_independence": {"status": "not_assessed", "reason": "no patient column supplied"},
        "split": {"status": "not_assessed", "reason": "no split column supplied"},
        "labels": {},
        "interpretation_limits": [
            "Hashed report identifiers do not establish unique patients.",
            "Exact normalized-text duplicates do not detect semantic near-duplicates.",
            (
                "Counts describe the supplied file only; cohort membership must be "
                "documented separately."
            ),
        ],
    }

    if patient_column:
        patients = frame[patient_column].astype("string")
        summary["patient_independence"] = {
            "status": "assessed",
            "column": patient_column,
            "missing": int(patients.isna().sum()),
            "unique_patients": int(patients.nunique(dropna=True)),
            "reports_per_patient": describe_numeric(patients.value_counts()),
        }
    if split_column:
        split_counts = frame[split_column].astype("string").value_counts(dropna=False)
        summary["split"] = {
            "status": "assessed",
            "column": split_column,
            "counts": {str(key): int(value) for key, value in split_counts.items()},
        }

    label_rows: list[dict[str, Any]] = []
    for label in labels:
        numeric = pd.to_numeric(frame[label], errors="coerce")
        valid = numeric.isin([1, 2, 3, 4])
        four_level = {str(level): int((numeric == level).sum()) for level in range(1, 5)}
        absent = int(numeric.isin([1, 2]).sum())
        present = int(numeric.isin([3, 4]).sum())
        details = {
            "missing": int(numeric.isna().sum()),
            "invalid": int((numeric.notna() & ~valid).sum()),
            "four_level": four_level,
            "core_absent": absent,
            "core_present": present,
        }
        summary["labels"][label] = details
        for level, count in four_level.items():
            label_rows.append(
                {
                    "dataset_id": dataset_id,
                    "label": label,
                    "scale": "four_level",
                    "value": level,
                    "count": count,
                }
            )
        label_rows.extend(
            [
                {
                    "dataset_id": dataset_id,
                    "label": label,
                    "scale": "core",
                    "value": "absent",
                    "count": absent,
                },
                {
                    "dataset_id": dataset_id,
                    "label": label,
                    "scale": "core",
                    "value": "present",
                    "count": present,
                },
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "cohort_audit.json", summary)
    atomic_write_csv(output_dir / "label_counts.csv", pd.DataFrame(label_rows))
    manifest = build_manifest(
        "audit",
        [dataset],
        {
            "dataset_id": dataset_id,
            "table": table,
            "id_column": id_column,
            "report_column": report_column,
            "labels": labels,
            "patient_column": patient_column,
            "split_column": split_column,
        },
    )
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    return summary


def audit_overlap(
    datasets: dict[str, Path],
    output_dir: Path,
    *,
    table: str = "reports",
    id_column: str = "Hashed_ReportURN",
    report_column: str = "Report",
    patient_column: str | None = None,
) -> dict[str, Any]:
    if len(datasets) < 2:
        raise ValueError("Overlap audit requires at least two named datasets")
    private_sets: dict[str, dict[str, set[str]]] = {}
    for name, path in datasets.items():
        columns = [id_column, report_column]
        if patient_column:
            columns.append(patient_column)
        frame = load_table(path, columns, table)
        ids = {private_digest(value) for value in frame[id_column].dropna().astype(str)}
        reports = {
            private_digest(normalized)
            for normalized in frame[report_column].map(normalize_report)
            if normalized
        }
        patients = (
            {private_digest(value) for value in frame[patient_column].dropna().astype(str)}
            if patient_column
            else set()
        )
        private_sets[name] = {"identifiers": ids, "reports": reports, "patients": patients}

    comparisons = []
    for left, right in combinations(sorted(datasets), 2):
        comparison = {
            "left": left,
            "right": right,
            "shared_report_identifiers": len(
                private_sets[left]["identifiers"] & private_sets[right]["identifiers"]
            ),
            "shared_exact_normalized_reports": len(
                private_sets[left]["reports"] & private_sets[right]["reports"]
            ),
        }
        if patient_column:
            comparison["shared_patient_identifiers"] = len(
                private_sets[left]["patients"] & private_sets[right]["patients"]
            )
        comparisons.append(comparison)
    result = {
        "schema_version": 1,
        "patient_overlap": {
            "status": "assessed" if patient_column else "not_assessed",
            "column": patient_column,
        },
        "comparisons": comparisons,
        "interpretation_limits": [
            "Zero shared report identifiers does not establish patient independence.",
            (
                "Patient overlap is interpretable only when the supplied key has stable, "
                "compatible semantics across cohorts."
            ),
            (
                "Exact normalized report matching does not detect paraphrased or templated "
                "near-duplicates."
            ),
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "overlap_audit.json", result)
    atomic_write_json(
        output_dir / "run_manifest.json",
        build_manifest(
            "overlap",
            list(datasets.values()),
            {
                "dataset_names": sorted(datasets),
                "table": table,
                "id_column": id_column,
                "report_column": report_column,
                "patient_column": patient_column,
            },
        ),
    )
    return result
