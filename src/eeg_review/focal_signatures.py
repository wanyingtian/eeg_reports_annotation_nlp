from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

KEY = "Hashed_ReportURN"
REPORT = "Report"
LABEL = "Focal Epi"

TRANSITION_GROUPS = (
    "both_correct_negative",
    "medgemma_only_false_positive",
    "mistral_only_false_positive",
    "both_false_positive",
    "both_true_positive",
    "medgemma_only_true_positive",
    "mistral_only_true_positive",
    "both_false_negative",
)

_TERMS = {
    "negation": re.compile(
        r"\b(?:no|not|none|without|absent|absence|negative for|free of)\b", re.I
    ),
    "epileptiform": re.compile(
        r"\b(?:epileptiform|epileptic|seizure|spike(?:s)?|"
        r"spike[- ]wave|sharp wave(?:s)?)\b",
        re.I,
    ),
    "seizure": re.compile(
        r"\b(?:seizure|seizures|epilepsy|epileptic|convulsion(?:s)?)\b", re.I
    ),
    "history_or_indication": re.compile(
        r"\b(?:history|indication|reason for (?:study|exam)|clinical history|"
        r"rule out|query)\b",
        re.I,
    ),
    "benign_or_artifact": re.compile(
        r"\b(?:benign|artifact|artefact|wicket|breach|vertex|mu rhythm|"
        r"lambda wave(?:s)?|small sharp spike(?:s)?)\b",
        re.I,
    ),
    "sharp": re.compile(r"\b(?:sharp|spike|spikes|spiky)\b", re.I),
    "focal": re.compile(
        r"\b(?:focal|focus|foci|localized|lateralized|temporal|frontal|parietal|"
        r"occipital|hemispher(?:e|ic))\b",
        re.I,
    ),
    "slowing_or_attenuation": re.compile(
        r"\b(?:slow(?:ing)?|attenuat(?:ed|ion)|asymmetr(?:y|ic)|suppression|"
        r"decreased amplitude)\b",
        re.I,
    ),
    "generalized_or_bilateral": re.compile(
        r"\b(?:generalized|generalised|bilateral|diffuse|symmetric|synchronous)\b",
        re.I,
    ),
    "uncertainty": re.compile(
        r"\b(?:possible|possibly|probable|probably|suggestive|suspicious|"
        r"questionable|may represent|cannot exclude|could represent)\b",
        re.I,
    ),
}

CUE_TERMS: Mapping[str, tuple[str, str]] = {
    "negated_epileptiform_segment": ("negation", "epileptiform"),
    "history_or_indication_with_seizure_segment": ("history_or_indication", "seizure"),
    "benign_or_artifact_with_sharp_segment": ("benign_or_artifact", "sharp"),
    "focal_slowing_or_attenuation_segment": ("focal", "slowing_or_attenuation"),
    "generalized_or_bilateral_epileptiform_segment": ("generalized_or_bilateral", "epileptiform"),
    "focal_epileptiform_wording_segment": ("focal", "epileptiform"),
    "uncertain_epileptiform_wording_segment": ("uncertainty", "epileptiform"),
}


def split_segments(report: str) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"(?:[\r\n]+|(?<=[.!?;:])\s+)", report or "")
        if segment.strip()
    ]


def cue_segments(report: str) -> dict[str, list[str]]:
    matches = {cue: [] for cue in CUE_TERMS}
    for segment in split_segments(report):
        for cue, (left, right) in CUE_TERMS.items():
            if _TERMS[left].search(segment) and _TERMS[right].search(segment):
                matches[cue].append(segment)
    return matches


def transition_group(reference_level: int, mistral_level: int, medgemma_level: int) -> str:
    reference = int(reference_level) > 1
    mistral = int(mistral_level) > 1
    medgemma = int(medgemma_level) > 1
    if not reference:
        if not mistral and not medgemma:
            return "both_correct_negative"
        if not mistral and medgemma:
            return "medgemma_only_false_positive"
        if mistral and not medgemma:
            return "mistral_only_false_positive"
        return "both_false_positive"
    if mistral and medgemma:
        return "both_true_positive"
    if not mistral and medgemma:
        return "medgemma_only_true_positive"
    if mistral and not medgemma:
        return "mistral_only_true_positive"
    return "both_false_negative"


def load_sqlite(path: Path, table: str, columns: list[str]) -> pd.DataFrame:
    quoted = ", ".join(f'"{column}"' for column in columns)
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query(f'SELECT {quoted} FROM "{table}"', connection)


def _validate_keys(frame: pd.DataFrame, *, name: str) -> set[str]:
    if frame[KEY].isna().any():
        raise ValueError(f"{name} contains missing report keys")
    keys = frame[KEY].astype(str)
    duplicates = keys[keys.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"{name} contains duplicate report keys: {duplicates[:3]}")
    return set(keys)


def audit_cohort(
    *,
    cohort: str,
    expected_records: int,
    reference_path: Path,
    mistral_path: Path,
    medgemma_path: Path,
    reference_table: str = "reports",
    prediction_table: str = "classifications",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference = load_sqlite(reference_path, reference_table, [KEY, REPORT, LABEL]).rename(
        columns={LABEL: "reference_level"}
    )
    mistral = load_sqlite(mistral_path, prediction_table, [KEY, LABEL]).rename(
        columns={LABEL: "mistral_level"}
    )
    medgemma = load_sqlite(medgemma_path, prediction_table, [KEY, LABEL]).rename(
        columns={LABEL: "medgemma_level"}
    )
    frames = {"reference": reference, "mistral": mistral, "medgemma": medgemma}
    key_sets = {name: _validate_keys(frame, name=name) for name, frame in frames.items()}
    if any(len(frame) != expected_records for frame in frames.values()):
        sizes = {name: len(frame) for name, frame in frames.items()}
        raise ValueError(f"{cohort} record counts differ from the frozen population: {sizes}")
    if len({frozenset(keys) for keys in key_sets.values()}) != 1:
        detail = {
            name: {
                "missing_from_reference": len(key_sets["reference"] - keys),
                "extra_vs_reference": len(keys - key_sets["reference"]),
            }
            for name, keys in key_sets.items()
        }
        raise ValueError(f"{cohort} prediction keys do not match the reference: {detail}")

    merged = reference.merge(mistral, on=KEY, validate="one_to_one").merge(
        medgemma, on=KEY, validate="one_to_one"
    )
    ledger_rows: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        matches = cue_segments(str(row[REPORT]))
        ledger_row: dict[str, Any] = {
            "cohort": cohort,
            KEY: str(row[KEY]),
            "reference_level": int(row["reference_level"]),
            "mistral_level": int(row["mistral_level"]),
            "medgemma_level": int(row["medgemma_level"]),
            "transition_group": transition_group(
                row["reference_level"], row["mistral_level"], row["medgemma_level"]
            ),
        }
        for cue, segments in matches.items():
            ledger_row[cue] = bool(segments)
            ledger_row[f"{cue}_source_segments"] = json.dumps(
                segments, ensure_ascii=False
            )
        ledger_rows.append(ledger_row)
    ledger = pd.DataFrame(ledger_rows)

    groups: dict[str, Any] = {}
    for group in TRANSITION_GROUPS:
        subset = ledger.loc[ledger["transition_group"] == group]
        groups[group] = {
            "reports": int(len(subset)),
            "cue_counts": {cue: int(subset[cue].sum()) for cue in CUE_TERMS},
            "cue_prevalence": {
                cue: None if subset.empty else float(subset[cue].mean())
                for cue in CUE_TERMS
            },
        }
    aggregate = {
        "cohort": cohort,
        "records": int(len(ledger)),
        "transition_groups": groups,
    }
    return ledger, aggregate
