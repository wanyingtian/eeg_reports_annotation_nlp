"""Reconcile a thesis-era explanation artifact without exposing governed rows.

This module does not regenerate explanations or polarity labels. It validates a
candidate historical artifact, computes aggregate-only checks, and keeps the
historical explanation-test surface distinct from the paper's classification
evaluation cohorts.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = (
    "Focal Epi",
    "Gen Epi",
    "Focal Non-epi",
    "Gen Non-epi",
    "Abnormality",
)
ID_COLUMN = "Hashed_ReportURN"
LEGACY_ID_COLUMN = "Hashed ID"
REPORT_COLUMN = "Report"
POLARITY_SUFFIX = " Reason Polarity"
REASON_SUFFIX = " Reasons"
VALID_LABELS = frozenset({1, 2, 3, 4})
VALID_POLARITIES = frozenset({-1, 0, 1})


@dataclass(frozen=True)
class TraceabilityUnit:
    """One report-category explanation selected by positive learned polarity."""

    row_number: int
    category: str
    report: str
    phrases: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    normalized = re.sub(r"[^\w\s]", " ", str(value).casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def split_reason_phrases(value: object) -> tuple[str, ...]:
    if value is None or pd.isna(value):
        return ()
    return tuple(
        normalized
        for part in str(value).split(";")
        if (normalized := normalize_text(part))
    )


def split_report_sentences(value: object) -> tuple[str, ...]:
    if value is None or pd.isna(value):
        return ()
    return tuple(
        normalized
        for part in re.split(r"(?:\n+|(?<=[.!?])\s+)", str(value))
        if (normalized := normalize_text(part))
    )


def _required_columns() -> set[str]:
    required = {ID_COLUMN, REPORT_COLUMN}
    required.update(CATEGORIES)
    required.update(f"{category}{REASON_SUFFIX}" for category in CATEGORIES)
    required.update(f"{category}{POLARITY_SUFFIX}" for category in CATEGORIES)
    return required


def load_explanation_artifact(path: Path) -> pd.DataFrame:
    """Load and strictly validate the candidate thesis-era Zoe artifact."""
    frame = pd.read_csv(path)
    if LEGACY_ID_COLUMN in frame.columns and ID_COLUMN not in frame.columns:
        frame = frame.rename(columns={LEGACY_ID_COLUMN: ID_COLUMN})
    missing = sorted(_required_columns() - set(frame.columns))
    if missing:
        raise ValueError(f"explanation artifact is missing columns: {missing}")
    if frame.empty:
        raise ValueError("explanation artifact is empty")
    raw_keys = frame[ID_COLUMN]
    missing_keys = raw_keys.isna() | raw_keys.astype("string").str.strip().eq("")
    if missing_keys.any():
        raise ValueError("explanation artifact has missing report keys")
    frame[ID_COLUMN] = raw_keys.astype(str)
    duplicates = int(frame[ID_COLUMN].duplicated().sum())
    if duplicates:
        raise ValueError(f"explanation artifact has {duplicates} duplicate report keys")
    for category in CATEGORIES:
        labels = pd.to_numeric(frame[category], errors="coerce")
        invalid_labels = labels.notna() & ~labels.isin(VALID_LABELS)
        if invalid_labels.any():
            raise ValueError(
                f"{category} has {int(invalid_labels.sum())} invalid four-level labels"
            )
        polarity = pd.to_numeric(frame[f"{category}{POLARITY_SUFFIX}"], errors="coerce")
        invalid_polarity = ~polarity.isin({-1, 0, 1, 2, 3, 4})
        if invalid_polarity.any():
            raise ValueError(
                f"{category} has {int(invalid_polarity.sum())} unreadable polarity values"
            )
    return frame


def artifact_census(frame: pd.DataFrame) -> dict:
    by_category = {}
    for category in CATEGORIES:
        model_label = pd.to_numeric(frame[category], errors="coerce")
        polarity = pd.to_numeric(frame[f"{category}{POLARITY_SUFFIX}"], errors="coerce")
        reasons = frame[f"{category}{REASON_SUFFIX}"]
        by_category[category] = {
            "model_positive_label": int(model_label.isin((3, 4)).sum()),
            "missing_model_label": int(model_label.isna().sum()),
            "positive_polarity": int((polarity == 1).sum()),
            "negative_polarity": int((polarity == -1).sum()),
            "zero_unscored_polarity": int((polarity == 0).sum()),
            "out_of_contract_polarity": int((~polarity.isin(VALID_POLARITIES)).sum()),
            "missing_reason": int(reasons.isna().sum()),
        }
    return {
        "rows": int(len(frame)),
        "unique_report_keys": int(frame[ID_COLUMN].nunique()),
        "unique_report_texts": int(frame[REPORT_COLUMN].nunique()),
        "positive_polarity_total": sum(
            category["positive_polarity"] for category in by_category.values()
        ),
        "model_positive_label_total": sum(
            category["model_positive_label"] for category in by_category.values()
        ),
        "categories": by_category,
    }


def label_polarity(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.map(lambda value: -1 if value in (1, 2) else 1 if value in (3, 4) else 0)


def _binary_metrics(truth: pd.Series, predicted: pd.Series) -> dict:
    truth = truth.astype(int)
    predicted = predicted.astype(int)
    negative_true = truth == -1
    positive_true = truth == 1
    negative_predicted = predicted == -1
    positive_predicted = predicted == 1
    tn = int((negative_true & negative_predicted).sum())
    fp = int((negative_true & positive_predicted).sum())
    fn = int((positive_true & negative_predicted).sum())
    tp = int((positive_true & positive_predicted).sum())

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    negative_precision = ratio(tn, tn + fn)
    negative_recall = ratio(tn, tn + fp)
    positive_precision = ratio(tp, tp + fp)
    positive_recall = ratio(tp, tp + fn)

    def f1(precision: float | None, recall: float | None) -> float | None:
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    return {
        "n": tn + fp + fn + tp,
        "accuracy": ratio(tn + tp, tn + fp + fn + tp),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "negative": {
            "precision": negative_precision,
            "recall": negative_recall,
            "f1": f1(negative_precision, negative_recall),
        },
        "positive": {
            "precision": positive_precision,
            "recall": positive_recall,
            "f1": f1(positive_precision, positive_recall),
        },
    }


def polarity_classifier_alignment(frame: pd.DataFrame, *, test_start: int = 200) -> dict:
    """Evaluate saved learned polarity labels on explicit post-training rows."""
    rows = frame.iloc[test_start:].copy()
    categories = {}
    for category in CATEGORIES:
        predicted = pd.to_numeric(rows[f"{category}{POLARITY_SUFFIX}"], errors="coerce")
        truth = label_polarity(rows[category])
        reason_present = rows[f"{category}{REASON_SUFFIX}"].notna()
        valid = predicted.isin((-1, 1)) & truth.isin((-1, 1)) & reason_present
        categories[category] = {
            **_binary_metrics(truth[valid], predicted[valid]),
            "candidate_rows": int(len(rows)),
            "excluded_unscored_or_invalid": int((~valid).sum()),
        }
    return {
        "surface": "saved_polarity_rows_after_fixed_first_200_training_rows",
        "training_rows": test_start,
        "candidate_test_rows": int(len(rows)),
        "categories": categories,
    }


def load_reference(reference_db: Path) -> pd.DataFrame:
    with sqlite3.connect(reference_db) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(reports)").fetchall()
        }
        source_id = ID_COLUMN if ID_COLUMN in columns else LEGACY_ID_COLUMN
        requested = [source_id, *CATEGORIES]
        quoted = ", ".join(f'"{column}"' for column in requested)
        frame = pd.read_sql_query(f"SELECT {quoted} FROM reports", connection)
    return frame.rename(columns={source_id: ID_COLUMN})


def reconcile_source_snapshot(artifact: pd.DataFrame, source_db: Path) -> dict:
    """Verify keys, order, and report text against the historical source snapshot."""
    with sqlite3.connect(source_db) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(reports)").fetchall()
        }
        source_id = ID_COLUMN if ID_COLUMN in columns else LEGACY_ID_COLUMN
        source = pd.read_sql_query(
            f'SELECT "{source_id}" AS "{ID_COLUMN}", "{REPORT_COLUMN}" FROM reports',
            connection,
        )
    source[ID_COLUMN] = source[ID_COLUMN].astype(str)
    if source[ID_COLUMN].duplicated().any():
        raise ValueError("source snapshot has duplicate report keys")
    source_by_key = source.set_index(ID_COLUMN)
    artifact_keys = artifact[ID_COLUMN].astype(str)
    missing = artifact_keys[~artifact_keys.isin(source_by_key.index)]
    if not missing.empty:
        raise ValueError(f"source snapshot is missing {len(missing)} artifact keys")
    source_reports = source_by_key.loc[artifact_keys, REPORT_COLUMN].reset_index(drop=True)
    artifact_reports = artifact[REPORT_COLUMN].reset_index(drop=True)
    text_exact = artifact_reports.fillna("").eq(source_reports.fillna(""))
    source_prefix = source[ID_COLUMN].iloc[: len(artifact)].reset_index(drop=True)
    return {
        "source_rows": int(len(source)),
        "artifact_rows": int(len(artifact)),
        "keys_matched": int(len(artifact_keys)),
        "artifact_order_matches_source_prefix": bool(
            artifact_keys.reset_index(drop=True).eq(source_prefix).all()
        ),
        "report_text_exact_matches": int(text_exact.sum()),
        "report_text_mismatches": int((~text_exact).sum()),
    }


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if LEGACY_ID_COLUMN in frame.columns and ID_COLUMN not in frame.columns:
        frame = frame.rename(columns={LEGACY_ID_COLUMN: ID_COLUMN})
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"manifest lacks {ID_COLUMN}")
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("manifest has duplicate report keys")
    return frame[[ID_COLUMN]].copy()


def join_reference(
    artifact: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    reference = reference.copy()
    reference[ID_COLUMN] = reference[ID_COLUMN].astype(str)
    if reference[ID_COLUMN].duplicated().any():
        raise ValueError("reference has duplicate report keys")
    if manifest is not None:
        keys = set(manifest[ID_COLUMN])
        missing_artifact = keys - set(artifact[ID_COLUMN])
        missing_reference = keys - set(reference[ID_COLUMN])
        if missing_artifact or missing_reference:
            raise ValueError(
                "manifest keys are incomplete: "
                f"artifact_missing={len(missing_artifact)}, "
                f"reference_missing={len(missing_reference)}"
            )
        artifact = manifest.merge(artifact, on=ID_COLUMN, validate="one_to_one")
        reference = manifest.merge(reference, on=ID_COLUMN, validate="one_to_one")
    return artifact.merge(
        reference[[ID_COLUMN, *CATEGORIES]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        suffixes=("_model", "_reference"),
    )


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def correctness_by_alignment(joined: pd.DataFrame, *, surface: str) -> dict:
    """Associate saved polarity/model agreement with reference correctness.

    The unit is one report-category decision. This is descriptive association,
    not a causal-faithfulness or clinical-validity analysis.
    """
    output = {}
    pooled_low_confidence = 0
    pooled_misaligned = 0
    for category in CATEGORIES:
        polarity = pd.to_numeric(joined[f"{category}{POLARITY_SUFFIX}"], errors="coerce")
        model_label = pd.to_numeric(joined[f"{category}_model"], errors="coerce")
        reference_label = pd.to_numeric(joined[f"{category}_reference"], errors="coerce")
        model_core = label_polarity(model_label)
        reference_core = label_polarity(reference_label)
        valid = polarity.isin((-1, 1)) & model_core.isin((-1, 1)) & reference_core.isin((-1, 1))
        aligned = polarity == model_core
        correct = model_core == reference_core
        low_confidence = model_label.isin((2, 3))

        counts = {}
        for name, group in (("aligned", aligned), ("misaligned", ~aligned)):
            group_valid = valid & group
            total = int(group_valid.sum())
            successes = int((group_valid & correct).sum())
            interval = _wilson(successes, total)
            counts[name] = {
                "correct": successes,
                "incorrect": total - successes,
                "total": total,
                "accuracy": successes / total if total else None,
                "accuracy_wilson_95": list(interval) if total else None,
            }

        aligned_rate = counts["aligned"]["accuracy"]
        misaligned_rate = counts["misaligned"]["accuracy"]
        aligned_interval = counts["aligned"]["accuracy_wilson_95"]
        misaligned_interval = counts["misaligned"]["accuracy_wilson_95"]
        risk_difference = None
        risk_difference_interval = None
        if aligned_rate is not None and misaligned_rate is not None:
            risk_difference = aligned_rate - misaligned_rate
            risk_difference_interval = [
                aligned_interval[0] - misaligned_interval[1],
                aligned_interval[1] - misaligned_interval[0],
            ]

        misaligned_valid = valid & ~aligned
        low_count = int((misaligned_valid & low_confidence).sum())
        misaligned_count = int(misaligned_valid.sum())
        pooled_low_confidence += low_count
        pooled_misaligned += misaligned_count
        output[category] = {
            **counts,
            "accuracy_difference_aligned_minus_misaligned": risk_difference,
            "accuracy_difference_conservative_95": risk_difference_interval,
            "misaligned_low_confidence": low_count,
            "misaligned_total": misaligned_count,
            "misaligned_low_confidence_fraction": (
                low_count / misaligned_count if misaligned_count else None
            ),
            "excluded_unscored_or_missing_reference": int((~valid).sum()),
        }
    return {
        "surface": surface,
        "unit": "report_category_decision",
        "interpretation": "descriptive association; not causal faithfulness or clinical validation",
        "categories": output,
        "pooled_misaligned_low_confidence": pooled_low_confidence,
        "pooled_misaligned_total": pooled_misaligned,
        "pooled_misaligned_low_confidence_fraction": (
            pooled_low_confidence / pooled_misaligned if pooled_misaligned else None
        ),
    }


def positive_traceability_units(frame: pd.DataFrame) -> list[TraceabilityUnit]:
    units = []
    for row_number, row in frame.iterrows():
        for category in CATEGORIES:
            if row[f"{category}{POLARITY_SUFFIX}"] != 1:
                continue
            units.append(
                TraceabilityUnit(
                    row_number=int(row_number),
                    category=category,
                    report="" if pd.isna(row[REPORT_COLUMN]) else str(row[REPORT_COLUMN]),
                    phrases=split_reason_phrases(row[f"{category}{REASON_SUFFIX}"]),
                )
            )
    return units


def deterministic_traceability(
    units: Sequence[TraceabilityUnit],
    *,
    fuzzy_ratio: Callable[[str, str], int] | None = None,
    fuzzy_threshold: int = 70,
    fuzzy_surface: str = "whole_report",
) -> list[str]:
    """Classify exact/fuzzy stages without loading an embedding model."""
    if fuzzy_surface not in {"whole_report", "sentences"}:
        raise ValueError("fuzzy_surface must be whole_report or sentences")
    stages = []
    for unit in units:
        report = normalize_text(unit.report)
        if any(phrase and phrase in report for phrase in unit.phrases):
            stages.append("normalized_exact_substring")
            continue
        if fuzzy_ratio is None:
            stages.append("unmatched")
            continue
        surfaces = (
            (report,)
            if fuzzy_surface == "whole_report"
            else split_report_sentences(unit.report)
        )
        if any(
            fuzzy_ratio(phrase, surface) >= fuzzy_threshold
            for phrase in unit.phrases
            for surface in surfaces
        ):
            stages.append(f"fuzzy_{fuzzy_surface}")
        else:
            stages.append("unmatched")
    return stages


def semantic_complete(
    units: Sequence[TraceabilityUnit],
    stages: Sequence[str],
    *,
    encoder: Callable[[Sequence[str]], np.ndarray],
    threshold: float = 0.70,
    semantic_surface: str = "whole_report",
) -> tuple[list[str], list[float | None]]:
    """Complete unmatched units against whole reports or report sentences."""
    if semantic_surface not in {"whole_report", "sentences"}:
        raise ValueError("semantic_surface must be whole_report or sentences")
    completed = list(stages)
    maxima: list[float | None] = [None] * len(units)
    candidate_pairs = []
    for index, (unit, stage) in enumerate(zip(units, stages, strict=True)):
        if stage != "unmatched":
            continue
        surfaces = (
            (normalize_text(unit.report),)
            if semantic_surface == "whole_report"
            else split_report_sentences(unit.report)
        )
        for phrase in unit.phrases:
            for surface in surfaces:
                candidate_pairs.append((index, phrase, surface))
    if not candidate_pairs:
        return completed, maxima
    phrases = list(dict.fromkeys(phrase for _, phrase, _ in candidate_pairs))
    surfaces = list(dict.fromkeys(surface for _, _, surface in candidate_pairs))
    phrase_vectors = encoder(phrases)
    surface_vectors = encoder(surfaces)
    phrase_map = dict(zip(phrases, phrase_vectors, strict=True))
    surface_map = dict(zip(surfaces, surface_vectors, strict=True))
    for index, phrase, surface in candidate_pairs:
        score = float(np.dot(phrase_map[phrase], surface_map[surface]))
        prior = maxima[index]
        maxima[index] = score if prior is None else max(prior, score)
    for index, maximum in enumerate(maxima):
        if completed[index] == "unmatched" and maximum is not None and maximum >= threshold:
            completed[index] = f"semantic_{semantic_surface}"
    return completed, maxima


def summarize_stages(stages: Iterable[str]) -> dict:
    counts = pd.Series(list(stages), dtype="object").value_counts().sort_index().to_dict()
    total = int(sum(counts.values()))
    unmatched = int(counts.get("unmatched", 0))
    return {
        "total": total,
        "matched": total - unmatched,
        "unmatched": unmatched,
        "match_fraction": (total - unmatched) / total if total else None,
        "stages": {str(key): int(value) for key, value in counts.items()},
    }
