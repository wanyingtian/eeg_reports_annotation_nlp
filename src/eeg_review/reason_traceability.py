"""Cross-generation traceability audit for saved EEG explanation evidence.

The audit deliberately separates source presence from progressively weaker
candidate links.  Only an unchanged substring is a verified quotation.  The
other stages help an author locate material for review; they do not establish
entailment, clinical correctness, or causal faithfulness.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .evidence_extraction import FALLBACK_EVIDENCE, JSON_KEYS, classification_levels
from .explanation_reconciliation import (
    CATEGORIES,
    ID_COLUMN,
    POLARITY_SUFFIX,
    REASON_SUFFIX,
    normalize_text,
    split_report_sentences,
)
from .source_grounding import inspect_reason, text_sha

FUZZY_THRESHOLD = 70
SEMANTIC_THRESHOLD = 0.70


@dataclass(frozen=True)
class EvidenceUnit:
    """One report-category explanation and its declared evidence segments."""

    report_key: str
    category: str
    report: str
    segments: tuple[str, ...]
    segment_roles: tuple[str, ...]
    source_kind: str

    def __post_init__(self) -> None:
        if len(self.segments) != len(self.segment_roles):
            raise ValueError("each evidence segment needs exactly one role")
        if not self.report_key or not self.report:
            raise ValueError("evidence unit requires a report key and report text")


def split_semicolon_segments(value: object) -> tuple[str, ...]:
    """Reproduce the public 2025 script's semicolon segmentation."""
    if value is None or pd.isna(value):
        return ()
    return tuple(part.strip() for part in str(value).split(";") if part.strip())


def split_declared_sentences(value: object) -> tuple[str, ...]:
    """Deterministic reconstruction of the thesis's sentence wording.

    The thesis does not name a tokenizer.  This lightweight rule is therefore
    a declared reconstruction, never silently labelled as the original code.
    """
    if value is None or pd.isna(value):
        return ()
    import re

    return tuple(
        part.strip()
        for part in re.split(r"(?:\n+|(?<=[.!?])\s+)", str(value))
        if part.strip()
    )


def historical_polarity_units(frame: pd.DataFrame) -> list[EvidenceUnit]:
    """Load the thesis-era 2,180-unit surface selected by learned polarity."""
    units: list[EvidenceUnit] = []
    for _, row in frame.iterrows():
        key = str(row[ID_COLUMN])
        report = "" if pd.isna(row["Report"]) else str(row["Report"])
        for category in CATEGORIES:
            polarity = pd.to_numeric(
                pd.Series([row[f"{category}{POLARITY_SUFFIX}"]]), errors="coerce"
            ).iloc[0]
            if polarity != 1:
                continue
            segments = split_semicolon_segments(row[f"{category}{REASON_SUFFIX}"])
            units.append(
                EvidenceUnit(
                    report_key=key,
                    category=category,
                    report=report,
                    segments=segments,
                    segment_roles=("abnormal_supporting",) * len(segments),
                    source_kind="historical_saved_polarity_positive",
                )
            )
    return units


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON field: {key}")
        output[key] = value
    return output


def structured_evidence_units(
    evidence: pd.DataFrame,
    reports: pd.DataFrame,
    *,
    source_kind: str,
    id_column: str = ID_COLUMN,
    report_column: str = "Report",
    explanation_column: str = "explanations",
    classification_column: str = "fixed_classifications",
) -> list[EvidenceUnit]:
    """Adapt saved fixed-decision or independent evidence JSON to one contract."""
    required_evidence = {id_column, explanation_column}
    required_reports = {id_column, report_column}
    if missing := sorted(required_evidence - set(evidence.columns)):
        raise ValueError(f"evidence is missing columns: {missing}")
    if missing := sorted(required_reports - set(reports.columns)):
        raise ValueError(f"reports are missing columns: {missing}")
    for name, frame in (("evidence", evidence), ("reports", reports)):
        keys = frame[id_column].astype("string")
        if keys.isna().any() or keys.str.strip().eq("").any():
            raise ValueError(f"{name} contains a missing report key")
        if keys.duplicated().any():
            raise ValueError(f"{name} contains duplicate report keys")
    report_map = reports.assign(**{id_column: reports[id_column].astype(str)}).set_index(id_column)
    evidence = evidence.assign(**{id_column: evidence[id_column].astype(str)})
    missing_keys = sorted(set(evidence[id_column]) - set(report_map.index))
    if missing_keys:
        raise ValueError(f"reports are missing {len(missing_keys)} evidence keys")

    units: list[EvidenceUnit] = []
    for _, row in evidence.iterrows():
        key = row[id_column]
        report = report_map.at[key, report_column]
        if not isinstance(report, str) or not report.strip():
            raise ValueError("report text is missing or blank")
        value = json.loads(str(row[explanation_column]), object_pairs_hook=_unique_object)
        if not isinstance(value, dict) or set(value) != set(JSON_KEYS):
            raise ValueError("evidence JSON must contain exactly the five category keys")
        fixed = None
        if classification_column in evidence.columns:
            fixed = classification_levels(str(row[classification_column]))
        for category in JSON_KEYS:
            cell = value[category]
            if not isinstance(cell, dict):
                raise ValueError("each evidence category must be an object")
            segments: list[str] = []
            roles: list[str] = []
            if set(cell) == {"decision", "reasons"}:
                if fixed is None:
                    raise ValueError("fixed-decision evidence requires fixed classifications")
                if type(cell["decision"]) is not int or cell["decision"] != fixed[category]:
                    raise ValueError("evidence decision does not copy the fixed classification")
                if not isinstance(cell["reasons"], list):
                    raise ValueError("evidence reasons must be a list")
                for reason in cell["reasons"]:
                    if not isinstance(reason, str):
                        raise ValueError("evidence reasons must be strings")
                    segments.append(reason)
                    roles.append(
                        "declared_no_evidence"
                        if reason == FALLBACK_EVIDENCE
                        else "decision_conditioned_reason"
                    )
            elif set(cell) == {
                "present_evidence",
                "absent_evidence",
                "qualification_evidence",
            }:
                for role in (
                    "present_evidence",
                    "absent_evidence",
                    "qualification_evidence",
                ):
                    values = cell[role]
                    if not isinstance(values, list):
                        raise ValueError("independent evidence fields must be lists")
                    for reason in values:
                        if not isinstance(reason, str) or not reason.strip():
                            raise ValueError("independent evidence must be nonblank text")
                        segments.append(reason)
                        roles.append(role)
            else:
                raise ValueError("unrecognized evidence category schema")
            units.append(
                EvidenceUnit(
                    report_key=key,
                    category=category,
                    report=report,
                    segments=tuple(segments),
                    segment_roles=tuple(roles),
                    source_kind=source_kind,
                )
            )
    return units


def _literal_stage(segment: str, report: str) -> str:
    inspected = inspect_reason(segment, report)
    mapping = {
        "exact": "verified_exact_substring",
        "casefold_only": "candidate_casefold_only",
        "whitespace_only": "candidate_whitespace_only",
        "typography_normalized_only": "candidate_typography_only",
        "typography_and_casefold_only": "candidate_typography_casefold",
        "unmatched_requires_review": "unresolved",
    }
    return mapping.get(inspected["status"], f"excluded_{inspected['status']}")


def audit_traceability(
    units: Sequence[EvidenceUnit],
    *,
    fuzzy_ratio: Callable[[str, str], int] | None = None,
    encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
    fuzzy_threshold: int = FUZZY_THRESHOLD,
    semantic_threshold: float = SEMANTIC_THRESHOLD,
) -> list[dict[str, Any]]:
    """Audit every evidence segment without altering or dropping any unit.

    Fuzzy and semantic stages are review candidates.  They never change the
    verified-quotation count, which is exact-source only.
    """
    rows: list[dict[str, Any]] = []
    pending: list[tuple[int, str, tuple[str, ...], str]] = []
    for unit_number, unit in enumerate(units):
        for segment_number, (segment, role) in enumerate(
            zip(unit.segments, unit.segment_roles, strict=True)
        ):
            stage = _literal_stage(segment, unit.report)
            row = {
                "report_key": unit.report_key,
                "report_text_sha256": text_sha(unit.report),
                "unit_number": unit_number,
                "segment_number": segment_number,
                "category": unit.category,
                "segment_role": role,
                "source_kind": unit.source_kind,
                "segment_sha256": text_sha(segment),
                "stage": stage,
                "verified_quote": stage == "verified_exact_substring",
                "fuzzy_sentence_max": None,
                "semantic_sentence_max": None,
                "semantic_whole_report": None,
            }
            rows.append(row)
            if stage == "unresolved":
                pending.append(
                    (
                        len(rows) - 1,
                        normalize_text(segment),
                        split_report_sentences(unit.report),
                        normalize_text(unit.report),
                    )
                )

    if fuzzy_ratio is not None:
        for row_index, segment, sentences, _ in pending:
            maximum = max((fuzzy_ratio(segment, sentence) for sentence in sentences), default=0)
            rows[row_index]["fuzzy_sentence_max"] = maximum
            if maximum >= fuzzy_threshold:
                rows[row_index]["stage"] = "candidate_fuzzy_sentence"

    if encoder is not None and pending:
        segments = list(dict.fromkeys(segment for _, segment, _, _ in pending if segment))
        sentences = list(
            dict.fromkeys(
                sentence for _, _, report_sentences, _ in pending for sentence in report_sentences
            )
        )
        reports = list(dict.fromkeys(report for _, _, _, report in pending if report))
        segment_vectors = dict(zip(segments, encoder(segments), strict=True))
        sentence_vectors = dict(zip(sentences, encoder(sentences), strict=True))
        report_vectors = dict(zip(reports, encoder(reports), strict=True))
        for row_index, segment, report_sentences, report in pending:
            if not segment:
                continue
            vector = segment_vectors[segment]
            sentence_max = max(
                (float(np.dot(vector, sentence_vectors[s])) for s in report_sentences),
                default=float("nan"),
            )
            whole = float(np.dot(vector, report_vectors[report]))
            rows[row_index]["semantic_sentence_max"] = sentence_max
            rows[row_index]["semantic_whole_report"] = whole
            if rows[row_index]["stage"] == "unresolved":
                if sentence_max >= semantic_threshold:
                    rows[row_index]["stage"] = "candidate_semantic_sentence"
                elif whole >= semantic_threshold:
                    rows[row_index]["stage"] = "candidate_semantic_whole_report"
    return rows


def summarize_traceability(units: Sequence[EvidenceUnit], rows: Sequence[dict[str, Any]]) -> dict:
    """Return aggregate-only counts at both segment and explanation-unit levels."""
    by_unit: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(units))}
    for row in rows:
        by_unit[int(row["unit_number"])].append(row)
    stage_counts = Counter(str(row["stage"]) for row in rows)
    role_counts = Counter(str(row["segment_role"]) for row in rows)
    category_counts: dict[str, Counter] = {}
    for row in rows:
        category_counts.setdefault(str(row["category"]), Counter())[str(row["stage"])] += 1

    def located(row: dict[str, Any]) -> bool:
        return str(row["stage"]).startswith(("verified_", "candidate_"))

    substantive_by_unit = {
        key: [row for row in values if not str(row["stage"]).startswith("excluded_")]
        for key, values in by_unit.items()
    }
    nonempty = [values for values in substantive_by_unit.values() if values]
    substantive_segments = sum(len(values) for values in substantive_by_unit.values())
    verified_segments = sum(bool(row["verified_quote"]) for row in rows)
    located_segments = sum(located(row) for row in rows)
    return {
        "units": len(units),
        "units_with_substantive_segments": len(nonempty),
        "units_without_substantive_segments": len(units) - len(nonempty),
        "segments": len(rows),
        "substantive_segments": substantive_segments,
        "excluded_blank_or_declared_absence_segments": len(rows) - substantive_segments,
        "verified_exact_segments": verified_segments,
        "verified_exact_segment_fraction": (
            verified_segments / substantive_segments if substantive_segments else None
        ),
        "review_candidate_segments": sum(
            str(row["stage"]).startswith("candidate_") for row in rows
        ),
        "unresolved_segments": stage_counts["unresolved"],
        "located_segment_fraction": (
            located_segments / substantive_segments if substantive_segments else None
        ),
        "units_with_any_verified_segment": sum(
            any(row["verified_quote"] for row in values) for values in nonempty
        ),
        "units_with_all_segments_verified": sum(
            all(row["verified_quote"] for row in values) for values in nonempty
        ),
        "units_with_any_located_segment": sum(
            any(located(row) for row in values) for values in nonempty
        ),
        "units_with_all_segments_located": sum(
            all(located(row) for row in values) for values in nonempty
        ),
        "stage_counts": dict(stage_counts),
        "role_counts": dict(role_counts),
        "by_category": {key: dict(value) for key, value in sorted(category_counts.items())},
        "interpretation": [
            "Only verified_exact_substring is an unchanged quotation from the source report.",
            (
                "Candidate stages locate material for review; they do not establish "
                "entailment or factuality."
            ),
            (
                "Any-segment and all-segment unit summaries are both retained because "
                "aggregation changes the question."
            ),
            (
                "Segments from one report and category are dependent observations, "
                "not independent samples."
            ),
            "No stage measures whether the quotation caused the model decision.",
        ],
    }
