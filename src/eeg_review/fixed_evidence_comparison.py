"""Paired, text-free summaries of saved fixed-decision evidence streams."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .evidence_extraction import classification_levels
from .io import load_table
from .reason_traceability import (
    audit_traceability,
    structured_evidence_units,
    summarize_traceability,
)


def _validated_keys(frame: pd.DataFrame, *, name: str, id_column: str) -> list[str]:
    if id_column not in frame:
        raise ValueError(f"{name} is missing the report-key column")
    keys = frame[id_column].astype("string")
    if keys.isna().any() or keys.str.strip().eq("").any():
        raise ValueError(f"{name} contains a missing report key")
    if keys.duplicated().any():
        raise ValueError(f"{name} contains duplicate report keys")
    return keys.astype(str).tolist()


def load_paired_evidence_surface(
    *,
    dataset,
    manifest,
    streams: dict[str, tuple[Any, str]],
    table: str = "reports",
    id_column: str = "Hashed_ReportURN",
    report_column: str = "Report",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load every named stream in one immutable manifest order.

    ``streams`` maps a name to ``(CSV path, classification-column name)``.  A
    stream may contain additional rows, but every manifest row must occur once
    and the returned view always follows the manifest exactly.
    """
    manifest_frame = pd.read_csv(manifest, usecols=[id_column])
    manifest_keys = _validated_keys(manifest_frame, name="manifest", id_column=id_column)
    reports = load_table(dataset, [id_column, report_column], table)
    report_keys = _validated_keys(reports, name="dataset", id_column=id_column)
    if missing := sorted(set(manifest_keys) - set(report_keys)):
        raise ValueError(f"dataset is missing {len(missing)} manifest keys")
    report_view = (
        reports.assign(**{id_column: reports[id_column].astype(str)})
        .set_index(id_column)
        .loc[manifest_keys]
        .reset_index()
    )
    blank_reports = report_view[report_column].isna() | report_view[
        report_column
    ].str.strip().eq("")
    if blank_reports.any():
        raise ValueError("dataset contains a missing or blank report")

    selected: dict[str, pd.DataFrame] = {}
    for name, (path, classification_column) in streams.items():
        frame = pd.read_csv(
            path,
            usecols=[id_column, classification_column, "explanations"],
        )
        keys = _validated_keys(frame, name=name, id_column=id_column)
        if missing := sorted(set(manifest_keys) - set(keys)):
            raise ValueError(f"{name} is missing {len(missing)} manifest keys")
        view = (
            frame.assign(**{id_column: frame[id_column].astype(str)})
            .set_index(id_column)
            .loc[manifest_keys]
            .reset_index()
            .rename(columns={classification_column: "fixed_classifications"})
        )
        for raw in view["fixed_classifications"]:
            classification_levels(str(raw))
        blank_explanations = view["explanations"].isna() | view[
            "explanations"
        ].astype(str).str.strip().eq("")
        if blank_explanations.any():
            raise ValueError(f"{name} contains a missing explanation")
        selected[name] = view
    return report_view, selected


def summarize_paired_evidence(
    reports: pd.DataFrame,
    streams: dict[str, pd.DataFrame],
    *,
    id_column: str = "Hashed_ReportURN",
    report_column: str = "Report",
) -> dict[str, Any]:
    """Apply one exact-source contract to every stream, without text output."""
    report_keys = reports[id_column].astype(str).tolist()
    output: dict[str, Any] = {}
    for name, frame in streams.items():
        if frame[id_column].astype(str).tolist() != report_keys:
            raise ValueError(f"{name} order differs from the frozen report surface")
        units = structured_evidence_units(
            frame,
            reports,
            source_kind=name,
            id_column=id_column,
            report_column=report_column,
        )
        rows = audit_traceability(units)
        output[name] = summarize_traceability(units, rows)
    return {
        "records": len(reports),
        "paired_same_report_surface": True,
        "primary_rule": "unchanged nonblank substring in the exact source report",
        "candidate_matching_used": False,
        "streams": output,
        "interpretation_boundaries": [
            "This is a development-surface transport diagnostic, not a held-out estimate.",
            (
                "A verified quotation establishes source presence, not entailment or "
                "clinical validity."
            ),
            (
                "Decision-conditioned evidence does not reveal hidden reasoning or "
                "causal faithfulness."
            ),
            (
                "Configured systems differ in model and interface; no base-weight effect "
                "is identified."
            ),
            (
                "Segment counts are repeated dependent observations and are not "
                "inferential sample sizes."
            ),
        ],
    }
