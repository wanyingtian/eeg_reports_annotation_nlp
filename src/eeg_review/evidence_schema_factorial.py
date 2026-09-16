"""Paired descriptive summaries for a two-model by two-evidence-schema design."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from eeg_review.evidence_extraction import JSON_KEYS
from eeg_review.reason_traceability import EvidenceUnit, structured_evidence_units

FACTOR_METRICS = (
    "evidence_coverage_units_fraction",
    "any_verified_all_units_fraction",
    "any_verified_evidence_units_fraction",
    "verified_exact_segment_fraction",
)


def factorial_evidence_units(
    evidence: pd.DataFrame,
    reports: pd.DataFrame,
    *,
    source_kind: str,
    id_column: str = "Hashed_ReportURN",
    report_column: str = "Report",
) -> list[EvidenceUnit]:
    """Parse valid rows and retain unparseable rows as five empty evidence units."""
    report_index = reports.assign(**{id_column: reports[id_column].astype(str)}).set_index(
        id_column
    )
    output: list[EvidenceUnit] = []
    for _, row in evidence.iterrows():
        key = str(row[id_column])
        one = pd.DataFrame([row])
        try:
            output.extend(
                structured_evidence_units(
                    one,
                    reports,
                    source_kind=source_kind,
                    id_column=id_column,
                    report_column=report_column,
                    classification_column="fixed_classifications",
                )
            )
        except (TypeError, ValueError) as exc:
            if bool(row.get("structured_output_valid", False)):
                raise
            report = report_index.at[key, report_column]
            if not isinstance(report, str) or not report.strip():
                raise ValueError("invalid evidence row lacks its source report") from exc
            output.extend(
                EvidenceUnit(
                    report_key=key,
                    category=category,
                    report=report,
                    segments=(),
                    segment_roles=(),
                    source_kind=source_kind,
                )
                for category in JSON_KEYS
            )
    return output


def factorial_contrasts(
    cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute within-model schema effects and a descriptive interaction."""
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in cells:
        key = (str(row["model_factor"]), str(row["evidence_schema_factor"]))
        if key in indexed:
            raise ValueError("duplicate factorial cell")
        indexed[key] = row
    models = sorted({key[0] for key in indexed})
    expected = {
        (model, schema)
        for model in models
        for schema in ("decision_conditioned", "independent_category_evidence")
    }
    if set(indexed) != expected or len(models) != 2:
        raise ValueError("factorial analysis requires exactly two complete model-by-schema cells")

    output: list[dict[str, Any]] = []
    effects: dict[tuple[str, str], float | None] = {}
    for model in models:
        conditioned = indexed[(model, "decision_conditioned")]
        independent = indexed[(model, "independent_category_evidence")]
        for metric in FACTOR_METRICS:
            left = conditioned[metric]
            right = independent[metric]
            effect = None if left is None or right is None else float(right) - float(left)
            effects[(model, metric)] = effect
            output.append(
                {
                    "contrast_type": "within_model_schema",
                    "model_factor": model,
                    "metric": metric,
                    "decision_conditioned": left,
                    "independent_category_evidence": right,
                    "effect": effect,
                }
            )
    reference_model, comparison_model = models
    for metric in FACTOR_METRICS:
        reference_effect = effects[(reference_model, metric)]
        comparison_effect = effects[(comparison_model, metric)]
        interaction = (
            None
            if reference_effect is None or comparison_effect is None
            else comparison_effect - reference_effect
        )
        output.append(
            {
                "contrast_type": "descriptive_interaction",
                "model_factor": f"{comparison_model}_minus_{reference_model}",
                "metric": metric,
                "decision_conditioned": None,
                "independent_category_evidence": None,
                "effect": interaction,
            }
        )
    return output


def paired_schema_transitions(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Count paired unit transitions between evidence schemas within each model."""
    indexed: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["model_factor"]),
            str(row["evidence_schema_factor"]),
            str(row["report_key"]),
            str(row["category"]),
        )
        if key in indexed:
            raise ValueError("duplicate factorial report-category unit")
        indexed[key] = row
    output: list[dict[str, Any]] = []
    for model in sorted({key[0] for key in indexed}):
        conditioned = {
            (key[2], key[3]): row
            for key, row in indexed.items()
            if key[0] == model and key[1] == "decision_conditioned"
        }
        independent = {
            (key[2], key[3]): row
            for key, row in indexed.items()
            if key[0] == model and key[1] == "independent_category_evidence"
        }
        if set(conditioned) != set(independent):
            raise ValueError("factorial schemas do not contain the same report-category units")
        for field in ("has_substantive_evidence", "has_any_verified_segment"):
            counts: Counter[str] = Counter()
            for key in sorted(conditioned):
                left = int(bool(conditioned[key][field]))
                right = int(bool(independent[key][field]))
                counts[f"{left}_to_{right}"] += 1
                for invariant in (
                    "prediction_level",
                    "reference_level",
                    "binary_agreement",
                    "exact_four_level_agreement",
                ):
                    if conditioned[key][invariant] != independent[key][invariant]:
                        raise ValueError(
                            "classification/reference invariant changed across schemas"
                        )
            for transition in ("0_to_0", "0_to_1", "1_to_0", "1_to_1"):
                output.append(
                    {
                        "model_factor": model,
                        "field": field,
                        "transition": transition,
                        "units": counts[transition],
                    }
                )
    return output
