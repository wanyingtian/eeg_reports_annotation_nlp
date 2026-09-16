"""Reference-label agreement joined to saved source-traceability evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from eeg_review.reason_traceability import EvidenceUnit
from eeg_review.source_grounding import text_sha


def binary_decision(level: int) -> int:
    """Collapse the study's four levels to absent (0) versus present (1)."""
    if level not in {1, 2, 3, 4}:
        raise ValueError("four-level decision must be in {1, 2, 3, 4}")
    return int(level >= 3)


def build_reference_aligned_rows(
    units: Sequence[EvidenceUnit],
    audit_rows: Sequence[Mapping[str, Any]],
    *,
    predictions: Mapping[tuple[str, str], int],
    references: Mapping[tuple[str, str], int],
    configured_system: str,
) -> list[dict[str, Any]]:
    """Build governed unit rows with agreement and traceability kept separate."""
    if not configured_system:
        raise ValueError("configured_system is required")
    indexed_audit: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        unit_number = int(row["unit_number"])
        if unit_number < 0 or unit_number >= len(units):
            raise ValueError("audit row references an unknown evidence unit")
        indexed_audit[unit_number].append(row)

    expected_keys = [(unit.report_key, unit.category) for unit in units]
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("evidence units contain duplicate report-category keys")
    if set(predictions) != set(expected_keys):
        raise ValueError("prediction keys do not match evidence units")
    if set(references) != set(expected_keys):
        raise ValueError("reference keys do not match evidence units")

    output: list[dict[str, Any]] = []
    for unit_number, unit in enumerate(units):
        key = (unit.report_key, unit.category)
        prediction_level = int(predictions[key])
        reference_level = int(references[key])
        prediction_binary = binary_decision(prediction_level)
        reference_binary = binary_decision(reference_level)
        rows = indexed_audit[unit_number]
        substantive = [
            row for row in rows if not str(row["stage"]).startswith("excluded_")
        ]
        verified = [row for row in substantive if bool(row["verified_quote"])]
        candidates = [
            row for row in substantive if str(row["stage"]).startswith("candidate_")
        ]
        unresolved = [row for row in substantive if str(row["stage"]) == "unresolved"]
        output.append(
            {
                "configured_system": configured_system,
                "report_key": unit.report_key,
                "report_text_sha256": text_sha(unit.report),
                "category": unit.category,
                "prediction_level": prediction_level,
                "reference_level": reference_level,
                "prediction_binary": prediction_binary,
                "reference_binary": reference_binary,
                "binary_agreement": prediction_binary == reference_binary,
                "exact_four_level_agreement": prediction_level == reference_level,
                "substantive_segments": len(substantive),
                "verified_exact_segments": len(verified),
                "candidate_segments": len(candidates),
                "unresolved_segments": len(unresolved),
                "excluded_segments": len(rows) - len(substantive),
                "has_substantive_evidence": bool(substantive),
                "has_any_verified_segment": bool(verified),
                "all_substantive_segments_verified": (
                    len(verified) == len(substantive) if substantive else None
                ),
            }
        )
    return output


def _aggregate_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    stratification: str,
    configured_system: str,
    category: str = "all",
    agreement_value: str = "all",
) -> dict[str, Any]:
    units = len(rows)
    evidence_rows = [row for row in rows if bool(row["has_substantive_evidence"])]
    evidence_units = len(evidence_rows)
    any_verified = sum(bool(row["has_any_verified_segment"]) for row in rows)
    all_verified = sum(
        bool(row["all_substantive_segments_verified"]) for row in evidence_rows
    )
    substantive_segments = sum(int(row["substantive_segments"]) for row in rows)
    verified_segments = sum(int(row["verified_exact_segments"]) for row in rows)
    return {
        "stratification": stratification,
        "configured_system": configured_system,
        "category": category,
        "agreement_value": agreement_value,
        "units": units,
        "reference_present_units": sum(int(row["reference_binary"]) for row in rows),
        "predicted_present_units": sum(int(row["prediction_binary"]) for row in rows),
        "binary_agreement_units": sum(bool(row["binary_agreement"]) for row in rows),
        "exact_four_level_agreement_units": sum(
            bool(row["exact_four_level_agreement"]) for row in rows
        ),
        "units_with_substantive_evidence": evidence_units,
        "evidence_coverage_units_fraction": evidence_units / units if units else None,
        "units_with_any_verified_segment": any_verified,
        "any_verified_all_units_fraction": any_verified / units if units else None,
        "any_verified_evidence_units_fraction": (
            any_verified / evidence_units if evidence_units else None
        ),
        "evidence_units_all_segments_verified": all_verified,
        "all_verified_evidence_units_fraction": (
            all_verified / evidence_units if evidence_units else None
        ),
        "substantive_segments": substantive_segments,
        "verified_exact_segments": verified_segments,
        "candidate_segments": sum(int(row["candidate_segments"]) for row in rows),
        "unresolved_segments": sum(int(row["unresolved_segments"]) for row in rows),
        "excluded_segments": sum(int(row["excluded_segments"]) for row in rows),
        "verified_exact_segment_fraction": (
            verified_segments / substantive_segments if substantive_segments else None
        ),
    }


def aggregate_reference_aligned_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the complete pre-specified aggregate strata."""
    if not rows:
        raise ValueError("reference-aligned rows are required")
    systems = sorted({str(row["configured_system"]) for row in rows})
    output: list[dict[str, Any]] = []
    for system in systems:
        system_rows = [row for row in rows if row["configured_system"] == system]
        categories = sorted({str(row["category"]) for row in system_rows})
        output.append(
            _aggregate_group(
                system_rows,
                stratification="overall",
                configured_system=system,
            )
        )
        for category in categories:
            category_rows = [row for row in system_rows if row["category"] == category]
            output.append(
                _aggregate_group(
                    category_rows,
                    stratification="category",
                    configured_system=system,
                    category=category,
                )
            )
        for field, label in (
            ("binary_agreement", "binary_agreement"),
            ("exact_four_level_agreement", "exact_four_level_agreement"),
        ):
            for value in (False, True):
                agreement_rows = [row for row in system_rows if bool(row[field]) is value]
                output.append(
                    _aggregate_group(
                        agreement_rows,
                        stratification=label,
                        configured_system=system,
                        agreement_value=str(value).lower(),
                    )
                )
                for category in categories:
                    category_rows = [
                        row
                        for row in agreement_rows
                        if row["category"] == category
                    ]
                    output.append(
                        _aggregate_group(
                            category_rows,
                            stratification=f"category_{label}",
                            configured_system=system,
                            category=category,
                            agreement_value=str(value).lower(),
                        )
                    )
    return output
