"""Blinded agreement and disagreement summaries for two evidence reviewers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from eeg_review.evidence_review_responses import (
    PAIR_FIELD,
    SYSTEM_FIELDS,
    validate_review_response,
)

SOURCE_FIELDS = ("reader_category_judgment",)


def _agreement_record(
    *,
    scope: str,
    field: str,
    first_values: Sequence[str],
    second_values: Sequence[str],
) -> dict[str, Any]:
    agreements = sum(left == right for left, right in zip(first_values, second_values, strict=True))
    total = len(first_values)
    return {
        "scope": scope,
        "field": field,
        "agreements": agreements,
        "disagreements": total - agreements,
        "total": total,
        "raw_agreement": agreements / total if total else None,
    }


def compare_blinded_reviews(
    first_payload: Mapping[str, Any],
    second_payload: Mapping[str, Any],
    *,
    expected_case_ids: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Compare two complete blinded reviews without copying free text.

    The returned queue contains only case identifiers, blinded system aliases,
    controlled field names and controlled response values. It deliberately
    excludes report text, evidence phrases and free-text notes.
    """
    first_cases = validate_review_response(
        first_payload,
        expected_case_ids=expected_case_ids,
    )
    second_cases = validate_review_response(
        second_payload,
        expected_case_ids=expected_case_ids,
    )
    first_code = str(first_payload["reviewer_code"])
    second_code = str(second_payload["reviewer_code"])
    if first_code == second_code:
        raise ValueError("reviewer_code values must be different")

    first_by_case = {str(case["case_id"]): case for case in first_cases}
    second_by_case = {str(case["case_id"]): case for case in second_cases}
    disagreements: list[dict[str, str]] = []
    agreement_rows: list[dict[str, Any]] = []

    def compare_field(
        *,
        scope: str,
        field: str,
        system: str | None = None,
    ) -> None:
        first_values: list[str] = []
        second_values: list[str] = []
        for case_id in expected_case_ids:
            left_case = first_by_case[str(case_id)]
            right_case = second_by_case[str(case_id)]
            if system is None:
                left = str(left_case["fields"][field])
                right = str(right_case["fields"][field])
            else:
                left_systems = {
                    str(item["system"]): item["fields"]
                    for item in left_case["systems"]
                }
                right_systems = {
                    str(item["system"]): item["fields"]
                    for item in right_case["systems"]
                }
                left = str(left_systems[system][field])
                right = str(right_systems[system][field])
            first_values.append(left)
            second_values.append(right)
            if left != right:
                disagreements.append(
                    {
                        "case_id": str(case_id),
                        "scope": scope,
                        "system": system or "",
                        "field": field,
                        "reviewer_1_code": first_code,
                        "reviewer_1_value": left,
                        "reviewer_2_code": second_code,
                        "reviewer_2_value": right,
                        "adjudication_outcome": "",
                        "adjudicated_value": "",
                        "adjudicator_code": "",
                        "adjudication_notes": "",
                    }
                )
        agreement_rows.append(
            _agreement_record(
                scope=scope,
                field=field,
                first_values=first_values,
                second_values=second_values,
            )
        )

    for field in SOURCE_FIELDS:
        compare_field(scope="source", field=field)
    compare_field(scope="pair", field=PAIR_FIELD)
    for system in ("System A", "System B"):
        for field in SYSTEM_FIELDS:
            compare_field(scope=system, field=field, system=system)

    summary = {
        "status": "complete_blinded_two_reader_comparison",
        "reviewers": [
            {
                "reviewer_code": first_code,
                "reviewer_role": str(first_payload["reviewer_role"]),
            },
            {
                "reviewer_code": second_code,
                "reviewer_role": str(second_payload["reviewer_role"]),
            },
        ],
        "cases_compared": len(expected_case_ids),
        "agreement_by_field": agreement_rows,
        "disagreement_items": len(disagreements),
        "next_status": (
            "awaiting_blinded_adjudication" if disagreements else "ready_for_unblinding_decision"
        ),
        "interpretation_boundaries": [
            "Raw agreement describes two readers on a purposive development sample.",
            "No chance-corrected coefficient or population inference is reported.",
            "System identities remain blinded during comparison and adjudication.",
            "Report text, evidence phrases and free-text notes are omitted.",
            "Disagreements require adjudication before any unblinding decision.",
        ],
    }
    return summary, disagreements
