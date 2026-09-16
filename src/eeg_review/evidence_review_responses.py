"""Validation and count summaries for the source-first evidence review."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

SYSTEM_FIELDS = (
    "source_presence",
    "category_relevance",
    "supports_stated_decision",
    "contradicts_stated_decision",
    "sufficient_as_explanation",
    "no_evidence_omission_reasonable",
)
PAIR_FIELD = "more_useful_evidence"


def validate_review_response(
    payload: Mapping[str, Any],
    *,
    expected_case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Reject incomplete, duplicate or unblinded response payloads."""
    if not str(payload.get("reviewer_code", "")).strip():
        raise ValueError("reviewer_code is required")
    if not str(payload.get("reviewer_role", "")).strip():
        raise ValueError("reviewer_role is required")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    expected = set(expected_case_ids)
    seen: set[str] = set()
    validated = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        case_id = str(case.get("case_id", ""))
        if case_id in seen:
            raise ValueError("duplicate case_id")
        seen.add(case_id)
        fields = case.get("fields")
        if not isinstance(fields, dict):
            raise ValueError(f"{case_id}: fields must be an object")
        if not str(fields.get("reader_category_judgment", "")).strip():
            raise ValueError(f"{case_id}: reader_category_judgment is required")
        if not str(fields.get(PAIR_FIELD, "")).strip():
            raise ValueError(f"{case_id}: {PAIR_FIELD} is required")
        systems = case.get("systems")
        if not isinstance(systems, list) or len(systems) != 2:
            raise ValueError(f"{case_id}: exactly two system reviews are required")
        aliases = {str(system.get("system", "")) for system in systems}
        if aliases != {"System A", "System B"}:
            raise ValueError(f"{case_id}: system aliases must be A and B")
        for system in systems:
            system_fields = system.get("fields")
            if not isinstance(system_fields, dict):
                raise ValueError(f"{case_id}: system fields must be an object")
            for field in SYSTEM_FIELDS:
                if not str(system_fields.get(field, "")).strip():
                    raise ValueError(f"{case_id} / {system['system']}: {field} is required")
        validated.append(case)
    if seen != expected:
        raise ValueError("response case ids do not match the frozen package")
    return validated


def summarize_review_response(
    payload: Mapping[str, Any],
    *,
    expected_case_ids: Sequence[str],
) -> dict[str, Any]:
    """Return counts only; never copy report, phrase or free-text note content."""
    cases = validate_review_response(payload, expected_case_ids=expected_case_ids)
    source_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    system_counts: dict[str, dict[str, Counter[str]]] = {
        alias: {field: Counter() for field in SYSTEM_FIELDS}
        for alias in ("System A", "System B")
    }
    for case in cases:
        source_counts[str(case["fields"]["reader_category_judgment"])] += 1
        pair_counts[str(case["fields"][PAIR_FIELD])] += 1
        for system in case["systems"]:
            alias = str(system["system"])
            for field in SYSTEM_FIELDS:
                system_counts[alias][field][str(system["fields"][field])] += 1
    return {
        "status": "complete_blinded_review",
        "reviewer_code": str(payload["reviewer_code"]),
        "reviewer_role": str(payload["reviewer_role"]),
        "cases_reviewed": len(cases),
        "system_reviews": 2 * len(cases),
        "source_judgment_counts": dict(sorted(source_counts.items())),
        "pair_preference_counts": dict(sorted(pair_counts.items())),
        "system_judgment_counts": {
            alias: {
                field: dict(sorted(counts.items()))
                for field, counts in fields.items()
            }
            for alias, fields in system_counts.items()
        },
        "interpretation_boundaries": [
            "Counts describe a purposive development review, not prevalence or performance.",
            "System identities remain blinded in this summary.",
            "Free-text notes, report text and evidence phrases are not copied into the summary.",
            "A full-cohort run remains a separate decision after review interpretation.",
        ],
    }
