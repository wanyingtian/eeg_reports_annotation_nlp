"""Deterministic paired sampling for a source-first evidence review.

The reviewer must not see these selection labels.  They exist only to make a
small review set cover the important automated traceability outcomes without
turning the review into a prevalence estimate.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

FOCUS_STRATA = ("unresolved", "no_evidence", "normalization_only", "all_exact")


def unit_traceability_status(rows: Sequence[Mapping[str, Any]]) -> str:
    """Reduce segment-level audit rows to one non-clinical unit status."""
    substantive = [
        row for row in rows if not str(row["stage"]).startswith("excluded_")
    ]
    if not substantive:
        return "no_evidence"
    stages = [str(row["stage"]) for row in substantive]
    if "unresolved" in stages:
        return "unresolved"
    if all(stage == "verified_exact_substring" for stage in stages):
        return "all_exact"
    if all(stage.startswith("candidate_") for stage in stages):
        return "normalization_only"
    return "mixed_exact_normalization"


def _pattern_priority(focus: str, statuses: tuple[str, str]) -> int:
    """Prefer pure paired examples, then a one-system contrast."""
    count = statuses.count(focus)
    if count == 2:
        return 0
    if count == 1:
        return 1
    raise ValueError("focus stratum is absent from candidate statuses")


def build_source_first_sample(
    pairs: Sequence[Mapping[str, Any]],
    *,
    categories: Sequence[str],
    focus_strata: Sequence[str] = FOCUS_STRATA,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one same-decision paired case per category and focus stratum.

    A report-category pair is eligible only when both configured systems made
    the same exact four-level decision.  That restriction keeps the human
    review focused on evidence quality instead of mixing it with classification
    disagreement.  Selection is deterministic and independent of input order.
    """
    category_order = tuple(categories)
    focus_order = tuple(focus_strata)
    if len(set(category_order)) != len(category_order) or not category_order:
        raise ValueError("categories must be unique and nonempty")
    if len(set(focus_order)) != len(focus_order) or not focus_order:
        raise ValueError("focus strata must be unique and nonempty")
    if not set(focus_order).issubset(FOCUS_STRATA):
        raise ValueError("unknown focus stratum")

    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for pair in pairs:
        record = dict(pair)
        identity = (str(record["report_key"]), str(record["category"]))
        if identity in identities:
            raise ValueError("duplicate report-category pair")
        identities.add(identity)
        statuses = (str(record["status_left"]), str(record["status_right"]))
        if record["decision_left"] != record["decision_right"]:
            continue
        fingerprint = hashlib.sha256(
            (
                f"{record['report_text_sha256']}|{record['category']}|"
                f"{record['decision_left']}|{'|'.join(statuses)}"
            ).encode()
        ).hexdigest()
        record["selection_fingerprint"] = fingerprint
        normalized.append(record)

    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    eligible_counts: Counter[tuple[str, str]] = Counter()
    for focus in focus_order:
        for category in category_order:
            eligible = []
            for pair in normalized:
                identity = (str(pair["report_key"]), str(pair["category"]))
                statuses = (str(pair["status_left"]), str(pair["status_right"]))
                if pair["category"] != category or focus not in statuses:
                    continue
                eligible_counts[(focus, category)] += 1
                if identity in used:
                    continue
                eligible.append(pair)
            if not eligible:
                raise ValueError(f"no unused same-decision pair for {focus} / {category}")
            chosen = min(
                eligible,
                key=lambda pair: (
                    _pattern_priority(
                        focus,
                        (str(pair["status_left"]), str(pair["status_right"])),
                    ),
                    pair["selection_fingerprint"],
                ),
            )
            identity = (str(chosen["report_key"]), str(chosen["category"]))
            used.add(identity)
            selected.append({**chosen, "focus_stratum": focus})

    selected.sort(
        key=lambda row: (
            focus_order.index(str(row["focus_stratum"])),
            category_order.index(str(row["category"])),
        )
    )
    summary = {
        "eligible_same_decision_pairs": len(normalized),
        "selected_pairs": len(selected),
        "selected_system_reviews": 2 * len(selected),
        "design": "one paired same-decision case per category and focus stratum",
        "focus_strata": list(focus_order),
        "categories": list(category_order),
        "eligible_by_focus_and_category": {
            f"{focus}::{category}": eligible_counts[(focus, category)]
            for focus in focus_order
            for category in category_order
        },
        "contains_report_keys_or_text": False,
    }
    return selected, summary


def counterbalanced_aliases(case_ids: Sequence[str]) -> dict[str, bool]:
    """Return whether the left stream is System A, balanced over all cases."""
    ranked = sorted(
        case_ids,
        key=lambda case_id: hashlib.sha256(
            f"source-first-blind|{case_id}".encode()
        ).hexdigest(),
    )
    midpoint = (len(ranked) + 1) // 2
    left_is_a = set(ranked[:midpoint])
    return {case_id: case_id in left_is_a for case_id in case_ids}
