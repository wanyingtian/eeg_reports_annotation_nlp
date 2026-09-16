from __future__ import annotations

import pytest

from eeg_review.source_first_review import (
    build_source_first_sample,
    counterbalanced_aliases,
    unit_traceability_status,
)


def _pair(
    number: int,
    category: str,
    left: str,
    right: str,
    *,
    left_decision: int = 1,
    right_decision: int = 1,
) -> dict[str, object]:
    return {
        "report_key": f"private-{number}",
        "report_text_sha256": f"{number:064x}",
        "category": category,
        "status_left": left,
        "status_right": right,
        "decision_left": left_decision,
        "decision_right": right_decision,
    }


@pytest.mark.parametrize(
    ("stages", "expected"),
    [
        ([], "no_evidence"),
        (["excluded_declared_no_evidence"], "no_evidence"),
        (["verified_exact_substring"], "all_exact"),
        (["candidate_whitespace_only"], "normalization_only"),
        (["verified_exact_substring", "candidate_whitespace_only"], "mixed_exact_normalization"),
        (["verified_exact_substring", "unresolved"], "unresolved"),
    ],
)
def test_unit_traceability_status(stages: list[str], expected: str) -> None:
    assert unit_traceability_status([{"stage": stage} for stage in stages]) == expected


def test_source_first_sample_is_paired_balanced_and_order_independent() -> None:
    categories = ["A", "B"]
    focus = ["unresolved", "no_evidence", "normalization_only", "all_exact"]
    pairs = []
    number = 1
    for stratum in focus:
        for category in categories:
            pairs.extend(
                [
                    _pair(number, category, stratum, stratum),
                    _pair(number + 1, category, stratum, "mixed_exact_normalization"),
                ]
            )
            number += 2
    selected, summary = build_source_first_sample(
        pairs,
        categories=categories,
        focus_strata=focus,
    )
    reverse, _ = build_source_first_sample(
        list(reversed(pairs)),
        categories=categories,
        focus_strata=focus,
    )
    assert len(selected) == 8
    assert summary["selected_system_reviews"] == 16
    assert [row["selection_fingerprint"] for row in selected] == [
        row["selection_fingerprint"] for row in reverse
    ]
    assert all(row["status_left"] == row["status_right"] for row in selected)
    assert summary["contains_report_keys_or_text"] is False
    assert "private-" not in str(summary)


def test_source_first_sample_excludes_decision_disagreement() -> None:
    pairs = [
        _pair(1, "A", "unresolved", "unresolved", left_decision=1, right_decision=4)
    ]
    with pytest.raises(ValueError, match="no unused same-decision pair"):
        build_source_first_sample(
            pairs,
            categories=["A"],
            focus_strata=["unresolved"],
        )


def test_source_first_sample_rejects_duplicate_pair() -> None:
    pair = _pair(1, "A", "unresolved", "unresolved")
    with pytest.raises(ValueError, match="duplicate report-category"):
        build_source_first_sample(
            [pair, dict(pair)],
            categories=["A"],
            focus_strata=["unresolved"],
        )


def test_counterbalanced_aliases_is_deterministic_and_balanced() -> None:
    case_ids = [f"C{index:03d}" for index in range(1, 21)]
    forward = counterbalanced_aliases(case_ids)
    reverse = counterbalanced_aliases(list(reversed(case_ids)))
    assert forward == reverse
    assert sum(forward.values()) == 10
