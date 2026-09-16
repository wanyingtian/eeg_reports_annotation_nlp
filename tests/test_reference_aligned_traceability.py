from __future__ import annotations

import pytest

from eeg_review.reason_traceability import EvidenceUnit
from eeg_review.reference_aligned_traceability import (
    aggregate_reference_aligned_rows,
    binary_decision,
    build_reference_aligned_rows,
)


def _unit(key: str, category: str) -> EvidenceUnit:
    return EvidenceUnit(
        report_key=key,
        category=category,
        report="Normal posterior rhythm. No epileptiform activity.",
        segments=("No epileptiform activity.",),
        segment_roles=("decision_conditioned_reason",),
        source_kind="test",
    )


def test_binary_decision_uses_nested_four_level_contract() -> None:
    assert [binary_decision(level) for level in (1, 2, 3, 4)] == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="four-level"):
        binary_decision(5)


def test_reference_aligned_rows_keep_agreement_and_traceability_separate() -> None:
    units = [_unit("R1", "focal"), _unit("R2", "focal")]
    audit = [
        {
            "unit_number": 0,
            "stage": "verified_exact_substring",
            "verified_quote": True,
        },
        {
            "unit_number": 1,
            "stage": "candidate_whitespace_only",
            "verified_quote": False,
        },
    ]
    rows = build_reference_aligned_rows(
        units,
        audit,
        predictions={("R1", "focal"): 4, ("R2", "focal"): 2},
        references={("R1", "focal"): 3, ("R2", "focal"): 3},
        configured_system="configured-test",
    )
    assert rows[0]["binary_agreement"] is True
    assert rows[0]["exact_four_level_agreement"] is False
    assert rows[0]["has_any_verified_segment"] is True
    assert rows[1]["binary_agreement"] is False
    assert rows[1]["candidate_segments"] == 1
    assert rows[1]["has_any_verified_segment"] is False


def test_reference_aligned_rows_require_complete_key_alignment() -> None:
    units = [_unit("R1", "focal")]
    with pytest.raises(ValueError, match="prediction keys"):
        build_reference_aligned_rows(
            units,
            [],
            predictions={},
            references={("R1", "focal"): 1},
            configured_system="configured-test",
        )


def test_aggregate_retains_empty_unfavorable_strata() -> None:
    units = [_unit("R1", "focal"), _unit("R2", "focal")]
    rows = build_reference_aligned_rows(
        units,
        [
            {
                "unit_number": 0,
                "stage": "verified_exact_substring",
                "verified_quote": True,
            }
        ],
        predictions={("R1", "focal"): 4, ("R2", "focal"): 2},
        references={("R1", "focal"): 3, ("R2", "focal"): 3},
        configured_system="configured-test",
    )
    aggregate = aggregate_reference_aligned_rows(rows)
    overall = next(row for row in aggregate if row["stratification"] == "overall")
    assert overall["units"] == 2
    assert overall["binary_agreement_units"] == 1
    assert overall["units_with_substantive_evidence"] == 1
    assert overall["units_with_any_verified_segment"] == 1
    empty_exact = next(
        row
        for row in aggregate
        if row["stratification"] == "exact_four_level_agreement"
        and row["agreement_value"] == "true"
    )
    assert empty_exact["units"] == 0
    assert empty_exact["evidence_coverage_units_fraction"] is None
