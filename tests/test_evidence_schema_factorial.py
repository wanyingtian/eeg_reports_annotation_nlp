from __future__ import annotations

import pytest

from eeg_review.evidence_schema_factorial import (
    factorial_contrasts,
    paired_schema_transitions,
)


def _cell(model: str, schema: str, value: float) -> dict:
    return {
        "model_factor": model,
        "evidence_schema_factor": schema,
        "evidence_coverage_units_fraction": value,
        "any_verified_all_units_fraction": value / 2,
        "any_verified_evidence_units_fraction": value / 3,
        "verified_exact_segment_fraction": value / 4,
    }


def test_factorial_contrasts_compute_within_model_effect_and_interaction() -> None:
    cells = [
        _cell("A", "decision_conditioned", 0.4),
        _cell("A", "independent_category_evidence", 0.6),
        _cell("B", "decision_conditioned", 0.3),
        _cell("B", "independent_category_evidence", 0.8),
    ]
    contrasts = factorial_contrasts(cells)
    a_coverage = next(
        row
        for row in contrasts
        if row["contrast_type"] == "within_model_schema"
        and row["model_factor"] == "A"
        and row["metric"] == "evidence_coverage_units_fraction"
    )
    interaction = next(
        row
        for row in contrasts
        if row["contrast_type"] == "descriptive_interaction"
        and row["metric"] == "evidence_coverage_units_fraction"
    )
    assert a_coverage["effect"] == pytest.approx(0.2)
    assert interaction["effect"] == pytest.approx(0.3)


def test_factorial_contrasts_require_complete_design() -> None:
    with pytest.raises(ValueError, match="two complete"):
        factorial_contrasts(
            [
                _cell("A", "decision_conditioned", 0.4),
                _cell("A", "independent_category_evidence", 0.6),
            ]
        )


def _unit(model: str, schema: str, *, evidence: bool, exact: bool) -> dict:
    return {
        "model_factor": model,
        "evidence_schema_factor": schema,
        "report_key": "R1",
        "category": "abnormality",
        "prediction_level": 4,
        "reference_level": 3,
        "binary_agreement": True,
        "exact_four_level_agreement": False,
        "has_substantive_evidence": evidence,
        "has_any_verified_segment": exact,
    }


def test_paired_transitions_preserve_classification_invariants() -> None:
    rows = [
        _unit("A", "decision_conditioned", evidence=False, exact=False),
        _unit("A", "independent_category_evidence", evidence=True, exact=True),
    ]
    transitions = paired_schema_transitions(rows)
    evidence = [row for row in transitions if row["field"] == "has_substantive_evidence"]
    assert sum(row["units"] for row in evidence) == 1
    assert next(row for row in evidence if row["transition"] == "0_to_1")["units"] == 1


def test_paired_transitions_reject_changed_decision() -> None:
    rows = [
        _unit("A", "decision_conditioned", evidence=False, exact=False),
        {
            **_unit("A", "independent_category_evidence", evidence=True, exact=True),
            "prediction_level": 3,
        },
    ]
    with pytest.raises(ValueError, match="invariant"):
        paired_schema_transitions(rows)
