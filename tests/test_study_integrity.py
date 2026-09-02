from __future__ import annotations

import json

import pandas as pd
import pytest

from eeg_review.study_integrity import Partition, audit_partitions, decision_lenses


def frame(*rows: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Hashed_ReportURN", "Report"])


def test_partition_audit_is_public_safe_and_disjoint() -> None:
    result = audit_partitions(
        {
            "development": Partition("development", frame(("private-a", "Report A"))),
            "evaluation": Partition(
                "held_out_evaluation", frame(("private-b", "Report B"), ("private-c", "Report C"))
            ),
        }
    )
    assert result["partition_separation_passed"] is True
    assert result["pairwise_overlap"]["development::evaluation"] == {
        "report_key_overlap": 0,
        "normalized_report_text_overlap": 0,
    }
    assert result["contains_report_keys_or_text"] is False
    assert "private-a" not in json.dumps(result)
    assert "Report A" not in json.dumps(result)


@pytest.mark.parametrize(
    "development,evaluation",
    [
        (frame(("same", "Report A")), frame(("same", "Report B"))),
        (frame(("a", "Report  A")), frame(("b", "  report a  "))),
    ],
)
def test_partition_audit_rejects_key_or_normalized_text_leakage(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="study partitions overlap"):
        audit_partitions(
            {
                "development": Partition("development", development),
                "evaluation": Partition("held_out_evaluation", evaluation),
            }
        )


def test_partition_audit_rejects_duplicate_or_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="blank or duplicate"):
        audit_partitions(
            {
                "development": Partition(
                    "development", frame(("same", "A"), ("same", "B"))
                ),
                "evaluation": Partition("held_out_evaluation", frame(("other", "C"))),
            }
        )
    with pytest.raises(ValueError, match="exactly one development"):
        audit_partitions(
            {
                "one": Partition("held_out_evaluation", frame(("a", "A"))),
                "two": Partition("held_out_evaluation", frame(("b", "B"))),
            }
        )


def test_four_level_decision_lenses_preserve_semantics_without_calibration_claim() -> None:
    assert decision_lenses(1) == {
        "four_level_decision": 1,
        "core_call": "absent",
        "declared_confidence": "confident",
        "probability_calibration_claimed": False,
    }
    assert decision_lenses("3")["declared_confidence"] == "low_confidence"
    assert decision_lenses(4.0)["core_call"] == "present"
    assert decision_lenses(4)["core_call"] == "present"
    for invalid in (True, 0, 5, 2.5, "low"):
        with pytest.raises(ValueError):
            decision_lenses(invalid)
