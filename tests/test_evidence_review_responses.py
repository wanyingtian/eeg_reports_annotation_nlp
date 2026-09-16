from __future__ import annotations

from copy import deepcopy

import pytest

from eeg_review.evidence_review_comparison import compare_blinded_reviews
from eeg_review.evidence_review_responses import summarize_review_response


def _payload() -> dict:
    system_fields = {
        "source_presence": "yes",
        "category_relevance": "yes",
        "supports_stated_decision": "partial",
        "contradicts_stated_decision": "no",
        "sufficient_as_explanation": "partial",
        "no_evidence_omission_reasonable": "not applicable",
        "system_notes": "private note that must not enter summary",
    }
    return {
        "reviewer_code": "R1",
        "reviewer_role": "technical reviewer",
        "cases": [
            {
                "case_id": "C001",
                "fields": {
                    "reader_category_judgment": "present",
                    "reader_key_source_passages": "private source text",
                    "reader_source_notes": "private note",
                    "more_useful_evidence": "tie",
                    "comparison_notes": "private comparison",
                },
                "systems": [
                    {"system": "System A", "fields": dict(system_fields)},
                    {"system": "System B", "fields": dict(system_fields)},
                ],
            }
        ],
    }


def test_response_summary_is_blinded_and_drops_free_text() -> None:
    summary = summarize_review_response(_payload(), expected_case_ids=["C001"])
    rendered = str(summary)
    assert summary["cases_reviewed"] == 1
    assert summary["pair_preference_counts"] == {"tie": 1}
    assert summary["system_judgment_counts"]["System A"]["source_presence"] == {
        "yes": 1
    }
    assert "private" not in rendered


def test_response_summary_rejects_incomplete_review() -> None:
    payload = _payload()
    payload["cases"][0]["systems"][0]["fields"]["category_relevance"] = ""
    with pytest.raises(ValueError, match="category_relevance is required"):
        summarize_review_response(payload, expected_case_ids=["C001"])


def test_response_summary_rejects_missing_case() -> None:
    with pytest.raises(ValueError, match="case ids do not match"):
        summarize_review_response(_payload(), expected_case_ids=["C001", "C002"])


def test_two_reader_comparison_reports_raw_agreement_without_private_text() -> None:
    first = _payload()
    second = deepcopy(first)
    second["reviewer_code"] = "R2"
    second["reviewer_role"] = "EEG-qualified reviewer"
    summary, queue = compare_blinded_reviews(
        first,
        second,
        expected_case_ids=["C001"],
    )
    assert summary["cases_compared"] == 1
    assert len(summary["agreement_by_field"]) == 14
    assert all(row["raw_agreement"] == 1.0 for row in summary["agreement_by_field"])
    assert queue == []
    assert "private" not in str(summary)


def test_two_reader_comparison_builds_blinded_disagreement_queue() -> None:
    first = _payload()
    second = deepcopy(first)
    second["reviewer_code"] = "R2"
    second["cases"][0]["fields"]["reader_category_judgment"] = "absent"
    second["cases"][0]["fields"]["more_useful_evidence"] = "System A"
    second["cases"][0]["systems"][1]["fields"]["source_presence"] = "no"
    summary, queue = compare_blinded_reviews(
        first,
        second,
        expected_case_ids=["C001"],
    )
    assert summary["disagreement_items"] == 3
    assert {row["scope"] for row in queue} == {"source", "pair", "System B"}
    assert {row["case_id"] for row in queue} == {"C001"}
    assert "private" not in str(queue)


def test_two_reader_comparison_rejects_duplicate_reviewer_code() -> None:
    with pytest.raises(ValueError, match="reviewer_code values must be different"):
        compare_blinded_reviews(
            _payload(),
            deepcopy(_payload()),
            expected_case_ids=["C001"],
        )
