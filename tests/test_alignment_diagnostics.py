import copy
import json

import pytest

from eeg_review.alignment_diagnostics import (
    diagnose_saved_evidence,
    legacy_rule_trace,
    prompt_consistency,
)
from eeg_review.evidence_extraction import JSON_KEYS
from eeg_review.source_grounding import double_asterisk_source_candidates, inspect_reason


@pytest.mark.parametrize(
    "text,polarity,basis",
    [
        ("", 0, "empty_unclear"),
        ("Unrecognized wording.", -1, "no_match_default_negative"),
        ("Normal background.", -1, "negative_pattern"),
        ("Background slowing.", 1, "positive_pattern"),
        ("This EEG is abnormal. No discharges.", 1, "explicit_abnormal"),
        ("No discharges. Background slowing.", -1, "negative_pattern"),
    ],
)
def test_historical_rule_is_preserved_and_its_basis_is_explicit(text, polarity, basis):
    trace = legacy_rule_trace(text)
    assert trace["historical_polarity"] == polarity
    assert trace["decision_basis"] == basis
    assert trace["semantic_alignment"] == "not_adjudicated"


def test_rule_flags_are_not_category_entailment():
    trace = legacy_rule_trace("No discharges. Background slowing.")
    assert trace["both_polarities_have_rule_matches"]
    # The inherited priority returns negative despite a positive pattern.
    assert trace["historical_polarity"] == -1
    trace = legacy_rule_trace("No epileptiform abnormalities.")
    assert trace["historical_polarity"] == -1
    assert trace["semantic_alignment"] == "not_adjudicated"


def test_marker_candidates_preserve_exact_source_offsets_and_original_status():
    report = "é **Focal** waves occurred. Focal ** \n **waves occurred."
    reason = "Focal waves occurred."
    candidates = double_asterisk_source_candidates(reason, report)
    assert len(candidates) == 2
    for candidate in candidates:
        source = report[candidate["start"]:candidate["end"]]
        assert source == candidate["source_quote"]
        assert " ".join(source.replace("**", "").split()) == reason
    assert inspect_reason(reason, report)["status"] == "unmatched_requires_review"
    assert not inspect_reason(reason, report)["accepted_as_verbatim"]


@pytest.mark.parametrize(
    "reason,report",
    [
        ("No discharges.", "**Discharges**."),
        ("20 Hz.", "**2** Hz."),
        ("no discharges.", "**No** discharges."),
        ("Left waves.", "**Right** waves."),
        ("Focal waves.", "*Focal* waves."),
        ("Focal waves.", "Focal waves."),
        ("", "**Focal** waves."),
        (" \n ", "**Focal** waves."),
    ],
)
def test_marker_recovery_cannot_change_words_case_numbers_or_negation(reason, report):
    assert double_asterisk_source_candidates(reason, report) == []


def test_both_prompt_constraints_are_checked_without_mutation():
    levels = dict.fromkeys(JSON_KEYS, 1)
    levels["abnormality"] = 4
    original = copy.deepcopy(levels)
    assert prompt_consistency(levels) == {
        "subtype_positive_overall_negative": False,
        "all_subtypes_negative_overall_positive": True,
    }
    assert levels == original
    levels["abnormality"] = 1
    levels[JSON_KEYS[0]] = 3
    assert prompt_consistency(levels)["subtype_positive_overall_negative"]


@pytest.mark.parametrize("bad", [True, 0, "1", None])
def test_consistency_requires_real_four_level_integers(bad):
    levels = dict.fromkeys(JSON_KEYS, 1)
    levels[JSON_KEYS[0]] = bad
    with pytest.raises(ValueError):
        prompt_consistency(levels)


def packet_fixture():
    def row(key):
        return {
            "Hashed_ReportURN": key,
            "Report": "**Focal** waves occurred.",
            "categories": {
                category: {
                    "predictions": {"medgemma_native_focal_v2": 1, "medgemma_native_v1": 1},
                    "evidence": {
                        "decision_copy_matches": True,
                        "reasons": [{
                            "original_reason": "Focal waves occurred.",
                            "status": "unmatched_requires_review",
                        }],
                    },
                }
                for category in JSON_KEYS
            },
        }
    return [row("synthetic-a"), row("synthetic-b")]


def test_sample_denominators_privacy_and_original_records_are_preserved():
    packet = packet_fixture()
    before = copy.deepcopy(packet)
    aggregate, details = diagnose_saved_evidence(packet, ["synthetic-a"])
    for sample in aggregate["evidence_samples"].values():
        assert sample["reports"] == 1 and sample["decision_cells"] == 5
        assert sample["double_asterisk_candidate_instances"] == 5
        assert sample["double_asterisk_unique_report_phrases"] == 1
        assert sample["rule_decision_bases"] == {"no_match_default_negative": 5}
    assert packet == before
    assert not aggregate["source_quote_acceptance_changed"]
    assert not aggregate["clinicalbert_alignment_score_reproduced"]
    encoded = json.dumps(aggregate)
    assert "synthetic-" not in encoded and "Focal waves" not in encoded
    assert details[1]["evidence_sample"] == "targeted_additions"


@pytest.mark.parametrize("mutation", ["duplicate", "reorder", "missing_evidence", "model_set"])
def test_population_and_evidence_changes_fail_closed(mutation):
    packet = packet_fixture()
    if mutation == "duplicate":
        packet[1]["Hashed_ReportURN"] = "synthetic-a"
    elif mutation == "reorder":
        packet.reverse()
    elif mutation == "missing_evidence":
        packet[0]["categories"][JSON_KEYS[0]]["evidence"] = None
    else:
        packet[0]["categories"][JSON_KEYS[0]]["predictions"].pop("medgemma_native_v1")
    with pytest.raises(ValueError):
        diagnose_saved_evidence(packet, ["synthetic-a"])


def test_decision_copy_mismatch_is_withheld():
    packet = packet_fixture()
    packet[0]["categories"][JSON_KEYS[0]]["evidence"]["decision_copy_matches"] = False
    aggregate, details = diagnose_saved_evidence(packet, ["synthetic-a"])
    assert aggregate["evidence_samples"]["first_sample"]["withheld_cells"] == 1
    assert details[0]["cells"][JSON_KEYS[0]]["status"].startswith("withheld")
