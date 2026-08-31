import json

import pandas as pd
import pytest

from eeg_review.evidence_extraction import JSON_KEYS
from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL
from eeg_review.prompt_diagnostics import KEY, build_diagnostic_packet, targeted_missing_evidence
from eeg_review.source_grounding import inspect_grounding


def fixtures():
    keys = ["synthetic-a", "synthetic-b"]
    reference = pd.DataFrame(
        {
            KEY: keys,
            "Report": ["normal", "abnormal"],
            **{label: [1, 4] for label in JSON_KEY_TO_LABEL.values()},
        }
    )

    def frame(values):
        return pd.DataFrame(
            {
                KEY: keys,
                "classifications": [
                    json.dumps(dict.fromkeys(JSON_KEYS, level)) for level in values
                ],
            }
        )

    versions = {
        "medgemma_native_v1": frame([1, 1]),
        "medgemma_native_focal_v2": frame([4, 4]),
        "mistral_historical_interface": frame([1, 4]),
    }
    evidence = [
        {
            KEY: keys[0],
            **inspect_grounding(
                json.dumps({key: {"decision": 4, "reasons": ["normal"]} for key in JSON_KEYS}),
                report="normal",
                fixed=versions["medgemma_native_focal_v2"].iloc[0]["classifications"],
            ),
        }
    ]
    return reference, versions, evidence


def test_packet_preserves_repairs_regressions_and_cross_model_disagreement():
    refs, versions, evidence = fixtures()
    summary, packet = build_diagnostic_packet(refs, versions, evidence)
    for counts in summary["by_category"].values():
        assert counts["reports"] == 2
        assert counts["v1_to_v2_repairs"] == 1 and counts["v1_to_v2_regressions"] == 1
        assert counts["cross_model_core_disagreements"] == 1
        assert counts["v2_errors_with_evidence"] == 1
    assert len(packet) == 2
    cell = packet[0]["categories"][JSON_KEYS[0]]
    assert cell["evidence"]["verified_quotes"][0]["text"] == "normal"
    assert cell["semantic_alignment"] == "not_adjudicated"
    assert cell["review_answers"] is None and cell["review_questions"]
    assert "synthetic-a" not in json.dumps(summary)


def test_missing_error_evidence_is_not_imputed():
    refs, versions, _ = fixtures()
    summary, packet = build_diagnostic_packet(refs, versions, [])
    for counts in summary["by_category"].values():
        assert counts["v2_errors_without_evidence"] == 1
    cell = packet[0]["categories"][JSON_KEYS[0]]
    assert cell["evidence"] is None and not cell["evidence_available"]
    assert any("No explanation" in question for question in cell["review_questions"])


def test_targeted_selector_reuses_existing_and_never_selects_repairs_only():
    refs, versions, evidence = fixtures()
    _, packet = build_diagnostic_packet(refs, versions, evidence)
    missing, plan = targeted_missing_evidence(packet)
    assert missing == [] and plan["selected_reports"] == 1
    assert plan["already_with_evidence"] == 1
    _, packet = build_diagnostic_packet(refs, versions, [])
    missing, plan = targeted_missing_evidence(packet)
    assert missing == ["synthetic-a"] and plan["new_classification_calls"] == 0
    assert plan["error_enriched_posthoc"] is True


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "extra_evidence", "invalid_reference"]
)
def test_bad_population_or_reference_rejected(mutation):
    refs, versions, evidence = fixtures()
    if mutation == "duplicate":
        refs.loc[1, KEY] = refs.loc[0, KEY]
    elif mutation == "missing":
        versions["mistral_historical_interface"] = versions["mistral_historical_interface"].iloc[:1]
    elif mutation == "extra_evidence":
        evidence[0][KEY] = "outside-cohort"
    else:
        refs.loc[0, "Focal Epi"] = 0
    with pytest.raises(ValueError):
        build_diagnostic_packet(refs, versions, evidence)
