from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "review/model-receipts/mistral-native-interface-sensitivity.preregistered.json"
)


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_mistral_interface_plan_changes_serialization_only() -> None:
    plan = _plan()
    interface = plan["classification_interface"]

    assert plan["model"]["weights_or_training_change_allowed"] is False
    assert interface["candidate_count"] == 1
    assert interface["grammar_applied"] is True
    assert interface["grammar_sha256"] == (
        "5237e13988062538cda9c21906f1f4e1fc8b99498e2462ea69fe24bface35016"
    )
    assert "instead of raw-completion serialization" in interface["only_planned_change"]


def test_mistral_plan_freezes_development_and_preserves_bad_results() -> None:
    plan = _plan()
    surface = plan["development_surface"]

    assert surface["records"] == 100
    assert surface["selection_rule"]["reference_metric_used_for_selection"] is False
    assert plan["historical_evidence_immutability"]["unfavorable_results_must_be_retained"]
    assert plan["evaluation_stage"][
        "protected_outcomes_may_not_be_used_for_interface_selection"
    ]


def test_explanation_followup_keeps_fixed_classification_and_grammar() -> None:
    followup = _plan()["explanation_interface_followup"]

    assert followup["classification_json_must_be_held_fixed_between_interfaces"]
    assert followup["grammar_applied_to_raw_and_native"]
    assert followup["causal_faithfulness_claim_allowed"] is False
    assert "decision_copy_fidelity" in followup["metrics"]
    assert "exact_substring_traceability_for_nonfallback_evidence" in followup["metrics"]


def test_governed_execution_remains_local_and_deferred() -> None:
    plan = _plan()
    policy = plan["execution_policy"]

    assert policy["run_after_active_medgemma_protected_job"]
    assert policy["local_inference_only"]
    assert policy["network_inference_allowed"] is False
    assert plan["distribution_status"]["keyed_outputs"] == "outputs_governed"
