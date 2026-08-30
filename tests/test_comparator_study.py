from __future__ import annotations

import json
from pathlib import Path

from eeg_review.comparator_study import validate_comparator_study

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLAN = REPOSITORY_ROOT / "review/model-receipts/medgemma-independent-comparator.preregistered.json"
NATIVE_INTERFACE_PLAN = (
    REPOSITORY_ROOT
    / "review/model-receipts/medgemma-native-interface-sensitivity.preregistered.json"
)
NATIVE_INTERFACE_FREEZE = (
    REPOSITORY_ROOT
    / "review/model-receipts/medgemma-native-interface-development.freeze.json"
)
NATIVE_INTERFACE_RESULT = (
    REPOSITORY_ROOT
    / "review/model-receipts/medgemma-native-interface-development.result.json"
)


def issue_fields(result: dict) -> set[str]:
    return {issue["field"] for issue in result["issues"] if issue["severity"] == "blocker"}


def write_variant(tmp_path: Path, update) -> Path:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    update(payload)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_preregistered_plan_is_structurally_ready_and_external_v5g_is_nonblocking() -> None:
    result = validate_comparator_study(PLAN)

    assert not issue_fields(result)
    assert result["ready_to_start_governed_inference"] is False
    assert result["independent_execution_blocked_by_external_configuration"] is False
    assert result["ready_for_manuscript_claim"] is False
    assert result["population_arithmetic"]["zoe_evaluation_1395"] == {
        "candidate": 1400,
        "complete": 1395,
        "excluded_incomplete": 5,
        "execute": 1395,
    }


def test_external_configuration_cannot_silently_block_independent_run(tmp_path: Path) -> None:
    path = write_variant(
        tmp_path,
        lambda payload: payload["external_configurations"][0].update(
            {"blocks_independent_execution": True}
        ),
    )

    result = validate_comparator_study(path)

    assert "external_configurations[0].blocks_independent_execution" in issue_fields(result)


def test_configuration_identifiers_cannot_collapse_distinct_lineages(tmp_path: Path) -> None:
    def duplicate(payload: dict) -> None:
        payload["external_configurations"][0]["configuration_id"] = payload[
            "independent_configuration"
        ]["configuration_id"]

    result = validate_comparator_study(write_variant(tmp_path, duplicate))

    assert "configurations" in issue_fields(result)


def test_population_arithmetic_and_complete_case_execution_are_frozen(tmp_path: Path) -> None:
    def change(payload: dict) -> None:
        payload["cohorts"][1]["population"]["complete_records"] = 1394

    result = validate_comparator_study(write_variant(tmp_path, change))

    assert "cohorts[1].population" in issue_fields(result)


def test_prompt_or_grammar_drift_blocks_execution(tmp_path: Path) -> None:
    def change(payload: dict) -> None:
        payload["independent_configuration"]["interface"]["prompt_sha256"] = "0" * 64

    result = validate_comparator_study(write_variant(tmp_path, change))

    assert "interface.prompt_sha256" in issue_fields(result)


def test_native_interface_sensitivity_cannot_replace_primary_or_start_evaluation() -> None:
    plan = json.loads(NATIVE_INTERFACE_PLAN.read_text(encoding="utf-8"))
    freeze = json.loads(NATIVE_INTERFACE_FREEZE.read_text(encoding="utf-8"))
    result = json.loads(NATIVE_INTERFACE_RESULT.read_text(encoding="utf-8"))

    assert plan["status"] == (
        "development_completed_configuration_frozen_evaluation_governance_locked"
    )
    assert plan["primary_result_immutability"]["completed_before_this_plan"] is True
    assert plan["primary_result_immutability"]["replacement_allowed"] is False
    assert plan["sensitivity_configuration"]["weights_or_training_change_allowed"] is False
    assert plan["development_stage"]["records"] == 100
    assert plan["development_stage"]["selection_rule"]["candidate_count"] == 1
    assert (
        plan["development_stage"]["selection_rule"]["reference_metric_used_for_selection"] is False
    )
    assert plan["evaluation_stage"]["status"] == "not_authorized"
    assert plan["development_stage"]["result_blind_freeze"]["selected"] is True
    assert (
        plan["development_stage"]["result_blind_freeze"][
            "selected_before_reference_metric_access"
        ]
        is True
    )
    assert freeze["status"] == "immutable_result_blind_configuration_freeze"
    assert freeze["selected_for_freeze"] is True
    assert freeze["blockers"] == []
    assert result["stage"] == "zoe_development_100_after_result_blind_freeze"
    assert result["analysis"]["selection_used_reference_metrics"] is False
    assert any("not run" in boundary for boundary in result["boundaries"])
    assert any(
        "H18-02728" in requirement
        for requirement in plan["evaluation_stage"]["unlock_requirements"]
    )
