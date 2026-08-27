from __future__ import annotations

import json
from pathlib import Path

from eeg_review.adaptation_plan import validate_adaptation_plan
from eeg_review.manifest import sha256_file

PLAN = (
    Path(__file__).parents[1] / "review/model-receipts/mistral-task-adaptation.preregistered.json"
)


def write_plan(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "adaptation-plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_preregistered_plan_is_valid_but_cannot_start_evaluation() -> None:
    result = validate_adaptation_plan(PLAN)

    assert result["design_valid"] is True
    assert result["ready_for_implementation"] is True
    assert result["ready_for_evaluation"] is False
    assert result["analysis_started"] is False
    assert result["issues"] == []
    assert result["signal_use_counts"] == {
        "context_only_prohibited_for_selection": 2,
        "design_prior": 2,
        "development": 1,
        "evaluation_only": 2,
    }
    rendered = json.dumps(result)
    assert "accuracy" not in rendered
    assert "patient" not in rendered.lower() or "patient key" in rendered.lower()


def test_frozen_plan_requires_and_verifies_adapter_and_freeze_receipt(tmp_path: Path) -> None:
    payload = load_plan()
    adapter = tmp_path / "adapter.json"
    receipt = tmp_path / "freeze-receipt.json"
    adapter.write_text('{"adapter":"frozen"}', encoding="utf-8")
    receipt.write_text('{"status":"frozen_before_evaluation"}', encoding="utf-8")
    payload["status"] = "frozen_before_evaluation"
    payload["task_adapter"]["artifact"] = {
        "path": adapter.name,
        "sha256": sha256_file(adapter),
    }
    payload["freeze"]["author_group_admitted"] = True
    payload["freeze"]["frozen_before_evaluation"] = True
    payload["freeze"]["receipt"] = {
        "path": receipt.name,
        "sha256": sha256_file(receipt),
    }
    plan = write_plan(tmp_path, payload)

    result = validate_adaptation_plan(plan, bundle_root=tmp_path, check_files=True)

    assert result["design_valid"] is True
    assert result["ready_for_implementation"] is False
    assert result["ready_for_evaluation"] is True
    assert result["artifact_validation"]["task_adapter.artifact"]["matches"] is True
    assert result["artifact_validation"]["freeze.receipt"]["matches"] is True


def test_evaluation_outcome_reuse_blocks_the_confirmatory_route(tmp_path: Path) -> None:
    payload = load_plan()
    zoe = next(signal for signal in payload["signals"] if signal["signal_id"] == "zoe_evaluation")
    zoe["used_for_parameter_or_variant_selection"] = True
    zoe["outcomes_inspected"] = True
    plan = write_plan(tmp_path, payload)

    result = validate_adaptation_plan(plan)

    assert result["design_valid"] is False
    fields = {issue["field"] for issue in result["issues"]}
    assert "signals[zoe_evaluation].outcomes_inspected" in fields
    assert any(
        "evaluation_only signals cannot select" in issue["message"] for issue in result["issues"]
    )


def test_medgemma_teacher_use_requires_a_separately_named_plan(tmp_path: Path) -> None:
    payload = load_plan()
    payload["task_adapter"]["teacher_model_outputs_used"] = True
    payload["task_adapter"]["methods"].append("teacher_student_distillation")
    predictions = next(
        signal for signal in payload["signals"] if signal["signal_id"] == "medgemma_v5g_predictions"
    )
    predictions["use"] = "development"
    predictions["used_for_parameter_or_variant_selection"] = True
    plan = write_plan(tmp_path, payload)

    result = validate_adaptation_plan(plan)

    assert result["design_valid"] is False
    assert any(
        "separately named distillation plan" in issue["message"] for issue in result["issues"]
    )
    assert any(
        "methods require a separately named experiment" in issue["message"]
        for issue in result["issues"]
    )
    assert any(
        issue["field"] == "signals[medgemma_v5g_predictions].use" for issue in result["issues"]
    )


def test_lora_cannot_be_added_silently_to_threshold_only_route(tmp_path: Path) -> None:
    payload = load_plan()
    payload["task_adapter"]["methods"].append("lora")
    plan = write_plan(tmp_path, payload)

    result = validate_adaptation_plan(plan)

    assert result["design_valid"] is False
    assert any("lora" in issue["message"] for issue in result["issues"])


def test_frozen_status_without_artifacts_or_author_admission_is_blocked(tmp_path: Path) -> None:
    payload = load_plan()
    payload["status"] = "frozen_before_evaluation"
    plan = write_plan(tmp_path, payload)

    result = validate_adaptation_plan(plan, check_files=True)

    assert result["ready_for_evaluation"] is False
    fields = {issue["field"] for issue in result["issues"]}
    assert "freeze.author_group_admitted" in fields
    assert "freeze.frozen_before_evaluation" in fields
    assert "task_adapter.artifact.path" in fields
    assert "freeze.receipt.path" in fields


def test_unrecognized_result_or_method_fields_cannot_enter_the_plan(tmp_path: Path) -> None:
    payload = load_plan()
    payload["observed_results"] = {"accuracy": 1.0}
    payload["task_adapter"]["methods"].append("secret_winning_method")
    plan = write_plan(tmp_path, payload)

    result = validate_adaptation_plan(plan)

    assert result["design_valid"] is False
    fields = {issue["field"] for issue in result["issues"]}
    assert "observed_results" in fields
    assert "task_adapter.methods" in fields
