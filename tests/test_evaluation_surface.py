from __future__ import annotations

import json
from pathlib import Path

from eeg_review.evaluation_surface import validate_evaluation_surface_registry

REGISTRY = Path(__file__).parents[1] / "review/model-receipts/jbhi-evaluation-surface-registry.json"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def write_registry(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "evaluation-surfaces.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def surface(payload: dict, surface_id: str) -> dict:
    return next(item for item in payload["surfaces"] if item["surface_id"] == surface_id)


def contrast(payload: dict, contrast_id: str) -> dict:
    return next(item for item in payload["contrasts"] if item["contrast_id"] == contrast_id)


def test_public_registry_is_valid_and_contains_no_result_values() -> None:
    result = validate_evaluation_surface_registry(REGISTRY)

    assert result["design_valid"] is True
    assert result["analysis_started"] is False
    assert result["factor_count"] == 11
    assert result["surface_count"] == 18
    assert result["contrast_count"] == 9
    assert result["design_family_count"] == 2
    assert result["issues"] == []
    payload = load_registry()
    assert all(item["result_values_in_registry"] is False for item in payload["surfaces"])
    serialized = json.dumps(payload)
    assert '"observed_results"' not in serialized
    assert '"metric_values"' not in serialized


def test_external_interface_policy_ablation_is_planned_and_result_blind() -> None:
    payload = load_registry()
    result = validate_evaluation_surface_registry(REGISTRY)

    assert result["design_valid"] is True
    baseline = surface(payload, "external-baseline-evaluation-1894-planned")
    candidate = surface(payload, "external-candidate-evaluation-1894-planned")
    comparison = contrast(payload, "external-candidate-policy-ablation-1894")
    assert baseline["status"] == candidate["status"] == "planned"
    assert baseline["result_values_in_registry"] is False
    assert candidate["result_values_in_registry"] is False
    assert comparison["status"] == "planned_not_run"
    assert comparison["declared_changed_factors"] == ["decision_policy"]
    differing = {
        key
        for key in baseline["factors"]
        if baseline["factors"][key] != candidate["factors"][key]
    }
    assert differing == {"decision_policy"}


def test_external_interface_policy_ablation_refuses_hidden_interface_change(tmp_path: Path) -> None:
    payload = load_registry()
    candidate = surface(payload, "external-candidate-evaluation-1894-planned")
    candidate["factors"]["interface_mode"] = "historical_raw_completion"
    path = write_registry(tmp_path, payload)

    result = validate_evaluation_surface_registry(path)

    assert result["design_valid"] is False
    assert any(
        "interface_mode" in issue["message"]
        for issue in result["issues"]
        if issue["field"].endswith("declared_changed_factors")
    )


def test_interface_ablation_cannot_hide_a_second_changed_factor(tmp_path: Path) -> None:
    payload = load_registry()
    native = surface(payload, "medgemma-q2-native-development-100")
    native["factors"]["grammar_mode"] = "external-consistency-grammar"
    path = write_registry(tmp_path, payload)

    result = validate_evaluation_surface_registry(path)

    assert result["design_valid"] is False
    issue = next(
        item for item in result["issues"] if item["field"].endswith("declared_changed_factors")
    )
    assert "grammar_mode" in issue["message"]
    assert any("controlled ablation" in item["message"] for item in result["issues"])


def test_model_native_comparison_rejects_mismatched_interface(tmp_path: Path) -> None:
    payload = load_registry()
    planned = surface(payload, "mistral-native-evaluation-1894-planned")
    planned["factors"]["interface_mode"] = "historical_raw_completion"
    item = contrast(payload, "native-mistral-versus-native-medgemma-symmetric-task-comparison")
    item["declared_changed_factors"].append("interface_mode")
    path = write_registry(tmp_path, payload)

    result = validate_evaluation_surface_registry(path)

    assert result["design_valid"] is False
    assert any(
        "both surfaces must use model_native_chat" in issue["message"] for issue in result["issues"]
    )


def test_held_out_surface_cannot_silently_include_development_cases(tmp_path: Path) -> None:
    payload = load_registry()
    item = surface(payload, "medgemma-q2-native-evaluation-1894")
    item["population"]["includes_development"] = True
    path = write_registry(tmp_path, payload)

    result = validate_evaluation_surface_registry(path)

    assert result["design_valid"] is False
    assert any("held-out evaluation" in issue["message"] for issue in result["issues"])


def test_population_arithmetic_must_reconcile(tmp_path: Path) -> None:
    payload = load_registry()
    item = surface(payload, "external-v5g-q2-labeled-summary")
    item["population"]["records"] = 2493
    path = write_registry(tmp_path, payload)

    result = validate_evaluation_surface_registry(path)

    assert result["design_valid"] is False
    assert any("population arithmetic" in issue["message"] for issue in result["issues"])


def test_unlabeled_surface_cannot_claim_reference_performance(tmp_path: Path) -> None:
    payload = load_registry()
    item = surface(payload, "external-v5g-q4-large-unlabeled-summary")
    item["metric_ids"].append("core_binary_f1")
    path = write_registry(tmp_path, payload)

    result = validate_evaluation_surface_registry(path)

    assert result["design_valid"] is False
    assert any("not valid without a reference" in issue["message"] for issue in result["issues"])


def test_external_summary_cannot_silently_become_an_admitted_result(tmp_path: Path) -> None:
    payload = load_registry()
    item = surface(payload, "external-v5g-q4-labeled-summary")
    item["status"] = "completed_governed"
    item["factors"]["interface_mode"] = None
    item["result_values_in_registry"] = True
    path = write_registry(tmp_path, payload)

    result = validate_evaluation_surface_registry(path)

    assert result["design_valid"] is False
    fields = {issue["field"] for issue in result["issues"]}
    assert any(field.endswith("factors.interface_mode") for field in fields)
    assert any(field.endswith("result_values_in_registry") for field in fields)


def test_ambiguous_certainty_f1_name_is_rejected(tmp_path: Path) -> None:
    payload = load_registry()
    payload["metric_definitions"].append(
        {
            "metric_id": "certainty_f1",
            "level": "label",
            "statistic": "unspecified",
            "reference_requirement": "required",
            "unlabeled_allowed": False,
            "definition_status": "pending_producing_bundle",
            "interpretation": "ambiguous test metric",
        }
    )
    path = write_registry(tmp_path, payload)

    result = validate_evaluation_surface_registry(path)

    assert result["design_valid"] is False
    assert any("ambiguous shorthand" in issue["message"] for issue in result["issues"])
