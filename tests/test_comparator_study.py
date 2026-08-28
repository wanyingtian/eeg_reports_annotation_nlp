from __future__ import annotations

import json
from pathlib import Path

from eeg_review.comparator_study import validate_comparator_study

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    REPOSITORY_ROOT
    / "review/model-receipts/medgemma-independent-comparator.preregistered.json"
)


def issue_fields(result: dict) -> set[str]:
    return {
        issue["field"]
        for issue in result["issues"]
        if issue["severity"] == "blocker"
    }


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
