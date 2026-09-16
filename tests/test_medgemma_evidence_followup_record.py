from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "review/model-receipts/medgemma-v1-evidence-development.preregistered.json"
RESULT_PATH = ROOT / "review/model-receipts/medgemma-v1-evidence-development.result.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_freezes_development_only_execution_and_implementation():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert plan["status"] == "frozen_before_100_report_evidence_inference"
    assert plan["classification_calls_planned"] == 0
    assert plan["evidence_calls_planned"] == 100
    assert plan["protected_evaluation_allowed"] is False
    assert plan["automatic_full_cohort_expansion_allowed"] is False
    assert plan["prompt_or_threshold_optimization_allowed"] is False
    paths = {
        "executor_sha256": "scripts/run_fixed_classification_explanations.py",
        "analysis_script_sha256": "scripts/analyze_development_evidence_transport.py",
        "comparison_module_sha256": "src/eeg_review/fixed_evidence_comparison.py",
        "grounding_module_sha256": "src/eeg_review/source_grounding.py",
        "protocol_sha256": "review/MEDGEMMA_EVIDENCE_EXTRACTION_FOLLOWUP_PROTOCOL.md",
    }
    for field, relative in paths.items():
        assert plan["implementation"][field] == sha256(ROOT / relative)


def test_public_result_is_complete_bounded_and_contains_no_case_payload():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert result["records"] == result["schema_valid_outputs"] == 100
    assert result["classification_calls"] == 0
    assert result["decision_copy_mismatches"] == 0
    assert result["candidate_or_semantic_matching_in_primary_result"] is False
    assert result["preregistered_plan_sha256"] == sha256(PLAN_PATH)
    medgemma = result["streams"]["medgemma_native_v1_fixed_decision"]
    mistral = result["streams"]["mistral_historical_interface_saved"]
    assert medgemma["verified_exact_segments"] == 330
    assert medgemma["substantive_segments"] == 728
    assert mistral["verified_exact_segments"] == 53
    assert mistral["substantive_segments"] == 295
    serialized = json.dumps(result)
    for prohibited in ["Hashed_ReportURN", "Report", "classifications", "explanations"]:
        assert prohibited not in serialized
