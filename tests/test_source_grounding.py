from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from eeg_review.evidence_extraction import FALLBACK_EVIDENCE, JSON_KEYS
from eeg_review.source_grounding import (
    aggregate_grounding,
    inspect_grounding,
    inspect_reason,
    whitespace_source_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "v2_audit", ROOT / "scripts/audit_medgemma_v2_evidence.py"
)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def fixed():
    return json.dumps(dict.fromkeys(JSON_KEYS, 1))


def response(reasons=None):
    return json.dumps({key: {"decision": 1, "reasons": reasons or ["normal"]} for key in JSON_KEYS})


def test_exact_offsets_handle_unicode_and_repeated_overlapping_matches():
    report = "é 😃 ababa"
    checked = inspect_reason("aba", report)
    assert checked["accepted_as_verbatim"] is True
    assert checked["source_spans"] == [{"start": 4, "end": 7}, {"start": 6, "end": 9}]
    for span in checked["source_spans"]:
        assert report[span["start"] : span["end"]] == "aba"


@pytest.mark.parametrize(
    "reason,report,status",
    [
        ("NORMAL", "normal", "casefold_only"),
        ("normal background", "normal\n background", "whitespace_only"),
        ("'normal'", "‘normal’", "typography_normalized_only"),
        ("'NORMAL'", "‘normal’", "typography_and_casefold_only"),
        ("No seizures", "Seizures were recorded", "unmatched_requires_review"),
        ("20 Hz", "2 Hz", "unmatched_requires_review"),
        ("no focal activity", "focal activity", "unmatched_requires_review"),
        ("abnormal", "normal", "unmatched_requires_review"),
        (" ", " ", "blank"),
        (None, "normal", "invalid_type"),
        (FALLBACK_EVIDENCE, FALLBACK_EVIDENCE, "declared_no_evidence"),
    ],
)
def test_only_literal_nonblank_source_can_become_verified_quote(reason, report, status):
    result = inspect_reason(reason, report)
    assert result["status"] == status
    assert result["accepted_as_verbatim"] is False
    assert result["source_spans"] == []


def test_mixed_record_keeps_original_and_abstains_without_changing_decisions():
    raw = response(["normal", "NORMAL", "unsupported statement"])
    result = inspect_grounding(raw, report="normal background", fixed=fixed())
    assert result["raw_response"] == raw
    assert result["fixed_classifications"] == json.loads(fixed())
    for cell in result["cells"].values():
        assert cell["status"] == "partial_verified_quotes"
        assert len(cell["reasons"]) == 3
        assert cell["verified_quotes"] == [
            {"text": "normal", "source_spans": [{"start": 0, "end": 6}]}
        ]


def test_decision_mismatch_withholds_even_literal_quotes():
    raw = json.loads(response())
    raw[JSON_KEYS[0]]["decision"] = 4
    result = inspect_grounding(json.dumps(raw), report="normal", fixed=fixed())
    cell = result["cells"][JSON_KEYS[0]]
    assert cell["decision"] == 1 and cell["generated_decision"] == 4
    assert cell["status"] == "abstain_decision_mismatch"
    assert cell["verified_quotes"] == []
    assert cell["reasons"][0]["status"] == "exact"


def test_whitespace_candidates_are_exact_source_not_relabelled_generated_quotes():
    report = " \té normal\n\tbackground. Another normal  background. "
    reason = "normal background"
    checked = inspect_reason(reason, report)
    assert not checked["accepted_as_verbatim"] and checked["source_spans"] == []
    assert len(checked["source_span_candidates"]) == 2
    for candidate in checked["source_span_candidates"]:
        assert report[candidate["start"] : candidate["end"]] == candidate["source_quote"]
        assert " ".join(candidate["source_quote"].split()) == reason
        assert candidate["source_quote"] != reason


@pytest.mark.parametrize(
    "reason,report",
    [
        ("no seizures", "seizures"),
        ("20 Hz", "2 Hz"),
        ("normal", "NORMAL"),
        ("", "normal"),
        (" \n ", "normal"),
    ],
)
def test_whitespace_recovery_never_changes_words_case_negation_or_numbers(reason, report):
    assert whitespace_source_candidates(reason, report) == []


@pytest.mark.parametrize("mutation", ["missing", "extra", "boolean", "duplicate", "invalid_json"])
def test_invalid_schema_retains_five_cells_with_no_verified_view(mutation):
    value = json.loads(response())
    if mutation == "missing":
        del value[JSON_KEYS[0]]
    elif mutation == "extra":
        value["unexpected"] = 1
    elif mutation == "boolean":
        value[JSON_KEYS[0]]["decision"] = True
    raw = json.dumps(value)
    if mutation == "duplicate":
        raw = raw.replace('"decision": 1', '"decision": 1, "decision": 1', 1)
    elif mutation == "invalid_json":
        raw = "{"
    result = inspect_grounding(raw, report="normal", fixed=fixed())
    assert not result["schema_valid"]
    assert len(result["cells"]) == 5
    assert all(not cell["verified_quotes"] for cell in result["cells"].values())
    assert result["raw_response"] == raw


def test_aggregate_keeps_invalid_and_blank_denominators_without_exporting_text():
    records = [
        inspect_grounding(response(["normal", " "]), report="normal", fixed=fixed()),
        inspect_grounding(response([FALLBACK_EVIDENCE]), report="normal", fixed=fixed()),
        inspect_grounding("{", report="normal", fixed=fixed()),
    ]
    result = aggregate_grounding(records)
    assert result["records"] == 3 and result["decision_cells"] == 15
    assert result["invalid_schema_records"] == 1
    assert result["cells_with_verified_quotes"] == 5 and result["cells_abstaining"] == 10
    assert result["reason_statuses"] == {"exact": 5, "blank": 5, "declared_no_evidence": 5}
    assert result["records_with_verified_quotes_for_all_five_categories"] == 1
    assert "raw_response" not in json.dumps(result) and "original_reason" not in json.dumps(result)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_fixture(root):
    for folder in ["inputs", "products", "analysis"]:
        (root / folder).mkdir(parents=True)
    keys = [f"synthetic-case-{i}" for i in range(100)]
    write_json(root / "state.json", {"status": "completed"})
    write_json(
        root / "job.json", {"study_id": audit_module.STUDY_ID, "repository_revision": "fixture"}
    )
    frame = pd.DataFrame(
        {audit_module.KEY: keys, "Report": ["normal"] * 100, "classifications": [fixed()] * 100}
    )
    with sqlite3.connect(root / "inputs/development.db") as connection:
        frame.drop(columns="classifications").to_sql("reports", connection, index=False)
    frame[[audit_module.KEY]].to_csv(root / "inputs/development.manifest.csv", index=False)
    frame[[audit_module.KEY]].iloc[:20].to_csv(root / "inputs/evidence.manifest.csv", index=False)
    frame[[audit_module.KEY, "classifications"]].to_csv(root / "products/v2.csv", index=False)
    frame[[audit_module.KEY, "classifications"]].to_csv(root / "inputs/v1.csv", index=False)
    evidence = pd.DataFrame(
        {
            audit_module.KEY: keys[:20],
            "fixed_classifications": [fixed()] * 20,
            "explanations": [response()] * 20,
        }
    )
    evidence.to_csv(root / "products/evidence-v2.csv", index=False)
    write_json(root / "products/evidence-v2.run.json", {})
    write_json(
        root / "analysis/author-summary.json",
        {
            "evidence_quality": {
                "records": 20,
                "exact_traceable_phrases": 100,
                "evidence_phrases": 100,
                "fallback_phrases": 0,
                "decision_copy_mismatches": 0,
            }
        },
    )
    paths = [
        root / "job.json",
        *sorted((root / "inputs").iterdir()),
        *sorted((root / "products").iterdir()),
        root / "analysis/author-summary.json",
    ]
    write_json(
        root / "final-scientific-manifest.json",
        {
            "files": [
                {"path": str(p.relative_to(root)), "sha256": audit_module.sha(p)} for p in paths
            ]
        },
    )
    return root


def test_complete_audit_uses_no_inference_and_preserves_source(tmp_path):
    root = source_fixture(tmp_path)
    before = audit_module.validate_source(root)
    result, records = audit_module.audit(root)
    assert result["aggregate"]["cells_with_verified_quotes"] == 100
    assert not result["inference_performed"] and not result["classifications_changed"]
    assert len(records) == 20
    assert "synthetic-case-" not in json.dumps(result)
    assert before == audit_module.validate_source(root)


@pytest.mark.parametrize("mutation", ["changed", "omitted", "duplicate", "traversal", "eclipsed"])
def test_source_receipt_drift_or_eclipse_is_rejected(tmp_path, mutation):
    root = source_fixture(tmp_path)
    path = root / "final-scientific-manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "changed":
        (root / "products/v2.csv").write_text("changed")
    elif mutation == "omitted":
        manifest["files"] = [x for x in manifest["files"] if x["path"] != "products/v2.csv"]
    elif mutation == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif mutation == "traversal":
        manifest["files"][0]["path"] = "../not-in-run"
    else:
        write_json(root / "ECLIPSED.json", {"reason": "synthetic governance exclusion"})
    write_json(path, manifest)
    with pytest.raises((ValueError, RuntimeError)):
        audit_module.validate_source(root)


@pytest.mark.parametrize("mutation", [None, "model", "offline", "count", "predictions"])
def test_targeted_receipt_is_separate_and_enforces_same_local_configuration(tmp_path, mutation):
    root = source_fixture(tmp_path)
    plan = audit_module.read(
        ROOT / "review/model-receipts/medgemma-native-focal-v2.development-plan.json"
    )
    write_json(root / "inputs/plan.json", plan)
    output = root / "targeted.csv"
    pd.DataFrame(
        [
            {
                audit_module.KEY: "synthetic-case-0",
                "fixed_classifications": fixed(),
                "explanations": response(),
            }
        ]
    ).to_csv(output, index=False)
    receipt = {
        "output": {"sha256": audit_module.sha(output)},
        "inputs": {
            "dataset_sha256": audit_module.sha(root / "inputs/development.db"),
            "fixed_predictions_sha256": audit_module.sha(root / "products/v2.csv"),
            "records": 1,
        },
        "interface": "native_chat",
        "classification_source_held_fixed": True,
        "model": {
            "sha256": plan["model"]["sha256"],
            "load_parameters": plan["runtime"]["parameters"],
            "artifact_access": {"mode": "local_cache_only"},
        },
        "prompt": {"sha256": plan["evidence"]["prompt_sha256"]},
        "grammar": {"sha256": plan["evidence"]["grammar_sha256"], "applied": True},
        "chat_template": {"sha256": plan["interface"]["chat_template_sha256"]},
        "sampling": {"temperature": 0.0, "top_k": 40, "top_p": 0.95, "max_tokens": 3000},
        "environment": {
            "hf_hub_offline": True,
            "hf_hub_telemetry_disabled": True,
            "git": {"revision": "fixture"},
        },
    }
    if mutation == "model":
        receipt["model"]["sha256"] = "0" * 64
    elif mutation == "offline":
        receipt["environment"]["hf_hub_offline"] = False
    elif mutation == "count":
        receipt["inputs"]["records"] = 2
    elif mutation == "predictions":
        receipt["inputs"]["fixed_predictions_sha256"] = "0" * 64
    write_json(output.with_suffix(".run.json"), receipt)
    if mutation:
        with pytest.raises(ValueError):
            audit_module.targeted_completion(root, output, ["synthetic-case-0"])
    else:
        records, result = audit_module.targeted_completion(root, output, ["synthetic-case-0"])
        assert len(records) == 1 and result["records"] == 1
        assert result["classification_calls"] == 0
        assert "not_pooled" in result["status"]
