import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "exporter",
    Path(__file__).resolve().parents[1] / "scripts/export_medgemma_interface_diagnostic.py",
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@pytest.fixture
def fixture(monkeypatch):
    contract = {
        "inputs": {"database": {"path": "/private/data.db", "sha256": "dbhash"}},
        "code": {"script.py": "codehash"},
        "versions": {"runtime": "version"},
    }
    calls = [
        {
            "actual_input_tokens_verified": True,
            "model_sha256": exporter.POLICY["model_sha256"],
            "grammar_sha256": exporter.POLICY["grammar_sha256"],
            "created_at_utc": "2026-08-31T12:00:00Z",
            "elapsed_seconds": 1.0,
            "usage": {"completion_tokens": 20},
            "finish_reason": "stop",
            "report_key": "SECRET_KEY",
            "text": "SECRET_OUTPUT",
            "prompt_text": "SECRET_REPORT",
            "input_token_ids": [111, 222],
            "levels": [1, 1, 1, 1, 1],
            "position": position,
            "arm": arm,
        }
        for position in exporter.POLICY["positions_zero_based"]
        for arm in exporter.POLICY["arms"]
    ]
    summary = {
        "call_receipts": {"case.json": "hash"},
        "completed_calls": 40,
        "invalid_outputs": 0,
        "comparisons": {},
        "saved_parent_replay": {},
        "saved_development_disagreements": {},
    }
    saved_contract = dict(contract)
    records = {
        "contract.json": saved_contract,
        "summary.json": summary,
        "execution.json": {"source_revision": "revision"},
    }
    monkeypatch.setattr(exporter.diagnostic, "intake", lambda _: (None, None, None, contract))
    monkeypatch.setattr(exporter.diagnostic, "read", lambda p: records[p.name])
    monkeypatch.setattr(exporter.diagnostic, "checkpoints", lambda *_: calls)
    monkeypatch.setattr(exporter.diagnostic, "publish", lambda *_: None)
    return SimpleNamespace(output_dir=Path("/private/run")), calls, summary, saved_contract


def test_export_positive_allowlist_does_not_publish_case_material(fixture):
    args, _calls, _summary, _contract = fixture
    report = exporter.build_report(args)
    text = json.dumps(report)
    for private in ("SECRET", "/private", "report_key", "prompt_text", "input_token_ids"):
        assert private not in text
    assert report["actual_inputs_verified"] == 40
    assert report["reports_replayed"] == 8
    assert report["execution"]["summed_inference_seconds"] == 40
    assert report["new_accuracy_estimate"] is False
    assert report["protected_evaluation"] is False
    assert report["comparisons"]["trim_only_vs_native_chat"]["same_five_labels"] == 8


def test_export_retains_invalid_outcomes(fixture):
    args, _calls, summary, _contract = fixture
    summary["invalid_outputs"] = 40
    assert exporter.build_report(args)["invalid_outputs"] == 40


def test_export_rejects_incomplete_diagnostic(fixture):
    args, calls, _summary, _contract = fixture
    calls.pop()
    with pytest.raises(ValueError, match="incomplete"):
        exporter.build_report(args)


def test_export_rejects_contract_drift(fixture):
    args, _calls, _summary, contract = fixture
    contract["extra"] = "changed"
    with pytest.raises(ValueError, match="contract changed"):
        exporter.build_report(args)


@pytest.mark.parametrize(
    "field", ["actual_input_tokens_verified", "model_sha256", "grammar_sha256"]
)
def test_export_rejects_unverified_input_or_changed_artifact(fixture, field):
    args, calls, _summary, _contract = fixture
    calls[0][field] = False
    with pytest.raises(ValueError):
        exporter.build_report(args)
