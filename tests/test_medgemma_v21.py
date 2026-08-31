from __future__ import annotations

import ast
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

from eeg_review import category_evidence
from eeg_review.evidence_extraction import JSON_KEYS
from eeg_review.native_interface import sha256_text
from eeg_review.prompt_versions import (
    CATEGORY_SCOPE,
    MEDGEMMA_FOCAL_V2,
    MEDGEMMA_SCOPE_V21,
    classification_prompt,
    scope_development_verdict,
)

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def base_prompt():
    tree = ast.parse((ROOT / "src/LLM_pipeline/pipeline.py").read_text())
    return next(
        ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PROMPT_CLASSIFY" for t in n.targets)
    )


@pytest.fixture
def queue():
    value = module("v21queue", "medgemma_prompt_v2.py")
    value.configure("v21")
    return value


def audit_output(phrase="Synthetic source text."):
    return json.dumps({key: {
        "present_evidence": [phrase], "absent_evidence": [], "qualification_evidence": []
    } for key in JSON_KEYS})


def test_only_named_scope_addition_changes_classification_and_freeze_matches(queue):
    base = base_prompt()
    old = classification_prompt(base, MEDGEMMA_FOCAL_V2)
    new = classification_prompt(base, MEDGEMMA_SCOPE_V21)
    assert new.replace("\n" + CATEGORY_SCOPE, "", 1) == old
    plan = queue.read(queue.PLAN)
    queue.validate_plan(plan)
    assert queue.validate_code(plan) == new
    assert sha256_text(new) == plan["interface"]["prompt_sha256"]


@pytest.mark.parametrize("mutation", ["labels", "feedback", "parent", "scope", "schema", "score"])
def test_upgrade_contract_rejects_scope_or_conditioning_drift(queue, mutation):
    plan = queue.read(queue.PLAN)
    if mutation == "labels":
        plan["evidence"]["classifications_supplied_to_model"] = True
    elif mutation == "feedback":
        plan["evidence"]["used_as_classification_input"] = True
    elif mutation == "parent":
        plan["analysis"]["all_parents"] = ["v2"]
    elif mutation == "scope":
        plan["classification_records"] = 1395
    elif mutation == "schema":
        plan["evidence"]["fields"] = ["decision", "reasons"]
    else:
        plan["evidence"]["paired_quality_superiority_claim_allowed"] = True
    with pytest.raises(ValueError):
        queue.validate_plan(plan)


def test_audit_layout_has_no_prediction_parameter_or_confidence_instructions():
    prompt = category_evidence.audit_prompt(base_prompt())
    message = category_evidence.messages(prompt, "Synthetic source text.")
    assert message == [{"role": "user", "content": category_evidence.task_message(
        prompt, "Synthetic source text."
    )}]
    assert "Confident no" not in prompt and "Classification JSON:" not in prompt
    with pytest.raises(TypeError):
        category_evidence.messages(prompt, "source", "must not enter the message")


@pytest.mark.parametrize("mutation", ["decision", "missing", "duplicate", "too_many", "blank"])
def test_audit_schema_fail_closed_without_imputing_labels(mutation):
    value = json.loads(audit_output())
    cell = value[JSON_KEYS[0]]
    if mutation == "decision":
        cell["decision"] = 4
    elif mutation == "missing":
        del cell["absent_evidence"]
    elif mutation == "too_many":
        cell["present_evidence"] *= 3
    elif mutation == "blank":
        cell["present_evidence"] = [" "]
    raw = json.dumps(value)
    if mutation == "duplicate":
        raw = raw.replace(
            '"absent_evidence": []', '"absent_evidence": [], "absent_evidence": []', 1
        )
    checked = category_evidence.inspect(raw, report="Synthetic source text.")
    assert not checked.structured_output_valid
    assert checked.error


def test_empty_lists_are_preserved_and_literal_quality_is_not_semantic_truth():
    value = {k: {f: [] for f in category_evidence.FIELDS} for k in JSON_KEYS}
    checked = category_evidence.inspect(json.dumps(value), report="Synthetic source text.")
    assert checked.structured_output_valid and checked.evidence_phrases == 0
    checked = category_evidence.inspect(audit_output(), report="Synthetic source\ntext.")
    assert checked.structured_output_valid and checked.exact_traceable_phrases == 0


def points():
    from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL
    return {label: {"tp": 9, "fn": 1, "fp": 2, "tn": 88} for label in JSON_KEY_TO_LABEL.values()}


@pytest.mark.parametrize("mutation,met", [
    ("repair", True), ("unchanged", False), ("other_error", False), ("rare_fn", False)
])
def test_rule_cannot_select_by_total_gain_alone(mutation, met):
    v1, v2, candidate = points(), points(), points()
    v2["Gen Non-epi"].update(tp=8, fn=2)
    if mutation == "unchanged":
        candidate["Gen Non-epi"].update(tp=8, fn=2)
    elif mutation == "other_error":
        candidate["Abnormality"].update(fp=3, tn=87)
    elif mutation == "rare_fn":
        candidate["Focal Epi"].update(tp=8, fn=2, fp=1, tn=89)
    result = scope_development_verdict(v1, v2, candidate)
    assert (result["status"] == "development_rule_met") is met
    assert not result["protected_evaluation_authorized"]


def test_upgraded_queue_uses_original_pipeline_and_distinct_outputs(queue, tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "products").mkdir()
    queue.atomic_json(tmp_path / "inputs/plan.json", queue.read(queue.PLAN))
    queue.atomic_json(tmp_path / "job.json", {"python_executable": sys.executable})
    command = queue.classification_command(tmp_path)
    assert command[command.index("--classification-prompt-version") + 1] == MEDGEMMA_SCOPE_V21
    assert command[command.index("--output-csv") + 1].endswith("products/v21.csv")
    assert "--resume-output" in command and "--local-model-only" in command
    assert all("--variant" in stage.command for stage in queue.stages(tmp_path)[1:])
    with pytest.raises(ValueError):
        queue.classification_command(tmp_path, 1894)
    with pytest.raises(ValueError, match="smoke"):
        queue.require_live_smoke(tmp_path)


def test_fake_independent_executor_interrupt_resume_and_completed_noop(tmp_path, monkeypatch):
    pytest.importorskip("llama_cpp")
    executor = module("independent_executor", "run_fixed_classification_explanations.py")
    dataset, predictions, manifest = [
        tmp_path / name for name in ["data.csv", "pred.csv", "keys.csv"]
    ]
    pd.DataFrame({"Hashed_ReportURN": ["synthetic-a", "synthetic-b"],
                  "Report": ["Synthetic source text."] * 2}).to_csv(dataset, index=False)
    pd.DataFrame({"Hashed_ReportURN": ["synthetic-a", "synthetic-b"],
                  "classifications": [json.dumps(dict.fromkeys(JSON_KEYS, 1)),
                                      json.dumps(dict.fromkeys(JSON_KEYS, 4))]}).to_csv(
        predictions, index=False
    )
    pd.DataFrame({"Hashed_ReportURN": ["synthetic-a", "synthetic-b"]}).to_csv(manifest, index=False)
    output = tmp_path / "output.csv"
    prompt = category_evidence.audit_prompt(base_prompt())
    argv = ["script", "--run-id", "synthetic-audit", "--dataset", str(dataset),
            "--predictions", str(predictions), "--manifest", str(manifest),
            "--output-csv", str(output), "--model", "medgemma-27b-q2-candidate",
            "--interface", "native_chat", "--evidence-mode", category_evidence.MODE,
            "--expected-dataset-sha256", executor.sha256_file(dataset),
            "--expected-predictions-sha256", executor.sha256_file(predictions),
            "--expected-manifest-sha256", executor.sha256_file(manifest),
            "--expected-chat-template-sha256", sha256_text("synthetic-template"),
            "--expected-prompt-sha256", sha256_text(prompt),
            "--expected-grammar-sha256", executor.sha256_file(
                ROOT / "src/LLM_pipeline/result_grammar_category_evidence.gbnf"
            ), "--flush-every", "1", "--resume"]
    monkeypatch.setattr(sys, "argv", argv)
    args = executor.parse_args()

    class FakeModel:
        metadata = {"tokenizer.chat_template": "synthetic-template"}
        calls = 0
        def create_chat_completion(self, messages, **kwargs):
            self.calls += 1
            assert messages == category_evidence.messages(prompt, "Synthetic source text.")
            if self.calls == 2:
                raise RuntimeError("synthetic interruption")
            return {"choices": [{"message": {"content": audit_output()}}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 50, "total_tokens": 80}}

    model = FakeModel()
    monkeypatch.setattr(executor.pipeline, "download_model_with_receipt", lambda *a, **k: (
        model, {"sha256": "0" * 64}
    ))
    monkeypatch.setattr(executor.pipeline, "load_gbnf", lambda *a: "synthetic-grammar")
    with pytest.raises(RuntimeError, match="interruption"):
        executor.run(args)
    assert len(pd.read_csv(output)) == 1
    receipt = executor.run(args)
    assert len(pd.read_csv(output)) == 2 and model.calls == 3
    assert receipt["classifications_supplied_to_model"] is False
    assert receipt["decision_copy_check_applicable"] is False
    monkeypatch.setattr(executor.pipeline, "download_model_with_receipt", lambda *a, **k: (
        pytest.fail("completed resume must not load a model")
    ))
    assert executor.run(args) == receipt
    args.evidence_mode = "fixed-classification"
    with pytest.raises(ValueError):
        executor.run(args)


def test_independent_grammar_compiles_without_a_model():
    llama = pytest.importorskip("llama_cpp")
    grammar = ROOT / "src/LLM_pipeline/result_grammar_category_evidence.gbnf"
    assert llama.LlamaGrammar.from_string(grammar.read_text(), verbose=False)


def test_all_three_versions_and_both_pairings_reach_real_analysis(queue, tmp_path):
    from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL

    for name in ["inputs", "products", "analysis"]:
        (tmp_path / name).mkdir()
    plan = queue.read(queue.PLAN)
    keys = [f"synthetic-{i}" for i in range(100)]
    reference = pd.DataFrame({"Hashed_ReportURN": keys,
                              "Report": ["Synthetic source text."] * 100,
                              **{label: [4] * 10 + [1] * 90
                                 for label in JSON_KEY_TO_LABEL.values()}})
    with sqlite3.connect(tmp_path / "inputs/development.db") as conn:
        reference.to_sql("reports", conn, index=False)
    pd.DataFrame({"Hashed_ReportURN": keys[:20]}).to_csv(
        tmp_path / "inputs/evidence.manifest.csv", index=False
    )
    for name, misses in [("v1", 2), ("v2", 3), ("v21", 1)]:
        folder = "products" if name == "v21" else "inputs"
        pd.DataFrame({"Hashed_ReportURN": keys,
                      "classifications": [json.dumps(dict.fromkeys(
                          JSON_KEYS, 4 if misses <= i < 10 else 1
                      )) for i in range(100)]}).to_csv(
            tmp_path / folder / f"{name}.csv", index=False
        )
    fixed = pd.read_csv(tmp_path / "products/v21.csv")
    evidence = fixed.iloc[:20].rename(columns={"classifications": "fixed_classifications"})
    evidence["explanations"] = audit_output()
    checked = category_evidence.inspect(audit_output(), report="Synthetic source text.")
    for key in ["structured_output_valid", "decision_copy_mismatches", "evidence_phrases",
                "fallback_phrases", "exact_traceable_phrases", "casefold_traceable_phrases"]:
        evidence[key] = getattr(checked, key)
    path = tmp_path / "products/evidence-v21.csv"
    evidence.to_csv(path, index=False)
    receipt = {
        "output": {"sha256": queue.sha256_file(path)},
        "classifications_supplied_to_model": False,
        "evidence_used_to_change_classifications": False,
        "decision_copy_check_applicable": False,
        "evidence_mode": category_evidence.MODE,
        "inputs": {
            "fixed_predictions_sha256": queue.sha256_file(tmp_path / "products/v21.csv"),
            "dataset_sha256": queue.sha256_file(tmp_path / "inputs/development.db"),
            "manifest_sha256": queue.sha256_file(tmp_path / "inputs/evidence.manifest.csv"),
        },
        "model": {"sha256": plan["model"]["sha256"],
                  "load_parameters": plan["runtime"]["parameters"]},
        "environment": {"git": {"revision": "synthetic", "worktree_dirty": False},
                        "hf_hub_offline": True, "hf_hub_telemetry_disabled": True},
        "prompt": {"sha256": plan["evidence"]["prompt_sha256"],
                   "task_message_template_sha256": plan["evidence"]["task_message_sha256"]},
        "grammar": {"sha256": plan["evidence"]["grammar_sha256"]},
        "chat_template": {"sha256": plan["interface"]["chat_template_sha256"]},
        "sampling": {"temperature": 0.0, "top_k": 40, "top_p": 0.95,
                     "max_tokens": plan["evidence"]["max_tokens"]},
    }
    queue.atomic_json(path.with_suffix(".run.json"), receipt)
    result = queue.analyze_v21(tmp_path, {"repository_revision": "synthetic"}, plan)
    assert set(result["results"]) == {"v1", "v2", "v21"}
    assert set(result["paired_comparisons"]) == {"v1", "v2"}
    assert result["development_verdict"]["status"] == "development_rule_met"
    assert not result["protected_evaluation_run"]
    assert not result["evidence_quality"]["classifications_supplied_to_model"]
    assert result["evidence_quality"]["records"] == 20
    assert "synthetic-0" not in json.dumps(result)
