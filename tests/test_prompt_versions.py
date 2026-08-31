from __future__ import annotations

import ast
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL
from eeg_review.native_interface import sha256_text
from eeg_review.prompt_versions import (
    FOCAL_DISAMBIGUATION,
    HISTORICAL_PROMPT_SHA256,
    HISTORICAL_PROMPT_VERSION,
    MEDGEMMA_FOCAL_V2,
    classification_prompt,
    development_verdict,
    prompt_row_identity,
    validate_prompt_resume,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "medgemma_prompt_v2", ROOT / "scripts/medgemma_prompt_v2.py"
)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def historical_prompt():
    tree = ast.parse((ROOT / "src/LLM_pipeline/pipeline.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PROMPT_CLASSIFY"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("historical prompt missing")


def test_only_focal_clarification_changes_and_hashes_match():
    base = historical_prompt()
    assert sha256_text(base) == HISTORICAL_PROMPT_SHA256
    assert classification_prompt(base) == base
    changed = classification_prompt(base, MEDGEMMA_FOCAL_V2)
    assert changed.replace("\n" + FOCAL_DISAMBIGUATION, "", 1) == base
    plan = queue.read(queue.PLAN)
    assert sha256_text(changed) == plan["interface"]["prompt_sha256"]
    assert prompt_row_identity(HISTORICAL_PROMPT_VERSION, base) == {}


@pytest.mark.parametrize("base,version", [("changed", MEDGEMMA_FOCAL_V2), ("", "unknown")])
def test_wrong_parent_or_unknown_version_rejected(base, version):
    with pytest.raises(ValueError):
        classification_prompt(base, version)


@pytest.mark.parametrize("mutation", ["legacy", "missing", "null", "mixed", "hash"])
def test_cannot_resume_unidentified_or_changed_prompt(mutation):
    prompt = classification_prompt(historical_prompt(), MEDGEMMA_FOCAL_V2)
    frame = pd.DataFrame([prompt_row_identity(MEDGEMMA_FOCAL_V2, prompt)] * 2)
    if mutation == "legacy":
        frame = pd.DataFrame({"case": [1, 2]})
    elif mutation == "missing":
        frame = frame.drop(columns="classification_prompt_sha256")
    elif mutation == "null":
        frame.loc[0, "classification_prompt_version"] = None
    elif mutation == "mixed":
        frame.loc[0, "classification_prompt_version"] = HISTORICAL_PROMPT_VERSION
    else:
        frame.loc[0, "classification_prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_prompt_resume(frame, MEDGEMMA_FOCAL_V2, prompt)


def test_legacy_resume_preserved_but_v2_cannot_become_legacy():
    base = historical_prompt()
    validate_prompt_resume(pd.DataFrame({"case": [1]}), HISTORICAL_PROMPT_VERSION, base)
    frame = pd.DataFrame(
        [prompt_row_identity(MEDGEMMA_FOCAL_V2, classification_prompt(base, MEDGEMMA_FOCAL_V2))]
    )
    with pytest.raises(ValueError):
        validate_prompt_resume(frame, HISTORICAL_PROMPT_VERSION, base)


@pytest.mark.parametrize(
    "field,value",
    [
        ("classification_records", 101),
        ("evidence_records", 21),
        ("evidence_positions", [20, 40]),
        ("candidate_count", 2),
        ("candidate_count", True),
        ("planned_committed_model_calls", 121),
        ("protected_evaluation_allowed", True),
        ("automatic_expansion_allowed", True),
        ("independent_test_claim_allowed", True),
        ("prior_protected_results_informed_hypothesis", False),
    ],
)
def test_frozen_scope_rejects_expansion_or_lost_caveat(field, value):
    plan = queue.read(queue.PLAN)
    queue.validate_plan(plan)
    plan[field] = value
    with pytest.raises(ValueError):
        queue.validate_plan(plan)


def points():
    return {label: {"tp": 9, "fn": 1, "fp": 2, "tn": 88} for label in JSON_KEY_TO_LABEL.values()}


@pytest.mark.parametrize(
    "change,met",
    [
        ("no_change", False),
        ("fewer_fp", True),
        ("new_fn", False),
        ("other_regression", False),
    ],
)
def test_descriptive_rule_keeps_every_category_and_never_unlocks_evaluation(change, met):
    a, b = points(), points()
    if change != "no_change":
        b["Focal Epi"].update(fp=1, tn=89)
    if change == "new_fn":
        b["Focal Epi"].update(fn=2, tp=8)
    if change == "other_regression":
        b["Gen Epi"].update(fn=2, tp=8)
    result = development_verdict(a, b)
    assert (result["status"] == "development_rule_met") is met
    assert result["protected_evaluation_authorized"] is False
    assert result["independent_confirmation"] is False


@pytest.mark.parametrize("mutation", ["missing_label", "wrong_count", "wrong_support"])
def test_rule_rejects_incomplete_or_incompatible_comparisons(mutation):
    a, b = points(), points()
    if mutation == "missing_label":
        del b["Gen Epi"]
    elif mutation == "wrong_count":
        b["Focal Epi"]["tn"] += 1
    else:
        b["Focal Epi"].update(tp=8, tn=89)
    with pytest.raises(ValueError):
        development_verdict(a, b)


def job_fixture(root):
    (root / "inputs").mkdir()
    (root / "products").mkdir()
    queue.atomic_json(root / "inputs/plan.json", queue.read(queue.PLAN))
    queue.atomic_json(root / "job.json", {"python_executable": sys.executable})
    return root


def test_queue_is_local_fixed_development_and_evidence_cannot_reclassify(tmp_path):
    root = job_fixture(tmp_path)
    stages = queue.stages(root)
    assert [stage.name for stage in stages] == [
        "classification_100",
        "validate_classification",
        "evidence_20",
        "analyze",
    ]
    command = queue.classification_command(root)
    assert command[command.index("--num-reports") + 1] == "100"
    assert "--local-model-only" in command and "--resume-output" in command
    assert "--classification-only" in command
    assert command[command.index("--classification-prompt-version") + 1] == MEDGEMMA_FOCAL_V2
    for count in [20, 101, 1395, 1894]:
        with pytest.raises(ValueError):
            queue.classification_command(root, count)


def test_incomplete_run_cannot_finalize(tmp_path):
    queue.atomic_json(tmp_path / "state.json", {"status": "running"})
    with pytest.raises(ValueError, match="incomplete"):
        queue.finalize(tmp_path)


@pytest.mark.parametrize("mutation", ["duplicate", "reorder", "missing"])
def test_manifest_validation_fails_before_receipt_analysis(tmp_path, mutation):
    root = job_fixture(tmp_path)
    manifest = pd.DataFrame({queue.KEY: ["synthetic-1", "synthetic-2", "synthetic-3"]})
    manifest.to_csv(root / "inputs/development.manifest.csv", index=False)
    output = manifest.copy()
    if mutation == "duplicate":
        output.loc[1, queue.KEY] = "synthetic-1"
    elif mutation == "reorder":
        output = output.iloc[::-1]
    else:
        output = output.iloc[:2]
    output.to_csv(root / "products/v2.csv", index=False)
    with pytest.raises(ValueError, match="manifest prefix"):
        queue.inspect_classification(root, 3)


def test_fake_model_v2_receipt_and_resume(tmp_path, monkeypatch):
    pytest.importorskip("llama_cpp")
    sys.path.insert(0, str(ROOT / "src/LLM_pipeline"))
    import pipeline

    class FakeModel:
        metadata = {"tokenizer.chat_template": "{{ messages[0]['content'] }}"}
        calls = 0

        def create_chat_completion(self, messages, **kwargs):
            self.calls += 1
            assert FOCAL_DISAMBIGUATION in messages[0]["content"]
            assert kwargs["grammar"] == "same-grammar"
            return {
                "choices": [
                    {"message": {"content": json.dumps({key: 1 for key in JSON_KEY_TO_LABEL})}}
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
            }

    model = FakeModel()
    config = pipeline.RunConfig(
        outdir=tmp_path,
        dataset_path=ROOT / "data/zoe_reports_sample.db",
        dataset_id="synthetic",
        model_name="medgemma-27b-q2-candidate",
        classification_interface="native_chat",
        classification_prompt_version=MEDGEMMA_FOCAL_V2,
        run_explanations=False,
    )
    receipt = {"registry_name": "synthetic", "sha256": "0" * 64, "load_parameters": {"n_ctx": 4096}}
    output = tmp_path / "raw.csv"
    pipeline.run_pipeline(
        model,
        receipt,
        pd.DataFrame([{queue.KEY: "synthetic-1", "Report": "Normal EEG."}]),
        pd.DataFrame(),
        "same-grammar",
        None,
        output,
        tmp_path / "config.json",
        config,
        flush_every=1,
    )
    assert model.calls == 1
    result = queue.read(output.with_suffix(".run.json"))
    assert result["prompts"]["classify"]["version"] == MEDGEMMA_FOCAL_V2
    assert (
        result["prompts"]["classify"]["sha256"]
        == queue.read(queue.PLAN)["interface"]["prompt_sha256"]
    )
    resumed, done = pipeline.process_completed_csv(
        output,
        run_explanations=False,
        classification_interface="native_chat",
        classification_prompt_version=MEDGEMMA_FOCAL_V2,
    )
    assert len(resumed) == 1 and done == {"synthetic-1"}
    with pytest.raises(ValueError, match="prompt identity"):
        pipeline.process_completed_csv(
            output, run_explanations=False, classification_interface="native_chat"
        )


def test_changed_completed_stage_cannot_be_silently_resealed(tmp_path, monkeypatch):
    root = job_fixture(tmp_path)
    queue.atomic_json(root / "state.json", {"status": "completed"})
    monkeypatch.setattr(queue, "stage_is_complete", lambda *_: False)
    with pytest.raises(ValueError, match="re-seal"):
        queue.finalize(root)


def test_full_analysis_with_synthetic_predictions_and_evidence(tmp_path, monkeypatch):
    from eeg_review.evidence_extraction import inspect_explanation

    root = job_fixture(tmp_path)
    (root / "receipts").mkdir()
    job = {"repository_revision": "synthetic"}
    plan = queue.read(queue.PLAN)
    monkeypatch.setattr(queue, "validate_job", lambda _: (job, plan))
    monkeypatch.setattr(queue, "inspect_classification", lambda _: {"records": 100})
    queue.atomic_json(root / "receipts/classification-complete.json", {"records": 100})
    records, evidence = [], []
    for i in range(100):
        level = 1 + i % 4
        classification = json.dumps({key: level for key in JSON_KEY_TO_LABEL})
        records.append(
            {
                queue.KEY: f"synthetic-{i}",
                "Report": "normal background",
                "classifications": classification,
                **{label: level for label in JSON_KEY_TO_LABEL.values()},
            }
        )
        if i < 20:
            explanation = json.dumps(
                {key: {"decision": level, "reasons": ["background"]} for key in JSON_KEY_TO_LABEL}
            )
            inspected = inspect_explanation(
                explanation, report="normal background", fixed_classification=classification
            )
            evidence.append(
                {
                    queue.KEY: f"synthetic-{i}",
                    "fixed_classifications": classification,
                    **vars(inspected),
                }
            )
    frame = pd.DataFrame(records)
    with sqlite3.connect(root / "inputs/development.db") as connection:
        frame.drop(columns="classifications").to_sql("reports", connection, index=False)
    for path in [root / "inputs/v1.csv", root / "products/v2.csv"]:
        frame[[queue.KEY, "classifications"]].to_csv(path, index=False)
    frame[[queue.KEY]].iloc[:20].to_csv(root / "inputs/evidence.manifest.csv", index=False)
    evidence_path = root / "products/evidence-v2.csv"
    pd.DataFrame(evidence).to_csv(evidence_path, index=False)
    queue.atomic_json(
        evidence_path.with_suffix(".run.json"),
        {"output": {"sha256": queue.sha256_file(evidence_path)}},
    )
    result = queue.analyze(root)
    assert result["classification_records"] == 100
    assert result["development_verdict"]["status"] == "development_rule_not_met"
    assert result["protected_evaluation_run"] is False
    assert len(result["paired_comparison"]) == 5
    assert result["same_core"] == dict.fromkeys(JSON_KEY_TO_LABEL.values(), 100)
    assert not any(f"synthetic-{i}" in json.dumps(result) for i in range(100))
