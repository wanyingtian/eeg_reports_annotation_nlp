from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import signal
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from eeg_review.evidence_extraction import JSON_KEYS, inspect_explanation
from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL
from eeg_review.native_interface import sha256_text
from eeg_review.protected_execution import ProtectedExecutionLocked

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


queue = load_script("mistral_interface_followup")


@pytest.fixture
def evidence():
    pytest.importorskip("llama_cpp")
    return load_script("run_fixed_classification_explanations")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def levels(value):
    return json.dumps({key: value for key in JSON_KEYS})


def reasons(value=1):
    return json.dumps({key: {"decision": value, "reasons": ["normal"]} for key in JSON_KEYS})


def inputs(root, count=2):
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "development.db"
    with sqlite3.connect(dataset) as connection:
        pd.DataFrame(
            [
                {
                    "Hashed_ReportURN": f"case-{i}",
                    "Report": "normal background",
                    **{label: 1 + i % 4 for label in JSON_KEY_TO_LABEL.values()},
                }
                for i in range(count)
            ]
        ).to_sql("reports", connection, index=False)
    frame = pd.DataFrame(
        {
            "Hashed_ReportURN": [f"case-{i}" for i in range(count)],
            "classifications": [levels(1 + i % 4) for i in range(count)],
        }
    )
    for column in [
        "classify_elapsed_seconds",
        "classify_prompt_tokens",
        "classify_completion_tokens",
    ]:
        frame[column] = 1
    frame.to_csv(root / "raw-development.csv", index=False)
    frame[["Hashed_ReportURN"]].to_csv(root / "development.manifest.csv", index=False)
    return frame


@pytest.mark.parametrize(
    "field,value",
    [
        ("classification_records", 101),
        ("evidence_records", 21),
        ("evidence_positions", [20, 40]),
        ("max_new_model_calls", 141),
        ("protected_evaluation_allowed", True),
        ("automatic_expansion_allowed", True),
    ],
)
def test_policy_rejects_expansion(field, value):
    policy = queue.read_json(queue.POLICY)
    queue.validate_policy(policy)
    policy[field] = value
    with pytest.raises(ValueError):
        queue.validate_policy(policy)


def dependency_fixture(root):
    write_json(root / "job.json", {"study_id": queue.DEPENDENCY_ID})
    write_json(root / "state.json", {"status": "running"})
    write_json(
        root / "status.json",
        {
            "study_id": queue.DEPENDENCY_ID,
            "configuration_id": "native",
            "completed_records": 1894,
            "target_records": 1894,
        },
    )
    write_json(root / "heartbeat.json", {"study_id": queue.DEPENDENCY_ID, "finalized": False})
    return {
        "run_dir": str(root),
        "study_id": queue.DEPENDENCY_ID,
        "job_sha256": queue.sha256_file(root / "job.json"),
        "configuration_id": "native",
        "public_status": str(root / "status.json"),
        "public_heartbeat": str(root / "heartbeat.json"),
    }


def test_dependency_waits_for_complete_finalized_and_released(tmp_path):
    dependency = dependency_fixture(tmp_path)
    assert not queue.dependency_readiness(dependency)[0]
    write_json(tmp_path / "state.json", {"status": "completed"})
    assert not queue.dependency_readiness(dependency)[0]
    write_json(tmp_path / "heartbeat.json", {"study_id": queue.DEPENDENCY_ID, "finalized": True})
    with (tmp_path / ".tiered-run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not queue.dependency_readiness(dependency)[0]
    assert queue.dependency_readiness(dependency)[0]


def test_dependency_rejects_changed_identity_and_failure(tmp_path):
    dependency = dependency_fixture(tmp_path)
    write_json(tmp_path / "state.json", {"status": "failed"})
    with pytest.raises(ValueError, match="failed"):
        queue.dependency_readiness(dependency)
    write_json(tmp_path / "job.json", {"study_id": "different"})
    with pytest.raises(ValueError, match="identity"):
        queue.dependency_readiness(dependency)


def test_dependency_eclipse_stops_dispatch(tmp_path):
    dependency = dependency_fixture(tmp_path)
    write_json(tmp_path / "ECLIPSED.json", {"status": "eclipsed"})
    with pytest.raises(ProtectedExecutionLocked, match="eclipsed"):
        queue.dependency_readiness(dependency)


def test_resume_contract_is_immutable(tmp_path, evidence):
    path = tmp_path / "contract.json"
    contract = {"prompt": "frozen", "classification": "fixed"}
    evidence.bind_execution_contract(path, contract, False)
    evidence.bind_execution_contract(path, contract, True)
    with pytest.raises(ValueError, match="contract changed"):
        evidence.bind_execution_contract(path, {**contract, "prompt": "changed"}, True)
    with pytest.raises(ValueError, match="lacks"):
        evidence.bind_execution_contract(tmp_path / "missing.json", contract, True)


def evidence_args(root, interface, evidence):
    return argparse.Namespace(
        dataset=root / "development.db",
        predictions=root / "raw-development.csv",
        manifest=root / "development.manifest.csv",
        output_csv=root / "evidence.csv",
        expected_dataset_sha256=queue.sha256_file(root / "development.db"),
        expected_predictions_sha256=queue.sha256_file(root / "raw-development.csv"),
        expected_manifest_sha256=queue.sha256_file(root / "development.manifest.csv"),
        resume=True,
        table="reports",
        id_column="Hashed_ReportURN",
        report_column="Report",
        classification_column="classifications",
        expected_records=2,
        n_ctx=4096,
        n_gpu_layers=30,
        n_batch=None,
        n_ubatch=None,
        n_threads=None,
        n_threads_batch=None,
        flash_attn=None,
        expected_grammar_sha256=queue.sha256_file(
            evidence.PIPELINE_ROOT / "result_grammar_exp.gbnf"
        ),
        expected_prompt_sha256=sha256_text(evidence.pipeline.PROMPT_EXPLAIN),
        expected_chat_template_sha256=sha256_text("native template"),
        run_id="test-fixed-evidence",
        model="mistral",
        interface=interface,
        temperature=0.0,
        top_k=40,
        top_p=0.95,
        max_tokens=3000,
        flush_every=1,
    )


@pytest.mark.parametrize("interface", ["raw_completion", "native_chat"])
def test_evidence_fake_model_interrupt_resume_and_complete_no_reload(
    tmp_path, monkeypatch, interface, evidence
):
    inputs(tmp_path)
    args = evidence_args(tmp_path, interface, evidence)
    model = SimpleNamespace(metadata={"tokenizer.chat_template": "native template"})
    loads, calls = [], []

    def download(*_args, **kwargs):
        assert kwargs["local_files_only"] is True
        loads.append(1)
        return model, {"sha256": "fake"}

    def call(**kwargs):
        assert kwargs["grammar"] == "test grammar"
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return SimpleNamespace(
            text=reasons(),
            elapsed_seconds=0.1,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

    monkeypatch.setattr(evidence.pipeline, "download_model_with_receipt", download)
    monkeypatch.setattr(evidence.pipeline, "load_gbnf", lambda _: "test grammar")
    monkeypatch.setattr(evidence.pipeline, "llm_chat_json_with_receipt", call)
    monkeypatch.setattr(evidence.pipeline, "llm_json_with_receipt", call)
    with pytest.raises(RuntimeError, match="interruption"):
        evidence.run(args)
    assert len(pd.read_csv(args.output_csv)) == 1
    receipt = evidence.run(args)
    assert receipt["aggregate_quality"]["records"] == 2
    assert receipt["aggregate_quality"]["decision_copy_mismatches"] == 5
    assert len(calls) == 3  # One failed in-flight call; completed first record was not rerun.
    evidence.run(args)
    assert len(loads) == 2
    args.temperature = 0.5
    with pytest.raises(ValueError, match="contract changed"):
        evidence.run(args)
    assert len(loads) == 2


def complete_fixture(root):
    frame = inputs(root / "inputs", 100)
    for name in ["products", "receipts", "analysis", "logs", "stages"]:
        (root / name).mkdir()
    plan = queue.read_json(queue.PLAN)
    plan["development_surface"]["source_database_sha256"] = queue.sha256_file(
        root / "inputs/development.db"
    )
    write_json(root / "inputs/scientific-plan.json", plan)
    write_json(
        root / "job.json",
        {"repository_revision": "test-revision", "python_executable": sys.executable},
    )
    native = frame.copy()
    native["classification_interface_mode"] = "native_chat"
    native.to_csv(root / "products/native-classification.csv", index=False)
    frame[["Hashed_ReportURN"]].iloc[:20].to_csv(root / "inputs/evidence.manifest.csv", index=False)
    interface = plan["classification_interface"]
    receipt = {
        "model": {
            "sha256": plan["model"]["sha256"],
            "load_parameters": {"n_ctx": 4096, "n_gpu_layers": 30},
            "artifact_access": {"mode": "local_cache_only"},
        },
        "prompts": {"classify": {"sha256": interface["classification_prompt_sha256"]}},
        "grammars": {"classify": {"sha256": interface["grammar_sha256"]}},
        "input_policy": {
            "classification_interface_mode": "native_chat",
            "embedded_chat_template": {"sha256": interface["chat_template_sha256"]},
            "task_message_template": {"sha256": interface["task_message_template_sha256"]},
        },
        "sampling": interface["sampling"],
        "reports_completed": 100,
        "output": {"sha256": queue.sha256_file(root / "products/native-classification.csv")},
        "dataset": {"sha256": plan["development_surface"]["source_database_sha256"]},
        "environment": {
            "git": {"revision": "test-revision", "worktree_dirty": False},
            "packages": {"synthetic": "1"},
        },
        "execution_surface": {"classification": True, "explanations": False},
    }
    write_json(root / "products/native-classification.run.json", receipt)
    write_json(root / "inputs/raw-parent.run.json", receipt)
    for name in ["raw_completion", "native_chat"]:
        rows = []
        for _, row in frame.iloc[:20].iterrows():
            checked = inspect_explanation(
                reasons(), report="normal background", fixed_classification=row["classifications"]
            )
            rows.append(
                {
                    "Hashed_ReportURN": row["Hashed_ReportURN"],
                    "fixed_classifications": row["classifications"],
                    "explanations": reasons(),
                    **vars(checked),
                    "elapsed_seconds": 1.0,
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                }
            )
        path = root / f"products/evidence-{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        write_json(path.with_suffix(".run.json"), {"output": {"sha256": queue.sha256_file(path)}})


def test_full_synthetic_freeze_and_existing_analysis_path(tmp_path):
    complete_fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        queue.analyze(tmp_path, iterations=10)
    freeze = queue.freeze_classification(tmp_path)
    assert freeze["selected_for_freeze"]
    result = queue.analyze(tmp_path, iterations=10)
    assert len(result["classification_agreement"]) == 5
    assert all(x["same_core"] == 100 for x in result["classification_agreement"].values())
    assert "case-" not in json.dumps(result)
    assert "case-" not in (tmp_path / "analysis/author-summary.md").read_text()
    assert list((tmp_path / "analysis/paired").glob("*.json"))


def test_freeze_rejects_duplicate_keys_and_preserves_rejected_output(tmp_path):
    complete_fixture(tmp_path)
    path = tmp_path / "products/native-classification.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "Hashed_ReportURN"] = "case-0"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="structural freeze rejected"):
        queue.freeze_classification(tmp_path)
    assert not queue.read_json(tmp_path / "receipts/structural-freeze.json")["selected_for_freeze"]
    assert path.exists()


def test_queued_commands_are_bounded_and_reuse_raw_classifications(tmp_path):
    complete_fixture(tmp_path)
    stages = queue.stages(tmp_path)
    assert [stage.target_rows for stage in stages] == [100, None, 20, 20, None]
    assert "--classification-only" in stages[0].command
    assert "--local-model-only" in stages[0].command
    for stage in stages[2:4]:
        assert str(tmp_path / "inputs/raw-development.csv") in stage.command
        assert stage.command[stage.command.index("--expected-records") + 1] == "20"
    assert "protected" not in " ".join(stages[0].command)


def test_supervisor_synthetic_stage_completion_and_no_repeat(tmp_path, monkeypatch):
    # Exercise the real subprocess supervisor, not a mocked success path.
    module = sys.modules["study_job"]
    write_json(tmp_path / "state.json", {"status": "queued", "stages": {}})
    write_json(tmp_path / "job.json", {"study_id": "synthetic"})
    (tmp_path / "stages").mkdir()
    product = tmp_path / "product.txt"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
        "p.write_text(p.read_text()+'x' if p.exists() else 'x')",
        str(product),
    ]
    stage = module.Stage("fake", command, (product,))
    previous = {s: signal.getsignal(s) for s in [signal.SIGTERM, signal.SIGINT]}
    progress = []
    try:
        supervisor = module.Supervisor(
            tmp_path, stages=[stage], progress_callback=lambda *_: progress.append(1)
        )
        supervisor.run()
        assert progress
        assert queue.read_json(tmp_path / "state.json")["status"] == "completed"
        supervisor.run()
        assert product.read_text() == "x"
        assert queue.read_json(tmp_path / "transfer-manifest.json")["study_id"] == "synthetic"
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def test_supervisor_pause_preserves_checkpoints_and_resumes(tmp_path):
    module = sys.modules["study_job"]
    write_json(tmp_path / "state.json", {"status": "queued", "stages": {}})
    write_json(tmp_path / "job.json", {"study_id": "synthetic"})
    (tmp_path / "stages").mkdir()
    checkpoint = tmp_path / "checkpoint.txt"
    checkpoint.write_text("retained")
    stage = module.Stage(
        "fake", [sys.executable, "-c", "import time; time.sleep(60)"], (checkpoint,)
    )
    previous = {s: signal.getsignal(s) for s in [signal.SIGTERM, signal.SIGINT]}
    try:
        supervisor = module.Supervisor(
            tmp_path,
            stages=[stage],
            progress_callback=lambda s, _: s.handle_signal(signal.SIGTERM, None),
        )
        supervisor.run()
        assert queue.read_json(tmp_path / "state.json")["status"] == "interrupted"
        assert checkpoint.read_text() == "retained"
        stage = module.Stage("fake", [sys.executable, "-c", "pass"], (checkpoint,))
        module.Supervisor(tmp_path, stages=[stage]).run()
        assert queue.read_json(tmp_path / "state.json")["status"] == "completed"
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def test_supervisor_callback_error_terminates_worker(tmp_path):
    module = sys.modules["study_job"]
    write_json(tmp_path / "state.json", {"status": "queued", "stages": {}})
    write_json(tmp_path / "job.json", {"study_id": "synthetic"})
    stage = module.Stage("fake", [sys.executable, "-c", "import time; time.sleep(60)"], ())
    previous = {s: signal.getsignal(s) for s in [signal.SIGTERM, signal.SIGINT]}

    def fail(*_):
        raise TimeoutError("synthetic safety limit")

    supervisor = module.Supervisor(tmp_path, stages=[stage], progress_callback=fail)
    try:
        with pytest.raises(TimeoutError, match="safety limit"):
            supervisor.run()
        assert supervisor.child.poll() is not None
        assert queue.run_lock_released(tmp_path)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
