from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from eeg_review.audit import DEFAULT_LABELS
from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL
from eeg_review.manifest import sha256_file
from eeg_review.reviewability import KEY

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "review_runner", ROOT / "scripts/prepare_comparison_review.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write_db(path, frame, table="reports"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        frame.to_sql(table, connection, index=False)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "COHORTS", {"zoe_evaluation_1395": 6, "maria_evaluation_499": 6})
    for name in ("reviewability.py", "source_grounding.py"):
        target = tmp_path / "src/eeg_review" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src/eeg_review" / name, target)
    base = tmp_path / "data/governed"
    native, original = base / "native", base / "original"
    for cohort in runner.COHORTS:
        key = [f"synthetic-{cohort}-{i}" for i in range(6)]
        report = "normal <script>alert('synthetic')</script> " + " ".join(
            f"word{i}" for i in range(40)
        )

        def frame(values, key=key, report=report):
            return pd.DataFrame(
                {KEY: key, "Report": [report] * 6, **{label: values for label in DEFAULT_LABELS}}
            )

        ref = frame([4, 4, 4, 1, 4, 1])
        a, b = frame([4, 1, 1, 4, 4, 1]), frame([1, 4, 1, 4, 4, 1])
        write_db(native / f"inputs/{cohort}.db", ref)
        write_db(native / f"products/{cohort}/processed/predictions.db", a, "classifications")
        path = native / f"comparators/{cohort}_reproduced_mistral.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        b.to_csv(path, index=False)
        raw = []
        for _, row in b.iterrows():
            levels = {k: row[label] for k, label in JSON_KEY_TO_LABEL.items()}
            raw.append(
                {
                    KEY: row[KEY],
                    "classifications": json.dumps(levels),
                    "explanations": json.dumps(
                        {
                            k: {"decision": value, "reasons": ["normal"]}
                            for k, value in levels.items()
                        }
                    ),
                }
            )
        path = original / f"products/llm/{cohort.split('_')[0]}/raw.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(raw).to_csv(path, index=False)
        path = native / f"analysis/{cohort}/vs_reproduced_mistral/paired_comparison_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "labels": {
                        label: {
                            "discordant_correctness": {
                                "core_accuracy": {"a_correct_b_wrong": 1, "a_wrong_b_correct": 1}
                            },
                            "model_a_point_estimates": {"fp": 1, "fn": 2},
                            "model_b_point_estimates": {"fp": 1, "fn": 2},
                        }
                        for label in DEFAULT_LABELS
                    }
                }
            )
        )
    dev = pd.DataFrame(
        {
            KEY: [f"development-{i}" for i in range(100)],
            "Report": ["tiny report"] * 100,
            **{label: [1] * 100 for label in DEFAULT_LABELS},
        }
    )
    write_db(original / "inputs/zoe_development_100.db", dev)
    files = [
        {"path": str(p.relative_to(native)), "sha256": sha256_file(p)}
        for p in native.rglob("*")
        if p.is_file()
    ]
    (native / "final-transfer-manifest.json").write_text(json.dumps({"files": files}))
    return Namespace(
        native_run=native,
        original_run=original,
        output_dir=base / "analysis-runs/test",
        acknowledge_governed_output=True,
        dry_run=False,
    )


def test_end_to_end_resume_privacy_and_html_escape(prepared, capsys):
    runner.run(prepared)
    output = prepared.output_dir
    summary = json.loads((output / "aggregate-summary.json").read_text())
    assert summary["selected_unique_reports"] == 12
    assert summary["selected_label_case_rows"] == 60
    assert summary["cohort_arithmetic"]["evaluation"] == 12
    assert summary["inference_performed"] is False
    assert "synthetic" not in (output / "aggregate-summary.json").read_text()
    page = (output / "case-review.html").read_text()
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "default-src 'none'" in page
    assert "synthetic-zoe" not in (output / "review.csv").read_text()
    complete_before = sha256_file(output / "COMPLETE.json")
    runner.run(prepared)
    assert sha256_file(output / "COMPLETE.json") == complete_before
    assert "no recomputation" in capsys.readouterr().out
    (output / "review.csv").write_text("tampered")
    with pytest.raises(ValueError, match="completed output changed"):
        runner.run(prepared)


def test_partial_resume_checks_checkpoint_hash(prepared):
    runner.run(prepared)
    (prepared.output_dir / "COMPLETE.json").unlink()
    checkpoint = next(prepared.output_dir.glob("near-duplicate-*.json"))
    checkpoint.write_text("{}")
    with pytest.raises(ValueError, match="partial checkpoint"):
        runner.run(prepared)


def test_dry_run_no_output_and_governance_gates(prepared):
    prepared.dry_run = True
    runner.run(prepared)
    assert not prepared.output_dir.exists()
    prepared.acknowledge_governed_output = False
    with pytest.raises(ValueError, match="acknowledgement"):
        runner.run(prepared)
    prepared.acknowledge_governed_output = True
    (prepared.native_run / "ECLIPSED.json").write_text("{}")
    with pytest.raises(Exception, match="eclipsed"):
        runner.run(prepared)


def test_bundle_tampering_and_escape_rejected(prepared):
    manifest = prepared.native_run / "final-transfer-manifest.json"
    original = json.loads(manifest.read_text())
    original["files"].append({"path": "../../escape", "sha256": "0" * 64})
    manifest.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="unsafe"):
        runner.run(prepared)
