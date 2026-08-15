#!/usr/bin/env python3
"""Resumable, portable supervisor for the JBHI study reproduction compute.

The run directory is governed and self-contained: it holds canonical input
snapshots, case-level products, aggregate receipts, logs, stage markers, and a
machine-readable transfer manifest. No governed output is written to Git.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]
SOURCE_COLUMNS = [
    "Hashed ID",
    "Physician",
    "Hospital",
    "Report",
    "Cluster code",
    *LABELS,
]
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def read_sqlite(path: Path) -> pd.DataFrame:
    resolved = path.expanduser().resolve(strict=True)
    with sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True) as connection:
        available = {
            row[1] for row in connection.execute("PRAGMA table_info(reports)").fetchall()
        }
        missing = sorted(set(SOURCE_COLUMNS) - available)
        if missing:
            raise ValueError(f"Missing source columns in {resolved.name}: {missing}")
        selection = ", ".join('"' + value.replace('"', '""') + '"' for value in SOURCE_COLUMNS)
        frame = pd.read_sql_query(f'SELECT {selection} FROM "reports"', connection)
    return frame.rename(columns={"Hashed ID": "Hashed_ReportURN"})


def select_ranges(frame: pd.DataFrame, ranges: list[tuple[int, int]]) -> pd.DataFrame:
    for start, end in ranges:
        if start < 0 or end <= start or end > len(frame):
            raise ValueError(f"Invalid snapshot range {start}:{end} for {len(frame)} rows")
    return pd.concat([frame.iloc[start:end] for start, end in ranges], ignore_index=True)


def atomic_sqlite(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with sqlite3.connect(temporary) as connection:
        frame.to_sql("reports", connection, index=False, if_exists="replace")
    temporary.replace(path)
    path.chmod(0o600)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    path.chmod(0o600)


def matched_annotation_surface(reference: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    if annotation["Hashed_ReportURN"].duplicated().any():
        raise ValueError("Second-annotator snapshot has duplicate report identifiers")
    columns = ["Hashed_ReportURN", *LABELS]
    return reference[["Hashed_ReportURN"]].merge(
        annotation[columns], on="Hashed_ReportURN", how="left", validate="one_to_one"
    )


def historical_mistral(path: Path, rows: int) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="classifications", nrows=rows)
    frame = frame.rename(columns={"Hashed ID": "Hashed_ReportURN"})
    return frame[["Hashed_ReportURN", *LABELS]].copy()


def historical_baseline(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    output = pd.DataFrame({"Hashed_ReportURN": source["Hashed ID"].astype("string")})
    for label in LABELS:
        output[f"{label} prediction"] = source[label]
        output[f"{label} probability"] = source[f"Prob_{label}"]
    return output


def input_receipt(path: Path, run_dir: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(run_dir)),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def initialize_run(run_dir: Path, source_root: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    if (run_dir / "job.json").exists():
        raise FileExistsError(f"Run is already initialized: {run_dir}")
    os.umask(0o077)
    for relative in ["inputs", "products", "logs", "stages", "cache"]:
        (run_dir / relative).mkdir(parents=True, exist_ok=True, mode=0o700)

    data_dir = source_root / "data"
    mistral_dir = source_root / "mistral_annotations"
    baseline_dir = source_root / "baseline_results"
    zoe_ra = read_sqlite(data_dir / "zoe_reports_LD_2000.db")
    zoe_sa = read_sqlite(data_dir / "zoe_reports_SG_1500.db")
    maria_ra = read_sqlite(data_dir / "maria_reports_LD.db")
    maria_sa = read_sqlite(data_dir / "maria_reports_SG.db")

    snapshots: dict[str, tuple[pd.DataFrame, str]] = {
        "zoe_source_2000.db": (select_ranges(zoe_ra, [(0, 2000)]), "LLM Zoe source"),
        "zoe_development_100.db": (
            select_ranges(zoe_ra, [(0, 100)]),
            "baseline and prompt-development snapshot",
        ),
        "zoe_evaluation_1400.db": (
            select_ranges(zoe_ra, [(100, 500), (1000, 2000)]),
            "Zoe evaluation candidates",
        ),
        "maria_source_500.db": (select_ranges(maria_ra, [(0, 500)]), "LLM Maria source"),
        "maria_evaluation_500.db": (
            select_ranges(maria_ra, [(0, 500)]),
            "Maria evaluation candidates",
        ),
    }
    receipts: list[dict[str, Any]] = []
    for filename, (frame, role) in snapshots.items():
        destination = run_dir / "inputs" / filename
        atomic_sqlite(destination, frame)
        receipts.append(input_receipt(destination, run_dir, role))

    zoe_eval = snapshots["zoe_evaluation_1400.db"][0]
    maria_eval = snapshots["maria_evaluation_500.db"][0]
    tabular_inputs = {
        "zoe_second_annotator.csv": (
            matched_annotation_surface(zoe_eval, zoe_sa),
            "Zoe second-annotator labels aligned to evaluation candidates",
        ),
        "maria_second_annotator.csv": (
            matched_annotation_surface(maria_eval, maria_sa),
            "Maria second-annotator labels aligned to evaluation candidates",
        ),
        "historical_zoe_mistral.csv": (
            historical_mistral(
                mistral_dir / "processed_zoe_mistral_2000_results.xlsx", 2000
            ),
            "submitted-study Zoe Mistral classifications",
        ),
        "historical_maria_mistral.csv": (
            historical_mistral(
                mistral_dir / "processed_maria_mistral_500_results.xlsx", 500
            ),
            "submitted-study Maria Mistral classifications",
        ),
    }
    for cohort in ["zoe", "maria"]:
        for model, filename in [
            ("bow", f"{cohort}_inference_results_bag_of_words_ep=0.1_v1.csv"),
            ("bert", f"{cohort}_inference_results_bert_base_ep=0.1_v1.csv"),
        ]:
            role_suffix = (
                "exact submitted rows"
                if cohort == "maria"
                else "uploaded historical version; not the producing submitted Zoe rows"
            )
            tabular_inputs[f"historical_{cohort}_{model}.csv"] = (
                historical_baseline(baseline_dir / filename),
                f"{cohort} {model} baseline predictions ({role_suffix})",
            )
    for filename, (frame, role) in tabular_inputs.items():
        destination = run_dir / "inputs" / filename
        atomic_csv(destination, frame)
        receipts.append(input_receipt(destination, run_dir, role))

    model_receipt_source = REPO_ROOT / "review/model-receipts/submitted-mistral.json"
    model_receipt_destination = run_dir / "inputs/submitted-mistral.json"
    shutil.copy2(model_receipt_source, model_receipt_destination)
    model_receipt_destination.chmod(0o600)
    receipts.append(
        input_receipt(
            model_receipt_destination, run_dir, "pinned submitted model receipt"
        )
    )

    job = {
        "schema_version": SCHEMA_VERSION,
        "study_id": "jbhi-02463-2026-native-reproduction",
        "created_at_utc": utc_now(),
        "repository_revision": git_revision(),
        "privacy_boundary": (
            "The run directory is governed. Input snapshots contain report text; "
            "case-level products contain pseudonymous identifiers."
        ),
        "targets": {"zoe_source": 2000, "maria_source": 500},
        "historical_selections": {
            "zoe_development": [[0, 100]],
            "zoe_evaluation": [[100, 500], [1000, 2000]],
            "maria_evaluation": [[0, 500]],
        },
        "inputs": receipts,
        "error_review_salt": secrets.token_hex(32),
    }
    atomic_json(run_dir / "job.json", job)
    atomic_json(
        run_dir / "state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "initialized",
            "updated_at_utc": utc_now(),
            "current_stage": None,
            "stages": {},
        },
    )
    return job


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    required: tuple[Path, ...]
    progress_csv: Path | None = None
    target_rows: int | None = None


def mappings(flag: str, suffix: str) -> list[str]:
    result: list[str] = []
    for label in LABELS:
        result.extend([flag, f"{label}={label}{suffix}"])
    return result


def stages_for(run_dir: Path) -> list[Stage]:
    py = sys.executable
    review = [py, "-m", "eeg_review.cli"]
    pipeline = [py, str(REPO_ROOT / "src/LLM_pipeline/pipeline.py")]
    process = [py, str(REPO_ROOT / "src/LLM_pipeline/process_output.py")]
    inputs = run_dir / "inputs"
    products = run_dir / "products"
    common_id = ["--id-column", "Hashed_ReportURN"]

    def audit(name: str, database: str, dataset_id: str, complete: bool = True) -> Stage:
        output = products / "audit" / name
        command = [
            *review,
            "audit",
            "--dataset",
            str(inputs / database),
            "--dataset-id",
            dataset_id,
            "--output-dir",
            str(output),
            *common_id,
        ]
        if complete:
            command.append("--require-complete-labels")
        return Stage(name, command, (output / "cohort_audit.json",))

    def evaluate(
        name: str,
        reference: str,
        predictions: Path,
        *,
        prediction_suffix: str = "",
        prediction_table: str = "classifications",
        bootstrap: int = 2000,
    ) -> Stage:
        output = products / "analysis" / name
        command = [
            *review,
            "evaluate",
            "--reference",
            str(inputs / reference),
            "--predictions",
            str(predictions),
            "--prediction-table",
            prediction_table,
            "--output-dir",
            str(output),
            *common_id,
            "--require-complete-reference",
            "--bootstrap-iterations",
            str(bootstrap),
        ]
        if prediction_suffix:
            command.extend(mappings("--prediction-column", prediction_suffix))
        return Stage(
            name,
            command,
            (
                output / "evaluation_summary.json",
                output / "metrics.csv",
                output / "confusion_matrices.json",
                output / "run_manifest.json",
            ),
        )

    def compare(
        name: str,
        reference: str,
        left: Path,
        right: Path,
        left_id: str,
        right_id: str,
        *,
        left_suffix: str = "",
        right_suffix: str = "",
        left_table: str = "classifications",
        right_table: str = "classifications",
    ) -> Stage:
        output = products / "analysis" / name
        command = [
            *review,
            "compare",
            "--reference",
            str(inputs / reference),
            "--predictions-a",
            str(left),
            "--predictions-b",
            str(right),
            "--model-a-id",
            left_id,
            "--model-b-id",
            right_id,
            "--prediction-a-table",
            left_table,
            "--prediction-b-table",
            right_table,
            "--output-dir",
            str(output),
            *common_id,
            "--require-complete-reference",
        ]
        if left_suffix:
            command.extend(mappings("--prediction-a-column", left_suffix))
        if right_suffix:
            command.extend(mappings("--prediction-b-column", right_suffix))
        return Stage(name, command, (output / "paired_comparison_summary.json",))

    def calibrate(name: str, reference: str, predictions: Path, model_id: str) -> Stage:
        output = products / "analysis" / name
        command = [
            *review,
            "calibrate",
            "--reference",
            str(inputs / reference),
            "--predictions",
            str(predictions),
            "--model-id",
            model_id,
            "--output-dir",
            str(output),
            *common_id,
            "--require-complete-reference",
            *mappings("--probability-column", " probability"),
        ]
        return Stage(name, command, (output / "calibration_summary.json",))

    def baseline_cv(model: str) -> Stage:
        name = f"baseline_{model}_cv"
        output = products / "baselines" / model / "development"
        command = [
            *review,
            "baseline-cv",
            "--dataset",
            str(inputs / "zoe_development_100.db"),
            "--model",
            "bag_of_words" if model == "bow" else "bert_base",
            "--output-dir",
            str(output),
            *common_id,
            "--folds",
            "5",
        ]
        if model == "bert":
            command.extend(
                ["--embedding-cache-dir", str(run_dir / "cache/bert/development")]
            )
        return Stage(
            name,
            command,
            (output / "baseline_summary.json", output / "oof_predictions.csv"),
        )

    def baseline_predict(model: str, cohort: str) -> Stage:
        name = f"baseline_{model}_predict_{cohort}"
        output = products / "baselines" / model / cohort
        dataset = "zoe_evaluation_1400.db" if cohort == "zoe" else "maria_evaluation_500.db"
        command = [
            *review,
            "baseline-predict",
            "--dataset",
            str(inputs / dataset),
            "--baseline-dir",
            str(products / "baselines" / model / "development"),
            "--model",
            "bag_of_words" if model == "bow" else "bert_base",
            "--output-dir",
            str(output),
            *common_id,
        ]
        if model == "bert":
            command.extend(["--embedding-cache-dir", str(run_dir / f"cache/bert/{cohort}")])
        return Stage(name, command, (output / "predictions.csv",))

    def baseline_oof(model: str) -> Stage:
        name = f"baseline_{model}_oof_evaluate"
        output = products / "analysis" / name
        command = [
            *review,
            "baseline-oof-evaluate",
            "--dataset",
            str(inputs / "zoe_development_100.db"),
            "--baseline-dir",
            str(products / "baselines" / model / "development"),
            "--model",
            "bag_of_words" if model == "bow" else "bert_base",
            "--output-dir",
            str(output),
            *common_id,
        ]
        return Stage(
            name,
            command,
            (
                output / "evaluation_summary.json",
                output / "metrics.csv",
                output / "fold_metrics.csv",
                output / "run_manifest.json",
            ),
        )

    def llm(cohort: str, target: int) -> Stage:
        name = f"llm_{cohort}_inference"
        output = products / "llm" / cohort
        raw = output / "raw.csv"
        database = "zoe_source_2000.db" if cohort == "zoe" else "maria_source_500.db"
        command = [
            *pipeline,
            "--num-reports",
            str(target),
            "--model",
            "mistral",
            "--dataset-id",
            f"{cohort}-native-reproduction",
            "--dataset-path",
            str(inputs / database),
            "--outdir",
            str(output),
            "--output-csv",
            str(raw),
            "--resume-output",
            "--flush-every",
            "1",
            "--comment",
            "JBHI native resumable reproduction; historical outputs remain authoritative",
        ]
        return Stage(
            name,
            command,
            (raw, raw.with_suffix(".run.json")),
            progress_csv=raw,
            target_rows=target,
        )

    def process_llm(cohort: str) -> Stage:
        name = f"llm_{cohort}_process"
        source = products / "llm" / cohort
        output = source / "processed"
        command = [
            *process,
            "raw.csv",
            "--input-dir",
            str(source),
            "--outdir",
            str(output),
            "--excel-name",
            "predictions.xlsx",
            "--sqlite-name",
            "predictions.db",
        ]
        return Stage(name, command, (output / "predictions.db", output / "predictions.xlsx"))

    def error_review(cohort: str) -> Stage:
        name = f"current_{cohort}_mistral_error_review"
        output = products / "analysis" / f"current_{cohort}_mistral_error_review"
        reference = "zoe_evaluation_1400.db" if cohort == "zoe" else "maria_evaluation_500.db"
        predictions = products / "llm" / cohort / "processed/predictions.db"
        job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
        command = [
            *review,
            "error-review",
            "--reference",
            str(inputs / reference),
            "--predictions",
            str(predictions),
            "--model-id",
            f"current-{cohort}-mistral",
            "--output-dir",
            str(output),
            *common_id,
            "--require-complete-reference",
            "--handle-salt",
            job["error_review_salt"],
            "--acknowledge-governed-output",
        ]
        return Stage(name, command, (output / "clinical_error_review_summary.json",))

    stages: list[Stage] = [
        audit("audit_zoe_development", "zoe_development_100.db", "zoe-development"),
        audit("audit_zoe_evaluation", "zoe_evaluation_1400.db", "zoe-evaluation"),
        audit("audit_maria_evaluation", "maria_evaluation_500.db", "maria-evaluation"),
    ]
    overlap_output = products / "audit/overlap"
    stages.append(
        Stage(
            "audit_cross_cohort_overlap",
            [
                *review,
                "overlap",
                "--dataset",
                f"development={inputs / 'zoe_development_100.db'}",
                "--dataset",
                f"zoe={inputs / 'zoe_evaluation_1400.db'}",
                "--dataset",
                f"maria={inputs / 'maria_evaluation_500.db'}",
                "--output-dir",
                str(overlap_output),
                *common_id,
            ],
            (overlap_output / "overlap_audit.json",),
        )
    )
    for cohort, reference in [
        ("zoe", "zoe_evaluation_1400.db"),
        ("maria", "maria_evaluation_500.db"),
    ]:
        historical_mistral_path = inputs / f"historical_{cohort}_mistral.csv"
        sa_path = inputs / f"{cohort}_second_annotator.csv"
        stages.extend(
            [
                evaluate(
                    f"historical_{cohort}_mistral_evaluate",
                    reference,
                    historical_mistral_path,
                ),
                compare(
                    f"historical_{cohort}_mistral_vs_sa",
                    reference,
                    historical_mistral_path,
                    sa_path,
                    "historical-mistral",
                    "second-annotator",
                ),
            ]
        )
    for model in ["bow", "bert"]:
        path = inputs / f"historical_maria_{model}.csv"
        stages.extend(
            [
                evaluate(
                    f"historical_maria_{model}_evaluate",
                    "maria_evaluation_500.db",
                    path,
                    prediction_suffix=" prediction",
                ),
                calibrate(
                    f"historical_maria_{model}_calibrate",
                    "maria_evaluation_500.db",
                    path,
                    f"historical-maria-{model}",
                ),
            ]
        )

    stages.extend([baseline_cv("bow"), baseline_oof("bow")])
    for cohort, reference in [
        ("zoe", "zoe_evaluation_1400.db"),
        ("maria", "maria_evaluation_500.db"),
    ]:
        stages.append(baseline_predict("bow", cohort))
        current = products / "baselines" / "bow" / cohort / "predictions.csv"
        stages.extend(
            [
                evaluate(
                    f"baseline_bow_evaluate_{cohort}",
                    reference,
                    current,
                    prediction_suffix=" prediction",
                ),
                calibrate(
                    f"baseline_bow_calibrate_{cohort}",
                    reference,
                    current,
                    f"current-{cohort}-bow",
                ),
            ]
        )

    stages.extend([llm("zoe", 2000), process_llm("zoe")])
    zoe_current = products / "llm/zoe/processed/predictions.db"
    stages.extend(
        [
            evaluate(
                "current_zoe_mistral_evaluate",
                "zoe_evaluation_1400.db",
                zoe_current,
            ),
            compare(
                "current_zoe_mistral_vs_historical",
                "zoe_evaluation_1400.db",
                zoe_current,
                inputs / "historical_zoe_mistral.csv",
                "current-mistral",
                "historical-mistral",
            ),
            compare(
                "current_zoe_mistral_vs_sa",
                "zoe_evaluation_1400.db",
                zoe_current,
                inputs / "zoe_second_annotator.csv",
                "current-mistral",
                "second-annotator",
            ),
            error_review("zoe"),
        ]
    )

    stages.extend([llm("maria", 500), process_llm("maria")])
    maria_current = products / "llm/maria/processed/predictions.db"
    stages.extend(
        [
            evaluate(
                "current_maria_mistral_evaluate",
                "maria_evaluation_500.db",
                maria_current,
            ),
            compare(
                "current_maria_mistral_vs_historical",
                "maria_evaluation_500.db",
                maria_current,
                inputs / "historical_maria_mistral.csv",
                "current-mistral",
                "historical-mistral",
            ),
            compare(
                "current_maria_mistral_vs_sa",
                "maria_evaluation_500.db",
                maria_current,
                inputs / "maria_second_annotator.csv",
                "current-mistral",
                "second-annotator",
            ),
            error_review("maria"),
        ]
    )

    stages.extend([baseline_cv("bert"), baseline_oof("bert")])
    for cohort, reference in [
        ("zoe", "zoe_evaluation_1400.db"),
        ("maria", "maria_evaluation_500.db"),
    ]:
        stages.append(baseline_predict("bert", cohort))
        current = products / "baselines" / "bert" / cohort / "predictions.csv"
        stages.extend(
            [
                evaluate(
                    f"baseline_bert_evaluate_{cohort}",
                    reference,
                    current,
                    prediction_suffix=" prediction",
                ),
                calibrate(
                    f"baseline_bert_calibrate_{cohort}",
                    reference,
                    current,
                    f"current-{cohort}-bert",
                ),
            ]
        )
    return stages


def marker_path(run_dir: Path, stage: Stage) -> Path:
    return run_dir / "stages" / f"{stage.name}.done.json"


def stage_is_complete(run_dir: Path, stage: Stage) -> bool:
    marker = marker_path(run_dir, stage)
    if not marker.exists():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected = payload.get("outputs", [])
    receipted_paths = {run_dir / item["path"] for item in expected}
    if not set(stage.required).issubset(receipted_paths):
        return False
    for item in expected:
        path = run_dir / item["path"]
        if not path.exists() or sha256_file(path) != item["sha256"]:
            return False
    return all(path.exists() for path in stage.required)


def csv_progress(path: Path, target_rows: int | None) -> dict[str, Any]:
    if not path.exists():
        return {"rows": 0, "mean_seconds_per_report": None, "eta_seconds": None}
    total_seconds = 0.0
    rows = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            try:
                total_seconds += float(row.get("classify_elapsed_seconds") or 0)
                total_seconds += float(row.get("explain_elapsed_seconds") or 0)
            except ValueError:
                pass
    mean = total_seconds / rows if rows and total_seconds else None
    eta = max((target_rows or rows) - rows, 0) * mean if mean is not None else None
    return {"rows": rows, "mean_seconds_per_report": mean, "eta_seconds": eta}


class Supervisor:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.child: subprocess.Popen[str] | None = None
        self.interrupted = False

    def handle_signal(self, signum: int, _frame: Any) -> None:
        self.interrupted = True
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signum)

    def update_state(self, **changes: Any) -> dict[str, Any]:
        state_path = self.run_dir / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update(changes)
        state["updated_at_utc"] = utc_now()
        atomic_json(state_path, state)
        return state

    def run(self, stop_after: str | None = None) -> None:
        lock_path = self.run_dir / "run.lock"
        with lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another supervisor already holds this run") from error
            signal.signal(signal.SIGTERM, self.handle_signal)
            signal.signal(signal.SIGINT, self.handle_signal)
            atomic_json(
                self.run_dir / "supervisor.json",
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "python": sys.version.split()[0],
                    "repository_revision_at_start": git_revision(),
                    "started_at_utc": utc_now(),
                },
            )
            self.update_state(status="running", current_stage=None)
            for stage in stages_for(self.run_dir):
                if stage_is_complete(self.run_dir, stage):
                    continue
                self.update_state(status="running", current_stage=stage.name)
                state = json.loads((self.run_dir / "state.json").read_text(encoding="utf-8"))
                stage_state = state.setdefault("stages", {}).setdefault(stage.name, {})
                stage_state.update(
                    status="running",
                    started_at_utc=utc_now(),
                    command=stage.command,
                    repository_revision_at_start=git_revision(),
                )
                atomic_json(self.run_dir / "state.json", state)
                log_path = self.run_dir / "logs" / f"{stage.name}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n[{utc_now()}] START {' '.join(stage.command)}\n")
                    log.flush()
                    self.child = subprocess.Popen(
                        stage.command,
                        cwd=REPO_ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )
                    returncode = self.child.wait()
                    self.child = None
                    log.write(f"[{utc_now()}] EXIT {returncode}\n")
                if self.interrupted:
                    self.update_state(status="interrupted", current_stage=stage.name)
                    return
                if returncode != 0:
                    state = json.loads(
                        (self.run_dir / "state.json").read_text(encoding="utf-8")
                    )
                    state["status"] = "failed"
                    state["current_stage"] = stage.name
                    state.setdefault("stages", {}).setdefault(stage.name, {})["status"] = "failed"
                    state["stages"][stage.name]["returncode"] = returncode
                    state["updated_at_utc"] = utc_now()
                    atomic_json(self.run_dir / "state.json", state)
                    raise RuntimeError(f"Stage failed: {stage.name}; inspect {log_path}")
                missing = [str(path) for path in stage.required if not path.exists()]
                if missing:
                    raise RuntimeError(f"Stage {stage.name} did not produce: {missing}")
                outputs = [
                    {
                        "path": str(path.relative_to(self.run_dir)),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in stage.required
                ]
                atomic_json(
                    marker_path(self.run_dir, stage),
                    {
                        "schema_version": 1,
                        "stage": stage.name,
                        "completed_at_utc": utc_now(),
                        "outputs": outputs,
                    },
                )
                state = json.loads((self.run_dir / "state.json").read_text(encoding="utf-8"))
                state.setdefault("stages", {}).setdefault(stage.name, {}).update(
                    status="completed", completed_at_utc=utc_now(), outputs=outputs
                )
                state["updated_at_utc"] = utc_now()
                atomic_json(self.run_dir / "state.json", state)
                if stop_after == stage.name:
                    self.update_state(status="paused", current_stage=None)
                    return
            self.update_state(status="completed", current_stage=None, completed_at_utc=utc_now())
            write_transfer_manifest(self.run_dir)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def status_payload(run_dir: Path) -> dict[str, Any]:
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    supervisor_path = run_dir / "supervisor.json"
    supervisor = (
        json.loads(supervisor_path.read_text(encoding="utf-8"))
        if supervisor_path.exists()
        else None
    )
    if supervisor and supervisor.get("hostname") == socket.gethostname():
        supervisor["alive"] = process_alive(int(supervisor["pid"]))
    stage_rows = []
    for stage in stages_for(run_dir):
        progress = (
            csv_progress(stage.progress_csv, stage.target_rows)
            if stage.progress_csv
            else {"rows": None, "mean_seconds_per_report": None, "eta_seconds": None}
        )
        rows = progress["rows"]
        stage_rows.append(
            {
                "name": stage.name,
                "complete": stage_is_complete(run_dir, stage),
                "rows": rows,
                "target_rows": stage.target_rows,
                "percent": round(rows / stage.target_rows * 100, 2)
                if rows is not None and stage.target_rows
                else None,
                "mean_seconds_per_report": progress["mean_seconds_per_report"],
                "eta_seconds": progress["eta_seconds"],
            }
        )
    return {"state": state, "supervisor": supervisor, "stages": stage_rows}


def print_status(run_dir: Path, as_json: bool) -> None:
    payload = status_payload(run_dir)
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    state = payload["state"]
    supervisor = payload["supervisor"] or {}
    print(f"status: {state['status']}")
    print(f"current stage: {state.get('current_stage') or '-'}")
    print(
        f"supervisor: pid={supervisor.get('pid', '-')} "
        f"alive={supervisor.get('alive', 'unknown')} host={supervisor.get('hostname', '-')}"
    )
    completed = sum(1 for stage in payload["stages"] if stage["complete"])
    print(f"stages: {completed}/{len(payload['stages'])} complete")
    for stage in payload["stages"]:
        if stage["name"] == state.get("current_stage") or stage["rows"]:
            progress = ""
            if stage["target_rows"]:
                progress = (
                    f" {stage['rows']}/{stage['target_rows']} ({stage['percent']:.2f}%)"
                )
                if stage["eta_seconds"] is not None:
                    progress += f", ETA {stage['eta_seconds'] / 3600:.2f}h"
            print(f"- {stage['name']}:{progress}")


def stop_run(run_dir: Path, timeout: int) -> None:
    supervisor_path = run_dir / "supervisor.json"
    if not supervisor_path.exists():
        raise FileNotFoundError("No supervisor receipt exists")
    supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
    if supervisor.get("hostname") != socket.gethostname():
        raise RuntimeError("Supervisor belongs to another host; stop it on that machine")
    pid = int(supervisor["pid"])
    if not process_alive(pid):
        print("Supervisor is not running")
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while process_alive(pid) and time.time() < deadline:
        time.sleep(0.25)
    if process_alive(pid):
        raise TimeoutError(f"Supervisor {pid} did not stop within {timeout}s")
    print(f"Stopped supervisor {pid}; partial atomic checkpoints are resumable")


def launch_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve(strict=True)
    supervisor_path = run_dir / "supervisor.json"
    if supervisor_path.exists():
        supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
        if (
            supervisor.get("hostname") == socket.gethostname()
            and process_alive(int(supervisor.get("pid", -1)))
        ):
            raise RuntimeError(f"Supervisor {supervisor['pid']} is already running")
    command = [sys.executable, str(Path(__file__).resolve()), "run", "--run-dir", str(run_dir)]
    if platform.system() == "Darwin" and Path("/usr/bin/caffeinate").exists():
        command = ["/usr/bin/caffeinate", "-dimsu", *command]
    log_path = run_dir / "logs/supervisor.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    receipt = {
        "launcher_pid": process.pid,
        "hostname": socket.gethostname(),
        "command": command,
        "launched_at_utc": utc_now(),
        "log": str(log_path.relative_to(run_dir)),
    }
    atomic_json(run_dir / "launcher.json", receipt)
    time.sleep(1)
    if process.poll() is not None:
        raise RuntimeError(f"Detached supervisor exited immediately; inspect {log_path}")
    return receipt


def sensitivity(path: Path, run_dir: Path) -> str:
    relative = path.relative_to(run_dir)
    if relative.parts[0] == "inputs":
        return "governed_report_data_or_case_level_input"
    if relative.parts[0] in {"products", "cache"}:
        return "governed_case_level_or_derived_product"
    return "operational_metadata"


def write_transfer_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "transfer-manifest.json"
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path in {manifest_path, run_dir / "run.lock"}:
            continue
        files.append(
            {
                "path": str(path.relative_to(run_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "sensitivity": sensitivity(path, run_dir),
            }
        )
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    generation_revision = git_revision()
    payload = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "study_id": "jbhi-02463-2026-native-reproduction",
        "repository_revision": job.get("repository_revision"),
        "job_repository_revision": job.get("repository_revision"),
        "manifest_generation_repository_revision": generation_revision,
        "transfer_rule": (
            "Transfer only through an approved governed channel. Preserve relative paths; "
            "verify every SHA-256 before resuming on another machine."
        ),
        "files": files,
    }
    atomic_json(manifest_path, payload)
    print(f"Wrote {manifest_path} with {len(files)} file receipts")
    return payload


def compare_raw(left: Path, right: Path) -> dict[str, Any]:
    keys = [
        "focal_epileptiform_activity",
        "generalized_epileptiform_activity",
        "focal_non_epileptiform_activity",
        "generalized_non_epileptiform_activity",
        "abnormality",
    ]

    def load(path: Path) -> dict[str, dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as stream:
            return {row["Hashed_ReportURN"]: row for row in csv.DictReader(stream)}

    left_rows = load(left)
    right_rows = load(right)
    shared = sorted(set(left_rows) & set(right_rows))
    exact = core = reasons = 0
    total = len(shared) * len(keys)
    for identifier in shared:
        left_class = json.loads(left_rows[identifier]["classifications"])
        right_class = json.loads(right_rows[identifier]["classifications"])
        left_explain = json.loads(left_rows[identifier]["explanations"])
        right_explain = json.loads(right_rows[identifier]["explanations"])
        for key in keys:
            left_value = int(left_class[key])
            right_value = int(right_class[key])
            exact += left_value == right_value
            core += (left_value >= 3) == (right_value >= 3)
            reasons += left_explain[key].get("reasons") == right_explain[key].get("reasons")
    return {
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "shared_rows": len(shared),
        "classification_cells": total,
        "exact_four_level_matches": exact,
        "core_matches": core,
        "exact_reason_list_matches": reasons,
    }


def compare_runs(left: Path, right: Path, output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "left": str(left),
        "right": str(right),
        "cohorts": {},
    }
    for cohort in ["zoe", "maria"]:
        left_raw = left / f"products/llm/{cohort}/raw.csv"
        right_raw = right / f"products/llm/{cohort}/raw.csv"
        if left_raw.exists() and right_raw.exists():
            result["cohorts"][cohort] = compare_raw(left_raw, right_raw)
    atomic_json(output, result)
    return result


def write_result_ledger(run_dir: Path) -> dict[str, Any]:
    from eeg_review.ledger import build_result_ledger

    analysis = run_dir / "products/analysis"

    def discover(filename: str) -> dict[str, Path]:
        return {
            path.parent.name: path
            for path in sorted(analysis.glob(f"*/{filename}"))
        }

    return build_result_ledger(
        run_dir / "products/aggregate-ledger",
        evaluations=discover("evaluation_summary.json"),
        calibrations=discover("calibration_summary.json"),
        comparisons=discover("paired_comparison_summary.json"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--run-dir", type=Path, required=True)
    initialize.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "data/governed/proton-2026-07-20",
    )
    run = commands.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--stop-after")
    launch = commands.add_parser("launch")
    launch.add_argument("--run-dir", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--json", action="store_true")
    stop = commands.add_parser("stop")
    stop.add_argument("--run-dir", type=Path, required=True)
    stop.add_argument("--timeout", type=int, default=30)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--run-dir", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    ledger = commands.add_parser("ledger")
    ledger.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        result = initialize_run(args.run_dir, args.source_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "run":
        Supervisor(args.run_dir.expanduser().resolve(strict=True)).run(args.stop_after)
    elif args.command == "launch":
        result = launch_run(args.run_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "status":
        print_status(args.run_dir.expanduser().resolve(strict=True), args.json)
    elif args.command == "stop":
        stop_run(args.run_dir.expanduser().resolve(strict=True), args.timeout)
    elif args.command == "manifest":
        write_transfer_manifest(args.run_dir.expanduser().resolve(strict=True))
    elif args.command == "compare":
        result = compare_runs(
            args.left.expanduser().resolve(strict=True),
            args.right.expanduser().resolve(strict=True),
            args.output.expanduser().resolve(),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        result = write_result_ledger(args.run_dir.expanduser().resolve(strict=True))
        print(json.dumps(result["row_counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
