#!/usr/bin/env python3
"""Prepare governed, exact-key MedGemma study inputs without running inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    path.chmod(0o600)


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def complete_predicate() -> str:
    return " AND ".join(
        f'CAST("{label}" AS INTEGER) IN (1, 2, 3, 4)' for label in LABELS
    )


def prepare_database(source: Path, destination: Path) -> list[str]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with sqlite3.connect(source) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(reports)")]
        if "Hashed_ReportURN" not in columns:
            raise ValueError(f"Missing Hashed_ReportURN in {source}")
        rows = connection.execute(
            f'SELECT * FROM "reports" WHERE {complete_predicate()} ORDER BY rowid'
        ).fetchall()
        keys = [str(row[columns.index("Hashed_ReportURN")]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate report keys in {source}")
        quoted = ", ".join(f'"{column.replace(chr(34), chr(34) * 2)}"' for column in columns)
        placeholders = ", ".join("?" for _ in columns)
    with sqlite3.connect(temporary) as output:
        with sqlite3.connect(source) as source_connection:
            schema = source_connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='reports'"
            ).fetchone()[0]
        output.execute(schema)
        output.executemany(
            f'INSERT INTO "reports" ({quoted}) VALUES ({placeholders})',
            rows,
        )
    temporary.replace(destination)
    destination.chmod(0o600)
    return keys


def write_manifest(path: Path, keys: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "Hashed_ReportURN"])
        writer.writerows((index, key) for index, key in enumerate(keys))
    temporary.replace(path)
    path.chmod(0o600)


def read_prediction_surface(path: Path, table: str | None = None) -> pd.DataFrame:
    if table is None:
        frame = pd.read_csv(path)
    else:
        with sqlite3.connect(path) as connection:
            frame = pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
    if "Hashed_ReportURN" not in frame.columns:
        raise ValueError(f"Missing Hashed_ReportURN in {path}")
    if frame["Hashed_ReportURN"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate prediction keys in {path}")
    return frame


def select_prediction_surface(frame: pd.DataFrame, keys: list[str], destination: Path) -> None:
    keyed = frame.assign(Hashed_ReportURN=frame["Hashed_ReportURN"].astype(str)).set_index(
        "Hashed_ReportURN", drop=False
    )
    missing = [key for key in keys if key not in keyed.index]
    if missing:
        raise ValueError(f"Prediction surface is missing {len(missing)} governed report keys")
    selected = keyed.loc[keys].reset_index(drop=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    selected.to_csv(temporary, index=False)
    temporary.replace(destination)
    destination.chmod(0o600)


def command_plan(run_dir: Path, cohorts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    py = "python"
    commands: list[dict[str, Any]] = []
    for cohort in cohorts:
        cohort_id = cohort["cohort_id"]
        records = cohort["records"]
        database = run_dir / cohort["database"]
        raw = run_dir / f"products/{cohort_id}/raw.csv"
        processed = run_dir / f"products/{cohort_id}/processed"
        commands.extend(
            [
                {
                    "stage": f"{cohort_id}_inference",
                    "command": [
                        py,
                        "src/LLM_pipeline/pipeline.py",
                        "--num-reports",
                        str(records),
                        "--model",
                        "medgemma-27b-q2-candidate",
                        "--dataset-id",
                        cohort_id,
                        "--dataset-path",
                        str(database),
                        "--outdir",
                        str(raw.parent),
                        "--output-csv",
                        str(raw),
                        "--resume-output",
                        "--flush-every",
                        "1",
                        "--classification-only",
                        "--temperature",
                        "0",
                        "--top-k",
                        "40",
                        "--top-p",
                        "0.95",
                        "--max-tokens",
                        "256",
                        "--comment",
                        "JBHI additive independent MedGemma matched-interface comparator",
                    ],
                },
                {
                    "stage": f"{cohort_id}_process",
                    "command": [
                        py,
                        "src/LLM_pipeline/process_output.py",
                        "raw.csv",
                        "--input-dir",
                        str(raw.parent),
                        "--outdir",
                        str(processed),
                        "--excel-name",
                        "predictions.xlsx",
                        "--sqlite-name",
                        "predictions.db",
                    ],
                },
            ]
        )
        if cohort["role"] == "evaluation":
            commands.extend(
                [
                    {
                        "stage": f"{cohort_id}_evaluate",
                        "command": [
                            py,
                            "-m",
                            "eeg_review.cli",
                            "evaluate",
                            "--reference",
                            str(database),
                            "--predictions",
                            str(processed / "predictions.db"),
                            "--output-dir",
                            str(run_dir / f"analysis/{cohort_id}/medgemma"),
                            "--require-complete-reference",
                            "--require-exact-key-set",
                            "--bootstrap-iterations",
                            "2000",
                            "--seed",
                            "20260718",
                        ],
                    },
                    {
                        "stage": f"{cohort_id}_compare_submitted",
                        "command": [
                            py,
                            "-m",
                            "eeg_review.cli",
                            "compare",
                            "--reference",
                            str(database),
                            "--predictions-a",
                            str(processed / "predictions.db"),
                            "--predictions-b",
                            str(run_dir / f"comparators/{cohort_id}_submitted_mistral.csv"),
                            "--model-a-id",
                            "medgemma-independent-matched-interface-q2-v1",
                            "--model-b-id",
                            "submitted-mistral",
                            "--output-dir",
                            str(run_dir / f"analysis/{cohort_id}/vs_submitted_mistral"),
                            "--require-complete-reference",
                            "--require-exact-key-set",
                            "--bootstrap-iterations",
                            "2000",
                            "--seed",
                            "20260718",
                            "--multiplicity",
                            "holm",
                        ],
                    },
                    {
                        "stage": f"{cohort_id}_compare_reproduced",
                        "command": [
                            py,
                            "-m",
                            "eeg_review.cli",
                            "compare",
                            "--reference",
                            str(database),
                            "--predictions-a",
                            str(processed / "predictions.db"),
                            "--predictions-b",
                            str(run_dir / f"comparators/{cohort_id}_reproduced_mistral.csv"),
                            "--model-a-id",
                            "medgemma-independent-matched-interface-q2-v1",
                            "--model-b-id",
                            "reproduced-mistral",
                            "--output-dir",
                            str(run_dir / f"analysis/{cohort_id}/vs_reproduced_mistral"),
                            "--require-complete-reference",
                            "--require-exact-key-set",
                            "--bootstrap-iterations",
                            "2000",
                            "--seed",
                            "20260718",
                            "--multiplicity",
                            "holm",
                        ],
                    },
                    {
                        "stage": f"{cohort_id}_compare_second_annotator",
                        "command": [
                            py,
                            "-m",
                            "eeg_review.cli",
                            "compare",
                            "--reference",
                            str(database),
                            "--predictions-a",
                            str(processed / "predictions.db"),
                            "--predictions-b",
                            str(run_dir / f"comparators/{cohort_id}_second_annotator.csv"),
                            "--model-a-id",
                            "medgemma-independent-matched-interface-q2-v1",
                            "--model-b-id",
                            "second-annotator",
                            "--output-dir",
                            str(run_dir / f"analysis/{cohort_id}/vs_second_annotator"),
                            "--require-complete-reference",
                            "--require-exact-key-set",
                            "--bootstrap-iterations",
                            "2000",
                            "--seed",
                            "20260718",
                            "--multiplicity",
                            "holm",
                        ],
                    },
                ]
            )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-governed-output", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_governed_output:
        parser.error("--acknowledge-governed-output is required")

    plan_path = args.plan.expanduser().resolve(strict=True)
    source_run = args.source_run.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output}")
    os.umask(0o077)
    for relative in ["inputs", "manifests", "comparators", "products", "analysis", "logs"]:
        (output / relative).mkdir(parents=True, exist_ok=True, mode=0o700)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    shutil.copy2(plan_path, output / "study-plan.json")
    (output / "study-plan.json").chmod(0o600)
    prepared: list[dict[str, Any]] = []
    for cohort in plan["cohorts"]:
        cohort_id = cohort["cohort_id"]
        source = source_run / cohort["input"]["path"]
        destination = output / "inputs" / f"{cohort_id}.db"
        keys = prepare_database(source, destination)
        expected = cohort["population"]["execute_records"]
        if len(keys) != expected:
            raise ValueError(f"{cohort_id}: selected {len(keys)} rows, expected {expected}")
        manifest = output / "manifests" / f"{cohort_id}.csv"
        write_manifest(manifest, keys)
        prepared.append(
            {
                "cohort_id": cohort_id,
                "role": cohort["role"],
                "records": len(keys),
                "database": str(destination.relative_to(output)),
                "database_sha256": sha256_file(destination),
                "manifest": str(manifest.relative_to(output)),
                "manifest_sha256": sha256_file(manifest),
            }
        )

        if cohort["role"] != "evaluation":
            continue
        prefix = "zoe" if cohort_id.startswith("zoe_") else "maria"
        surfaces = {
            "submitted_mistral": read_prediction_surface(
                source_run / f"inputs/historical_{prefix}_mistral.csv"
            ),
            "reproduced_mistral": read_prediction_surface(
                source_run / f"products/llm/{prefix}/processed/predictions.db",
                "classifications",
            ),
            "second_annotator": read_prediction_surface(
                source_run / f"inputs/{prefix}_second_annotator.csv"
            ),
        }
        for name, frame in surfaces.items():
            select_prediction_surface(
                frame,
                keys,
                output / "comparators" / f"{cohort_id}_{name}.csv",
            )

    job = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "study_id": plan["study_id"],
        "configuration_id": plan["independent_configuration"]["configuration_id"],
        "repository_revision_at_preparation": git_revision(),
        "source_run": {
            "study_id": json.loads((source_run / "job.json").read_text(encoding="utf-8"))[
                "study_id"
            ],
            "job_sha256": sha256_file(source_run / "job.json"),
            "transfer_manifest_sha256": sha256_file(source_run / "transfer-manifest.json"),
        },
        "cohorts": prepared,
        "commands": command_plan(output, prepared),
        "status": "prepared_no_inference",
        "privacy_boundary": (
            "This run directory is governed. Inputs, manifests, and comparator surfaces contain "
            "report-level data or pseudonymous keys."
        ),
    }
    atomic_json(output / "job.json", job)
    atomic_json(
        output / "state.json",
        {
            "schema_version": 1,
            "status": "prepared_no_inference",
            "current_stage": None,
            "completed_stages": [],
        },
    )
    print(json.dumps({"prepared": True, "cohorts": prepared}, indent=2))


if __name__ == "__main__":
    main()
