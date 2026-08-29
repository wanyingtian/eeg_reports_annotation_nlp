#!/usr/bin/env python3
"""Benchmark a preregistered MedGemma runtime without reading reference outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "src/LLM_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

import llm_models  # noqa: E402
import pipeline  # noqa: E402


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    path.chmod(0o600)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_revision() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"revision": revision, "worktree_dirty": dirty}


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()


def system_snapshot() -> dict[str, Any]:
    return {
        "memory_pressure": command_output(["memory_pressure", "-Q"]),
        "thermal": command_output(["pmset", "-g", "therm"]),
        "power": command_output(["pmset", "-g", "batt"]),
    }


def read_rows(path: Path, count: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < count:
        raise ValueError(f"{path.name}: expected at least {count} rows, found {len(rows)}")
    return rows[:count]


def parsed_classification(row: dict[str, str]) -> dict[str, int]:
    value = json.loads(row["classifications"])
    if set(value) != set(pipeline.JSON_KEY_TO_LABEL):
        raise ValueError("classification keys do not match the frozen five-label interface")
    parsed = {key: int(item) for key, item in value.items()}
    if any(item not in {1, 2, 3, 4} for item in parsed.values()):
        raise ValueError("classification contains a value outside levels 1--4")
    return parsed


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    elapsed = [float(row["classify_elapsed_seconds"]) for row in rows]
    keys = [row["Hashed_ReportURN"] for row in rows]
    valid = 0
    for row in rows:
        try:
            parsed_classification(row)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        valid += 1
    return {
        "records": len(rows),
        "unique_keys": len(set(keys)),
        "missing_keys": sum(not key for key in keys),
        "duplicate_keys": len(keys) - len(set(keys)),
        "valid_structured_outputs": valid,
        "invalid_structured_outputs": len(rows) - valid,
        "latency_seconds": {
            "mean": statistics.mean(elapsed),
            "median": statistics.median(elapsed),
            "minimum": min(elapsed),
            "maximum": max(elapsed),
        },
        "reports_per_hour": 3600 / statistics.mean(elapsed),
        "prompt_tokens_total": sum(int(float(row["classify_prompt_tokens"])) for row in rows),
        "completion_tokens_total": sum(
            int(float(row["classify_completion_tokens"])) for row in rows
        ),
    }


def validate_plan(plan: dict[str, Any], run_dir: Path, records: int) -> None:
    if plan["status"] != "preregistered_before_optimization_benchmark":
        raise ValueError("Runtime benchmark plan is not preregistered")
    if records != int(plan["promotion_gates"]["minimum_records"]):
        raise ValueError("Benchmark record count differs from the preregistered count")
    if read_json(run_dir / "state.json")["status"] != "stopped":
        raise ValueError("The producing run must remain stopped during the benchmark")
    invariants = plan["invariants"]
    if sha256_text(pipeline.PROMPT_CLASSIFY) != invariants["prompt_sha256"]:
        raise ValueError("Classification prompt hash differs from the benchmark plan")
    grammar = PIPELINE_ROOT / "result_grammar.gbnf"
    if sha256_file(grammar) != invariants["grammar_sha256"]:
        raise ValueError("Classification grammar hash differs from the benchmark plan")
    model = llm_models.MODEL_CONFIGS["medgemma-27b-q2-candidate"]
    if model["sha256"] != invariants["model_sha256"]:
        raise ValueError("Pinned model hash differs from the benchmark plan")


def candidate_by_id(plan: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    candidates = {
        item["runtime_profile_id"]: item for item in plan["candidates"]
    }
    if candidate_id not in candidates:
        raise ValueError(f"Candidate is not preregistered: {candidate_id}")
    return candidates[candidate_id]


def parse_maximum_rss(stderr: str) -> int | None:
    for line in stderr.splitlines():
        if "maximum resident set size" in line:
            try:
                return int(line.strip().split()[0])
            except (IndexError, ValueError):
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--records", type=int, default=12)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve(strict=True)
    plan_path = args.plan.expanduser().resolve(strict=True)
    plan = read_json(plan_path)
    validate_plan(plan, run_dir, args.records)
    candidate = candidate_by_id(plan, args.candidate)
    repository = git_revision()
    if repository["worktree_dirty"]:
        raise RuntimeError("Commit the runtime benchmark implementation before execution")

    benchmark_dir = run_dir / "maintenance/runtime-optimization" / args.candidate
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output = benchmark_dir / "raw.csv"
    receipt_path = benchmark_dir / "benchmark-receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"Immutable benchmark receipt already exists: {receipt_path}")

    baseline_path = run_dir / "products/zoe_development_transport_100/raw.csv"
    baseline_rows = read_rows(baseline_path, args.records)
    parameters = candidate["parameters"]
    command = [
        sys.executable,
        str(PIPELINE_ROOT / "pipeline.py"),
        "--num-reports",
        str(args.records),
        "--model",
        "medgemma-27b-q2-candidate",
        "--dataset-id",
        "zoe_development_runtime_benchmark_12",
        "--dataset-path",
        str(run_dir / "inputs/zoe_development_transport_100.db"),
        "--outdir",
        str(benchmark_dir),
        "--output-csv",
        str(output),
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
        "--runtime-profile-id",
        args.candidate,
        "--n-ctx",
        str(parameters["n_ctx"]),
        "--n-gpu-layers",
        str(parameters["n_gpu_layers"]),
        "--n-batch",
        str(parameters["n_batch"]),
        "--n-ubatch",
        str(parameters["n_ubatch"]),
        "--n-threads",
        str(parameters["n_threads"]),
        "--n-threads-batch",
        str(parameters["n_threads_batch"]),
        "--flash-attn" if parameters["flash_attn"] else "--no-flash-attn",
        "--comment",
        "Result-blind MedGemma Metal runtime optimization benchmark",
    ]
    log_path = benchmark_dir / "benchmark.log"
    started_at = utc_now()
    system_before = system_snapshot()
    completed = subprocess.run(
        ["/usr/bin/time", "-lp", *command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    system_after = system_snapshot()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "plan_sha256": sha256_file(plan_path),
        "candidate": candidate,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "repository": repository,
        "command": command,
        "exit_code": completed.returncode,
        "maximum_resident_set_size_bytes": parse_maximum_rss(completed.stderr),
        "system_before": system_before,
        "system_after": system_after,
        "log_sha256": sha256_file(log_path),
        "reference_outcomes_accessed": False,
        "partial_performance_metrics_computed": False,
        "manuscript_claim_available": False,
    }
    if completed.returncode != 0:
        payload["status"] = "failed_operational_benchmark"
        atomic_json(receipt_path, payload)
        raise SystemExit(completed.returncode)

    candidate_rows = read_rows(output, args.records)
    baseline = metrics(baseline_rows)
    observed = metrics(candidate_rows)
    paired = 0
    exact = 0
    baseline_by_key = {row["Hashed_ReportURN"]: row for row in baseline_rows}
    for row in candidate_rows:
        reference_row = baseline_by_key.get(row["Hashed_ReportURN"])
        if reference_row is None:
            continue
        paired += 1
        if parsed_classification(row) == parsed_classification(reference_row):
            exact += 1
    speedup = baseline["latency_seconds"]["mean"] / observed["latency_seconds"]["mean"]
    equivalence = exact / paired if paired else 0.0
    gates = plan["promotion_gates"]
    gate_results = {
        "record_count": len(candidate_rows) == int(gates["minimum_records"]),
        "structural_validity": (
            observed["valid_structured_outputs"] / len(candidate_rows)
            == float(gates["valid_structured_output_fraction"])
        ),
        "key_integrity": observed["missing_keys"] + observed["duplicate_keys"] == 0,
        "exact_classification_equivalence": (
            equivalence >= float(gates["exact_classification_equivalence_fraction"])
        ),
        "throughput_speedup": speedup >= float(gates["minimum_throughput_speedup"]),
        "no_thermal_or_performance_warning": (
            "No thermal warning level has been recorded" in system_after["thermal"]
            and "No performance warning level has been recorded" in system_after["thermal"]
        ),
    }
    run_receipt = output.with_suffix(".run.json")
    payload.update(
        {
            "status": "completed_result_blind_benchmark",
            "baseline": baseline,
            "observed": observed,
            "speedup": speedup,
            "paired_records": paired,
            "exact_classification_matches": exact,
            "exact_classification_equivalence_fraction": equivalence,
            "gate_results": gate_results,
            "promotion_gate_passed": all(gate_results.values()),
            "output_sha256": sha256_file(output),
            "pipeline_run_receipt_sha256": sha256_file(run_receipt),
            "pipeline_run_receipt": read_json(run_receipt),
            "interpretation": (
                "This benchmark compares runtime and model-output equivalence only. It does "
                "not read human reference outcomes or estimate clinical performance."
            ),
        }
    )
    atomic_json(receipt_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
