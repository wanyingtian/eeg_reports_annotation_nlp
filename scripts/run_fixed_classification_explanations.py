#!/usr/bin/env python3
"""Run grammar-constrained evidence extraction from immutable classifications."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "src/LLM_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

import pipeline  # noqa: E402

from eeg_review.evidence_extraction import (  # noqa: E402
    aggregate_inspections,
    inspect_explanation,
    load_fixed_evidence_inputs,
)
from eeg_review.io import atomic_write_csv, atomic_write_json  # noqa: E402
from eeg_review.native_interface import (  # noqa: E402
    NATIVE_CHAT_INTERFACE_MODE,
    RAW_COMPLETION_INTERFACE_MODE,
    embedded_chat_template_receipt,
    explanation_input,
    explanation_task_message_template,
    native_explanation_messages,
    sha256_text,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str, name: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, found {observed}")
    return observed


def git_receipt() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "worktree_dirty": dirty}


def completed_prefix(output: Path, expected_keys: list[str], interface: str) -> pd.DataFrame:
    if not output.exists():
        return pd.DataFrame()
    frame = pd.read_csv(output)
    required = {
        "Hashed_ReportURN",
        "explanation_interface_mode",
        "fixed_classifications",
        "explanations",
    }
    if not required.issubset(frame.columns):
        raise ValueError("resumed output lacks fixed-evidence identity columns")
    keys = frame["Hashed_ReportURN"].astype(str).tolist()
    if keys != expected_keys[: len(keys)]:
        raise ValueError("resumed output is not an exact prefix of the frozen manifest")
    interfaces = set(frame["explanation_interface_mode"].dropna().astype(str))
    if interfaces != {interface}:
        raise ValueError("resumed output mixes explanation interfaces")
    return frame


def bind_execution_contract(path: Path, contract: dict[str, Any], output_exists: bool) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ValueError("fixed-evidence execution contract changed; refuse mixed resume")
    elif output_exists:
        raise ValueError("partial evidence output lacks its pre-inference execution contract")
    else:
        atomic_write_json(path, contract)
        path.chmod(0o600)


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.umask(0o077)
    dataset = args.dataset.expanduser().resolve(strict=True)
    predictions = args.predictions.expanduser().resolve(strict=True)
    manifest = args.manifest.expanduser().resolve(strict=True)
    output = args.output_csv.expanduser().resolve()
    receipt_path = output.with_suffix(".run.json")
    if output.exists() and not args.resume:
        raise FileExistsError(f"output exists; use --resume: {output}")

    dataset_sha = verify_hash(dataset, args.expected_dataset_sha256, "dataset")
    predictions_sha = verify_hash(
        predictions, args.expected_predictions_sha256, "fixed predictions"
    )
    manifest_sha = verify_hash(manifest, args.expected_manifest_sha256, "manifest")
    fixed = load_fixed_evidence_inputs(
        dataset=dataset,
        predictions=predictions,
        manifest=manifest,
        table=args.table,
        id_column=args.id_column,
        report_column=args.report_column,
        classification_column=args.classification_column,
    )
    if args.expected_records is not None and len(fixed) != args.expected_records:
        raise ValueError(
            f"manifest population mismatch: expected {args.expected_records}, found {len(fixed)}"
        )

    existing = completed_prefix(
        output,
        fixed[args.id_column].astype(str).tolist(),
        args.interface,
    )
    completed = len(existing)
    for index, row in existing.iterrows():
        if row["fixed_classifications"] != fixed.iloc[index][args.classification_column]:
            raise ValueError("resumed fixed classification differs from the frozen source")

    load_overrides = {
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "n_batch": args.n_batch,
        "n_ubatch": args.n_ubatch,
        "n_threads": args.n_threads,
        "n_threads_batch": args.n_threads_batch,
        "flash_attn": args.flash_attn,
    }
    grammar_path = PIPELINE_ROOT / "result_grammar_exp.gbnf"
    grammar_sha = sha256_file(grammar_path)
    if grammar_sha != args.expected_grammar_sha256:
        raise ValueError("explanation grammar SHA-256 does not match the frozen plan")
    prompt_sha = sha256_text(pipeline.PROMPT_EXPLAIN)
    if prompt_sha != args.expected_prompt_sha256:
        raise ValueError("explanation prompt SHA-256 does not match the frozen plan")
    from llm_models import MODEL_CONFIGS

    contract = {
        "run_id": args.run_id,
        "dataset_sha256": dataset_sha,
        "predictions_sha256": predictions_sha,
        "manifest_sha256": manifest_sha,
        "model": MODEL_CONFIGS[args.model],
        "load_overrides": load_overrides,
        "interface": args.interface,
        "chat_template_sha256": args.expected_chat_template_sha256,
        "prompt_sha256": prompt_sha,
        "grammar_sha256": grammar_sha,
        "sampling": [args.temperature, args.top_k, args.top_p, args.max_tokens],
        "records": len(fixed),
        "columns": [args.table, args.id_column, args.report_column, args.classification_column],
        "source_revision": git_receipt(),
    }
    bind_execution_contract(output.with_suffix(".execution.json"), contract, output.exists())
    if completed == len(fixed) and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["output"]["sha256"] != sha256_file(output):
            raise ValueError("completed evidence output checksum changed")
        return receipt  # A completed resume must not load the model again.
    model, model_receipt = pipeline.download_model_with_receipt(
        args.model,
        load_overrides=load_overrides,
        local_files_only=True,
    )
    grammar = pipeline.load_gbnf(grammar_path)
    template_receipt = (
        embedded_chat_template_receipt(model)
        if args.interface == NATIVE_CHAT_INTERFACE_MODE
        else None
    )
    if template_receipt and template_receipt["sha256"] != args.expected_chat_template_sha256:
        raise ValueError("embedded chat-template SHA-256 does not match the frozen plan")

    rows = existing.to_dict(orient="records")
    for _offset, row in fixed.iloc[completed:].iterrows():
        report = str(row[args.report_column])
        classification = str(row[args.classification_column])
        if args.interface == NATIVE_CHAT_INTERFACE_MODE:
            call = pipeline.llm_chat_json_with_receipt(
                model=model,
                messages=native_explanation_messages(
                    pipeline.PROMPT_EXPLAIN,
                    report,
                    classification,
                ),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stop=None,
                grammar=grammar,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        else:
            call = pipeline.llm_json_with_receipt(
                model=model,
                prompt=explanation_input(
                    pipeline.PROMPT_EXPLAIN,
                    report,
                    classification,
                ),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stop=None,
                grammar=grammar,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        checked = inspect_explanation(
            call.text,
            report=report,
            fixed_classification=classification,
        )
        rows.append(
            {
                "Hashed_ReportURN": str(row[args.id_column]),
                "explanation_interface_mode": args.interface,
                "fixed_classifications": classification,
                "explanations": call.text,
                "structured_output_valid": checked.structured_output_valid,
                "decision_copy_mismatches": checked.decision_copy_mismatches,
                "evidence_phrases": checked.evidence_phrases,
                "fallback_phrases": checked.fallback_phrases,
                "exact_traceable_phrases": checked.exact_traceable_phrases,
                "casefold_traceable_phrases": checked.casefold_traceable_phrases,
                "validation_error": checked.error,
                "elapsed_seconds": call.elapsed_seconds,
                "prompt_tokens": call.prompt_tokens,
                "completion_tokens": call.completion_tokens,
                "total_tokens": call.total_tokens,
            }
        )
        if (len(rows) % args.flush_every) == 0:
            atomic_write_csv(output, pd.DataFrame(rows))
            output.chmod(0o600)

    result_frame = pd.DataFrame(rows)
    atomic_write_csv(output, result_frame)
    output.chmod(0o600)
    aggregate = aggregate_inspections(result_frame)
    receipt = {
        "schema_version": 1,
        "receipt_type": "fixed_classification_evidence_extraction_run",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "interface": args.interface,
        "classification_source_held_fixed": True,
        "causal_faithfulness_claim": False,
        "inputs": {
            "dataset_sha256": dataset_sha,
            "fixed_predictions_sha256": predictions_sha,
            "manifest_sha256": manifest_sha,
            "records": len(fixed),
        },
        "model": model_receipt,
        "prompt": {
            "sha256": sha256_text(pipeline.PROMPT_EXPLAIN),
            "task_message_template_sha256": sha256_text(
                explanation_task_message_template(pipeline.PROMPT_EXPLAIN)
            ),
        },
        "grammar": {
            "filename": grammar_path.name,
            "sha256": grammar_sha,
            "applied": True,
        },
        "chat_template": template_receipt,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        "aggregate_quality": aggregate,
        "output": {"filename": output.name, "sha256": sha256_file(output)},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git": git_receipt(),
            "hf_hub_offline": os.getenv("HF_HUB_OFFLINE") == "1",
            "hf_hub_telemetry_disabled": os.getenv("HF_HUB_DISABLE_TELEMETRY") == "1",
        },
        "privacy_boundary": (
            "The keyed output and evidence strings remain governed. This receipt contains "
            "only hashes, configuration, counts, and aggregate quality measures."
        ),
    }
    atomic_write_json(receipt_path, receipt)
    receipt_path.chmod(0o600)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract grammar-constrained evidence from frozen classification JSON."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model", choices=pipeline.get_available_models(), required=True)
    parser.add_argument(
        "--interface",
        choices=[RAW_COMPLETION_INTERFACE_MODE, NATIVE_CHAT_INTERFACE_MODE],
        required=True,
    )
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-predictions-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-chat-template-sha256")
    parser.add_argument(
        "--expected-prompt-sha256",
        default="09e7e46d13d9fd1f6ebd14e4caecf32766354ead048ef22aa834c5b6064cd05f",
    )
    parser.add_argument(
        "--expected-grammar-sha256",
        default="718d3b0b16499d04d97723893f5e1de67aa1f342ba8b455a293a3d93084cd315",
    )
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--table", default="reports")
    parser.add_argument("--id-column", default="Hashed_ReportURN")
    parser.add_argument("--report-column", default="Report")
    parser.add_argument("--classification-column", default="classifications")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=30)
    parser.add_argument("--n-batch", type=int)
    parser.add_argument("--n-ubatch", type=int)
    parser.add_argument("--n-threads", type=int)
    parser.add_argument("--n-threads-batch", type=int)
    parser.add_argument(
        "--flash-attn",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.flush_every < 1:
        raise ValueError("--flush-every must be positive")
    if args.interface == NATIVE_CHAT_INTERFACE_MODE and not args.expected_chat_template_sha256:
        raise ValueError("native chat requires --expected-chat-template-sha256")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    receipt = run(args)
    print(json.dumps(receipt["aggregate_quality"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
