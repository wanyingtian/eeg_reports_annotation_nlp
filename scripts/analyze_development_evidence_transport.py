#!/usr/bin/env python3
"""Compare saved Mistral and MedGemma evidence on one frozen development surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from eeg_review.fixed_evidence_comparison import (
    load_paired_evidence_surface,
    summarize_paired_evidence,
)
from eeg_review.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return observed


def git_receipt() -> dict[str, object]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    return {"revision": revision, "worktree_dirty": dirty}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--medgemma-evidence", type=Path, required=True)
    parser.add_argument("--mistral-saved", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-medgemma-evidence-sha256", required=True)
    parser.add_argument("--expected-mistral-saved-sha256", required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("analysis receipt exists; never overwrite a prior result")
    hashes = {
        "dataset": verified(args.dataset, args.expected_dataset_sha256, "dataset"),
        "manifest": verified(args.manifest, args.expected_manifest_sha256, "manifest"),
        "medgemma_evidence": verified(
            args.medgemma_evidence,
            args.expected_medgemma_evidence_sha256,
            "MedGemma evidence",
        ),
        "mistral_saved": verified(
            args.mistral_saved,
            args.expected_mistral_saved_sha256,
            "saved Mistral stream",
        ),
    }
    reports, streams = load_paired_evidence_surface(
        dataset=args.dataset,
        manifest=args.manifest,
        streams={
            "medgemma_native_v1_fixed_decision": (
                args.medgemma_evidence,
                "fixed_classifications",
            ),
            "mistral_historical_interface_saved": (
                args.mistral_saved,
                "classifications",
            ),
        },
    )
    if len(reports) != args.expected_records:
        raise ValueError(
            "development population mismatch: "
            f"expected {args.expected_records}, found {len(reports)}"
        )
    summary = summarize_paired_evidence(reports, streams)
    receipt = {
        "schema_version": 1,
        "analysis_id": "jbhi-medgemma-v1-evidence-development-20260916",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed_development_transport_diagnostic_not_manuscript_admitted",
        "new_model_inference_performed": False,
        "classification_calls_performed": 0,
        "candidate_or_semantic_matching_performed": False,
        "input_hashes": hashes,
        "summary": summary,
        "implementation": git_receipt(),
        "runtime": {"python": platform.python_version()},
        "privacy": (
            "This aggregate contains no report key, report text, classification JSON, "
            "or extracted phrase. Keyed evidence remains governed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, receipt)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
