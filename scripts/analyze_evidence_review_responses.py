#!/usr/bin/env python3
"""Validate a completed source-first review and write a blinded count summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from eeg_review.evidence_review_responses import summarize_review_response
from eeg_review.io import atomic_write_json

PACKAGE = Path(
    "/Users/sbergner/Research/eeg/eeg_reports_annotation_nlp/data/governed/study-runs/"
    "jbhi-medgemma-v1-evidence-development-20260916/review-source-first-v2"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_IMPLEMENTATION = (
    Path(__file__).resolve(),
    REPO_ROOT / "src/eeg_review/evidence_review_responses.py",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path, help="JSON downloaded from review_form.html")
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE / "blinded_review_summary.json",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Hash receipt path; defaults beside the response payload.",
    )
    args = parser.parse_args()
    payload = json.loads(args.responses.read_text(encoding="utf-8"))
    expected = pd.read_csv(PACKAGE / "01_source_first_review.csv")["case_id"].tolist()
    summary = summarize_review_response(payload, expected_case_ids=expected)
    atomic_write_json(args.output, summary)
    args.output.chmod(0o600)
    completion = PACKAGE / "COMPLETE.json"
    receipt_path = args.receipt or args.responses.with_suffix(".receipt.json")
    receipt = {
        "status": "complete_blinded_review_pass",
        "reviewer_code": summary["reviewer_code"],
        "reviewer_role": summary["reviewer_role"],
        "cases_reviewed": summary["cases_reviewed"],
        "system_reviews": summary["system_reviews"],
        "frozen_package_complete_sha256": sha256_path(completion),
        "analysis_implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_path(path)
            for path in ANALYSIS_IMPLEMENTATION
        },
        "response_sha256": sha256_path(args.responses),
        "blinded_summary_sha256": sha256_path(args.output),
        "response_path": str(args.responses),
        "blinded_summary_path": str(args.output),
        "unblinded": False,
    }
    atomic_write_json(receipt_path, receipt)
    receipt_path.chmod(0o600)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
