#!/usr/bin/env python3
"""Compare two independent evidence reviews while system identity stays blinded."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from eeg_review.evidence_review_comparison import compare_blinded_reviews
from eeg_review.io import atomic_write_csv, atomic_write_json

PACKAGE = Path(
    "/Users/sbergner/Research/eeg/eeg_reports_annotation_nlp/data/governed/study-runs/"
    "jbhi-medgemma-v1-evidence-development-20260916/review-source-first-v2"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = (
    Path(__file__).resolve(),
    REPO_ROOT / "src/eeg_review/evidence_review_comparison.py",
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
    parser.add_argument("review_one", type=Path)
    parser.add_argument("review_two", type=Path)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PACKAGE / "blinded_multi_reader_summary.json",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=PACKAGE / "blinded_disagreement_queue.csv",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=PACKAGE / "blinded_multi_reader_receipt.json",
    )
    args = parser.parse_args()
    first = json.loads(args.review_one.read_text(encoding="utf-8"))
    second = json.loads(args.review_two.read_text(encoding="utf-8"))
    expected = pd.read_csv(PACKAGE / "01_source_first_review.csv")["case_id"].tolist()
    summary, queue = compare_blinded_reviews(
        first,
        second,
        expected_case_ids=expected,
    )
    atomic_write_json(args.summary, summary)
    atomic_write_csv(
        args.queue,
        pd.DataFrame(
            queue,
            columns=[
                "case_id",
                "scope",
                "system",
                "field",
                "reviewer_1_code",
                "reviewer_1_value",
                "reviewer_2_code",
                "reviewer_2_value",
                "adjudication_outcome",
                "adjudicated_value",
                "adjudicator_code",
                "adjudication_notes",
            ],
        ),
    )
    receipt = {
        "status": "complete_blinded_two_reader_comparison",
        "unblinded": False,
        "frozen_package_complete_sha256": sha256_path(PACKAGE / "COMPLETE.json"),
        "review_response_sha256": {
            str(args.review_one): sha256_path(args.review_one),
            str(args.review_two): sha256_path(args.review_two),
        },
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_path(path)
            for path in IMPLEMENTATION
        },
        "output_sha256": {
            str(args.summary): sha256_path(args.summary),
            str(args.queue): sha256_path(args.queue),
        },
    }
    atomic_write_json(args.receipt, receipt)
    for path in (args.summary, args.queue, args.receipt):
        path.chmod(0o600)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
