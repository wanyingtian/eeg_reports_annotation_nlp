#!/usr/bin/env python3
"""Validate a completed source-first review and write a blinded count summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eeg_review.evidence_review_responses import summarize_review_response
from eeg_review.io import atomic_write_json

PACKAGE = Path(
    "/Users/sbergner/Research/eeg/eeg_reports_annotation_nlp/data/governed/study-runs/"
    "jbhi-medgemma-v1-evidence-development-20260916/review-source-first-v2"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path, help="JSON downloaded from review_form.html")
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE / "blinded_review_summary.json",
    )
    args = parser.parse_args()
    payload = json.loads(args.responses.read_text(encoding="utf-8"))
    expected = pd.read_csv(PACKAGE / "01_source_first_review.csv")["case_id"].tolist()
    summary = summarize_review_response(payload, expected_case_ids=expected)
    atomic_write_json(args.output, summary)
    args.output.chmod(0o600)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
