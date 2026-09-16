#!/usr/bin/env python3
"""Verify the committed evidence-profile catalog against the typed registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eeg_review.evidence_profiles import (
    DECISION_CONDITIONED,
    INDEPENDENT_CATEGORY,
    VERBATIM_PROVENANCE,
    catalog,
    profile_selection_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "review/model-receipts/evidence-profile-catalog.v1.json"
FACTORIAL_MAP = (
    ROOT / "review/model-receipts/evidence-schema-factorial.profile-map.json"
)
FACTORIAL_RESULT = (
    ROOT / "review/model-receipts/evidence-schema-factorial.result.json"
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", dest="print_catalog")
    args = parser.parse_args()
    expected = catalog()
    actual = json.loads(CATALOG.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("committed evidence-profile catalog differs from typed registry")
    factorial_map = json.loads(FACTORIAL_MAP.read_text(encoding="utf-8"))
    if factorial_map["catalog_sha256"] != sha256_path(CATALOG):
        raise ValueError("factorial profile map does not bind the current catalog")
    if factorial_map["factorial_result_receipt_sha256"] != sha256_path(FACTORIAL_RESULT):
        raise ValueError("factorial profile map does not bind the result receipt")
    expected_selections = {
        "decision_conditioned_cells": profile_selection_receipt(
            DECISION_CONDITIONED,
            VERBATIM_PROVENANCE,
        ),
        "independent_category_cells": profile_selection_receipt(
            INDEPENDENT_CATEGORY,
            VERBATIM_PROVENANCE,
        ),
    }
    if factorial_map["selections"] != expected_selections:
        raise ValueError("factorial profile map contains an invalid profile selection")
    if args.print_catalog:
        print(json.dumps(expected, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "generation_profiles": len(expected["generation_profiles"]),
                    "claim_profiles": len(expected["claim_profiles"]),
                    "factorial_profile_map": "verified",
                    "status": "verified",
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
