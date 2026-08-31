#!/usr/bin/env python3
"""Separate structural-anchor inventory; never recover names or infer patient identity.

Policy fixed before this scan: search only the frozen diagnostic's report text
for opaque hex tokens (16--64 characters), UUIDs, and explicitly labelled
patient ID/URN/key/hash values in those opaque formats. Do not search for names,
addresses, dates of birth, numeric health numbers or external identifying data.
Repeated tokens are candidates, not verified identities. Generic redaction
placeholders and common dates are not accepted as anchors. Zero matches means
only that these particular formats were absent. This inventory does not alter
the earlier frozen similarity/LLM-review experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from eeg_review.io import atomic_write_json
from eeg_review.manifest import sha256_file
from eeg_review.protected_execution import assert_governed_run_active

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_linkage_diagnostic import stage_done  # noqa: E402

OPAQUE = r"(?:[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}|[a-f0-9]{16,64})"
PATTERNS = {
    "opaque_hex_or_uuid": rf"(?<!\w)({OPAQUE})(?!\w)",
    "explicit_patient_opaque_key": (
        rf"\b(?:hashed[ _-]*)?patient[ _-]*(?:id|urn|key|hash)\b\s*[:=]\s*({OPAQUE})(?!\w)"
    ),
}


def scan(texts):
    summary, details = {}, {}
    for name, pattern in PATTERNS.items():
        values = defaultdict(set)
        for index, text in enumerate(texts):
            for found in re.finditer(pattern, text, re.IGNORECASE):
                values[found.group(1).casefold()].add(index)
        repeated = {k: sorted(v) for k, v in values.items() if len(v) > 1}
        summary[name] = {
            "distinct_tokens": len(values),
            "reports_with_token": len(set().union(*values.values())) if values else 0,
            "tokens_repeated_across_reports": len(repeated),
        }
        details[name] = repeated
    return summary, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-governed-output", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    base = (ROOT / "data/governed/analysis-runs").resolve()
    if not args.acknowledge_governed_output or root == base or not root.is_relative_to(base):
        raise ValueError("dedicated governed diagnostic and acknowledgement required")
    assert_governed_run_active(root)
    if not stage_done(root, "prepare"):
        raise ValueError("verified prepared diagnostic required")
    records = root / "records.json"
    rows = json.loads(records.read_text())
    result, candidates = scan([r["Report"] for r in rows])
    receipt = {
        "diagnostic_id": "jbhi-02463/diagnostic/opaque-linkage-anchor-inventory/v1",
        "records": len(rows), "source_sha256": sha256_file(records),
        "code_sha256": sha256_file(Path(__file__)), "patterns": PATTERNS,
        "summary": result, "patient_identity_confirmed": False,
        "external_identity_sources_accessed": False,
        "repeated_token_candidates_governed": candidates,
    }
    target = root / "anchor-inventory.json"
    if target.exists() and json.loads(target.read_text()) != receipt:
        raise ValueError("immutable anchor-inventory receipt mismatch")
    assert_governed_run_active(root)
    os.umask(0o077)
    atomic_write_json(target, receipt)
    print(json.dumps({"records": len(rows), "summary": result,
                      "patient_identity_confirmed": False}))


if __name__ == "__main__":
    main()
