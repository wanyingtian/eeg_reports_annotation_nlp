#!/usr/bin/env python3
"""Exact report-export reconciliation; never infer patient membership from text.

Reads only report keys/text from the prepared diagnostic, repository example,
and four original annotation snapshots. No fuzzy fallback, row-position match,
label access, source mutation, or new patient map. The output is aggregate-only
but remains beside its governed source receipts.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from eeg_review.io import atomic_write_json
from eeg_review.manifest import sha256_file
from eeg_review.protected_execution import assert_governed_run_active

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_linkage_diagnostic import stage_done  # noqa: E402

KEY = "Hashed_ReportURN"
SNAPSHOTS = {
    "zoe_LD": "zoe_reports_LD_2000.db",
    "zoe_SG": "zoe_reports_SG_1500.db",
    "maria_LD": "maria_reports_LD.db",
    "maria_SG": "maria_reports_SG.db",
}
COUNTS = {"development": 100, "zoe_evaluation": 1395, "maria_evaluation": 499}


def report_index(rows):
    result = {}
    for key, text in rows:
        if not isinstance(key, str) or not key.strip() or not isinstance(text, str):
            raise ValueError("report key/text must be non-null strings")
        if key in result:
            raise ValueError("duplicate report key; cannot silently collapse records")
        result[key] = text
    return result


def load_reports(path):
    path = path.resolve(strict=True)
    before = sha256_file(path)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ]
        columns = [r[1] for r in conn.execute('PRAGMA table_info("reports")')]
        aliases = [k for k in [KEY, "Hashed ID"] if k in columns]
        if len(aliases) != 1 or "Report" not in columns:
            raise ValueError("one explicit report-key column and report text required")
        # Column is from the fixed alias allowlist, not arbitrary source SQL.
        records = report_index(conn.execute(f'SELECT "{aliases[0]}", "Report" FROM reports'))
    if sha256_file(path) != before:
        raise ValueError("source database changed during read-only audit")
    return records, {
        "sha256": before,
        "records": len(records),
        "tables": tables,
        "columns": columns,
        "report_key_column": aliases[0],
    }


def compare(query, reference):
    common = query.keys() & reference.keys()
    exact = sum(query[k] == reference[k] for k in common)
    return {
        "query_reports": len(query),
        "reference_reports": len(reference),
        "keys_found": len(common),
        "keys_missing": len(query) - len(common),
        "exact_text_matches_on_same_key": exact,
        "text_mismatches_on_same_key": len(common) - exact,
        "complete_exact_report_join": bool(query) and exact == len(query),
        "patient_membership_established": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-governed-output", action="store_true")
    args = parser.parse_args()
    root, snapshot = args.run_dir.resolve(), args.snapshot_dir.resolve()
    base = (ROOT / "data/governed/analysis-runs").resolve()
    if not args.acknowledge_governed_output or root == base or not root.is_relative_to(base):
        raise ValueError("dedicated governed diagnostic and acknowledgement required")
    assert_governed_run_active(root)
    assert_governed_run_active(snapshot)
    if not stage_done(root, "prepare"):
        raise ValueError("verified prepared diagnostic required")
    source = root / "records.json"
    before = sha256_file(source)
    rows = json.loads(source.read_text())
    if len(rows) != sum(COUNTS.values()) or {r["cohort"] for r in rows} != set(COUNTS):
        raise ValueError("fixed diagnostic cohort mismatch")
    cohorts = {}
    for cohort, count in COUNTS.items():
        selected = [r for r in rows if r["cohort"] == cohort]
        cohorts[cohort] = report_index((r[KEY], r["Report"]) for r in selected)
        if len(cohorts[cohort]) != count:
            raise ValueError("fixed diagnostic cohort count mismatch")
    # Also reject report keys reused across cohorts; do not silently deduplicate.
    report_index((r[KEY], r["Report"]) for r in rows)
    sample_path = ROOT / "data/zoe_reports_sample.db"
    sample, sample_metadata = load_reports(sample_path)
    sources = {str(sample_path): sample_metadata}
    checks = {}
    for name, filename in SNAPSHOTS.items():
        path = snapshot / filename
        reference, metadata = load_reports(path)
        sources[str(path)] = metadata
        applicable = (
            ["maria_evaluation"] if name.startswith("maria") else ["development", "zoe_evaluation"]
        )
        checks[name] = {c: compare(cohorts[c], reference) for c in applicable}
        if name.startswith("zoe"):
            checks[name]["repository_example"] = compare(sample, reference)
    receipt = {
        "audit_id": "jbhi-02463/diagnostic/exact-report-snapshot-join/v1",
        "code_sha256": sha256_file(Path(__file__)),
        "prepared_records_sha256": before,
        "sources": sources,
        "checks": checks,
        "unique_study_reports": len(rows),
        "all_checked_joins_exact": all(
            v["complete_exact_report_join"] for c in checks.values() for v in c.values()
        ),
        "only_report_key_and_text_values_loaded": True,
        "fuzzy_fallback_used": False,
        "patient_map_created": False,
        "original_databases_modified": False,
    }
    for path, metadata in sources.items():
        if sha256_file(Path(path)) != metadata["sha256"]:
            raise ValueError("source changed during audit")
    if sha256_file(source) != before or not stage_done(root, "prepare"):
        raise ValueError("prepared diagnostic changed during audit")
    target = root / "exact-report-snapshot-join.json"
    if target.exists() and json.loads(target.read_text()) != receipt:
        raise ValueError("immutable report-join audit receipt mismatch")
    assert_governed_run_active(root)
    assert_governed_run_active(snapshot)
    os.umask(0o077)
    atomic_write_json(target, receipt)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in [
                    "unique_study_reports",
                    "checks",
                    "all_checked_joins_exact",
                    "patient_map_created",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
