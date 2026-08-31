#!/usr/bin/env python3
"""Read frozen v2 outputs; write a separate governed literal-source audit, without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from eeg_review.evidence_extraction import load_fixed_evidence_inputs
from eeg_review.io import atomic_write_csv, atomic_write_json, load_table
from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL
from eeg_review.prompt_diagnostics import build_diagnostic_packet, targeted_missing_evidence
from eeg_review.protected_execution import assert_governed_run_active
from eeg_review.source_grounding import POLICY_ID, aggregate_grounding, inspect_grounding

ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "jbhi-02463-medgemma-native-focal-v2-development"
KEY = "Hashed_ReportURN"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source(root):
    assert_governed_run_active(root)
    if read(root / "state.json")["status"] != "completed":
        raise ValueError("source run is not completed")
    if read(root / "job.json")["study_id"] != STUDY_ID:
        raise ValueError("wrong producing study")
    manifest = read(root / "final-scientific-manifest.json")
    seen = set()
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or path in seen:
            raise ValueError("invalid or duplicate manifest path")
        seen.add(path)
        if sha(path) != item["sha256"]:
            raise ValueError("frozen source changed: " + item["path"])
    needed = [
        "job.json",
        "inputs/development.db",
        "inputs/development.manifest.csv",
        "inputs/evidence.manifest.csv",
        "inputs/v1.csv",
        "products/v2.csv",
        "products/evidence-v2.csv",
        "products/evidence-v2.run.json",
        "analysis/author-summary.json",
    ]
    if not {(root / p).resolve() for p in needed}.issubset(seen):
        raise ValueError("scientific manifest omitted required source artifacts")
    return {"files_verified": len(seen), "sha256": sha(root / "final-scientific-manifest.json")}


def audit(root):
    source = validate_source(root)
    fixed = load_fixed_evidence_inputs(
        dataset=root / "inputs/development.db",
        predictions=root / "products/v2.csv",
        manifest=root / "inputs/evidence.manifest.csv",
    )
    dev = pd.read_csv(root / "inputs/development.manifest.csv")
    if len(dev) != 100 or dev[KEY].isna().any() or dev[KEY].duplicated().any():
        raise ValueError("invalid development manifest")
    if len(fixed) != 20 or fixed[KEY].tolist() != dev[KEY].tolist()[:20]:
        raise ValueError("wrong first-20 development evidence population")
    frame = pd.read_csv(root / "products/evidence-v2.csv")
    if frame[KEY].tolist() != fixed[KEY].tolist():
        raise ValueError("evidence missing, duplicate or reordered keys")
    records = []
    for index, row in frame.iterrows():
        parent = fixed.iloc[index]
        if row["fixed_classifications"] != parent["classifications"]:
            raise ValueError("fixed classification drift")
        records.append(
            {
                KEY: row[KEY],
                **inspect_grounding(
                    row["explanations"], report=parent["Report"], fixed=parent["classifications"]
                ),
            }
        )
    summary = aggregate_grounding(records)
    original = read(root / "analysis/author-summary.json")["evidence_quality"]
    checks = {
        "records": summary["records"],
        "exact_traceable_phrases": summary["reason_statuses"].get("exact", 0),
        "evidence_phrases": summary["nonfallback_nonblank_phrases"],
        "fallback_phrases": summary["reason_statuses"].get("declared_no_evidence", 0),
        "decision_copy_mismatches": summary["decision_copy_mismatches"],
    }
    if any(original[key] != value for key, value in checks.items()):
        raise ValueError("literal audit does not reproduce frozen counters")
    if source != validate_source(root):
        raise ValueError("source changed during read-only analysis")
    result = {
        "policy_id": POLICY_ID,
        "study_id": STUDY_ID,
        "status": "completed_posthoc_source_audit_author_review_only",
        "source_manifest": source,
        "source_summary_sha256": sha(root / "analysis/author-summary.json"),
        "producing_commit": read(root / "job.json")["repository_revision"],
        "audit_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "audit_worktree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "inference_performed": False,
        "classifications_changed": False,
        "normalization_used_for_acceptance": False,
        "offset_units": "zero-based Unicode code points, end exclusive, exact report text",
        "aggregate": summary,
    }
    return result, records


def comparison_sources(root, source):
    """Use only source-hashed saved Mistral predictions; do not load a model."""
    assert_governed_run_active(source)
    job = read(source / "job.json")
    if job["study_id"] != "jbhi-02463-mistral-native-interface-small-followup-v1":
        raise ValueError("wrong Mistral comparison study")
    if read(source / "state.json")["status"] != "completed":
        raise ValueError("Mistral comparison is incomplete")
    if sha(root / "inputs/development.db") != sha(source / "inputs/development.db"):
        raise ValueError("Mistral reference corpus differs")
    files = {}
    for item in read(source / "transfer-manifest.json")["files"]:
        if item["path"] in files:
            raise ValueError("duplicate Mistral source manifest path")
        files[item["path"]] = item["sha256"]
    selected = {
        "mistral_historical_interface": "inputs/raw-development.csv",
        "mistral_native_interface": "products/native-classification.csv",
    }
    receipts = [
        "inputs/development.db",
        "inputs/raw-parent.run.json",
        "products/native-classification.run.json",
        "job.json",
    ]
    hashes = {}
    for path in [*selected.values(), *receipts]:
        if path not in files or sha(source / path) != files[path]:
            raise ValueError("changed or unreceipted Mistral comparison source")
        hashes[path] = files[path]
    return (
        {name: pd.read_csv(source / path) for name, path in selected.items()},
        {
            "producing_commit": job["repository_revision"],
            "files": hashes,
            "scope": "only named saved prediction inputs; not a new Mistral evaluation",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--comparison-run", type=Path)
    parser.add_argument("--acknowledge-governed-output", action="store_true", required=True)
    args = parser.parse_args()
    root, output = args.run_dir.resolve(strict=True), args.output_dir.resolve()
    if output.is_relative_to(root):
        raise ValueError("audit must not modify the completed run bundle")
    if output.exists():
        raise FileExistsError("audit output exists; never overwrite an earlier receipt")
    result, records = audit(root)
    versions = {
        "medgemma_native_v1": pd.read_csv(root / "inputs/v1.csv"),
        "medgemma_native_focal_v2": pd.read_csv(root / "products/v2.csv"),
    }
    if args.comparison_run:
        others, comparison_receipt = comparison_sources(
            root, args.comparison_run.resolve(strict=True)
        )
        versions.update(others)
        result["comparison_sources"] = comparison_receipt
    reference = load_table(
        root / "inputs/development.db", [KEY, "Report", *JSON_KEY_TO_LABEL.values()]
    )
    result["diagnostic_context"], packet = build_diagnostic_packet(reference, versions, records)
    target_keys, result["targeted_evidence_plan"] = targeted_missing_evidence(packet)
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    atomic_write_json(output / "aggregate.json", result)
    atomic_write_json(output / "governed-source-spans.json", records)
    atomic_write_json(output / "governed-diagnostic-packet.json", packet)
    atomic_write_csv(output / "targeted-evidence.manifest.csv", pd.DataFrame({KEY: target_keys}))
    atomic_write_json(
        output / "manifest.json",
        {
            "source_manifest_sha256": result["source_manifest"]["sha256"],
            "policy_id": POLICY_ID,
            "files": [
                {"path": name, "sha256": sha(output / name)}
                for name in [
                    "aggregate.json",
                    "governed-source-spans.json",
                    "governed-diagnostic-packet.json",
                    "targeted-evidence.manifest.csv",
                ]
            ],
            "distribution": "governed storage; aggregate requires author review before publication",
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
