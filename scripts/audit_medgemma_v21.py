#!/usr/bin/env python3
"""Interpret frozen v2.1 products in a separate governed directory, without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from eeg_review.development_interpretation import KEY, exact_index, interpret
from eeg_review.io import atomic_write_json, load_table
from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL
from eeg_review.protected_execution import assert_governed_run_active

ROOT = Path(__file__).resolve().parents[1]
STUDY = "jbhi-02463-medgemma-native-scope-v21-development"
REQUIRED = {
    "job.json",
    "inputs/plan.json",
    "inputs/development.db",
    "inputs/development.manifest.csv",
    "inputs/evidence.manifest.csv",
    "inputs/v1.csv",
    "inputs/v2.csv",
    "products/v21.csv",
    "products/v21.run.json",
    "products/evidence-v21.csv",
    "products/evidence-v21.run.json",
    "analysis/author-summary.json",
    "receipts/classification-complete.json",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source(root):
    assert_governed_run_active(root)
    if read(root / "state.json")["status"] != "completed":
        raise ValueError("source is incomplete")
    if read(root / "job.json")["study_id"] != STUDY:
        raise ValueError("wrong study")
    manifest = read(root / "final-scientific-manifest.json")
    seen, resolved = set(), set()
    for entry in manifest["files"]:
        relative = entry["path"]
        path = (root / relative).resolve()
        if (
            relative in seen
            or path in resolved
            or not path.is_relative_to(root)
            or Path(relative).is_absolute()
        ):
            raise ValueError("invalid source manifest path")
        seen.add(relative)
        resolved.add(path)
        if sha(path) != entry["sha256"]:
            raise ValueError("changed source artifact")
    if not REQUIRED.issubset(seen):
        raise ValueError("manifest omitted required scientific source")
    return {"files_verified": len(seen), "sha256": sha(root / "final-scientific-manifest.json")}


def audit(root):
    source = validate_source(root)
    manifest = pd.read_csv(root / "inputs/development.manifest.csv")[KEY].tolist()
    evidence_keys = pd.read_csv(root / "inputs/evidence.manifest.csv")[KEY].tolist()
    if len(manifest) != 100 or len(evidence_keys) != 20:
        raise ValueError("frozen study population changed")
    reference = load_table(
        root / "inputs/development.db", [KEY, "Report", *JSON_KEY_TO_LABEL.values()]
    )
    # Check membership before explicit manifest ordering; never silently drop an extra row.
    if (
        len(reference) != 100
        or set(reference[KEY]) != set(manifest)
        or reference[KEY].duplicated().any()
    ):
        raise ValueError("database and manifest populations differ")
    reference = reference.set_index(KEY).loc[manifest].reset_index()
    exact_index(reference, manifest)
    versions = {
        name: pd.read_csv(root / path)
        for name, path in {
            "v1": "inputs/v1.csv",
            "v2": "inputs/v2.csv",
            "v21": "products/v21.csv",
        }.items()
    }
    summary, packet = interpret(
        reference, versions, pd.read_csv(root / "products/evidence-v21.csv"), evidence_keys
    )
    frozen = read(root / "analysis/author-summary.json")
    receipt = read(root / "products/evidence-v21.run.json")
    if (
        receipt["classifications_supplied_to_model"] is not False
        or receipt["evidence_used_to_change_classifications"] is not False
    ):
        raise ValueError("independent audit interface changed")
    for name, labels in summary["binary_counts"].items():
        for label, counts in labels.items():
            if any(
                frozen["results"][name][label]["point_estimates"][key] != value
                for key, value in counts.items()
            ):
                raise ValueError("classification counts differ from frozen analysis")
    for parent, labels in summary["paired_changes"].items():
        for label, counts in labels.items():
            pair = frozen["paired_comparisons"][parent][label]["discordant_correctness"]
            if (
                counts["repair"] != pair["core_accuracy"]["a_correct_b_wrong"]
                or counts["regression"] != pair["core_accuracy"]["a_wrong_b_correct"]
            ):
                raise ValueError("paired transitions differ from frozen analysis")
            if (
                counts["four_level_repair"]
                != pair["certainty_adjusted_accuracy"]["a_correct_b_wrong"]
                or counts["four_level_regression"]
                != pair["certainty_adjusted_accuracy"]["a_wrong_b_correct"]
            ):
                raise ValueError("four-level transitions differ from frozen analysis")
    evidence = summary["independent_evidence"]
    if (
        evidence["phrase_instances"] != frozen["evidence_quality"]["evidence_phrases"]
        or evidence["phrase_statuses"].get("exact", 0)
        != frozen["evidence_quality"]["exact_traceable_phrases"]
    ):
        raise ValueError("literal source counters differ from frozen analysis")
    if source != validate_source(root):
        raise ValueError("source changed during analysis")
    return {
        "study_id": STUDY,
        "status": "completed_readonly_interpretation_author_working",
        "source_manifest": source,
        "source_summary_sha256": sha(root / "analysis/author-summary.json"),
        "producing_commit": read(root / "job.json")["repository_revision"],
        "audit_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "audit_worktree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "diagnostics": summary,
    }, packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-governed-output", action="store_true", required=True)
    args = parser.parse_args()
    root, output = args.run_dir.resolve(strict=True), args.output_dir.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("interpretation must be separate from the producing bundle")
    if output.exists():
        raise FileExistsError("never overwrite a previous interpretation")
    result, packet = audit(root)
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    atomic_write_json(output / "interpretation-summary.json", result)
    atomic_write_json(output / "governed-review-packet.json", packet)
    atomic_write_json(
        output / "interpretation-manifest.json",
        {
            "distribution": "governed storage only; packet contains report text and keys",
            "files": [{"path": p.name, "sha256": sha(p)} for p in sorted(output.glob("*.json"))],
            "source_manifest": result["source_manifest"],
        },
    )
    print(
        json.dumps(
            {
                "records": result["diagnostics"]["records"],
                "source_files_verified": result["source_manifest"]["files_verified"],
                "inference_performed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
