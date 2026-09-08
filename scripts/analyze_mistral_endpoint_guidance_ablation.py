#!/usr/bin/env python3
"""Validate and summarize the fixed Mistral endpoint-guidance development ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from eeg_review.evidence_extraction import classification_levels
from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL
KEY = "Hashed_ReportURN"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prediction_table(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    rows = []
    for _, row in source.iterrows():
        parsed = classification_levels(row["classifications"])
        rows.append({KEY: str(row[KEY]), **{label: parsed[field] for field, label in JSON_KEY_TO_LABEL.items()}})
    return pd.DataFrame(rows)


def consistency_violations(frame: pd.DataFrame) -> dict[str, int]:
    subtypes = [label for label in JSON_KEY_TO_LABEL.values() if label != "Abnormality"]
    all_subtypes_absent = (frame[subtypes] <= 2).all(axis=1)
    any_subtype_present = (frame[subtypes] >= 3).any(axis=1)
    abnormal = frame["Abnormality"] >= 3
    return {
        "all_subtypes_absent_but_abnormality_present": int((all_subtypes_absent & abnormal).sum()),
        "any_subtype_present_but_abnormality_absent": int((any_subtype_present & ~abnormal).sum()),
    }


def metrics(reference: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    merged = reference.merge(predictions, on=KEY, suffixes=("_reference", "_prediction"), validate="one_to_one")
    output: dict[str, dict[str, Any]] = {}
    for label in JSON_KEY_TO_LABEL.values():
        ref = merged[f"{label}_reference"].astype(int)
        pred = merged[f"{label}_prediction"].astype(int)
        output[label] = {
            "records": len(merged),
            "core_agreement": float(((ref >= 3) == (pred >= 3)).mean()),
            "exact_four_level_agreement": float((ref == pred).mean()),
            "level_counts": {str(level): int((pred == level).sum()) for level in range(1, 5)},
        }
    return output


def analyze(run_dir: Path) -> dict[str, Any]:
    inputs, products = run_dir / "inputs", run_dir / "products"
    analysis = run_dir / "analysis"
    if analysis.exists():
        raise FileExistsError("refusing to overwrite an existing analysis directory")
    plan = read_json(inputs / "preregistered.json")
    parent_path, candidate_path = inputs / "parent.csv", products / "candidate.csv"
    manifest_path, database_path = inputs / "development.manifest.csv", inputs / "development.db"
    receipt_path = candidate_path.with_suffix(".run.json")
    receipt = read_json(receipt_path)

    expected_hashes = {
        database_path: plan["development_surface"]["database_sha256"],
        manifest_path: plan["development_surface"]["manifest_sha256"],
        parent_path: plan["development_surface"]["saved_parent_subset_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input checksum mismatch: {path.name}")
    if receipt["model"]["sha256"] != plan["model"]["sha256"]:
        raise ValueError("candidate model differs from the preregistration")
    if receipt["prompts"]["classify"]["sha256"] != plan["factors"]["candidate_prompt_sha256"]:
        raise ValueError("candidate prompt differs from the preregistration")
    if receipt["grammars"]["classify"]["sha256"] != plan["decoding"]["grammar_sha256"]:
        raise ValueError("candidate grammar differs from the preregistration")
    if receipt["environment"]["git"].get("worktree_dirty") is not False:
        raise ValueError("candidate must come from a clean producing worktree")
    if receipt["model"]["artifact_access"]["mode"] != "local_cache_only":
        raise ValueError("candidate model was not resolved locally")
    if receipt["execution_surface"] != {"classification": True, "explanations": False}:
        raise ValueError("unexpected candidate execution surface")
    if receipt["reports_completed"] != 100 or sha256_file(candidate_path) != receipt["output"]["sha256"]:
        raise ValueError("candidate output is incomplete or its receipt is stale")

    manifest = pd.read_csv(manifest_path, dtype={KEY: str})
    if len(manifest) != 100 or manifest[KEY].duplicated().any():
        raise ValueError("the fixed development manifest must contain 100 unique keys")
    keys = manifest[KEY].tolist()
    parent, candidate = prediction_table(parent_path), prediction_table(candidate_path)
    for name, frame in [("parent", parent), ("candidate", candidate)]:
        if frame[KEY].tolist() != keys:
            raise ValueError(f"{name} keys do not match the frozen manifest in order")

    with sqlite3.connect(database_path) as connection:
        reference = pd.read_sql_query("SELECT * FROM reports", connection)
    reference[KEY] = reference[KEY].astype(str)
    reference = reference.set_index(KEY).loc[keys].reset_index()
    labels = list(JSON_KEY_TO_LABEL.values())
    reference = reference[[KEY, *labels]]

    transitions: list[dict[str, Any]] = []
    paired: dict[str, dict[str, Any]] = {}
    for label in labels:
        old, new = parent[label].astype(int), candidate[label].astype(int)
        old_core, new_core = old >= 3, new >= 3
        table = Counter(zip(old.tolist(), new.tolist()))
        paired[label] = {
            "same_core": int((old_core == new_core).sum()),
            "same_four_level": int((old == new).sum()),
            "core_absent_to_present": int((~old_core & new_core).sum()),
            "core_present_to_absent": int((old_core & ~new_core).sum()),
            "four_level_transitions": {
                f"{source}->{target}": table.get((source, target), 0)
                for source in range(1, 5)
                for target in range(1, 5)
                if table.get((source, target), 0)
            },
        }
        for index in range(len(keys)):
            if old.iloc[index] != new.iloc[index]:
                transitions.append(
                    {
                        KEY: keys[index],
                        "category": label,
                        "parent_level": int(old.iloc[index]),
                        "candidate_level": int(new.iloc[index]),
                        "parent_core": int(old_core.iloc[index]),
                        "candidate_core": int(new_core.iloc[index]),
                    }
                )

    parent_metrics, candidate_metrics = metrics(reference, parent), metrics(reference, candidate)
    for label in labels:
        paired[label]["candidate_minus_parent_core_agreement"] = (
            candidate_metrics[label]["core_agreement"] - parent_metrics[label]["core_agreement"]
        )
        paired[label]["candidate_minus_parent_exact_agreement"] = (
            candidate_metrics[label]["exact_four_level_agreement"]
            - parent_metrics[label]["exact_four_level_agreement"]
        )

    analysis.mkdir(mode=0o700)
    pd.DataFrame(transitions).to_csv(analysis / "governed-paired-transitions.csv", index=False)
    result = {
        "study_id": plan["study_id"],
        "status": "completed_development_ablation_not_manuscript_admitted",
        "records": 100,
        "changed_factor": plan["factors"]["changed_factor"],
        "parent": {
            "prompt_sha256": plan["factors"]["parent_prompt_sha256"],
            "metrics": parent_metrics,
            "consistency_violations": consistency_violations(parent),
        },
        "candidate": {
            "prompt_sha256": plan["factors"]["candidate_prompt_sha256"],
            "metrics": candidate_metrics,
            "consistency_violations": consistency_violations(candidate),
        },
        "paired": paired,
        "receipts": {
            "preregistration_sha256": sha256_file(inputs / "preregistered.json"),
            "candidate_output_sha256": sha256_file(candidate_path),
            "candidate_run_receipt_sha256": sha256_file(receipt_path),
            "governed_transition_count": len(transitions),
        },
        "boundaries": plan["interpretation"]["not_supported"],
        "historical_14_percent_claim_restored": False,
        "protected_evaluation_run": False,
        "automatic_manuscript_admission": False,
    }
    write_json(analysis / "aggregate.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir.expanduser().resolve(strict=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
