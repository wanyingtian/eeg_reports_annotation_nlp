#!/usr/bin/env python3
"""Freeze the result-blind selection receipt for native-chat development."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LABELS = [
    "focal_epileptiform_activity",
    "generalized_epileptiform_activity",
    "focal_non_epileptiform_activity",
    "generalized_non_epileptiform_activity",
    "abnormality",
]
EXPECTED = {
    "model": "b137aac80f2bcb1c1ed35bfe13387bc496eb18898d5f46425687604f0f714481",
    "prompt": "52198221d8330e9857b51a7ad99b017aa18836e1718b08dd0ae355820f5a5e69",
    "grammar": "5237e13988062538cda9c21906f1f4e1fc8b99498e2462ea69fe24bface35016",
    "chat_template": "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4",
    "task_message": "c0a84d01c54e20a90e8885650fc34955719d0d744ac4ac9a0a5497171d55b441",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_keys(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if "Hashed_ReportURN" not in (reader.fieldnames or []):
            raise ValueError("Manifest lacks Hashed_ReportURN")
        return [str(row["Hashed_ReportURN"]) for row in reader]


def finalize(raw: Path, run_receipt: Path, manifest: Path) -> dict[str, Any]:
    receipt = json.loads(run_receipt.read_text(encoding="utf-8"))
    with raw.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    manifest_keys = load_keys(manifest)
    output_keys = [str(row.get("Hashed_ReportURN", "")) for row in rows]
    blockers: list[str] = []
    if len(rows) != 100:
        blockers.append(f"expected 100 rows, found {len(rows)}")
    if len(output_keys) != len(set(output_keys)):
        blockers.append("duplicate output keys")
    if output_keys != manifest_keys:
        blockers.append("output keys do not match the frozen ordered manifest")

    patterns: Counter[tuple[int, ...]] = Counter()
    levels = {label: Counter() for label in LABELS}
    invalid = 0
    for row in rows:
        try:
            parsed = json.loads(row.get("classifications", ""))
            if set(parsed) != set(LABELS):
                raise ValueError("label set mismatch")
            pattern = tuple(int(parsed[label]) for label in LABELS)
            if any(value not in {1, 2, 3, 4} for value in pattern):
                raise ValueError("invalid four-level value")
        except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
            invalid += 1
            continue
        patterns[pattern] += 1
        for label, value in zip(LABELS, pattern, strict=True):
            levels[label][str(value)] += 1
    if invalid:
        blockers.append(f"{invalid} invalid structured outputs")
    if len(patterns) <= 1:
        blockers.append("degenerate constant five-label output pattern")

    observed = {
        "model": receipt["model"]["sha256"],
        "prompt": receipt["prompts"]["classify"]["sha256"],
        "grammar": receipt["grammars"]["classify"]["sha256"],
        "chat_template": receipt["input_policy"]["embedded_chat_template"]["sha256"],
        "task_message": receipt["input_policy"]["task_message_template"]["sha256"],
    }
    for name, expected in EXPECTED.items():
        if observed[name] != expected:
            blockers.append(f"{name} hash mismatch")
    if receipt["input_policy"].get("classification_interface_mode") != "native_chat":
        blockers.append("classification interface is not native_chat")
    if receipt["reports_completed"] != len(rows):
        blockers.append("run receipt count differs from output")
    if receipt["output"]["sha256"] != sha256_file(raw):
        blockers.append("run receipt output hash mismatch")

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "study_id": "jbhi-02463-post-submission-medgemma-native-interface-sensitivity-v1",
        "configuration_id": (
            "jbhi-02463/comparator/medgemma-27b-text-it/configuration/"
            "independent-native-interface-q2-v1"
        ),
        "stage": "zoe_development_100_result_blind_selection",
        "selection_rule": "singleton structural transport rule; no reference metric used",
        "selected_for_freeze": not blockers,
        "blockers": blockers,
        "population": {
            "manifest_records": len(manifest_keys),
            "output_records": len(rows),
            "unique_output_keys": len(set(output_keys)),
            "invalid_outputs": invalid,
            "full_pattern_cardinality": len(patterns),
        },
        "predicted_level_counts": {
            label: dict(sorted(counter.items())) for label, counter in levels.items()
        },
        "identity": {
            **observed,
            "manifest_sha256": sha256_file(manifest),
            "output_sha256": sha256_file(raw),
            "run_receipt_sha256": sha256_file(run_receipt),
            "repository_revision": receipt["environment"]["git"]["revision"],
            "worktree_dirty": receipt["environment"]["git"]["worktree_dirty"],
        },
        "boundaries": [
            "No reference outcome or development accuracy selected this singleton configuration.",
            "This receipt does not authorize protected-cohort evaluation.",
            "This configuration cannot replace the completed matched-historical Q2 result.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--run-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.raw, args.run_receipt, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    args.output.chmod(0o600)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["selected_for_freeze"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
