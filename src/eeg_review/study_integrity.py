"""Study-partition and decision-lens integrity helpers.

The helpers expose aggregate checks only. Report keys and text are used in
memory to detect overlap and are never returned in the public-safe result.
"""

from __future__ import annotations

import hashlib
import numbers
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Partition:
    role: str
    frame: pd.DataFrame


def normalized_report(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("report text must be a nonblank string")
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def decision_lenses(level: object) -> dict[str, object]:
    """Expose Chris's four-level output as core and declared-confidence lenses."""
    if isinstance(level, bool):
        raise ValueError("four-level decision must be an integer from 1 to 4")
    if isinstance(level, str):
        if level.strip() not in {"1", "2", "3", "4"}:
            raise ValueError("four-level decision must be an integer from 1 to 4")
        value = int(level)
    elif isinstance(level, numbers.Real) and float(level).is_integer():
        value = int(level)
    else:
        raise ValueError("four-level decision must be an integer from 1 to 4")
    if value not in {1, 2, 3, 4}:
        raise ValueError("four-level decision must be an integer from 1 to 4")
    return {
        "four_level_decision": value,
        "core_call": "absent" if value in {1, 2} else "present",
        "declared_confidence": "confident" if value in {1, 4} else "low_confidence",
        "probability_calibration_claimed": False,
    }


def _partition_summary(
    name: str,
    partition: Partition,
    *,
    key_column: str,
    report_column: str,
) -> tuple[dict[str, object], set[str], set[str]]:
    if partition.role not in {"development", "held_out_evaluation"}:
        raise ValueError(f"invalid partition role for {name}")
    frame = partition.frame
    missing = sorted({key_column, report_column} - set(frame.columns))
    if missing:
        raise ValueError(f"partition {name} is missing columns: {missing}")
    keys = frame[key_column].map(lambda value: str(value).strip())
    if (keys == "").any() or keys.duplicated().any():
        raise ValueError(f"partition {name} has blank or duplicate report keys")
    normalized = frame[report_column].map(normalized_report)
    if normalized.duplicated().any():
        raise ValueError(f"partition {name} has duplicate normalized report text")
    text_hashes = normalized.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    rows = sorted(f"{key}\0{text_hash}" for key, text_hash in zip(keys, text_hashes, strict=True))
    digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
    return (
        {
            "role": partition.role,
            "records": len(frame),
            "unique_report_keys": len(set(keys)),
            "unique_normalized_reports": len(set(text_hashes)),
            "partition_identity_digest": digest,
        },
        set(keys),
        set(text_hashes),
    )


def audit_partitions(
    partitions: Mapping[str, Partition],
    *,
    key_column: str = "Hashed_ReportURN",
    report_column: str = "Report",
) -> dict[str, object]:
    if len(partitions) < 2:
        raise ValueError("at least two study partitions are required")
    if sum(item.role == "development" for item in partitions.values()) != 1:
        raise ValueError("exactly one development partition is required")
    if not any(item.role == "held_out_evaluation" for item in partitions.values()):
        raise ValueError("at least one held-out evaluation partition is required")

    summaries: dict[str, dict[str, object]] = {}
    key_sets: dict[str, set[str]] = {}
    text_sets: dict[str, set[str]] = {}
    for name, partition in sorted(partitions.items()):
        summary, keys, texts = _partition_summary(
            name,
            partition,
            key_column=key_column,
            report_column=report_column,
        )
        summaries[name] = summary
        key_sets[name] = keys
        text_sets[name] = texts

    pairwise: dict[str, dict[str, int]] = {}
    names = sorted(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            key_overlap = len(key_sets[left] & key_sets[right])
            text_overlap = len(text_sets[left] & text_sets[right])
            pairwise[f"{left}::{right}"] = {
                "report_key_overlap": key_overlap,
                "normalized_report_text_overlap": text_overlap,
            }
            if key_overlap or text_overlap:
                raise ValueError(f"study partitions overlap: {left} and {right}")

    return {
        "partition_separation_passed": True,
        "partitions": summaries,
        "pairwise_overlap": pairwise,
        "contains_report_keys_or_text": False,
        "interpretation": (
            "No report-key or normalized-report-text overlap was found among the "
            "supplied partitions. This does not establish patient independence."
        ),
    }
