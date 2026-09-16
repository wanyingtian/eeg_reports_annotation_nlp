#!/usr/bin/env python3
"""Build a blinded, stratified human-review package for MedGemma vs. saved
Mistral evidence on the frozen 100-report development surface.

Governance: report text and phrase text are read only from governed storage
and written only back into governed storage. Nothing with report or phrase
text is written into the git-tracked repository. Model identity is blinded
in the reviewer-facing file; the unblinding key is a separate governed file.

This does not compute a new primary result. It selects a deterministic,
bounded sample for a human to judge whether an extracted phrase supports its
named category -- a question no automated string match can answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from eeg_review.io import atomic_write_csv, atomic_write_json, load_table
from eeg_review.reason_traceability import (
    REVIEW_SAMPLE_QUOTAS,
    EvidenceUnit,
    audit_traceability,
    build_review_queue,
    structured_evidence_units,
)

ID_COLUMN = "Hashed_ReportURN"
REPORT_COLUMN = "Report"
STUDY_ID = "jbhi-02463-medgemma-v1-evidence-development-20260916"

GOVERNED_ROOT = Path(
    "/Users/sbergner/Research/eeg/eeg_reports_annotation_nlp/data/governed/study-runs"
)
DATASET = GOVERNED_ROOT / "jbhi-medgemma-native-chat-development-20260829/inputs/zoe_development_native_100.db"
MANIFEST = GOVERNED_ROOT / "jbhi-medgemma-native-chat-development-20260829/manifests/zoe_development_native_100.csv"
MEDGEMMA_EVIDENCE = GOVERNED_ROOT / "jbhi-medgemma-v1-evidence-development-20260916/products/medgemma-v1-evidence-all100.csv"
MISTRAL_SAVED = GOVERNED_ROOT / "jbhi-native-20260814/products/llm/zoe/raw.csv"

OUTPUT_DIR = GOVERNED_ROOT / "jbhi-medgemma-v1-evidence-development-20260916" / "review"
NO_EVIDENCE_QUOTA_PER_STREAM_CATEGORY = 2

STREAM_LABELS = {
    "medgemma_native_v1_fixed_decision": ("MedGemma-27B Q2_K (native chat, fixed decisions)", "fixed_classifications"),
    "mistral_historical_interface_saved": ("Saved Mistral-7B (historical raw-completion interface)", "classifications"),
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_units(stream: str, path: Path, classification_column: str, reports: pd.DataFrame) -> list[EvidenceUnit]:
    manifest_keys = pd.read_csv(MANIFEST, usecols=[ID_COLUMN])[ID_COLUMN].astype(str).tolist()
    frame = pd.read_csv(path, usecols=[ID_COLUMN, classification_column, "explanations"])
    frame = frame.assign(**{ID_COLUMN: frame[ID_COLUMN].astype(str)}).set_index(ID_COLUMN).loc[manifest_keys].reset_index()
    return structured_evidence_units(
        frame,
        reports,
        source_kind=stream,
        classification_column=classification_column,
    )


def rows_for(stream: str, units: list[EvidenceUnit]) -> list[dict]:
    rows = audit_traceability(units)
    for row in rows:
        row["stream"] = stream
    return rows


def find_no_evidence_units(stream: str, units: list[EvidenceUnit], rows: list[dict]) -> list[dict]:
    """Units where every declared segment was excluded (declared no evidence,
    or a blank/fallback list). build_review_queue skips these entirely, but
    they are exactly the "no-evidence" stratum the review needs."""
    by_unit: dict[int, list[dict]] = {}
    for row in rows:
        if row["stream"] == stream:
            by_unit.setdefault(row["unit_number"], []).append(row)

    candidates = []
    for unit_number, unit in enumerate(units):
        unit_rows = by_unit.get(unit_number, [])
        substantive = [r for r in unit_rows if not str(r["stage"]).startswith("excluded_")]
        if substantive:
            continue
        fingerprint = sha256_text(f"{stream}|{sha256_text(unit.report)}|{unit.category}|{unit_number}")
        candidates.append(
            {
                "stream": stream,
                "unit_number": unit_number,
                "category": unit.category,
                "stratum": "no_evidence",
                "selection_fingerprint": fingerprint,
            }
        )

    selected = []
    for category in sorted({c["category"] for c in candidates}):
        eligible = sorted(
            (c for c in candidates if c["category"] == category),
            key=lambda c: c["selection_fingerprint"],
        )
        selected.extend(eligible[:NO_EVIDENCE_QUOTA_PER_STREAM_CATEGORY])
    return selected


def blind_label(stream: str) -> str:
    # Stable across a run, opaque without the separately stored key.
    order = sorted(STREAM_LABELS, key=lambda name: sha256_text(f"{STUDY_ID}|blind-order|{name}"))
    return "System A" if stream == order[0] else "System B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print counts only; write nothing.")
    args = parser.parse_args()

    reports = load_table(DATASET, [ID_COLUMN, REPORT_COLUMN])

    units_by_stream: dict[str, list[EvidenceUnit]] = {}
    rows: list[dict] = []
    for stream, (_, classification_column) in STREAM_LABELS.items():
        path = MEDGEMMA_EVIDENCE if stream == "medgemma_native_v1_fixed_decision" else MISTRAL_SAVED
        units = load_units(stream, path, classification_column, reports)
        units_by_stream[stream] = units
        rows.extend(rows_for(stream, units))

    selected, summary = build_review_queue(rows)
    no_evidence_selected: list[dict] = []
    for stream in STREAM_LABELS:
        no_evidence_selected.extend(
            find_no_evidence_units(stream, units_by_stream[stream], rows)
        )

    # Hydrate each selected unit with real text, and blind the model identity.
    review_rows = []
    selection_counter = 0
    for item in [*selected, *no_evidence_selected]:
        stream = item["stream"]
        unit_number = item["unit_number"]
        unit = units_by_stream[stream][unit_number]
        selection_counter += 1
        selection_id = f"R{selection_counter:04d}"
        unit_rows = [
            r for r in rows if r["stream"] == stream and r["unit_number"] == unit_number
        ]
        stage_by_segment = {r["segment_number"]: r["stage"] for r in unit_rows}
        if unit.segments:
            for segment_number, segment_text in enumerate(unit.segments):
                review_rows.append(
                    {
                        "selection_id": selection_id,
                        "stratum": item["stratum"],
                        "system": blind_label(stream),
                        "category": unit.category,
                        "report_text": unit.report,
                        "extracted_phrase": segment_text,
                        "audit_stage": stage_by_segment.get(segment_number, "n/a"),
                        "reviewer_supports_category": "",
                        "reviewer_notes": "",
                    }
                )
        else:
            review_rows.append(
                {
                    "selection_id": selection_id,
                    "stratum": item["stratum"],
                    "system": blind_label(stream),
                    "category": unit.category,
                    "report_text": unit.report,
                    "extracted_phrase": "(no phrase declared)",
                    "audit_stage": "no_evidence",
                    "reviewer_supports_category": "",
                    "reviewer_notes": "",
                }
            )

    review_frame = pd.DataFrame(review_rows)
    unblinding_key = {
        blind_label(stream): {"stream": stream, "label": label}
        for stream, (label, _) in STREAM_LABELS.items()
    }
    receipt = {
        "study_id": STUDY_ID,
        "purpose": "Blinded human review of a deterministic, stratified evidence sample",
        "inputs": {
            "dataset": str(DATASET),
            "manifest": str(MANIFEST),
            "medgemma_evidence": str(MEDGEMMA_EVIDENCE),
            "mistral_saved": str(MISTRAL_SAVED),
        },
        "selection_summary": summary,
        "no_evidence_stratum": {
            "quota_per_stream_category": NO_EVIDENCE_QUOTA_PER_STREAM_CATEGORY,
            "selected_units": len(no_evidence_selected),
            "note": "build_review_queue skips units with no substantive segments; this stratum is added separately.",
        },
        "total_reviewer_rows": len(review_rows),
        "blinding": "System A / System B labels only in the reviewer-facing CSV; real identity in unblinding_key.json.",
        "instructions_for_reviewer": [
            "For each selection_id + extracted_phrase, judge only: does this phrase, read against the report, support the presence or absence decision named by 'category'?",
            "Ignore 'audit_stage' while judging; it is analysis metadata, not a hint.",
            "Mark reviewer_supports_category as one of: yes / no / partial / unclear.",
            "This is not a diagnostic-correctness review. It asks whether the model's stated reason is actually grounded in and relevant to the report.",
        ],
        "boundaries": [
            "This is a bounded review workload, not a prevalence sample or performance estimate.",
            "Contains real de-identified report text and model output text; must not leave governed storage.",
            "Not required for or referenced by the current JBHI revision.",
        ],
    }

    if args.dry_run:
        print(json.dumps({"total_reviewer_rows": len(review_rows), "summary": summary}, indent=2))
        return

    atomic_write_csv(OUTPUT_DIR / "evidence_review_package.csv", review_frame)
    atomic_write_json(OUTPUT_DIR / "unblinding_key.json", unblinding_key)
    atomic_write_json(OUTPUT_DIR / "package_receipt.json", receipt)
    print(f"Wrote {len(review_rows)} reviewer rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
