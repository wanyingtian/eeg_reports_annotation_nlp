#!/usr/bin/env python3
"""Build or verify an independent clinical-reader delivery packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from eeg_review.io import atomic_write_json

GOVERNED_ROOT = Path(
    "/Users/sbergner/Research/eeg/eeg_reports_annotation_nlp/data/governed/study-runs"
)
PACKAGE = (
    GOVERNED_ROOT
    / "jbhi-medgemma-v1-evidence-development-20260916/review-source-first-v2"
)
OUTPUT = (
    GOVERNED_ROOT
    / "jbhi-medgemma-v1-evidence-development-20260916/clinical-reader-independent-v1"
)
DELIVERY_FILES = (
    "01_source_first_review.csv",
    "02_blinded_evidence_review.csv",
    "03_blinded_pair_comparison.csv",
    "README.txt",
    "review_form.html",
)
FORBIDDEN_FILENAMES = {
    "analysis_metadata.json",
    "unblinding_key.json",
    "blinded_review_summary.json",
    "claude-technical-reader-pass1.json",
}
FORBIDDEN_TEXT = (
    "MedGemma",
    "Mistral",
    "technical-reader",
    "technical reader",
    "priority case",
    "focus_stratum",
    "normalization_only",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _protect_tree(path: Path) -> None:
    path.chmod(0o700)
    for child in path.iterdir():
        if child.is_dir():
            _protect_tree(child)
        else:
            child.chmod(0o600)


def _delivery_hashes(path: Path) -> dict[str, str]:
    missing = [name for name in DELIVERY_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"clinical-reader delivery is missing files: {missing}")
    return {name: sha256_path(path / name) for name in DELIVERY_FILES}


def verify_delivery(path: Path = OUTPUT) -> dict[str, Any]:
    completion = json.loads(
        (path / "DELIVERY_COMPLETE.json").read_text(encoding="utf-8")
    )
    actual = _delivery_hashes(path)
    if actual != completion["output_sha256"]:
        raise ValueError("clinical-reader delivery hashes do not match completion record")
    present = {item.name for item in path.rglob("*") if item.is_file()}
    leaked_files = sorted(present & FORBIDDEN_FILENAMES)
    if leaked_files:
        raise ValueError(f"clinical-reader delivery contains withheld files: {leaked_files}")
    reviewer_text = "\n".join(
        (path / name).read_text(encoding="utf-8") for name in DELIVERY_FILES
    )
    leaked_text = [value for value in FORBIDDEN_TEXT if value in reviewer_text]
    if leaked_text:
        raise ValueError(f"clinical-reader delivery leaks withheld context: {leaked_text}")
    allowed_top_level = set(DELIVERY_FILES) | {"DELIVERY_COMPLETE.json"}
    unexpected_top_level = sorted(
        item.name
        for item in path.iterdir()
        if item.is_file() and item.name not in allowed_top_level
    )
    if unexpected_top_level:
        raise ValueError(
            f"clinical-reader delivery contains unexpected files: {unexpected_top_level}"
        )
    if any(item.stat().st_mode & 0o077 for item in path.rglob("*")):
        raise ValueError("clinical-reader delivery contains accessible files")
    response_files = list((path / "responses").glob("*.json"))
    return {
        "files_verified": len(actual),
        "independent_reader_results_present": bool(response_files),
        "withheld_files_present": False,
    }


def build_delivery() -> dict[str, Any]:
    package_complete = PACKAGE / "COMPLETE.json"
    if not package_complete.is_file():
        raise ValueError("frozen source-first review package is incomplete")
    OUTPUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in DELIVERY_FILES:
        if name == "README.txt":
            continue
        shutil.copyfile(PACKAGE / name, OUTPUT / name)
    _write_text(
        OUTPUT / "README.txt",
        "INDEPENDENT CLINICAL EVIDENCE REVIEW\n\n"
        "Purpose: complete an independent review of all 20 blinded cases.\n\n"
        "1. Do not consult another reader's notes, counts or interpretation.\n"
        "2. Open review_form.html locally; it makes no network requests.\n"
        "3. Read each source report before opening either blinded system panel.\n"
        "4. Judge both systems independently, then record the paired preference.\n"
        "5. Use 'unclear' rather than infer beyond your clinical qualifications.\n"
        "6. Download the completed JSON and move it immediately to an authorized "
        "governed return location. Do not return it by ordinary email.\n\n"
        "The CSV files provide the same staged material for table-oriented review.\n"
        "This packet contains de-identified clinical text and must remain in governed "
        "storage. It is a purposive development review, not a prevalence sample or "
        "clinical-performance estimate. System identity remains concealed until the "
        "independent reviews have been compared and disagreements adjudicated.\n",
    )
    (OUTPUT / "responses").mkdir(exist_ok=True, mode=0o700)
    completion = {
        "study_id": "jbhi-02463-evidence-independent-clinical-review-v1",
        "status": "complete_frozen_awaiting_independent_reader",
        "source_instrument_complete_sha256": sha256_path(package_complete),
        "output_sha256": _delivery_hashes(OUTPUT),
        "exclusions": [
            "other-reader responses and summaries",
            "system identities",
            "selection and automation metadata",
            "automated focus strata",
        ],
    }
    atomic_write_json(OUTPUT / "DELIVERY_COMPLETE.json", completion)
    _protect_tree(OUTPUT)
    return verify_delivery(OUTPUT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_delivery() if args.verify_only else build_delivery()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
