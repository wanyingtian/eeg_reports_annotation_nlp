#!/usr/bin/env python3
# ruff: noqa: E501
"""Build and verify a compact, source-first, blinded evidence-review package.

All report and evidence text stays in governed storage. The package separates
the review into three steps: read the report first, judge each blinded system,
then compare the pair. Selection metadata and identities are kept outside the
reviewer-facing files.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from eeg_review.evidence_extraction import JSON_KEYS, classification_levels
from eeg_review.io import atomic_write_csv, atomic_write_json, load_table
from eeg_review.reason_traceability import (
    EvidenceUnit,
    audit_traceability,
    structured_evidence_units,
)
from eeg_review.source_first_review import (
    FOCUS_STRATA,
    build_source_first_sample,
    counterbalanced_aliases,
    unit_traceability_status,
)
from eeg_review.source_grounding import text_sha

ID_COLUMN = "Hashed_ReportURN"
REPORT_COLUMN = "Report"
STUDY_ID = "jbhi-02463-medgemma-v1-evidence-source-first-review-20260916-v2"
REPO_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = (
    REPO_ROOT / "scripts/build_evidence_review_package.py",
    REPO_ROOT / "src/eeg_review/source_first_review.py",
    REPO_ROOT / "src/eeg_review/reason_traceability.py",
)

GOVERNED_ROOT = Path(
    "/Users/sbergner/Research/eeg/eeg_reports_annotation_nlp/data/governed/study-runs"
)
DATASET = (
    GOVERNED_ROOT
    / "jbhi-medgemma-native-chat-development-20260829/inputs/zoe_development_native_100.db"
)
MANIFEST = (
    GOVERNED_ROOT
    / "jbhi-medgemma-native-chat-development-20260829/manifests/zoe_development_native_100.csv"
)
MEDGEMMA_EVIDENCE = (
    GOVERNED_ROOT
    / "jbhi-medgemma-v1-evidence-development-20260916/products/medgemma-v1-evidence-all100.csv"
)
MISTRAL_SAVED = GOVERNED_ROOT / "jbhi-native-20260814/products/llm/zoe/raw.csv"
OUTPUT_DIR = (
    GOVERNED_ROOT
    / "jbhi-medgemma-v1-evidence-development-20260916/review-source-first-v2"
)
SUPERSEDED_DIR = GOVERNED_ROOT / "jbhi-medgemma-v1-evidence-development-20260916/review"
FROZEN_PACKAGE_FILES = (
    "01_source_first_review.csv",
    "02_blinded_evidence_review.csv",
    "03_blinded_pair_comparison.csv",
    "README.txt",
    "analysis_metadata.json",
    "package_receipt.json",
    "review_form.html",
    "unblinding_key.json",
)

STREAMS = {
    "medgemma_native_v1_fixed_decision": {
        "path": MEDGEMMA_EVIDENCE,
        "classification_column": "fixed_classifications",
        "label": "MedGemma-27B Q2_K native-chat fixed-decision evidence",
    },
    "mistral_historical_interface_saved": {
        "path": MISTRAL_SAVED,
        "classification_column": "classifications",
        "label": "saved Mistral-7B historical-interface evidence",
    },
}
LEFT_STREAM, RIGHT_STREAM = tuple(STREAMS)

CATEGORY_LABELS = {
    "abnormality": "Overall abnormality",
    "focal_epileptiform_activity": "Focal epileptiform activity",
    "generalized_epileptiform_activity": "Generalized epileptiform activity",
    "focal_non_epileptiform_activity": "Focal non-epileptiform activity",
    "generalized_non_epileptiform_activity": "Generalized non-epileptiform activity",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stream(
    stream: str,
    reports: pd.DataFrame,
    manifest_keys: list[str],
) -> tuple[
    dict[tuple[str, str], EvidenceUnit],
    dict[tuple[str, str], int],
    dict[int, list[dict[str, Any]]],
]:
    config = STREAMS[stream]
    classification_column = str(config["classification_column"])
    frame = pd.read_csv(
        Path(config["path"]),
        usecols=[ID_COLUMN, classification_column, "explanations"],
    )
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    frame = frame.set_index(ID_COLUMN).loc[manifest_keys].reset_index()
    levels = {
        str(row[ID_COLUMN]): classification_levels(str(row[classification_column]))
        for _, row in frame.iterrows()
    }
    units = structured_evidence_units(
        frame,
        reports,
        source_kind=stream,
        classification_column=classification_column,
    )
    rows_by_unit: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_traceability(units):
        rows_by_unit[int(row["unit_number"])].append(row)
    unit_map: dict[tuple[str, str], EvidenceUnit] = {}
    decision_map: dict[tuple[str, str], int] = {}
    for unit_number, unit in enumerate(units):
        key = (unit.report_key, unit.category)
        unit_map[key] = unit
        decision_map[key] = int(levels[unit.report_key][unit.category])
        rows_by_unit.setdefault(unit_number, [])
    return unit_map, decision_map, rows_by_unit


def build_pairs(
    units_by_stream: dict[str, dict[tuple[str, str], EvidenceUnit]],
    decisions_by_stream: dict[str, dict[tuple[str, str], int]],
    rows_by_stream: dict[str, dict[int, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    left_units = units_by_stream[LEFT_STREAM]
    right_units = units_by_stream[RIGHT_STREAM]
    if set(left_units) != set(right_units):
        raise ValueError("configured streams do not cover the same report-category pairs")
    unit_numbers = {
        stream: {key: number for number, key in enumerate(units_by_stream[stream])}
        for stream in STREAMS
    }
    pairs = []
    for key, left_unit in left_units.items():
        right_unit = right_units[key]
        if left_unit.report != right_unit.report:
            raise ValueError("paired streams use different report text")
        pairs.append(
            {
                "report_key": key[0],
                "category": key[1],
                "report_text_sha256": text_sha(left_unit.report),
                "status_left": unit_traceability_status(
                    rows_by_stream[LEFT_STREAM][unit_numbers[LEFT_STREAM][key]]
                ),
                "status_right": unit_traceability_status(
                    rows_by_stream[RIGHT_STREAM][unit_numbers[RIGHT_STREAM][key]]
                ),
                "decision_left": decisions_by_stream[LEFT_STREAM][key],
                "decision_right": decisions_by_stream[RIGHT_STREAM][key],
            }
        )
    return pairs


def _numbered_phrases(unit: EvidenceUnit) -> str:
    substantive = [
        phrase
        for phrase, role in zip(unit.segments, unit.segment_roles, strict=True)
        if role != "declared_no_evidence" and phrase.strip()
    ]
    if not substantive:
        return "[No evidence phrase returned]"
    return "\n".join(f"{index}. {phrase}" for index, phrase in enumerate(substantive, 1))


def _binary_decision(level: int) -> str:
    return "present" if level >= 3 else "absent"


def _system_alias(stream: str, left_is_a: bool) -> str:
    if stream == LEFT_STREAM:
        return "System A" if left_is_a else "System B"
    return "System B" if left_is_a else "System A"


def reviewer_rows(
    selected: list[dict[str, Any]],
    units_by_stream: dict[str, dict[tuple[str, str], EvidenceUnit]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    case_ids = [f"C{index:03d}" for index in range(1, len(selected) + 1)]
    aliases = counterbalanced_aliases(case_ids)
    source_rows = []
    evidence_rows = []
    pair_rows = []
    metadata: dict[str, Any] = {}
    unblinding: dict[str, Any] = {}
    for case_id, pair in zip(case_ids, selected, strict=True):
        key = (str(pair["report_key"]), str(pair["category"]))
        unit = units_by_stream[LEFT_STREAM][key]
        level = int(pair["decision_left"])
        source_rows.append(
            {
                "case_id": case_id,
                "category": CATEGORY_LABELS[unit.category],
                "report_text": unit.report,
                "reader_category_judgment": "",
                "reader_key_source_passages": "",
                "reader_source_notes": "",
            }
        )
        case_evidence: dict[str, str] = {}
        case_mapping: dict[str, str] = {}
        for stream in STREAMS:
            alias = _system_alias(stream, aliases[case_id])
            phrases = _numbered_phrases(units_by_stream[stream][key])
            case_evidence[alias] = phrases
            case_mapping[alias] = stream
            evidence_rows.append(
                {
                    "case_id": case_id,
                    "system": alias,
                    "category": CATEGORY_LABELS[unit.category],
                    "decision_level": level,
                    "binary_decision": _binary_decision(level),
                    "evidence_phrases": phrases,
                    "source_presence": "",
                    "category_relevance": "",
                    "supports_stated_decision": "",
                    "contradicts_stated_decision": "",
                    "sufficient_as_explanation": "",
                    "no_evidence_omission_reasonable": "",
                    "reviewer_notes": "",
                }
            )
        pair_rows.append(
            {
                "case_id": case_id,
                "category": CATEGORY_LABELS[unit.category],
                "shared_decision_level": level,
                "shared_binary_decision": _binary_decision(level),
                "system_a_evidence": case_evidence["System A"],
                "system_b_evidence": case_evidence["System B"],
                "more_useful_evidence": "",
                "comparison_notes": "",
            }
        )
        metadata[case_id] = {
            "focus_stratum": pair["focus_stratum"],
            "category": unit.category,
            "report_key": pair["report_key"],
            "report_text_sha256": pair["report_text_sha256"],
            "selection_fingerprint": pair["selection_fingerprint"],
            "stream_statuses": {
                LEFT_STREAM: pair["status_left"],
                RIGHT_STREAM: pair["status_right"],
            },
        }
        unblinding[case_id] = {
            alias: {"stream": stream, "label": STREAMS[stream]["label"]}
            for alias, stream in case_mapping.items()
        }
    return (
        pd.DataFrame(source_rows),
        pd.DataFrame(evidence_rows),
        pd.DataFrame(pair_rows),
        metadata,
        unblinding,
    )


def render_html(
    source: pd.DataFrame,
    evidence: pd.DataFrame,
    pairs: pd.DataFrame,
) -> str:
    cards = []
    for source_row in source.to_dict(orient="records"):
        case_id = str(source_row["case_id"])
        system_rows = evidence[evidence["case_id"] == case_id].sort_values("system")
        pair = pairs[pairs["case_id"] == case_id].iloc[0]
        systems = []
        for row in system_rows.to_dict(orient="records"):
            systems.append(
                f"""
                <section class="system">
                  <h4>{html.escape(str(row['system']))}</h4>
                  <p><b>Declared decision:</b> level {row['decision_level']}
                     ({html.escape(str(row['binary_decision']))})</p>
                  <pre>{html.escape(str(row['evidence_phrases']))}</pre>
                  <label>Source presence<select data-field="source_presence"><option></option><option>yes</option><option>partial</option><option>no</option><option>unclear</option></select></label>
                  <label>Relevant to category<select data-field="category_relevance"><option></option><option>yes</option><option>partial</option><option>no</option><option>unclear</option></select></label>
                  <label>Supports stated decision<select data-field="supports_stated_decision"><option></option><option>yes</option><option>partial</option><option>no</option><option>unclear</option></select></label>
                  <label>Contradicts stated decision<select data-field="contradicts_stated_decision"><option></option><option>yes</option><option>partial</option><option>no</option><option>unclear</option></select></label>
                  <label>Sufficient as explanation<select data-field="sufficient_as_explanation"><option></option><option>yes</option><option>partial</option><option>no</option><option>unclear</option></select></label>
                  <label>If no phrase was returned, was that omission reasonable?<select data-field="no_evidence_omission_reasonable"><option></option><option>yes</option><option>no</option><option>unclear</option><option>not applicable</option></select></label>
                  <label>Notes<textarea data-field="system_notes"></textarea></label>
                </section>"""
            )
        cards.append(
            f"""
            <article class="case" data-case="{case_id}">
              <h2>{case_id} · {html.escape(str(source_row['category']))}</h2>
              <h3>Step 1 — Read the source before opening model evidence</h3>
              <pre class="report">{html.escape(str(source_row['report_text']))}</pre>
              <label>Your category judgment<select data-field="reader_category_judgment"><option></option><option>present</option><option>absent</option><option>unclear</option><option>not qualified</option></select></label>
              <label>Key source passages<textarea data-field="reader_key_source_passages"></textarea></label>
              <label>Source notes<textarea data-field="reader_source_notes"></textarea></label>
              <details>
                <summary>Step 2 — Reveal blinded system evidence only after Step 1</summary>
                <div class="systems">{''.join(systems)}</div>
                <h3>Step 3 — Compare the evidence sets</h3>
                <p>Both systems made the same level-{pair['shared_decision_level']} ({html.escape(str(pair['shared_binary_decision']))}) decision.</p>
                <label>More useful evidence<select data-field="more_useful_evidence"><option></option><option>System A</option><option>System B</option><option>tie</option><option>neither</option><option>unclear</option></select></label>
                <label>Comparison notes<textarea data-field="comparison_notes"></textarea></label>
              </details>
            </article>"""
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Source-first evidence review</title>
<style>
body{{font:16px/1.45 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17202a}}
.notice{{background:#eef6f8;border-left:5px solid #24788f;padding:1rem}} .case{{border-top:3px solid #37474f;padding:1.5rem 0}}
.report,pre{{white-space:pre-wrap;background:#f5f5f2;padding:1rem;border-radius:6px}} details{{margin-top:1rem;border:1px solid #b0bec5;padding:1rem;border-radius:6px}}
summary{{font-weight:700;cursor:pointer}} .systems{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}} .system{{border:1px solid #d0d7de;padding:1rem;border-radius:6px}}
label{{display:block;margin:.7rem 0;font-weight:600}} select,textarea{{display:block;width:100%;max-width:100%;margin-top:.25rem;font:inherit}} textarea{{min-height:4rem}}
button{{font:inherit;padding:.7rem 1rem;background:#245b78;color:white;border:0;border-radius:5px;cursor:pointer}} @media(max-width:750px){{.systems{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Source-first, blinded evidence review</h1>
<div class="notice"><b>Governed local document.</b> Read and record Step 1 before revealing the system evidence. System identities and automated match labels are intentionally absent. This review assesses source grounding, relevance and support—not diagnostic correctness, prevalence or model superiority. Nothing is transmitted by this file.</div>
<label>Reviewer code<input id="reviewer-code"></label>
<label>Reviewer role or qualification<input id="reviewer-role"></label>
<p><button id="save">Download review responses as JSON</button></p>
{''.join(cards)}
<script>
document.getElementById('save').addEventListener('click',()=>{{
 const cases=[]; document.querySelectorAll('.case').forEach(c=>{{
   const row={{case_id:c.dataset.case,fields:{{}},systems:[]}};
   c.querySelectorAll(':scope > label [data-field], :scope > details > label [data-field]').forEach(x=>row.fields[x.dataset.field]=x.value);
   c.querySelectorAll('.system').forEach(s=>{{const z={{system:s.querySelector('h4').textContent,fields:{{}}}};s.querySelectorAll('[data-field]').forEach(x=>z.fields[x.dataset.field]=x.value);row.systems.push(z)}});
   cases.push(row)
 }}); const out={{reviewer_code:document.getElementById('reviewer-code').value,reviewer_role:document.getElementById('reviewer-role').value,cases}};const b=new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='evidence-review-responses.json';a.click();URL.revokeObjectURL(a.href)
}})
</script></body></html>"""


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _output_hashes(path: Path) -> dict[str, str]:
    missing = [name for name in FROZEN_PACKAGE_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"frozen review package is missing files: {missing}")
    return {name: sha256_path(path / name) for name in FROZEN_PACKAGE_FILES}


def verify_package(path: Path) -> dict[str, Any]:
    completion = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
    actual = _output_hashes(path)
    if actual != completion["output_sha256"]:
        raise ValueError("review package output hashes do not match COMPLETE.json")
    source = pd.read_csv(path / "01_source_first_review.csv")
    evidence = pd.read_csv(path / "02_blinded_evidence_review.csv")
    pairs = pd.read_csv(path / "03_blinded_pair_comparison.csv")
    if len(source) != 20 or source["case_id"].nunique() != 20:
        raise ValueError("source-first review must contain exactly 20 unique cases")
    if len(evidence) != 40 or not evidence.groupby("case_id")["system"].nunique().eq(2).all():
        raise ValueError("each source-first case must have two blinded system rows")
    if len(pairs) != 20 or pairs["case_id"].nunique() != 20:
        raise ValueError("pair comparison must contain exactly 20 unique cases")
    reviewer_text = "\n".join(
        (path / name).read_text(encoding="utf-8")
        for name in (
            "README.txt",
            "01_source_first_review.csv",
            "02_blinded_evidence_review.csv",
            "03_blinded_pair_comparison.csv",
            "review_form.html",
        )
    )
    forbidden = [
        "MedGemma",
        "Mistral",
        "focus_stratum",
        "audit_stage",
        "normalization_only",
        "unresolved",
    ]
    leaked = [value for value in forbidden if value in reviewer_text]
    if leaked:
        raise ValueError(f"reviewer-facing files leak withheld metadata: {leaked}")
    if any(item.stat().st_mode & 0o077 for item in path.rglob("*")):
        raise ValueError("governed package contains group- or world-accessible files")
    return {
        "cases": len(source),
        "system_reviews": len(evidence),
        "hashes_verified": len(actual),
    }


def build_package() -> dict[str, Any]:
    reports = load_table(DATASET, [ID_COLUMN, REPORT_COLUMN])
    manifest_keys = pd.read_csv(MANIFEST, usecols=[ID_COLUMN])[ID_COLUMN].astype(str).tolist()
    units_by_stream = {}
    decisions_by_stream = {}
    rows_by_stream = {}
    for stream in STREAMS:
        units, decisions, rows = load_stream(stream, reports, manifest_keys)
        units_by_stream[stream] = units
        decisions_by_stream[stream] = decisions
        rows_by_stream[stream] = rows
    pairs = build_pairs(units_by_stream, decisions_by_stream, rows_by_stream)
    selected, summary = build_source_first_sample(
        pairs,
        categories=JSON_KEYS,
        focus_strata=FOCUS_STRATA,
    )
    source, evidence, comparisons, metadata, unblinding = reviewer_rows(
        selected, units_by_stream
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_csv(OUTPUT_DIR / "01_source_first_review.csv", source)
    atomic_write_csv(OUTPUT_DIR / "02_blinded_evidence_review.csv", evidence)
    atomic_write_csv(OUTPUT_DIR / "03_blinded_pair_comparison.csv", comparisons)
    atomic_write_json(OUTPUT_DIR / "analysis_metadata.json", metadata)
    atomic_write_json(OUTPUT_DIR / "unblinding_key.json", unblinding)
    _write_text(OUTPUT_DIR / "review_form.html", render_html(source, evidence, comparisons))
    _write_text(
        OUTPUT_DIR / "README.txt",
        "SOURCE-FIRST BLINDED EVIDENCE REVIEW\n\n"
        "1. Open review_form.html locally. It makes no network requests.\n"
        "2. For each case, record Step 1 before opening the system evidence.\n"
        "3. Judge System A and System B independently, then record the paired preference.\n"
        "4. Download the JSON response file before closing the browser.\n\n"
        "5. Validate and summarize it with scripts/analyze_evidence_review_responses.py.\n\n"
        "The three CSV files provide an equivalent staged form for table-oriented review.\n"
        "Do not open analysis_metadata.json or unblinding_key.json until review is complete.\n"
        "This package contains de-identified clinical text and must remain in governed storage.\n"
        "It is a purposive 20-case review, not a performance or prevalence sample.\n",
    )
    receipt = {
        "study_id": STUDY_ID,
        "status": "frozen_awaiting_human_review",
        "purpose": "source-first blinded review of 20 paired development report-category cases",
        "inputs": {
            str(path): sha256_path(path)
            for path in (DATASET, MANIFEST, MEDGEMMA_EVIDENCE, MISTRAL_SAVED)
        },
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_path(path)
            for path in IMPLEMENTATION_FILES
        },
        "selection_summary": summary,
        "design": [
            "one same-four-level-decision pair per category in each of four focus strata",
            "20 report-category cases and 40 blinded configured-system reviews",
            "source judgment is recorded before model evidence is revealed",
            "A/B assignment is counterbalanced per case",
            "automated strata, report keys and model identities are withheld from review files",
        ],
        "allowed_values": {
            "individual judgments": ["yes", "partial", "no", "unclear"],
            "no-evidence omission": ["yes", "no", "unclear", "not applicable"],
            "pair preference": ["System A", "System B", "tie", "neither", "unclear"],
        },
        "decision_gate": (
            "Do not launch full-cohort evidence extraction until completed human review shows "
            "that source presence, category relevance and decision support are worth scaling."
        ),
        "boundaries": [
            "purposive review workload, not a prevalence or performance estimate",
            "development surface only",
            "same exact four-level decision required across paired systems",
            "not diagnostic correctness, clinical validation or mechanistic explanation",
            "not part of or required by the current JBHI revision",
        ],
    }
    atomic_write_json(OUTPUT_DIR / "package_receipt.json", receipt)
    _protect_tree(OUTPUT_DIR)
    completion = {
        "study_id": STUDY_ID,
        "status": "complete_and_frozen",
        "output_sha256": _output_hashes(OUTPUT_DIR),
    }
    atomic_write_json(OUTPUT_DIR / "COMPLETE.json", completion)
    _protect_tree(OUTPUT_DIR)
    return verify_package(OUTPUT_DIR)


def mark_superseded() -> None:
    if not SUPERSEDED_DIR.exists():
        return
    marker = {
        "status": "superseded_do_not_review",
        "replacement": str(OUTPUT_DIR),
        "reasons": [
            "phrase-row design was not source-first",
            "automated match labels were visible to the reviewer",
            "systems were not consistently paired on the same report-category case",
            "workload was larger and imbalanced across systems",
        ],
    }
    atomic_write_json(SUPERSEDED_DIR / "SUPERSEDED_DO_NOT_USE.json", marker)
    _protect_tree(SUPERSEDED_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_package(OUTPUT_DIR), indent=2, sort_keys=True))
        return
    result = build_package()
    mark_superseded()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
