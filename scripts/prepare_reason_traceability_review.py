#!/usr/bin/env python3
"""Build a governed human-review queue from the frozen traceability ledger."""

from __future__ import annotations

import argparse
import html
import json
import os
import secrets
from pathlib import Path

import pandas as pd

from eeg_review.error_review import review_handle
from eeg_review.explanation_reconciliation import (
    ID_COLUMN,
    load_explanation_artifact,
    sha256_file,
)
from eeg_review.io import atomic_write_csv, atomic_write_json
from eeg_review.reason_traceability import (
    build_review_queue,
    historical_polarity_units,
)
from eeg_review.source_grounding import text_sha
from eeg_review.study_integrity import decision_lenses

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_STREAM = "historical_mistral_saved_polarity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traceability-run", type=Path, required=True)
    parser.add_argument("--historical-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--acknowledge-governed-output", action="store_true")
    return parser.parse_args()


def read_json(file_path: Path) -> dict:
    return json.loads(file_path.read_text(encoding="utf-8"))


def optional_number(value):
    return None if value is None or pd.isna(value) else float(value)


def verify_traceability_run(run_root: Path) -> tuple[Path, dict]:
    completion = read_json(run_root / "COMPLETE.json")
    aggregate_path = run_root / "aggregate-traceability.json"
    ledger_path = run_root / "governed-segment-ledger.csv"
    if sha256_file(aggregate_path) != completion["aggregate_sha256"]:
        raise ValueError("traceability aggregate changed after completion")
    if sha256_file(ledger_path) != completion["governed_case_ledger_sha256"]:
        raise ValueError("traceability ledger changed after completion")
    aggregate = read_json(aggregate_path)
    governed = aggregate["governed_case_ledger"]
    if governed["sha256"] != sha256_file(ledger_path):
        raise ValueError("aggregate and completion receipts disagree on the ledger")
    if governed["contains_report_or_reason_text"] is not False:
        raise ValueError("unexpected traceability-ledger disclosure state")
    return ledger_path, aggregate


def verify_historical_rows(
    ledger: pd.DataFrame,
    units,
) -> list[dict]:
    required = {
        "stream",
        "report_key",
        "report_text_sha256",
        "unit_number",
        "segment_number",
        "category",
        "segment_role",
        "source_kind",
        "segment_sha256",
        "stage",
        "verified_quote",
    }
    if missing := sorted(required - set(ledger.columns)):
        raise ValueError(f"traceability ledger is missing columns: {missing}")
    frame = ledger.loc[ledger["stream"] == HISTORICAL_STREAM].copy()
    expected_segments = sum(len(unit.segments) for unit in units)
    if len(frame) != expected_segments:
        raise ValueError("historical ledger segment denominator changed")
    seen: set[tuple[int, int]] = set()
    records: list[dict] = []
    for row in frame.to_dict("records"):
        unit_number = int(row["unit_number"])
        segment_number = int(row["segment_number"])
        if (unit_number, segment_number) in seen:
            raise ValueError("historical ledger contains a duplicate segment")
        seen.add((unit_number, segment_number))
        if not 0 <= unit_number < len(units):
            raise ValueError("historical ledger unit number is out of range")
        unit = units[unit_number]
        if not 0 <= segment_number < len(unit.segments):
            raise ValueError("historical ledger segment number is out of range")
        segment = unit.segments[segment_number]
        expected = {
            "report_key": unit.report_key,
            "report_text_sha256": text_sha(unit.report),
            "category": unit.category,
            "segment_role": unit.segment_roles[segment_number],
            "source_kind": unit.source_kind,
            "segment_sha256": text_sha(segment),
        }
        if any(str(row[key]) != str(value) for key, value in expected.items()):
            raise ValueError("historical source and traceability ledger disagree")
        records.append(row)
    return records


def render_review_html(items: list[dict]) -> str:
    sections = [
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Governed EEG evidence review</title><style>
body{font:17px/1.5 Georgia,serif;max-width:980px;margin:2em auto;padding:0 1em;color:#182330}
article{border-top:2px solid #ccd7df;margin:2em 0;padding-top:1em}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:1em;
font:14px/1.45 monospace}
summary{cursor:pointer;color:#234f73;font-weight:bold}.meta{color:#53616d}.warning{background:#fff1d8;padding:1em}
</style><h1>Evidence traceability review</h1>
<p class="warning"><strong>Governed local review only.</strong> Do not commit, email,
or circulate this case-level document. Review source text before opening the saved
model output. A located phrase is not
automatically relevant, clinically correct, or the cause of a model decision.</p>
<p>The queue includes every unresolved historical evidence unit plus deterministic contrast samples.
It is a workload, not a prevalence sample or performance estimate. Record judgments in
<code>review-form.csv</code>.</p>"""
    ]
    for item in items:
        sections.append(
            f'<article id="{html.escape(item["case_handle"])}">'
            f'<h2>{html.escape(item["case_handle"])} — {html.escape(item["category"])}</h2>'
            f'<p class="meta">Priority: {html.escape(item["stratum"])}; '
            f'{len(item["segments"])} saved segment(s).</p>'
            f'<h3>1. Read the report</h3><pre>{html.escape(item["report"])}</pre>'
            '<details><summary>2. Reveal the saved model decision and evidence</summary>'
            f'<p>Four-level decision: <strong>{item["decision"]["four_level_decision"]}</strong>; '
            f'core call: <strong>{html.escape(item["decision"]["core_call"])}</strong>; '
            "declared confidence: "
            f'<strong>{html.escape(item["decision"]["declared_confidence"])}</strong>. '
            "This is an ordinal model output, not a calibrated probability.</p><ol>"
        )
        for segment in item["segments"]:
            scores = []
            for name in (
                "fuzzy_sentence_max",
                "semantic_sentence_max",
                "semantic_whole_report",
            ):
                value = segment.get(name)
                if value is not None and not pd.isna(value):
                    scores.append(f"{name}={value}")
            score_text = "; ".join(scores) if scores else "no weaker-stage score"
            sections.append(
                f'<li><pre>{html.escape(segment["text"])}</pre>'
                f'<p class="meta">Audit stage: {html.escape(segment["stage"])}; '
                f'{html.escape(score_text)}</p></li>'
            )
        sections.append("</ol></details></article>")
    return "".join(sections) + "</html>\n"


def run(args: argparse.Namespace) -> None:
    os.umask(0o077)
    run_root = args.traceability_run.expanduser().resolve(strict=True)
    artifact_path = args.historical_artifact.expanduser().resolve(strict=True)
    output_root = args.output_dir.expanduser().resolve()
    governed_root = (ROOT / "data/governed/analysis-runs").resolve()
    if not output_root.is_relative_to(governed_root) or output_root == governed_root:
        raise ValueError("output must be a dedicated directory under governed analysis-runs")

    ledger_path, aggregate = verify_traceability_run(run_root)
    artifact = load_explanation_artifact(artifact_path)
    artifact_by_key = artifact.set_index(ID_COLUMN)
    units = historical_polarity_units(artifact)
    rows = verify_historical_rows(pd.read_csv(ledger_path), units)
    selected, public_summary = build_review_queue(rows)
    safe_preview = {
        **public_summary,
        "decision_lens": {
            "four_level_semantics": (
                "1 confident absent; 2 low-confidence absent; "
                "3 low-confidence present; 4 confident present"
            ),
            "core_mapping": "1-2 absent; 3-4 present",
            "probability_calibration_claimed": False,
        },
        "inference_performed": False,
        "historical_units_verified": len(units),
        "historical_segments_verified": len(rows),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": "passed", **safe_preview}, indent=2))
        return
    if not args.acknowledge_governed_output:
        raise ValueError("explicit governed-output acknowledgement is required")

    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    intake = {
        "schema_version": 1,
        "traceability_analysis_id": aggregate["analysis_id"],
        "traceability_aggregate_sha256": sha256_file(
            run_root / "aggregate-traceability.json"
        ),
        "traceability_ledger_sha256": sha256_file(ledger_path),
        "historical_artifact_sha256": sha256_file(artifact_path),
        "implementation": {
            "review_runner_sha256": sha256_file(Path(__file__)),
            "traceability_library_sha256": sha256_file(
                ROOT / "src/eeg_review/reason_traceability.py"
            ),
            "study_integrity_library_sha256": sha256_file(
                ROOT / "src/eeg_review/study_integrity.py"
            ),
        },
        "inference_performed": False,
    }
    intake_path = output_root / "intake.json"
    if intake_path.exists() and read_json(intake_path) != intake:
        raise ValueError("resume refused: review inputs or implementation changed")
    atomic_write_json(intake_path, intake)
    completion_path = output_root / "COMPLETE.json"
    if completion_path.exists():
        for name, digest in read_json(completion_path)["outputs"].items():
            if sha256_file(output_root / name) != digest:
                raise ValueError("completed review output changed")
        print("Completed traceability review package verified; no recomputation.")
        return

    salt_path = output_root / "private-handle-salt.json"
    if not salt_path.exists():
        atomic_write_json(salt_path, {"salt": secrets.token_hex(32)})
    salt = read_json(salt_path)["salt"]

    selected_items = []
    private_index = []
    review_rows = []
    for position, selection in enumerate(selected, 1):
        unit = units[int(selection["unit_number"])]
        matching = sorted(
            (
                row
                for row in rows
                if int(row["unit_number"]) == int(selection["unit_number"])
            ),
            key=lambda row: int(row["segment_number"]),
        )
        case_handle = review_handle(
            f'{HISTORICAL_STREAM}:{unit.report_key}:{unit.category}', salt
        )
        decision = decision_lenses(artifact_by_key.at[unit.report_key, unit.category])
        segments = []
        for row in matching:
            segment_number = int(row["segment_number"])
            segments.append(
                {
                    "segment_number": segment_number,
                    "text": unit.segments[segment_number],
                    "stage": str(row["stage"]),
                    "fuzzy_sentence_max": optional_number(row.get("fuzzy_sentence_max")),
                    "semantic_sentence_max": optional_number(
                        row.get("semantic_sentence_max")
                    ),
                    "semantic_whole_report": optional_number(
                        row.get("semantic_whole_report")
                    ),
                }
            )
            review_rows.append(
                {
                    "queue_position": position,
                    "case_handle": case_handle,
                    "category": unit.category,
                    "stratum": selection["stratum"],
                    "four_level_decision": decision["four_level_decision"],
                    "core_call": decision["core_call"],
                    "declared_confidence": decision["declared_confidence"],
                    "segment_number": segment_number,
                    "audit_stage": str(row["stage"]),
                    "source_present": "",
                    "category_role": "",
                    "clinically_relevant": "",
                    "reviewer": "",
                    "review_date": "",
                    "notes": "",
                }
            )
        selected_items.append(
            {
                "case_handle": case_handle,
                "category": unit.category,
                "stratum": selection["stratum"],
                "decision": decision,
                "report": unit.report,
                "segments": segments,
            }
        )
        private_index.append(
            {
                "case_handle": case_handle,
                "report_key": unit.report_key,
                "unit_number": int(selection["unit_number"]),
                "report_text_sha256": text_sha(unit.report),
                "category": unit.category,
            }
        )

    atomic_write_csv(output_root / "review-form.csv", pd.DataFrame(review_rows))
    atomic_write_csv(output_root / "private-case-index.csv", pd.DataFrame(private_index))
    atomic_write_json(output_root / "case-sources.json", selected_items)
    temporary_html = output_root / "review.html.tmp"
    temporary_html.write_text(render_review_html(selected_items), encoding="utf-8")
    temporary_html.replace(output_root / "review.html")
    atomic_write_json(output_root / "aggregate-summary.json", safe_preview)
    output_names = [
        "aggregate-summary.json",
        "case-sources.json",
        "intake.json",
        "private-case-index.csv",
        "private-handle-salt.json",
        "review-form.csv",
        "review.html",
    ]
    outputs = {name: sha256_file(output_root / name) for name in output_names}
    atomic_write_json(
        completion_path,
        {
            "schema_version": 1,
            "outputs": outputs,
            "inference_performed": False,
            "distribution": "governed_local_author_review_only",
        },
    )
    print(json.dumps(safe_preview, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
