#!/usr/bin/env python3
"""Prepare local-only paired review and lexical overlap checks from frozen runs."""

from __future__ import annotations

import argparse
import html
import json
import os
import secrets
from pathlib import Path

import pandas as pd

from eeg_review.audit import DEFAULT_LABELS
from eeg_review.error_review import review_handle
from eeg_review.io import atomic_write_csv, atomic_write_json, load_table
from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL
from eeg_review.manifest import build_manifest, sha256_file
from eeg_review.protected_execution import assert_governed_run_active
from eeg_review.reviewability import KEY, POLICY, cohort_pairs, exact_frame, near_duplicate_pair
from eeg_review.reviewability import paired_packet as make_packet
from eeg_review.source_grounding import inspect_grounding, text_sha

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {"zoe_evaluation_1395": 1395, "maria_evaluation_499": 499}


def read(path):
    return json.loads(path.read_text())


def verify_bundle(root):
    assert_governed_run_active(root)
    manifest = root / "final-transfer-manifest.json"
    seen = set()
    for item in read(manifest)["files"]:
        path = (root / item["path"]).resolve()
        if path in seen or not path.is_relative_to(root) or Path(item["path"]).is_absolute():
            raise ValueError("unsafe or duplicate manifest path")
        seen.add(path)
        if sha256_file(path) != item["sha256"]:
            raise ValueError("frozen source manifest mismatch")
    return {"files_verified": len(seen), "manifest_sha256": sha256_file(manifest)}


def source_paths(native, original):
    result = {"development": original / "inputs/zoe_development_100.db"}
    for cohort in COHORTS:
        short = cohort.split("_")[0]
        result.update(
            {
                f"{cohort}/reference": native / f"inputs/{cohort}.db",
                f"{cohort}/medgemma": native / f"products/{cohort}/processed/predictions.db",
                f"{cohort}/mistral": native / f"comparators/{cohort}_reproduced_mistral.csv",
                f"{cohort}/mistral_raw": original / f"products/llm/{short}/raw.csv",
                f"{cohort}/comparison": native
                / f"analysis/{cohort}/vs_reproduced_mistral/paired_comparison_summary.json",
            }
        )
    return result


def review_html(records):
    parts = [
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Governed paired case review</title><style>
body{font:17px/1.5 Georgia,serif;max-width:950px;margin:2em auto;padding:0 1em;color:#182330}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.5 monospace;
background:#f3f5f7;padding:1em}
article{border-top:2px solid #ced7df;margin:2em 0;padding-top:1em}
summary{cursor:pointer;color:#254e74}
nav{columns:3}table{border-collapse:collapse}td,th{padding:.3em .6em;border-bottom:1px solid #ccc}
</style><h1>Paired case review</h1><p><strong>Governed: local author review only. Do not email,
commit, or circulate this case-level file.</strong></p><p>Read the source first, record your
own assessment in review.csv, then open the reference/model comparison. Labels are relative
to the Reference Annotator, not adjudicated clinical truth. The sample deliberately includes
corrections, regressions, shared errors and correct controls; it is not prevalence-representative.
Model explanations are decision-conditioned quotations, not causal reasoning. MedGemma's
full evaluation was classification-only: no matched explanation exists for these cases.
Clinical review has not yet occurred. No patient linkage is assumed.</p><nav>"""
    ]
    for row in records:
        handle = html.escape(row["case_handle"])
        parts.append(f'<p><a href="#{handle}">{handle}</a></p>')
    parts.append("</nav>")
    for row in records:
        handle = html.escape(row["case_handle"])
        parts.append(
            f'<article id="{handle}"><h2>{handle}</h2><pre>'
            + html.escape(row["report_text"])
            + "</pre>"
        )
        parts.append(
            "<details><summary>Reference/model comparison and saved Mistral evidence</summary>"
        )
        parts.append(
            "<table><tr><th>Category</th><th>Reference</th><th>MedGemma</th><th>Mistral</th></tr>"
        )
        for cell in row["selected_cells"]:
            parts.append(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(cell[k]))}</td>"
                    for k in ("label", "reference_level", "medgemma_level", "mistral_level")
                )
                + "</tr>"
            )
        parts.append(
            "</table><p>All original Mistral phrases are retained below. Only exact"
            " source spans are accepted as verbatim; other matches remain diagnostics.</p><pre>"
        )
        parts.append(
            html.escape(json.dumps(row["mistral_grounding"], indent=2, ensure_ascii=False))
        )
        parts.append("</pre></details></article>")
    return "".join(parts) + "</html>\n"


def run(args):
    os.umask(0o077)
    native, original, output = (
        p.expanduser().resolve() for p in (args.native_run, args.original_run, args.output_dir)
    )
    governed = ROOT / "data/governed/analysis-runs"
    if not output.is_relative_to(governed.resolve()) or output == governed.resolve():
        raise ValueError("output must be a dedicated directory under data/governed/analysis-runs")
    if not args.acknowledge_governed_output:
        raise ValueError("explicit governed-output acknowledgement is required")
    assert_governed_run_active(original)
    assert_governed_run_active(output)
    source_receipt = verify_bundle(native)
    paths = source_paths(native, original)
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    code_hashes = {
        p.name: sha256_file(p)
        for p in (
            Path(__file__),
            ROOT / "src/eeg_review/reviewability.py",
            ROOT / "src/eeg_review/source_grounding.py",
        )
    }
    signature = {
        "policy": POLICY,
        "inputs": hashes,
        "code": code_hashes,
        "native_bundle": source_receipt,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": "passed",
                    "input_files": len(paths),
                    "source_bundle_files": source_receipt["files_verified"],
                    "inference": False,
                    "policy": POLICY,
                },
                indent=2,
            )
        )
        return
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (output / "intake.json").exists():
        if read(output / "intake.json") != signature:
            raise ValueError("resume refused: source, policy or code changed")
    elif any(output.iterdir()):
        raise ValueError("refusing an unrelated nonempty output directory")
    else:
        atomic_write_json(output / "intake.json", signature)
    if (output / "COMPLETE.json").exists():
        for name, digest in read(output / "COMPLETE.json")["outputs"].items():
            if sha256_file(output / name) != digest:
                raise ValueError("completed output changed")
        print("Completed review package verified; no recomputation.")
        return
    salt_path = output / "private-handle-salt.json"
    if not salt_path.exists():
        atomic_write_json(salt_path, {"salt": secrets.token_hex(32)})
    salt = read(salt_path)["salt"]
    summary = {
        "schema_version": 1,
        "policy": POLICY,
        "source_bundle": source_receipt,
        "cohorts": {},
        "near_duplicates": {},
        "inference_performed": False,
        "classifications_changed": False,
        "cohort_exclusions_changed": False,
        "patient_grouped_analysis": {
            "status": "not_run_missing_validated_patient_key",
            "no_patient_identity_inferred": True,
        },
    }
    dev = load_table(paths["development"], [KEY, "Report", *DEFAULT_LABELS])
    if len(dev) != 100:
        raise ValueError("development denominator changed")
    exact_frame(dev, dev[KEY].tolist())
    cohorts = {"zoe_development_100": dev}
    records, packets, private_index = [], [], []
    for cohort, expected in COHORTS.items():
        ref = load_table(paths[f"{cohort}/reference"], [KEY, "Report", *DEFAULT_LABELS])
        a = load_table(paths[f"{cohort}/medgemma"], [KEY, *DEFAULT_LABELS], "classifications")
        b = load_table(paths[f"{cohort}/mistral"], [KEY, *DEFAULT_LABELS])
        if len(ref) != expected:
            raise ValueError("evaluation denominator changed")
        cohorts[cohort] = ref
        counts, packet = make_packet(ref, a, b, cohort=cohort, salt=salt)
        frozen = read(paths[f"{cohort}/comparison"])
        for label, row in counts["labels"].items():
            original_comparison = frozen["labels"][label]
            discord = original_comparison["discordant_correctness"]["core_accuracy"]
            if (
                row["eligible"]["medgemma_correct_mistral_wrong"] != discord["a_correct_b_wrong"]
                or row["eligible"]["medgemma_wrong_mistral_correct"] != discord["a_wrong_b_correct"]
            ):
                raise ValueError("paired counts disagree with frozen comparison")
            for model, pointkey in (
                ("medgemma", "model_a_point_estimates"),
                ("mistral", "model_b_point_estimates"),
            ):
                point = original_comparison[pointkey]
                if row[f"{model}_errors"] != point["fp"] + point["fn"]:
                    raise ValueError("error counts disagree with frozen comparison")
        raw = pd.read_csv(paths[f"{cohort}/mistral_raw"])
        if raw[KEY].isna().any() or raw[KEY].duplicated().any():
            raise ValueError("ambiguous original Mistral outputs")
        if not set(ref[KEY]).issubset(set(raw[KEY])):
            raise ValueError("Mistral raw outputs omit evaluation cases")
        raw = raw.set_index(KEY)
        b = b.set_index(KEY)
        selected = set(packet.case_handle)
        evidence_statuses = {}
        for _, row in ref.iterrows():
            key = row[KEY]
            handle = review_handle(f"{cohort}:{key}", salt)
            private_index.append({KEY: key, "cohort": cohort, "case_handle": handle})
            if handle not in selected:
                continue
            saved = raw.loc[key]
            fixed = json.loads(saved.classifications)
            if any(fixed[k] != int(b.loc[key, label]) for k, label in JSON_KEY_TO_LABEL.items()):
                raise ValueError("raw Mistral labels differ from evaluated predictions")
            grounding = inspect_grounding(
                saved.explanations, report=row.Report, fixed=saved.classifications
            )
            for cell in grounding["cells"].values():
                for reason in cell["reasons"]:
                    status = reason["status"]
                    evidence_statuses[status] = evidence_statuses.get(status, 0) + 1
            records.append(
                {
                    "case_handle": handle,
                    "cohort": cohort,
                    "report_text": row.Report,
                    "report_text_sha256": text_sha(row.Report),
                    "selected_cells": packet.loc[packet.case_handle == handle].to_dict("records"),
                    "medgemma_explanation_status": "not_collected_classification_only",
                    "mistral_grounding": grounding,
                }
            )
        counts["mistral_phrase_statuses_in_selected_reports"] = evidence_statuses
        counts["medgemma_explanation_records"] = 0
        summary["cohorts"][cohort] = counts
        packets.append(packet)
    for frame in cohorts.values():
        if frame[KEY].duplicated().any() or frame.Report.isna().any():
            raise ValueError("duplicate keys or missing text in cohort")
    for left, right in cohort_pairs(cohorts):
        checkpoint = output / f"near-duplicate-{left}--{right}.json"
        checksum = checkpoint.with_suffix(".receipt.json")
        if checkpoint.exists():
            if not checksum.exists() or read(checksum).get("sha256") != sha256_file(checkpoint):
                raise ValueError("partial checkpoint changed or lacks a completion receipt")
            saved = read(checkpoint)
        else:
            counts, pairs = near_duplicate_pair(
                cohorts[left], cohorts[right], left_name=left, right_name=right, salt=salt
            )
            saved = {"summary": counts, "pairs": pairs}
            atomic_write_json(checkpoint, saved)
            atomic_write_json(checksum, {"sha256": sha256_file(checkpoint)})
        summary["near_duplicates"][f"{left}--{right}"] = saved["summary"]
        print(json.dumps({"completed_pair": f"{left}--{right}", **saved["summary"]}))
    if source_receipt != verify_bundle(native) or hashes != {
        key: sha256_file(path) for key, path in paths.items()
    }:
        raise ValueError("source changed during analysis")
    assert_governed_run_active(original)
    assert_governed_run_active(output)
    packet = pd.concat(packets, ignore_index=True)
    atomic_write_csv(output / "review.csv", packet)
    atomic_write_csv(output / "private-case-index.csv", pd.DataFrame(private_index))
    atomic_write_json(output / "case-sources.json", records)
    (output / "case-review.html").write_text(review_html(records), encoding="utf-8")
    summary["selected_unique_reports"] = len(records)
    summary["selected_label_case_rows"] = len(packet)
    summary["cohort_arithmetic"] = {
        "transport": 100,
        "zoe_evaluation": COHORTS["zoe_evaluation_1395"],
        "maria_evaluation": COHORTS["maria_evaluation_499"],
        "evaluation": sum(COHORTS.values()),
        "transport_plus_evaluation": 100 + sum(COHORTS.values()),
    }
    atomic_write_json(output / "aggregate-summary.json", summary)
    atomic_write_json(
        output / "run_manifest.json",
        build_manifest(
            "prepare-comparison-review",
            list(paths.values()),
            POLICY,
            privacy_boundary=(
                "governed case review, source text and key map; aggregate summary separate"
            ),
        ),
    )
    outputs = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
    atomic_write_json(output / "COMPLETE.json", {"outputs": outputs, "inference_performed": False})
    print(
        json.dumps(
            {
                "completed": True,
                "selected_unique_reports": len(records),
                "selected_label_case_rows": len(packet),
                "inference_performed": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-governed-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())
