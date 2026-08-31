#!/usr/bin/env python3
"""Export a complete, revalidated interface diagnostic without case material."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean

from eeg_review.interface_diagnostic import POLICY, digest
from eeg_review.io import atomic_write_json

spec = importlib.util.spec_from_file_location(
    "diagnostic", Path(__file__).with_name("diagnose_medgemma_interface.py")
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def build_report(args):
    frame, _prompt, _template, contract = diagnostic.intake(args)
    if diagnostic.read(args.output_dir / "contract.json") != contract:
        raise ValueError("producing contract changed")
    calls = diagnostic.checkpoints(args, contract)
    if len(calls) != POLICY["max_model_calls"]:
        raise ValueError("do not export an incomplete diagnostic as complete")
    for call in calls:
        if not call["actual_input_tokens_verified"]:
            raise ValueError("actual input was not verified")
        for field in ("model_sha256", "grammar_sha256"):
            if call[field] != POLICY[field]:
                raise ValueError("model or grammar changed within diagnostic")
    diagnostic.publish(args, frame, contract)
    summary = diagnostic.read(args.output_dir / "summary.json")
    execution = diagnostic.read(args.output_dir / "execution.json")
    lookup = {(c["position"], c["arm"]): c for c in calls}
    pairs = [
        (lookup[p, "trim_only"], lookup[p, "native_chat"])
        for p in POLICY["positions_zero_based"]
    ]
    valid = [(a, b) for a, b in pairs if a["levels"] is not None and b["levels"] is not None]
    comparisons = dict(summary["comparisons"])
    comparisons["trim_only_vs_native_chat"] = {
        "completed_pairs": len(pairs),
        "valid_pairs": len(valid),
        "same_five_labels": sum(a["levels"] == b["levels"] for a, b in valid),
        "same_output_text": sum(a["text"] == b["text"] for a, b in valid),
        "same_input_tokens": sum(a["input_token_ids"] == b["input_token_ids"] for a, b in pairs),
    }
    # Positive allowlist: no input paths, report keys, text, token lists,
    # classifications, case positions or generated quotations enter the export.
    return {
        "schema_version": 1,
        "scope": "posthoc_development_interface_mechanics_not_accuracy_estimation",
        "source_revision": execution["source_revision"],
        "contract_sha256": digest(contract),
        "policy_sha256": digest(POLICY),
        "input_sha256": {k: v["sha256"] for k, v in contract["inputs"].items()},
        "producing_code_sha256": contract["code"],
        "runtime_versions": contract["versions"],
        "model_sha256": POLICY["model_sha256"],
        "prompt_sha256": POLICY["prompt_sha256"],
        "grammar_sha256": POLICY["grammar_sha256"],
        "template_sha256": POLICY["template_sha256"],
        "completed_at_utc": max(c["created_at_utc"] for c in calls),
        "call_receipts_sha256": digest(summary["call_receipts"]),
        "reports_replayed": len(POLICY["positions_zero_based"]),
        "completed_calls": summary["completed_calls"],
        "invalid_outputs": summary["invalid_outputs"],
        "actual_inputs_verified": len(calls),
        "comparisons": comparisons,
        "saved_parent_replay": summary["saved_parent_replay"],
        "saved_development_disagreements": summary["saved_development_disagreements"],
        "execution": {
            "mean_seconds_per_call": mean(c["elapsed_seconds"] for c in calls),
            "summed_inference_seconds": sum(c["elapsed_seconds"] for c in calls),
            "completion_token_range": [
                min(c["usage"]["completion_tokens"] for c in calls),
                max(c["usage"]["completion_tokens"] for c in calls),
            ],
            "finish_reasons": sorted({c["finish_reason"] for c in calls}),
        },
        "protected_evaluation": False,
        "reference_labels_used": False,
        "new_accuracy_estimate": False,
        "clinical_or_neural_mechanism_claim": False,
        "author_working_not_submission_admitted": True,
        "case_material_included": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-run", "historical-run", "output-dir", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name in ("native_run", "historical_run", "output_dir", "receipt"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    report = build_report(args)
    if args.check or args.receipt.exists():
        if not args.receipt.exists() or json.loads(args.receipt.read_text()) != report:
            raise ValueError("aggregate receipt differs; do not overwrite a frozen result")
    else:
        atomic_write_json(args.receipt, report)
    print("Verified complete aggregate-only interface receipt")


if __name__ == "__main__":
    main()
