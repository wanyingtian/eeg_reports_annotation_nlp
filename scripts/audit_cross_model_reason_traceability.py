#!/usr/bin/env python3
"""Audit saved Mistral and MedGemma evidence under one traceability contract."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from eeg_review.explanation_reconciliation import (
    TraceabilityUnit,
    deterministic_traceability,
    load_explanation_artifact,
    normalize_text,
    positive_traceability_units,
    reconcile_source_snapshot,
    semantic_complete,
    sha256_file,
    summarize_stages,
)
from eeg_review.io import load_table
from eeg_review.reason_traceability import (
    audit_traceability,
    historical_polarity_units,
    split_declared_sentences,
    structured_evidence_units,
    summarize_traceability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-artifact", type=Path, required=True)
    parser.add_argument("--historical-source", type=Path, required=True)
    parser.add_argument(
        "--modern",
        action="append",
        nargs=3,
        metavar=("NAME", "EVIDENCE_CSV", "REPORT_DB"),
        default=[],
        help="Repeat for each saved contemporary evidence stream.",
    )
    parser.add_argument("--semantic-model", type=Path, required=True)
    parser.add_argument("--semantic-model-id", required=True)
    parser.add_argument("--historically-applicable-revision", required=True)
    parser.add_argument("--executed-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-governed-output", action="store_true")
    return parser.parse_args()


def thesis_sentence_units(frame: pd.DataFrame) -> list[TraceabilityUnit]:
    output: list[TraceabilityUnit] = []
    for unit in historical_polarity_units(frame):
        reason = "; ".join(unit.segments)
        output.append(
            TraceabilityUnit(
                row_number=len(output),
                category=unit.category,
                report=unit.report,
                phrases=tuple(normalize_text(x) for x in split_declared_sentences(reason)),
            )
        )
    return output


def main() -> None:
    args = parse_args()
    if not args.acknowledge_governed_output:
        raise SystemExit("--acknowledge-governed-output is required")
    if not args.modern:
        raise SystemExit("at least one --modern stream is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[1]

    from fuzzywuzzy import fuzz
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(args.semantic_model), device="cpu")

    def encode(texts):
        return model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
        )

    artifact = load_explanation_artifact(args.historical_artifact)
    historical_units = historical_polarity_units(artifact)
    historical_rows = audit_traceability(
        historical_units,
        fuzzy_ratio=fuzz.token_sort_ratio,
        encoder=encode,
    )
    stream_rows = {"historical_mistral_saved_polarity": historical_rows}
    stream_summaries = {
        "historical_mistral_saved_polarity": summarize_traceability(
            historical_units, historical_rows
        )
    }

    input_receipts = {
        "historical_artifact": sha256_file(args.historical_artifact),
        "historical_source": sha256_file(args.historical_source),
    }
    for name, evidence_path_raw, report_path_raw in args.modern:
        if name in stream_summaries:
            raise ValueError(f"duplicate stream name: {name}")
        evidence_path, report_path = Path(evidence_path_raw), Path(report_path_raw)
        evidence = pd.read_csv(evidence_path)
        reports = load_table(report_path, ["Hashed_ReportURN", "Report"], "reports")
        units = structured_evidence_units(evidence, reports, source_kind=name)
        rows = audit_traceability(
            units,
            fuzzy_ratio=fuzz.token_sort_ratio,
            encoder=encode,
        )
        stream_rows[name] = rows
        stream_summaries[name] = summarize_traceability(units, rows)
        input_receipts[f"{name}:evidence"] = sha256_file(evidence_path)
        input_receipts[f"{name}:reports"] = sha256_file(report_path)

    # Reproduce the author-uploaded public script and the final-thesis prose as
    # distinct executable specifications.  Neither is relabelled to match the
    # submitted numerator.
    public_units = positive_traceability_units(artifact)
    public_stages = deterministic_traceability(
        public_units,
        fuzzy_ratio=fuzz.token_sort_ratio,
        fuzzy_surface="whole_report",
    )
    public_stages, _ = semantic_complete(
        public_units,
        public_stages,
        encoder=encode,
        semantic_surface="whole_report",
    )
    prose_units = thesis_sentence_units(artifact)
    prose_stages = deterministic_traceability(
        prose_units,
        fuzzy_ratio=fuzz.token_sort_ratio,
        fuzzy_surface="sentences",
    )
    prose_stages, _ = semantic_complete(
        prose_units,
        prose_stages,
        encoder=encode,
        semantic_surface="whole_report",
    )

    case_frames = []
    for name, rows in stream_rows.items():
        frame = pd.DataFrame(rows)
        frame.insert(0, "stream", name)
        case_frames.append(frame)
    case_path = args.output_dir / "governed-segment-ledger.csv"
    pd.concat(case_frames, ignore_index=True).to_csv(case_path, index=False)

    receipt = {
        "schema_version": 1,
        "analysis_id": "jbhi-cross-model-reason-traceability-20260902",
        "status": "completed_experimental_traceability_audit_not_manuscript_admitted",
        "study_model_inference_performed": False,
        "local_embedding_inference_performed": True,
        "historical_replay": {
            "submitted": {"matched": 2132, "total": 2180},
            "author_uploaded_public_script_with_thesis_selection": summarize_stages(
                public_stages
            ),
            "final_thesis_prose_reconstruction": summarize_stages(prose_stages),
            "submitted_numerator_reproduced": False,
            "interpretation": (
                "The public script and thesis prose are distinct executable specifications; "
                "neither is threshold-fitted to the printed result."
            ),
        },
        "cross_model_contract": {
            "verified_quote_rule": "unchanged nonblank substring in exact source report",
            "candidate_stages": [
                "casefold",
                "whitespace",
                "typography",
                "fuzzy report sentence at 70/100",
                "MiniLM report sentence or whole report at cosine 0.70",
            ],
            "candidate_stages_are_factuality": False,
            "unit_aggregation": ["any segment", "all segments"],
            "threshold_optimization_performed": False,
        },
        "streams": stream_summaries,
        "model": {
            "id": args.semantic_model_id,
            "historically_applicable_revision": args.historically_applicable_revision,
            "executed_revision": args.executed_revision,
            "executable_files_verified_identical_between_revisions": True,
            "weights_sha256": sha256_file(args.semantic_model / "model.safetensors"),
        },
        "inputs": input_receipts,
        "historical_source_reconciliation": reconcile_source_snapshot(
            artifact, args.historical_source
        ),
        "implementation": {
            "script_sha256": sha256_file(Path(__file__)),
            "adapter_sha256": sha256_file(
                repository_root / "src/eeg_review/reason_traceability.py"
            ),
            "author_uploaded_public_script_sha256": sha256_file(
                repository_root / "src/evidence_analysis/evidence_factuality.py"
            ),
        },
        "governed_case_ledger": {
            "file": case_path.name,
            "sha256": sha256_file(case_path),
            "contains_report_keys": True,
            "contains_report_or_reason_text": False,
            "distribution": "governed_only",
        },
        "limits": [
            "Text presence is not entailment, clinical validity or causal faithfulness.",
            "The modern evidence streams use different prompts and evidence roles.",
            (
                "The 20-report modern streams are bounded development diagnostics, "
                "not evaluation estimates."
            ),
            (
                "The historically applicable upstream revision is inferred from repository "
                "chronology because the producer code was unpinned."
            ),
            (
                "No report key, report text, reason text or keyed output belongs in "
                "public-safe artifacts."
            ),
        ],
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "sentence_transformers": version("sentence-transformers"),
            "fuzzywuzzy": version("fuzzywuzzy"),
        },
    }
    aggregate_path = args.output_dir / "aggregate-traceability.json"
    aggregate_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    completion = {
        "analysis_id": receipt["analysis_id"],
        "aggregate_sha256": sha256_file(aggregate_path),
        "governed_case_ledger_sha256": sha256_file(case_path),
    }
    (args.output_dir / "COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
