#!/usr/bin/env python3
"""Produce governed aggregate receipts for a thesis-era explanation artifact."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from eeg_review.explanation_reconciliation import (
    CATEGORIES,
    ID_COLUMN,
    artifact_census,
    correctness_by_alignment,
    deterministic_traceability,
    join_reference,
    load_explanation_artifact,
    load_manifest,
    load_reference,
    polarity_classifier_alignment,
    positive_traceability_units,
    reconcile_source_snapshot,
    semantic_complete,
    sha256_file,
    summarize_stages,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact", type=Path, required=True)
    result.add_argument("--reference", type=Path, required=True)
    result.add_argument("--evaluation-reference", type=Path, required=True)
    result.add_argument("--evaluation-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--semantic-model", type=Path)
    result.add_argument("--semantic-model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    result.add_argument("--semantic-model-revision")
    result.add_argument("--acknowledge-governed-output", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if not args.acknowledge_governed_output:
        raise SystemExit("--acknowledge-governed-output is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_explanation_artifact(args.artifact)
    full_reference = load_reference(args.reference)
    full_join = join_reference(artifact, full_reference)
    evaluation_manifest = load_manifest(args.evaluation_manifest)
    evaluation_reference = load_reference(args.evaluation_reference)
    evaluation_join = join_reference(
        artifact,
        evaluation_reference,
        manifest=evaluation_manifest,
    )

    units = positive_traceability_units(artifact)
    from fuzzywuzzy import fuzz

    historical_stages = deterministic_traceability(
        units,
        fuzzy_ratio=fuzz.token_sort_ratio,
        fuzzy_surface="whole_report",
    )
    sentence_stages = deterministic_traceability(
        units,
        fuzzy_ratio=fuzz.token_sort_ratio,
        fuzzy_surface="sentences",
    )
    semantic_receipts = {}
    stage_rows = {
        "row_number": [unit.row_number for unit in units],
        "category": [unit.category for unit in units],
        "historical_public_script_stage": historical_stages,
        "sentence_diagnostic_stage": sentence_stages,
    }
    if args.semantic_model:
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

        historical_stages, historical_maxima = semantic_complete(
            units,
            historical_stages,
            encoder=encode,
            semantic_surface="whole_report",
        )
        sentence_stages, sentence_maxima = semantic_complete(
            units,
            sentence_stages,
            encoder=encode,
            semantic_surface="sentences",
        )
        stage_rows["historical_public_script_stage"] = historical_stages
        stage_rows["sentence_diagnostic_stage"] = sentence_stages
        stage_rows["historical_public_script_max_similarity"] = historical_maxima
        stage_rows["sentence_diagnostic_max_similarity"] = sentence_maxima
        semantic_receipts = {
            "model_id": args.semantic_model_id,
            "model_revision": args.semantic_model_revision,
            "local_model_path_sha256": sha256_file(args.semantic_model / "model.safetensors"),
            "sentence_transformers_version": version("sentence-transformers"),
            "semantic_threshold": 0.70,
        }

    receipt = {
        "schema_version": 1,
        "analysis_id": "jbhi-explanation-reconciliation-20260901",
        "status": "aggregate_reconciliation_not_manuscript_admission",
        "artifact": {
            "sha256": sha256_file(args.artifact),
            "census": artifact_census(artifact),
        },
        "reference": {
            "sha256": sha256_file(args.reference),
            "rows": int(len(full_reference)),
            "source_snapshot_reconciliation": reconcile_source_snapshot(
                artifact, args.reference
            ),
        },
        "declared_evaluation": {
            "reference_sha256": sha256_file(args.evaluation_reference),
            "manifest_sha256": sha256_file(args.evaluation_manifest),
            "manifest_rows": int(len(evaluation_manifest)),
        },
        "polarity_classifier_alignment": polarity_classifier_alignment(artifact),
        "correctness_association": {
            "historical_polarity_test_surface": correctness_by_alignment(
                full_join.iloc[200:].copy(),
                surface="artifact_rows_200_to_1999_with_available_reference_labels",
            ),
            "declared_revision_evaluation_surface": correctness_by_alignment(
                evaluation_join,
                surface="author_confirmed_zoe_evaluation_1395_with_saved_polarity_available",
            ),
        },
        "traceability": {
            "selection": "saved_learned_polarity_equals_positive",
            "historical_public_script_replay": summarize_stages(historical_stages),
            "sentence_level_diagnostic_not_original_method": summarize_stages(sentence_stages),
            "submitted_claim": {"matched": 2132, "total": 2180, "fraction": 2132 / 2180},
            "semantic_receipt": semantic_receipts,
        },
        "limits": [
            (
                "The recovered artifact is a strong producing-artifact candidate, "
                "not an author confirmation."
            ),
            "The sentence-level diagnostic is not interchangeable with the submitted method.",
            (
                "Polarity correctness association is descriptive and does not establish "
                "causal faithfulness."
            ),
            (
                "The declared 1395-report classification evaluation and historical polarity "
                "test surface differ."
            ),
            (
                "No report text, report key, reason text, or keyed prediction may leave "
                "governed storage."
            ),
        ],
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "fuzzywuzzy": version("fuzzywuzzy"),
        },
    }
    aggregate_path = args.output_dir / "aggregate-reconciliation.json"
    aggregate_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(stage_rows).to_csv(
        args.output_dir / "governed-traceability-cases.csv", index=False
    )

    case_rows = []
    for surface_name, joined in (
        ("historical_polarity_test_surface", full_join.iloc[200:].copy()),
        ("declared_revision_evaluation_surface", evaluation_join),
    ):
        for category in CATEGORIES:
            polarity = joined[f"{category} Reason Polarity"]
            model = joined[f"{category}_model"]
            reference = joined[f"{category}_reference"]
            valid = (
                polarity.isin((-1, 1))
                & model.isin((1, 2, 3, 4))
                & reference.isin((1, 2, 3, 4))
            )
            model_core = model.isin((3, 4))
            reference_core = reference.isin((3, 4))
            aligned = polarity.eq(1) == model_core
            review = valid & ((~aligned) | (model_core != reference_core))
            for _, row in joined.loc[review].iterrows():
                case_rows.append(
                    {
                        ID_COLUMN: row[ID_COLUMN],
                        "surface": surface_name,
                        "category": category,
                        "polarity": int(row[f"{category} Reason Polarity"]),
                        "model_label": int(row[f"{category}_model"]),
                        "reference_label": int(row[f"{category}_reference"]),
                    }
                )
    pd.DataFrame(case_rows).to_csv(
        args.output_dir / "governed-alignment-cases.csv", index=False
    )
    completion = {
        "analysis_id": receipt["analysis_id"],
        "aggregate_sha256": sha256_file(aggregate_path),
        "case_outputs": [
            "governed-traceability-cases.csv",
            "governed-alignment-cases.csv",
        ],
    }
    (args.output_dir / "COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
