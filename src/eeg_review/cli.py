from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import DEFAULT_LABELS, audit_dataset, audit_overlap
from .baseline import run_baseline_cv
from .calibration import calibrate_predictions
from .compare import compare_predictions
from .error_review import build_error_review_packet
from .metrics import evaluate_predictions


def named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=/path/to/dataset")
    return name, Path(path)


def prediction_mapping(value: str) -> tuple[str, str]:
    reference, separator, prediction = value.partition("=")
    if not separator or not reference or not prediction:
        raise argparse.ArgumentTypeError("expected REFERENCE_COLUMN=PREDICTION_COLUMN")
    return reference, prediction


def row_range(value: str) -> tuple[int, int]:
    start_text, separator, end_text = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("expected START:END")
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("START and END must be integers") from error
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= START < END")
    return start, end


def add_schema_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--table", default="reports")
    parser.add_argument("--id-column", default="Hashed_ReportURN")
    parser.add_argument("--report-column", default="Report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eeg-review",
        description="Aggregate-only audit and evaluation receipts for JBHI-02463-2026",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit one cohort without emitting rows")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--dataset-id", required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--label", action="append", dest="labels")
    audit.add_argument("--patient-column")
    audit.add_argument("--split-column")
    audit.add_argument(
        "--row-range",
        action="append",
        type=row_range,
        dest="row_ranges",
        help="Half-open positional source range START:END; repeat for disjoint ranges",
    )
    audit.add_argument(
        "--require-complete-labels",
        action="store_true",
        help="Exclude a candidate unless all requested labels are valid four-level values",
    )
    add_schema_arguments(audit)

    overlap = subparsers.add_parser("overlap", help="Count exact overlap across cohorts")
    overlap.add_argument("--dataset", action="append", type=named_path, required=True)
    overlap.add_argument("--output-dir", type=Path, required=True)
    overlap.add_argument("--patient-column")
    add_schema_arguments(overlap)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate paired four-level predictions")
    evaluate.add_argument("--reference", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--reference-table", default="reports")
    evaluate.add_argument("--prediction-table", default="classifications")
    evaluate.add_argument("--id-column", default="Hashed_ReportURN")
    evaluate.add_argument("--label", action="append", dest="labels")
    evaluate.add_argument("--prediction-column", action="append", type=prediction_mapping)
    evaluate.add_argument("--cluster-column")
    evaluate.add_argument(
        "--reference-range",
        action="append",
        type=row_range,
        dest="reference_row_ranges",
        help="Half-open positional source range START:END; repeat for disjoint ranges",
    )
    evaluate.add_argument(
        "--require-complete-reference",
        action="store_true",
        help=(
            "Exclude a candidate unless all requested reference labels are valid four-level values"
        ),
    )
    evaluate.add_argument(
        "--fold-column",
        help="Prediction-file column identifying the held-out fold for each row",
    )
    evaluate.add_argument("--bootstrap-iterations", type=int, default=2000)
    evaluate.add_argument("--seed", type=int, default=20260718)

    compare = subparsers.add_parser(
        "compare", help="Run paired same-case inference for two prediction surfaces"
    )
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--predictions-a", type=Path, required=True)
    compare.add_argument("--predictions-b", type=Path, required=True)
    compare.add_argument("--model-a-id", required=True)
    compare.add_argument("--model-b-id", required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--reference-table", default="reports")
    compare.add_argument("--prediction-a-table", default="classifications")
    compare.add_argument("--prediction-b-table", default="classifications")
    compare.add_argument("--id-column", default="Hashed_ReportURN")
    compare.add_argument("--label", action="append", dest="labels")
    compare.add_argument("--prediction-a-column", action="append", type=prediction_mapping)
    compare.add_argument("--prediction-b-column", action="append", type=prediction_mapping)
    compare.add_argument("--cluster-column")
    compare.add_argument(
        "--reference-range",
        action="append",
        type=row_range,
        dest="reference_row_ranges",
    )
    compare.add_argument("--require-complete-reference", action="store_true")
    compare.add_argument("--bootstrap-iterations", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=20260718)
    compare.add_argument("--multiplicity", choices=["holm", "none"], default="holm")

    calibrate = subparsers.add_parser(
        "calibrate", help="Evaluate binary probability calibration for probability-bearing models"
    )
    calibrate.add_argument("--reference", type=Path, required=True)
    calibrate.add_argument("--predictions", type=Path, required=True)
    calibrate.add_argument("--model-id", required=True)
    calibrate.add_argument("--output-dir", type=Path, required=True)
    calibrate.add_argument("--reference-table", default="reports")
    calibrate.add_argument("--prediction-table", default="classifications")
    calibrate.add_argument("--id-column", default="Hashed_ReportURN")
    calibrate.add_argument("--label", action="append", dest="labels")
    calibrate.add_argument("--probability-column", action="append", type=prediction_mapping)
    calibrate.add_argument("--cluster-column")
    calibrate.add_argument(
        "--reference-range",
        action="append",
        type=row_range,
        dest="reference_row_ranges",
    )
    calibrate.add_argument("--require-complete-reference", action="store_true")
    calibrate.add_argument("--bins", type=int, default=10)
    calibrate.add_argument("--bootstrap-iterations", type=int, default=2000)
    calibrate.add_argument("--seed", type=int, default=20260718)

    error_review = subparsers.add_parser(
        "error-review", help="Create a governed FN/FP packet for authorized clinical review"
    )
    error_review.add_argument("--reference", type=Path, required=True)
    error_review.add_argument("--predictions", type=Path, required=True)
    error_review.add_argument("--model-id", required=True)
    error_review.add_argument("--output-dir", type=Path, required=True)
    error_review.add_argument("--reference-table", default="reports")
    error_review.add_argument("--prediction-table", default="classifications")
    error_review.add_argument("--id-column", default="Hashed_ReportURN")
    error_review.add_argument("--label", action="append", dest="labels")
    error_review.add_argument("--prediction-column", action="append", type=prediction_mapping)
    error_review.add_argument("--cluster-column")
    error_review.add_argument(
        "--reference-range", action="append", type=row_range, dest="reference_row_ranges"
    )
    error_review.add_argument("--require-complete-reference", action="store_true")
    error_review.add_argument("--max-per-stratum", type=int, default=25)
    error_review.add_argument("--seed", type=int, default=20260718)
    error_review.add_argument(
        "--handle-salt",
        required=True,
        help="Study-controlled salt used only to derive portable case handles",
    )
    error_review.add_argument(
        "--acknowledge-governed-output",
        action="store_true",
        help="Required acknowledgement that the case-level output stays governed",
    )

    baseline = subparsers.add_parser(
        "baseline-cv", help="Run leakage-safe OOF evaluation for the submitted baseline families"
    )
    baseline.add_argument("--dataset", type=Path, required=True)
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--model", choices=["bag_of_words", "bert_base"], required=True)
    baseline.add_argument("--label", action="append", dest="labels")
    baseline.add_argument("--patient-column")
    baseline.add_argument("--folds", type=int, default=5)
    baseline.add_argument("--seed", type=int, default=20260718)
    baseline.add_argument("--epsilon", type=float, default=0.1)
    baseline.add_argument("--batch-size", type=int, default=16)
    add_schema_arguments(baseline)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_dataset(
            args.dataset,
            args.dataset_id,
            args.output_dir,
            table=args.table,
            id_column=args.id_column,
            report_column=args.report_column,
            labels=args.labels or DEFAULT_LABELS,
            patient_column=args.patient_column,
            split_column=args.split_column,
            row_ranges=args.row_ranges,
            require_complete_labels=args.require_complete_labels,
        )
    elif args.command == "overlap":
        datasets = dict(args.dataset)
        if len(datasets) != len(args.dataset):
            parser.error("dataset names must be unique")
        result = audit_overlap(
            datasets,
            args.output_dir,
            table=args.table,
            id_column=args.id_column,
            report_column=args.report_column,
            patient_column=args.patient_column,
        )
    elif args.command == "evaluate":
        mappings = dict(args.prediction_column or [])
        labels = args.labels or DEFAULT_LABELS
        result = evaluate_predictions(
            args.reference,
            args.predictions,
            args.output_dir,
            reference_table=args.reference_table,
            prediction_table=args.prediction_table,
            id_column=args.id_column,
            labels=labels,
            prediction_columns=mappings or None,
            cluster_column=args.cluster_column,
            fold_column=args.fold_column,
            reference_row_ranges=args.reference_row_ranges,
            require_complete_reference=args.require_complete_reference,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
    elif args.command == "compare":
        result = compare_predictions(
            args.reference,
            args.predictions_a,
            args.predictions_b,
            args.output_dir,
            model_a_id=args.model_a_id,
            model_b_id=args.model_b_id,
            reference_table=args.reference_table,
            prediction_a_table=args.prediction_a_table,
            prediction_b_table=args.prediction_b_table,
            id_column=args.id_column,
            labels=args.labels or DEFAULT_LABELS,
            prediction_a_columns=dict(args.prediction_a_column or []) or None,
            prediction_b_columns=dict(args.prediction_b_column or []) or None,
            cluster_column=args.cluster_column,
            reference_row_ranges=args.reference_row_ranges,
            require_complete_reference=args.require_complete_reference,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
            multiplicity=args.multiplicity,
        )
    elif args.command == "calibrate":
        result = calibrate_predictions(
            args.reference,
            args.predictions,
            args.output_dir,
            model_id=args.model_id,
            reference_table=args.reference_table,
            prediction_table=args.prediction_table,
            id_column=args.id_column,
            labels=args.labels or DEFAULT_LABELS,
            probability_columns=dict(args.probability_column or []) or None,
            cluster_column=args.cluster_column,
            reference_row_ranges=args.reference_row_ranges,
            require_complete_reference=args.require_complete_reference,
            bins=args.bins,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
    elif args.command == "error-review":
        result = build_error_review_packet(
            args.reference,
            args.predictions,
            args.output_dir,
            model_id=args.model_id,
            acknowledge_governed_output=args.acknowledge_governed_output,
            reference_table=args.reference_table,
            prediction_table=args.prediction_table,
            id_column=args.id_column,
            labels=args.labels or DEFAULT_LABELS,
            prediction_columns=dict(args.prediction_column or []) or None,
            cluster_column=args.cluster_column,
            reference_row_ranges=args.reference_row_ranges,
            require_complete_reference=args.require_complete_reference,
            max_per_stratum=args.max_per_stratum,
            seed=args.seed,
            handle_salt=args.handle_salt,
        )
    else:
        result = run_baseline_cv(
            args.dataset,
            args.output_dir,
            model_name=args.model,
            table=args.table,
            id_column=args.id_column,
            report_column=args.report_column,
            labels=args.labels or DEFAULT_LABELS,
            patient_column=args.patient_column,
            folds=args.folds,
            seed=args.seed,
            epsilon=args.epsilon,
            batch_size=args.batch_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
