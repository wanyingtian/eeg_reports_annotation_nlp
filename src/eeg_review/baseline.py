from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import DEFAULT_LABELS
from .io import atomic_write_csv, atomic_write_json, load_table
from .manifest import build_manifest

BERT_CHECKPOINT = "bert-base-uncased"
BOW_PARAMETERS = {
    "max_features": 10000,
    "stop_words": "english",
    "ngram_range": (1, 5),
    "token_pattern": r"\b[a-zA-Z]{3,}\b",
}


def probability_to_level(probability: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """Reproduce the paper code's four bins around probability 0.5."""
    return np.select(
        [probability < 0.5 - epsilon, probability < 0.5, probability < 0.5 + epsilon],
        [1, 2, 3],
        default=4,
    ).astype(int)


def _splitter(folds: int, seed: int, groups: pd.Series | None):
    if groups is None:
        from sklearn.model_selection import StratifiedKFold

        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), None
    from sklearn.model_selection import StratifiedGroupKFold

    return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed), groups


def _bert_embeddings(reports: list[str], batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT)
    model = AutoModel.from_pretrained(BERT_CHECKPOINT).to(device)
    model.eval()
    token_lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in reports]
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(reports), batch_size):
            encoded = tokenizer(
                reports[start : start + batch_size],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(device)
            batches.append(model(**encoded).last_hidden_state[:, 0, :].cpu().numpy())
    return np.concatenate(batches), {
        "checkpoint": BERT_CHECKPOINT,
        "pooling": "final_hidden_state_CLS",
        "frozen": True,
        "max_length": 512,
        "truncation": "tokenizer default: truncate sequence end",
        "device": str(device),
        "token_length": {
            "minimum": int(min(token_lengths)),
            "median": float(np.median(token_lengths)),
            "mean": float(np.mean(token_lengths)),
            "maximum": int(max(token_lengths)),
            "reports_over_512": int(sum(length > 512 for length in token_lengths)),
        },
    }


def run_baseline_cv(
    dataset: Path,
    output_dir: Path,
    *,
    model_name: str,
    table: str = "reports",
    id_column: str = "Hashed_ReportURN",
    report_column: str = "Report",
    labels: list[str] | None = None,
    patient_column: str | None = None,
    folds: int = 5,
    seed: int = 20260718,
    epsilon: float = 0.1,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Generate leakage-safe OOF receipts and explicitly refitted native baselines."""
    if model_name not in {"bag_of_words", "bert_base"}:
        raise ValueError("model_name must be bag_of_words or bert_base")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    labels = labels or DEFAULT_LABELS
    columns = [id_column, report_column, *labels]
    if patient_column:
        columns.append(patient_column)
    frame = load_table(dataset, columns, table)
    reports = frame[report_column].fillna("").astype(str)
    identifiers = frame[id_column].astype("string")
    if identifiers.isna().any() or identifiers.duplicated().any():
        raise ValueError("Baseline receipt requires complete, unique report identifiers")
    if patient_column:
        patients = frame[patient_column].astype("string")
        if patients.isna().any():
            raise ValueError("Patient-grouped CV requires a complete patient key")
        if patients.nunique() < folds:
            raise ValueError("Patient-grouped CV requires at least one patient per fold")

    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    bert_matrix: np.ndarray | None = None
    representation: dict[str, Any]
    if model_name == "bert_base":
        bert_matrix, representation = _bert_embeddings(reports.tolist(), batch_size)
    else:
        representation = {"kind": "raw_count_BoW", **BOW_PARAMETERS}

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    oof = pd.DataFrame({id_column: identifiers})
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": model_name,
        "records": int(len(frame)),
        "folds_requested": folds,
        "split_unit": "patient" if patient_column else "report",
        "representation": representation,
        "labels": {},
    }

    for label in labels:
        numeric = pd.to_numeric(frame[label], errors="coerce")
        valid = numeric.isin([1, 2, 3, 4])
        y = numeric.loc[valid].isin([3, 4]).astype(int)
        counts = y.value_counts().to_dict()
        label_summary: dict[str, Any] = {
            "valid_records": int(valid.sum()),
            "core_absent": int(counts.get(0, 0)),
            "core_present": int(counts.get(1, 0)),
        }
        if len(counts) < 2 or min(counts.values()) < folds:
            label_summary.update(
                status="skipped",
                reason="each core class must contain at least one record per requested fold",
            )
            summary["labels"][label] = label_summary
            continue

        valid_positions = np.flatnonzero(valid.to_numpy())
        y_array = y.to_numpy()
        groups = frame.loc[valid, patient_column] if patient_column else None
        splitter, split_groups = _splitter(folds, seed, groups)
        split_input = (
            reports.loc[valid].to_numpy()
            if model_name == "bag_of_words"
            else bert_matrix[valid]
        )
        probabilities = np.full(len(frame), np.nan)
        assignments = np.full(len(frame), np.nan)

        for fold_index, (train, test) in enumerate(
            splitter.split(split_input, y_array, split_groups), start=1
        ):
            classifier = LogisticRegression(max_iter=1000, random_state=seed)
            if model_name == "bag_of_words":
                from sklearn.feature_extraction.text import CountVectorizer

                estimator = Pipeline(
                    [("vectorizer", CountVectorizer(**BOW_PARAMETERS)), ("classifier", classifier)]
                )
            else:
                estimator = classifier
            estimator.fit(split_input[train], y_array[train])
            positions = valid_positions[test]
            probabilities[positions] = estimator.predict_proba(split_input[test])[:, 1]
            assignments[positions] = fold_index

        final_classifier = LogisticRegression(max_iter=1000, random_state=seed)
        if model_name == "bag_of_words":
            from sklearn.feature_extraction.text import CountVectorizer

            final_estimator = Pipeline(
                [
                    ("vectorizer", CountVectorizer(**BOW_PARAMETERS)),
                    ("classifier", final_classifier),
                ]
            )
        else:
            final_estimator = final_classifier
        final_estimator.fit(split_input, y_array)
        safe_label = label.lower().replace(" ", "_").replace("-", "_")
        model_path = model_dir / f"{safe_label}_{model_name}.joblib"
        dump(final_estimator, model_path)

        predicted_levels = np.full(len(frame), np.nan)
        predicted_levels[valid_positions] = probability_to_level(
            probabilities[valid_positions], epsilon
        )
        oof[f"{label} prediction"] = pd.Series(predicted_levels, dtype="Int64")
        oof[f"{label} probability"] = probabilities
        oof[f"{label} fold"] = pd.Series(assignments, dtype="Int64")
        label_summary.update(
            status="completed",
            oof_records=int(np.isfinite(probabilities).sum()),
            final_fit_records=int(valid.sum()),
            model_artifact=str(model_path.relative_to(output_dir)),
        )
        summary["labels"][label] = label_summary

    atomic_write_csv(output_dir / "oof_predictions.csv", oof)
    atomic_write_json(output_dir / "baseline_summary.json", summary)
    manifest = build_manifest(
        "baseline-cv",
        [dataset],
        {
            "model": model_name,
            "table": table,
            "id_column": id_column,
            "report_column": report_column,
            "patient_column": patient_column,
            "labels": labels,
            "folds": folds,
            "seed": seed,
            "epsilon": epsilon,
            "logistic_regression": {"max_iter": 1000, "random_state": seed},
            "representation": representation,
        },
    )
    manifest["privacy_boundary"] = (
        "case-level output contains pseudonymous report identifiers but no report text; "
        "keep inside the authorized analysis environment"
    )
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    return summary
