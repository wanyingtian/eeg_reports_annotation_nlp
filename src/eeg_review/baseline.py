from __future__ import annotations

import hashlib
import json
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


def _reports_digest(reports: list[str]) -> str:
    digest = hashlib.sha256()
    for report in reports:
        encoded = report.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _atomic_save_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def _bert_embeddings(
    reports: list[str],
    batch_size: int,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT)
    model = AutoModel.from_pretrained(BERT_CHECKPOINT).to(device)
    model.eval()
    token_lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in reports]
    report_digest = _reports_digest(reports)
    cache_manifest_path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_manifest_path = cache_dir / "embedding_cache.json"
        expected_cache = {
            "schema_version": 1,
            "checkpoint": BERT_CHECKPOINT,
            "report_digest": report_digest,
            "records": len(reports),
            "batch_size": batch_size,
        }
        if cache_manifest_path.exists():
            observed = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            for key, value in expected_cache.items():
                if observed.get(key) != value:
                    raise ValueError(
                        f"BERT embedding cache mismatch for {key}: "
                        f"expected {value!r}, found {observed.get(key)!r}"
                    )
        else:
            atomic_write_json(cache_manifest_path, {**expected_cache, "completed_batches": []})

    batches: list[np.ndarray] = []
    completed_batches: list[str] = []
    with torch.no_grad():
        for start in range(0, len(reports), batch_size):
            end = min(start + batch_size, len(reports))
            chunk_name = f"batch_{start:06d}_{end:06d}.npy"
            chunk_path = cache_dir / chunk_name if cache_dir is not None else None
            if chunk_path is not None and chunk_path.exists():
                chunk = np.load(chunk_path, allow_pickle=False)
                if chunk.shape != (end - start, 768):
                    raise ValueError(f"Unexpected cached BERT batch shape in {chunk_path}")
            else:
                encoded = tokenizer(
                    reports[start:end],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                ).to(device)
                chunk = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
                if chunk_path is not None:
                    _atomic_save_numpy(chunk_path, chunk)
            batches.append(chunk)
            completed_batches.append(chunk_name)
            if cache_manifest_path is not None:
                atomic_write_json(
                    cache_manifest_path,
                    {
                        "schema_version": 1,
                        "checkpoint": BERT_CHECKPOINT,
                        "report_digest": report_digest,
                        "records": len(reports),
                        "batch_size": batch_size,
                        "completed_batches": completed_batches,
                    },
                )
    return np.concatenate(batches), {
        "checkpoint": BERT_CHECKPOINT,
        "pooling": "final_hidden_state_CLS",
        "frozen": True,
        "max_length": 512,
        "truncation": "tokenizer default: truncate sequence end",
        "device": str(device),
        "report_digest": report_digest,
        "embedding_cache": str(cache_dir) if cache_dir is not None else None,
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
    embedding_cache_dir: Path | None = None,
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
        bert_matrix, representation = _bert_embeddings(
            reports.tolist(), batch_size, embedding_cache_dir
        )
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
        if len(counts) < 2:
            label_summary.update(
                status="skipped",
                reason="a binary final fit requires both core classes",
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
        cv_eligible = min(counts.values()) >= folds
        if cv_eligible:
            for fold_index, (train, test) in enumerate(
                splitter.split(split_input, y_array, split_groups), start=1
            ):
                classifier = LogisticRegression(max_iter=1000, random_state=seed)
                if model_name == "bag_of_words":
                    from sklearn.feature_extraction.text import CountVectorizer

                    estimator = Pipeline(
                        [
                            ("vectorizer", CountVectorizer(**BOW_PARAMETERS)),
                            ("classifier", classifier),
                        ]
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
            status="completed" if cv_eligible else "external_fit_only",
            oof_records=int(np.isfinite(probabilities).sum()),
            final_fit_records=int(valid.sum()),
            model_artifact=str(model_path.relative_to(output_dir)),
        )
        if not cv_eligible:
            label_summary["reason"] = (
                "five-fold stratification unavailable because the minority core class "
                f"has {min(counts.values())} records; final external-inference model was "
                "fit on all valid development records"
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
            "embedding_cache_dir": str(embedding_cache_dir)
            if embedding_cache_dir is not None
            else None,
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


def run_baseline_predict(
    dataset: Path,
    baseline_dir: Path,
    output_dir: Path,
    *,
    model_name: str,
    table: str = "reports",
    id_column: str = "Hashed_ReportURN",
    report_column: str = "Report",
    labels: list[str] | None = None,
    epsilon: float = 0.1,
    batch_size: int = 16,
    embedding_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Apply the explicitly refitted native baseline to a frozen external cohort."""
    if model_name not in {"bag_of_words", "bert_base"}:
        raise ValueError("model_name must be bag_of_words or bert_base")
    labels = labels or DEFAULT_LABELS
    baseline_dir = baseline_dir.expanduser().resolve(strict=True)
    summary_path = baseline_dir / "baseline_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing baseline summary: {summary_path}")

    training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if training_summary.get("model") != model_name:
        raise ValueError(
            f"Baseline model mismatch: expected {model_name}, "
            f"found {training_summary.get('model')}"
        )

    frame = load_table(dataset, [id_column, report_column], table)
    identifiers = frame[id_column].astype("string")
    if identifiers.isna().any() or identifiers.duplicated().any():
        raise ValueError("Baseline inference requires complete, unique report identifiers")
    reports = frame[report_column].fillna("").astype(str)

    if model_name == "bert_base":
        representation_input, representation = _bert_embeddings(
            reports.tolist(), batch_size, embedding_cache_dir
        )
    else:
        representation_input = reports.to_numpy()
        representation = {"kind": "raw_count_BoW", **BOW_PARAMETERS}

    from joblib import load

    predictions = pd.DataFrame({id_column: identifiers})
    model_paths: list[Path] = []
    label_summary: dict[str, Any] = {}
    for label in labels:
        training_label = training_summary.get("labels", {}).get(label, {})
        if training_label.get("status") not in {"completed", "external_fit_only"}:
            label_summary[label] = {
                "status": "skipped",
                "reason": training_label.get("reason", "training label was not completed"),
            }
            continue
        relative_model = training_label.get("model_artifact")
        if not relative_model:
            raise ValueError(f"Missing model artifact receipt for {label}")
        model_path = (baseline_dir / relative_model).resolve(strict=True)
        model_paths.append(model_path)
        estimator = load(model_path)
        probabilities = estimator.predict_proba(representation_input)[:, 1]
        predictions[f"{label} prediction"] = probability_to_level(probabilities, epsilon)
        predictions[f"{label} probability"] = probabilities
        label_summary[label] = {
            "status": "completed",
            "records": int(len(frame)),
            "model_artifact": str(model_path.relative_to(baseline_dir)),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "predictions.csv", predictions)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": model_name,
        "records": int(len(frame)),
        "training_records": int(training_summary.get("records", 0)),
        "epsilon": epsilon,
        "representation": representation,
        "labels": label_summary,
    }
    atomic_write_json(output_dir / "prediction_summary.json", summary)
    manifest = build_manifest(
        "baseline-predict",
        [dataset, summary_path, *model_paths],
        {
            "model": model_name,
            "table": table,
            "id_column": id_column,
            "report_column": report_column,
            "labels": labels,
            "epsilon": epsilon,
            "batch_size": batch_size,
            "embedding_cache_dir": str(embedding_cache_dir)
            if embedding_cache_dir is not None
            else None,
            "representation": representation,
        },
    )
    manifest["privacy_boundary"] = (
        "case-level output contains pseudonymous report identifiers but no report text; "
        "keep inside the authorized analysis environment"
    )
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    return summary
