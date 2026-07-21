from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import DEFAULT_LABELS
from .io import atomic_write_csv, atomic_write_json, load_table
from .manifest import build_manifest
from .metrics import select_reference_rows


def review_handle(value: str, salt: str, *, prefix: str = "case") -> str:
    digest = hashlib.sha256(f"{prefix}:{salt}:{value}".encode()).hexdigest()
    return f"{prefix}-{digest[:20]}"


def _select_stratum(
    frame: pd.DataFrame,
    *,
    maximum: int,
    rng: np.random.Generator,
    cluster_column: str | None,
) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    if not cluster_column:
        selected = rng.choice(frame.index.to_numpy(), size=maximum, replace=False)
        return frame.loc[np.sort(selected)]

    cluster_values = frame[cluster_column].astype(str)
    clusters = pd.unique(cluster_values)
    chosen_clusters = rng.choice(clusters, size=min(maximum, len(clusters)), replace=False)
    selected_indices: list[int] = []
    for cluster in chosen_clusters:
        candidates = frame.index[cluster_values == cluster].to_numpy()
        selected_indices.append(int(rng.choice(candidates)))
    if len(selected_indices) < maximum:
        remaining = frame.index.difference(pd.Index(selected_indices)).to_numpy()
        fill = rng.choice(
            remaining,
            size=min(maximum - len(selected_indices), len(remaining)),
            replace=False,
        )
        selected_indices.extend(int(index) for index in fill)
    return frame.loc[sorted(selected_indices)]


def build_error_review_packet(
    reference_path: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    model_id: str,
    acknowledge_governed_output: bool,
    reference_table: str = "reports",
    prediction_table: str = "classifications",
    id_column: str = "Hashed_ReportURN",
    labels: list[str] | None = None,
    prediction_columns: dict[str, str] | None = None,
    cluster_column: str | None = None,
    reference_row_ranges: list[tuple[int, int]] | None = None,
    require_complete_reference: bool = False,
    max_per_stratum: int = 25,
    seed: int = 20260718,
    handle_salt: str,
) -> dict[str, Any]:
    if not acknowledge_governed_output:
        raise ValueError(
            "Error review packets are case-level governed outputs; pass explicit acknowledgement"
        )
    if max_per_stratum < 1:
        raise ValueError("max_per_stratum must be at least 1")
    if not handle_salt:
        raise ValueError("handle_salt must not be empty")

    labels = labels or DEFAULT_LABELS
    prediction_columns = prediction_columns or {label: label for label in labels}
    missing_mappings = sorted(set(labels) - set(prediction_columns))
    if missing_mappings:
        raise ValueError(f"Missing prediction-column mappings: {missing_mappings}")

    reference_columns = [id_column, *labels]
    if cluster_column:
        reference_columns.append(cluster_column)
    reference = load_table(reference_path, reference_columns, reference_table)
    reference, selection = select_reference_rows(reference, reference_row_ranges)
    if reference[id_column].duplicated().any():
        raise ValueError("Reference identifiers are not unique")

    if require_complete_reference:
        complete = pd.DataFrame(
            {
                label: pd.to_numeric(reference[label], errors="coerce").isin([1, 2, 3, 4])
                for label in labels
            }
        ).all(axis=1)
        selection["complete_reference_required"] = True
        selection["excluded_incomplete_reference_records"] = int((~complete).sum())
        reference = reference.loc[complete].reset_index(drop=True)
    else:
        selection["complete_reference_required"] = False
        selection["excluded_incomplete_reference_records"] = 0

    prediction_columns_to_load = [id_column, *[prediction_columns[label] for label in labels]]
    predictions = load_table(predictions_path, prediction_columns_to_load, prediction_table)
    if predictions[id_column].duplicated().any():
        raise ValueError("Prediction identifiers are not unique")
    predictions = predictions.rename(
        columns={prediction_columns[label]: f"{label}__prediction" for label in labels}
    )
    merged = reference.merge(predictions, on=id_column, how="inner", validate="one_to_one")
    if cluster_column and merged[cluster_column].isna().any():
        raise ValueError("Missing cluster identifiers among matched reference records")

    rng = np.random.default_rng(seed)
    packet_rows: list[dict[str, Any]] = []
    label_summary: dict[str, Any] = {}
    for label in labels:
        reference_values = pd.to_numeric(merged[label], errors="coerce")
        prediction_values = pd.to_numeric(merged[f"{label}__prediction"], errors="coerce")
        valid = reference_values.isin([1, 2, 3, 4]) & prediction_values.isin([1, 2, 3, 4])
        working_columns = [id_column]
        if cluster_column:
            working_columns.append(cluster_column)
        working = merged.loc[valid, working_columns].copy()
        working["reference_level"] = reference_values[valid].astype(int)
        working["prediction_level"] = prediction_values[valid].astype(int)
        working["reference_core_positive"] = working["reference_level"].isin([3, 4])
        working["prediction_core_positive"] = working["prediction_level"].isin([3, 4])
        working["error_type"] = ""
        false_negative = working["reference_core_positive"] & ~working["prediction_core_positive"]
        false_positive = ~working["reference_core_positive"] & working["prediction_core_positive"]
        working.loc[false_negative, "error_type"] = "false_negative"
        working.loc[false_positive, "error_type"] = "false_positive"

        counts: dict[str, Any] = {
            "valid_pairs": int(valid.sum()),
            "excluded_invalid_or_missing_pairs": int((~valid).sum()),
            "false_negative_total": int(false_negative.sum()),
            "false_positive_total": int(false_positive.sum()),
        }
        for error_type in ("false_negative", "false_positive"):
            stratum = working.loc[working["error_type"] == error_type]
            chosen = _select_stratum(
                stratum,
                maximum=max_per_stratum,
                rng=rng,
                cluster_column=cluster_column,
            )
            counts[f"{error_type}_selected"] = int(len(chosen))
            for _, row in chosen.iterrows():
                packet_rows.append(
                    {
                        "case_handle": review_handle(str(row[id_column]), handle_salt),
                        "label": label,
                        "model_id": model_id,
                        "error_type": error_type,
                        "reference_level": int(row["reference_level"]),
                        "prediction_level": int(row["prediction_level"]),
                        "review_status": "pending",
                        "clinical_salience": "",
                        "reference_ambiguity": "",
                        "likely_workflow_consequence": "",
                        "escalation_or_override_would_catch": "",
                        "reviewer_role": "",
                        "review_notes": "",
                    }
                )
        label_summary[label] = counts

    packet = pd.DataFrame(
        packet_rows,
        columns=[
            "case_handle",
            "label",
            "model_id",
            "error_type",
            "reference_level",
            "prediction_level",
            "review_status",
            "clinical_salience",
            "reference_ambiguity",
            "likely_workflow_consequence",
            "escalation_or_override_would_catch",
            "reviewer_role",
            "review_notes",
        ],
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model_id": model_id,
        "reference_records": int(len(reference)),
        "prediction_records": int(len(predictions)),
        "matched_records": int(len(merged)),
        "unmatched_reference_records": int(len(reference) - len(merged)),
        "reference_selection": selection,
        "sampling": {
            "seed": seed,
            "max_per_label_and_error_type": max_per_stratum,
            "unit": "patient_cluster" if cluster_column else "report",
            "one_case_per_cluster_preferred": bool(cluster_column),
        },
        "selected_case_rows": int(len(packet)),
        "labels": label_summary,
        "interpretation_limits": [
            "This packet supports governed clinical error review; it does not adjudicate truth.",
            "Report text and source identifiers are intentionally absent from the portable packet.",
        ],
    }
    if not cluster_column:
        summary["interpretation_limits"].append(
            "No patient/cluster column supplied; sampled rows may include repeated patients."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "clinical_error_review_packet.csv", packet)
    atomic_write_json(output_dir / "clinical_error_review_summary.json", summary)
    atomic_write_json(
        output_dir / "run_manifest.json",
        build_manifest(
            "error-review",
            [reference_path, predictions_path],
            {
                "model_id": model_id,
                "reference_table": reference_table,
                "prediction_table": prediction_table,
                "id_column": id_column,
                "labels": labels,
                "prediction_columns": prediction_columns,
                "cluster_column": cluster_column,
                "reference_row_ranges": reference_row_ranges,
                "require_complete_reference": require_complete_reference,
                "max_per_stratum": max_per_stratum,
                "seed": seed,
                "handle_salt_sha256": hashlib.sha256(handle_salt.encode()).hexdigest(),
            },
            privacy_boundary=(
                "governed case-level handles and label pairs; no report text, source identifiers, "
                "or patient identifiers emitted"
            ),
        ),
    )
    return summary
