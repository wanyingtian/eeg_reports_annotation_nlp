from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adaptation_plan import (
    AdaptationPlanStatus,
    CertaintyMapping,
    parse_adaptation_plan,
    validate_adaptation_plan,
)
from .audit import DEFAULT_LABELS
from .io import atomic_write_csv, atomic_write_json, load_table
from .manifest import build_manifest, sha256_file

DEVELOPMENT_COHORT_ID = "zoe_development_first_100_ra"
DEVELOPMENT_RECORDS = 100
BINARY_CORE_ADAPTER_MODE = "binary_core_certainty_adapter"


@dataclass(frozen=True)
class MarginSelection:
    margin: float
    fitted: bool
    reason: str
    valid_pairs: int
    core_negative_pairs: int
    core_positive_pairs: int
    candidate_scores: tuple[dict[str, float | int | None], ...]


def map_probability_to_certainty(
    probability: np.ndarray,
    *,
    margin: float,
    core_boundary: float = 0.5,
) -> np.ndarray:
    """Map core-positive token probabilities to the thesis's four certainty levels."""
    if not 0 < margin < core_boundary:
        raise ValueError("Margin must be greater than zero and below the core boundary")
    values = np.asarray(probability, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    lower = core_boundary - margin
    upper = core_boundary + margin
    return np.select(
        [values < lower, values < core_boundary, values < upper],
        [1, 2, 3],
        default=4,
    ).astype(int)


def select_certainty_margin(
    reference: np.ndarray,
    probability: np.ndarray,
    specification: CertaintyMapping,
) -> MarginSelection:
    reference = np.asarray(reference, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if reference.size != probability.size:
        raise ValueError("Reference and probability arrays must have the same length")
    if reference.size and not np.isin(reference, [1, 2, 3, 4]).all():
        raise ValueError("Reference values must use four-level labels 1 through 4")
    if probability.size and (
        np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1))
    ):
        raise ValueError("Probabilities must be finite and within [0, 1]")

    core_positive = np.isin(reference, [3, 4])
    positive_pairs = int(core_positive.sum())
    negative_pairs = int((~core_positive).sum())
    valid_pairs = int(reference.size)
    minimum_valid = int(specification.minimum_valid_pairs or 0)
    minimum_side = int(specification.minimum_pairs_per_core_side or 0)
    historical_margin = float(specification.historical_margin or 0.1)

    candidate_scores: list[dict[str, float | int | None]] = []
    for margin in specification.candidate_margins:
        mapped = map_probability_to_certainty(
            probability,
            margin=margin,
            core_boundary=float(specification.core_boundary or 0.5),
        )
        candidate_scores.append(
            {
                "margin": margin,
                "lower_threshold": float((specification.core_boundary or 0.5) - margin),
                "upper_threshold": float((specification.core_boundary or 0.5) + margin),
                "exact_matches": int(np.sum(reference == mapped)),
                "exact_four_level_agreement": (
                    float(np.mean(reference == mapped)) if valid_pairs else None
                ),
            }
        )

    support_sufficient = (
        valid_pairs >= minimum_valid
        and positive_pairs >= minimum_side
        and negative_pairs >= minimum_side
    )
    if not support_sufficient:
        return MarginSelection(
            margin=historical_margin,
            fitted=False,
            reason=(
                "insufficient_development_support; retained historical margin without fitting"
            ),
            valid_pairs=valid_pairs,
            core_negative_pairs=negative_pairs,
            core_positive_pairs=positive_pairs,
            candidate_scores=tuple(candidate_scores),
        )

    best_agreement = max(float(row["exact_four_level_agreement"]) for row in candidate_scores)
    tied = [
        row
        for row in candidate_scores
        if math.isclose(float(row["exact_four_level_agreement"]), best_agreement)
    ]
    selected = min(
        tied,
        key=lambda row: (
            abs(float(row["margin"]) - historical_margin),
            float(row["margin"]),
        ),
    )
    return MarginSelection(
        margin=float(selected["margin"]),
        fitted=True,
        reason="selected_by_preregistered_development_objective",
        valid_pairs=valid_pairs,
        core_negative_pairs=negative_pairs,
        core_positive_pairs=positive_pairs,
        candidate_scores=tuple(candidate_scores),
    )


def _wilson_interval(successes: int, total: int) -> dict[str, float | None]:
    if total <= 0:
        return {"low": None, "high": None}
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1 + z**2 / total
    centre = (estimate + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(estimate * (1 - estimate) / total + z**2 / (4 * total**2))
        / denominator
    )
    return {"low": centre - half_width, "high": centre + half_width}


def _leave_one_out_diagnostic(
    reference: np.ndarray,
    probability: np.ndarray,
    specification: CertaintyMapping,
) -> dict[str, Any]:
    if reference.size == 0:
        return {
            "method": specification.crossfit_diagnostic,
            "n": 0,
            "exact_matches": 0,
            "exact_four_level_agreement": None,
            "wilson_interval_95": _wilson_interval(0, 0),
            "fallback_folds": 0,
            "selected_margin_frequency": {},
            "interpretation": (
                "Unavailable because no valid development pairs were supplied."
            ),
        }
    predictions: list[int] = []
    selected_margins: list[float] = []
    fallback_folds = 0
    for index in range(reference.size):
        keep = np.arange(reference.size) != index
        selection = select_certainty_margin(reference[keep], probability[keep], specification)
        if not selection.fitted:
            fallback_folds += 1
        selected_margins.append(selection.margin)
        prediction = map_probability_to_certainty(
            probability[index : index + 1],
            margin=selection.margin,
            core_boundary=float(specification.core_boundary or 0.5),
        )[0]
        predictions.append(int(prediction))
    predicted = np.asarray(predictions, dtype=int)
    matches = int(np.sum(predicted == reference))
    return {
        "method": specification.crossfit_diagnostic,
        "n": int(reference.size),
        "exact_matches": matches,
        "exact_four_level_agreement": float(matches / reference.size),
        "wilson_interval_95": _wilson_interval(matches, int(reference.size)),
        "fallback_folds": fallback_folds,
        "selected_margin_frequency": {
            str(margin): selected_margins.count(margin)
            for margin in specification.candidate_margins
            if margin in selected_margins
        },
        "interpretation": (
            "Development-only cross-fitting diagnostic; not an external performance estimate."
        ),
    }


def _bootstrap_stability(
    reference: np.ndarray,
    probability: np.ndarray,
    specification: CertaintyMapping,
    *,
    seed: int,
) -> dict[str, Any]:
    if reference.size == 0:
        return {
            "method": specification.resampling_method,
            "iterations": 0,
            "seed": seed,
            "selection_counts": {},
            "selection_frequencies": {},
            "fallback_replicates": 0,
            "interpretation": "Unavailable because no valid development pairs were supplied.",
        }
    rng = np.random.default_rng(seed)
    core_positive = np.isin(reference, [3, 4])
    strata = [np.flatnonzero(~core_positive), np.flatnonzero(core_positive)]
    frequencies = {margin: 0 for margin in specification.candidate_margins}
    fallback_replicates = 0
    for _ in range(int(specification.resampling_iterations or 0)):
        selected = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in strata
                if len(indices)
            ]
        )
        selection = select_certainty_margin(
            reference[selected], probability[selected], specification
        )
        if not selection.fitted:
            fallback_replicates += 1
        frequencies[selection.margin] = frequencies.get(selection.margin, 0) + 1
    iterations = int(specification.resampling_iterations or 0)
    return {
        "method": specification.resampling_method,
        "iterations": iterations,
        "seed": seed,
        "selection_counts": {str(key): value for key, value in frequencies.items()},
        "selection_frequencies": {
            str(key): value / iterations if iterations else None
            for key, value in frequencies.items()
        },
        "fallback_replicates": fallback_replicates,
        "interpretation": (
            "Threshold-selection stability on the development set; these are not confidence "
            "intervals for external performance."
        ),
    }


def _key_set_digest(values: pd.Series) -> str:
    normalized = sorted(str(value) for value in values)
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_plan(contract_path: Path) -> tuple[dict[str, Any], Any]:
    validation = validate_adaptation_plan(contract_path)
    if not validation["design_valid"]:
        raise ValueError("Adaptation plan is invalid; run adaptation-plan-validate for details")
    with contract_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    plan = parse_adaptation_plan(payload)
    if plan.status != AdaptationPlanStatus.PREREGISTERED_UNFROZEN:
        raise ValueError("Threshold fitting requires a valid preregistered-unfrozen plan")
    return validation, plan


def fit_certainty_adapter(
    contract_path: Path,
    reference_path: Path,
    predictions_path: Path,
    prediction_run_receipt_path: Path,
    development_manifest_path: Path,
    output_dir: Path,
    *,
    reference_table: str = "reports",
    prediction_table: str = "classifications",
    manifest_table: str = "manifest",
    id_column: str = "Hashed_ReportURN",
    labels: list[str] | None = None,
    probability_columns: dict[str, str] | None = None,
    acknowledge_governed_inputs: bool = False,
) -> dict[str, Any]:
    """Fit an aggregate-only four-level certainty mapper on the fixed Zoe development set."""
    if not acknowledge_governed_inputs:
        raise ValueError("Explicit acknowledgement of governed keyed inputs is required")
    labels = labels or DEFAULT_LABELS
    if labels != DEFAULT_LABELS:
        raise ValueError("A freeze-review adapter must cover all five labels in canonical order")
    probability_columns = probability_columns or {label: f"Prob_{label}" for label in labels}
    if set(probability_columns) != set(labels):
        raise ValueError("Every canonical label requires exactly one probability-column mapping")

    contract_path = contract_path.expanduser().resolve(strict=True)
    reference_path = reference_path.expanduser().resolve(strict=True)
    predictions_path = predictions_path.expanduser().resolve(strict=True)
    prediction_run_receipt_path = prediction_run_receipt_path.expanduser().resolve(strict=True)
    development_manifest_path = development_manifest_path.expanduser().resolve(strict=True)
    _validation, plan = _load_plan(contract_path)
    specification = plan.certainty_mapping
    declared_manifest = specification.development_manifest
    if not declared_manifest.path or not declared_manifest.sha256:
        raise ValueError(
            "The governed development manifest path and SHA-256 must be declared before fitting"
        )
    actual_manifest_sha256 = sha256_file(development_manifest_path)
    if actual_manifest_sha256.lower() != declared_manifest.sha256.lower():
        raise ValueError("Development manifest checksum does not match the preregistered plan")

    with prediction_run_receipt_path.open(encoding="utf-8") as stream:
        prediction_run_receipt = json.load(stream)
    if not isinstance(prediction_run_receipt, dict):
        raise ValueError("Prediction run receipt must be a JSON object")
    instrumentation = prediction_run_receipt.get("calibration_instrumentation")
    output_identity = prediction_run_receipt.get("output")
    if not isinstance(instrumentation, dict) or not isinstance(output_identity, dict):
        raise ValueError("Prediction run receipt lacks instrumentation or output identity")
    if instrumentation.get("enabled") is not True:
        raise ValueError("Prediction run did not enable log-probability instrumentation")
    if instrumentation.get("classification_mode") != BINARY_CORE_ADAPTER_MODE:
        raise ValueError("Prediction run did not use the preregistered binary-core mode")
    if output_identity.get("sha256") != sha256_file(predictions_path):
        raise ValueError("Prediction CSV checksum does not match its producing run receipt")
    model_identity = prediction_run_receipt.get("model")
    prompts = prediction_run_receipt.get("prompts")
    grammars = prediction_run_receipt.get("grammars")
    if not all(isinstance(value, dict) for value in (model_identity, prompts, grammars)):
        raise ValueError("Prediction run receipt lacks model, prompt, or grammar identity")
    if not isinstance(model_identity.get("sha256"), str):
        raise ValueError("Prediction run receipt lacks the model artifact checksum")
    producing_artifacts = (
        ("classify prompt", prompts.get("classify")),
        ("classify grammar", grammars.get("classify")),
    )
    for field, block in producing_artifacts:
        if not isinstance(block, dict) or not isinstance(block.get("sha256"), str):
            raise ValueError(f"Prediction run receipt lacks the {field} checksum")

    manifest = load_table(development_manifest_path, [id_column], manifest_table)
    reference = load_table(reference_path, [id_column, *labels], reference_table)
    predictions = load_table(
        predictions_path,
        [
            id_column,
            "adaptation_classification_mode",
            *[probability_columns[label] for label in labels],
        ],
        prediction_table,
    )
    for name, frame in (
        ("development manifest", manifest),
        ("reference", reference),
        ("predictions", predictions),
    ):
        if frame[id_column].isna().any():
            raise ValueError(f"{name} contains missing report keys")
        if frame[id_column].duplicated().any():
            raise ValueError(f"{name} contains duplicate report keys")
    if len(manifest) != DEVELOPMENT_RECORDS:
        raise ValueError(
            f"{DEVELOPMENT_COHORT_ID} must contain exactly {DEVELOPMENT_RECORDS} report keys"
        )
    modes = set(predictions["adaptation_classification_mode"].dropna().astype(str))
    if modes != {BINARY_CORE_ADAPTER_MODE}:
        raise ValueError("Prediction rows do not carry one consistent binary-core mode marker")

    manifest_keys = set(manifest[id_column])
    reference_keys = set(reference[id_column])
    prediction_keys = set(predictions[id_column])
    missing_reference = manifest_keys - reference_keys
    missing_predictions = manifest_keys - prediction_keys
    extra_predictions = prediction_keys - manifest_keys
    if missing_reference or missing_predictions or extra_predictions:
        raise ValueError(
            "Exact development key alignment failed: "
            f"reference_missing={len(missing_reference)}, "
            f"predictions_missing={len(missing_predictions)}, "
            f"predictions_extra={len(extra_predictions)}"
        )

    reference = manifest.merge(reference, on=id_column, how="left", validate="one_to_one")
    predictions = manifest.merge(predictions, on=id_column, how="left", validate="one_to_one")
    merged = reference.merge(predictions, on=id_column, how="inner", validate="one_to_one")

    label_receipts: dict[str, Any] = {}
    adapter_labels: dict[str, Any] = {}
    candidate_rows: list[dict[str, Any]] = []
    for offset, label in enumerate(labels):
        reference_level = pd.to_numeric(merged[label], errors="coerce")
        probability = pd.to_numeric(merged[probability_columns[label]], errors="coerce")
        valid = reference_level.isin([1, 2, 3, 4]) & probability.between(
            0, 1, inclusive="both"
        )
        reference_values = reference_level[valid].to_numpy(dtype=int)
        probability_values = probability[valid].to_numpy(dtype=float)
        selection = select_certainty_margin(reference_values, probability_values, specification)
        mapped = map_probability_to_certainty(
            probability_values,
            margin=selection.margin,
            core_boundary=float(specification.core_boundary or 0.5),
        )
        matches = int(np.sum(reference_values == mapped))
        core_matches = int(
            np.sum(np.isin(reference_values, [3, 4]) == np.isin(mapped, [3, 4]))
        )
        crossfit = _leave_one_out_diagnostic(
            reference_values, probability_values, specification
        )
        stability = _bootstrap_stability(
            reference_values,
            probability_values,
            specification,
            seed=int(specification.resampling_seed or 0) + offset,
        )
        lower = float((specification.core_boundary or 0.5) - selection.margin)
        upper = float((specification.core_boundary or 0.5) + selection.margin)
        adapter_labels[label] = {
            "selected_margin": selection.margin,
            "lower_threshold": lower,
            "core_boundary": specification.core_boundary,
            "upper_threshold": upper,
            "fitted": selection.fitted,
            "selection_reason": selection.reason,
        }
        label_receipts[label] = {
            "manifest_records": len(manifest),
            "valid_pairs": selection.valid_pairs,
            "excluded_invalid_or_missing_pairs": int((~valid).sum()),
            "core_negative_pairs": selection.core_negative_pairs,
            "core_positive_pairs": selection.core_positive_pairs,
            "selected_margin": selection.margin,
            "lower_threshold": lower,
            "upper_threshold": upper,
            "fitted": selection.fitted,
            "selection_reason": selection.reason,
            "apparent_development_diagnostic": {
                "exact_matches": matches,
                "exact_four_level_agreement": (
                    matches / len(reference_values) if len(reference_values) else None
                ),
                "wilson_interval_95": _wilson_interval(matches, len(reference_values)),
                "core_matches_at_fixed_0_5_boundary": core_matches,
                "core_agreement_at_fixed_0_5_boundary": (
                    core_matches / len(reference_values) if len(reference_values) else None
                ),
                "interpretation": (
                    "Optimistically selected on this development set; not external evidence."
                ),
            },
            "leave_one_out_diagnostic": crossfit,
            "bootstrap_selection_stability": stability,
        }
        for row in selection.candidate_scores:
            candidate_rows.append({"label": label, **row})

    adapter = {
        "schema_version": 1,
        "adapter_id": plan.task_adapter.adapter_id,
        "proposed_evidence_layer_id": plan.proposed_evidence_layer_id,
        "base_evidence_layer": plan.base_evidence_layer,
        "development_cohort_id": DEVELOPMENT_COHORT_ID,
        "development_manifest_sha256": actual_manifest_sha256,
        "report_key_set_sha256": _key_set_digest(manifest[id_column]),
        "input_feature": specification.input_feature,
        "output_interpretation": specification.output_interpretation,
        "mapping": {
            "level_1": "p < lower_threshold",
            "level_2": "lower_threshold <= p < 0.5",
            "level_3": "0.5 <= p < upper_threshold",
            "level_4": "p >= upper_threshold",
        },
        "labels": adapter_labels,
        "parameter_update": "certainty_threshold_margins_only; model weights unchanged",
        "teacher_model_outputs_used": False,
        "claim_boundary": (
            "Development-fitted adapter artifact only; not admitted for evaluation until frozen, "
            "checksummed, and admitted by the author group."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "certainty_adapter.json"
    atomic_write_json(adapter_path, adapter)
    atomic_write_csv(
        output_dir / "certainty_margin_candidates.csv", pd.DataFrame(candidate_rows)
    )
    labels_using_fallback = [label for label in labels if not adapter_labels[label]["fitted"]]
    receipt = {
        "schema_version": 1,
        "plan_id": plan.plan_id,
        "contract_sha256": sha256_file(contract_path),
        "adapter_artifact": {
            "path": adapter_path.name,
            "sha256": sha256_file(adapter_path),
        },
        "development_cohort_id": DEVELOPMENT_COHORT_ID,
        "development_manifest": {
            "sha256": actual_manifest_sha256,
            "records": len(manifest),
            "report_key_set_sha256": _key_set_digest(manifest[id_column]),
            "exact_prediction_key_set": True,
        },
        "producing_surface": {
            "prediction_run_receipt_sha256": sha256_file(prediction_run_receipt_path),
            "prediction_csv_sha256": sha256_file(predictions_path),
            "model_artifact_sha256": model_identity["sha256"],
            "classification_prompt_sha256": prompts["classify"]["sha256"],
            "classification_grammar_sha256": grammars["classify"]["sha256"],
            "classification_mode": instrumentation["classification_mode"],
        },
        "selection_specification": asdict(specification),
        "labels": label_receipts,
        "labels_fitted": [label for label in labels if adapter_labels[label]["fitted"]],
        "labels_using_preregistered_fallback": labels_using_fallback,
        "ready_for_freeze_review": True,
        "ready_for_evaluation": False,
        "analysis_scope": "development_only",
        "interpretation_boundary": specification.interpretation_boundary,
        "privacy_boundary": (
            "Aggregate diagnostics and threshold artifacts only; no report text, report key, "
            "patient key, or case-level prediction emitted."
        ),
    }
    atomic_write_json(output_dir / "certainty_adapter_fit_receipt.json", receipt)
    atomic_write_json(
        output_dir / "run_manifest.json",
        build_manifest(
            "certainty-adapter-fit",
            [
                contract_path,
                reference_path,
                predictions_path,
                prediction_run_receipt_path,
                development_manifest_path,
            ],
            {
                "development_cohort_id": DEVELOPMENT_COHORT_ID,
                "reference_table": reference_table,
                "prediction_table": prediction_table,
                "manifest_table": manifest_table,
                "id_column": id_column,
                "labels": labels,
                "probability_columns": probability_columns,
                "selection_specification": asdict(specification),
            },
            privacy_boundary=(
                "aggregate outputs and threshold artifact only; keyed inputs remain governed"
            ),
        ),
    )
    return receipt
