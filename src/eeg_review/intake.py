from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from .audit import DEFAULT_LABELS
from .io import atomic_write_json, load_table
from .manifest import build_manifest, sha256_file


class EvidenceLayer(StrEnum):
    SUBMITTED_MISTRAL = "submitted_mistral"
    REPRODUCED_MISTRAL = "reproduced_mistral"
    POST_SUBMISSION_MEDGEMMA = "post_submission_medgemma"


class IssueSeverity(StrEnum):
    BLOCKER = "blocker"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    severity: IssueSeverity
    field: str
    message: str


@dataclass(frozen=True)
class ArtifactIdentity:
    path: str | None
    sha256: str | None


@dataclass(frozen=True)
class ModelIdentity:
    upstream_repo_id: str | None
    upstream_revision: str | None
    artifact: ArtifactIdentity
    size_bytes: int | None
    quantization: str | None
    license: str | None


@dataclass(frozen=True)
class ChatTemplateIdentity:
    mode: str | None
    source: str | None
    artifact: ArtifactIdentity
    applied: bool | None


@dataclass(frozen=True)
class RuntimeIdentity:
    engine: str | None
    engine_version: str | None
    engine_revision: str | None
    chat_template: ChatTemplateIdentity
    n_ctx: int | None
    n_gpu_layers: int | None
    temperature: float | None
    top_k: int | None
    top_p: float | None
    seed: int | None
    max_tokens: int | None
    hardware: str | None
    operating_system: str | None


@dataclass(frozen=True)
class PromptIdentity:
    prompt_id: str | None
    artifact: ArtifactIdentity
    development_population: str | None
    reference_outcomes_inspected_during_selection: bool | None
    frozen_before_final_evaluation: bool | None
    stopping_rule: str | None
    selection_history_artifact: ArtifactIdentity


@dataclass(frozen=True)
class GrammarIdentity:
    mode: str | None
    artifact: ArtifactIdentity
    purpose: str | None


@dataclass(frozen=True)
class KeyContract:
    report_key_column: str | None
    report_key_namespace: str | None
    report_key_normalization: str | None
    patient_key_column: str | None
    patient_key_namespace: str | None
    patient_key_semantics_confirmed: bool | None


@dataclass(frozen=True)
class PopulationCounts:
    source_records: int | None
    candidate_records: int | None
    included_records: int | None
    excluded_records_by_reason: dict[str, int]
    reference_complete_records: int | None
    prediction_expected_records: int | None


@dataclass(frozen=True)
class TabularArtifact:
    artifact: ArtifactIdentity
    table: str


@dataclass(frozen=True)
class PredictionSurface:
    tabular: TabularArtifact
    report_key_column: str | None
    label_columns: dict[str, str | None]
    invalid_records: int | None
    unfinished_records: int | None


@dataclass(frozen=True)
class CohortContract:
    cohort_id: str | None
    role: str | None
    manifest: TabularArtifact
    population: PopulationCounts
    predictions: PredictionSurface


@dataclass(frozen=True)
class ComparatorIntake:
    schema_version: int
    status: str | None
    evidence_layer: EvidenceLayer | None
    model_identity: ModelIdentity
    runtime: RuntimeIdentity
    prompt: PromptIdentity
    grammar: GrammarIdentity
    key_contract: KeyContract
    cohorts: tuple[CohortContract, ...]
    privacy_boundary: str | None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _artifact(value: Any) -> ArtifactIdentity:
    data = _mapping(value)
    return ArtifactIdentity(path=data.get("path"), sha256=data.get("sha256"))


def _tabular(value: Any, *, default_table: str) -> TabularArtifact:
    data = _mapping(value)
    return TabularArtifact(
        artifact=_artifact(data.get("artifact")),
        table=str(data.get("table") or default_table),
    )


def parse_intake(payload: dict[str, Any]) -> ComparatorIntake:
    model = _mapping(payload.get("model_identity"))
    runtime = _mapping(payload.get("runtime"))
    template = _mapping(runtime.get("chat_template"))
    prompt = _mapping(payload.get("prompt"))
    grammar = _mapping(payload.get("grammar"))
    keys = _mapping(payload.get("key_contract"))
    layer_value = payload.get("evidence_layer")
    try:
        layer = EvidenceLayer(layer_value) if layer_value is not None else None
    except (TypeError, ValueError):
        layer = None

    cohorts: list[CohortContract] = []
    raw_cohorts = payload.get("cohorts")
    if isinstance(raw_cohorts, list):
        for raw_cohort in raw_cohorts:
            cohort = _mapping(raw_cohort)
            population = _mapping(cohort.get("population"))
            predictions = _mapping(cohort.get("predictions"))
            exclusions = _mapping(population.get("excluded_records_by_reason"))
            labels = _mapping(predictions.get("label_columns"))
            cohorts.append(
                CohortContract(
                    cohort_id=cohort.get("cohort_id"),
                    role=cohort.get("role"),
                    manifest=_tabular(cohort.get("manifest"), default_table="reports"),
                    population=PopulationCounts(
                        source_records=population.get("source_records"),
                        candidate_records=population.get("candidate_records"),
                        included_records=population.get("included_records"),
                        excluded_records_by_reason={
                            str(key): value for key, value in exclusions.items()
                        },
                        reference_complete_records=population.get("reference_complete_records"),
                        prediction_expected_records=population.get("prediction_expected_records"),
                    ),
                    predictions=PredictionSurface(
                        tabular=_tabular(
                            predictions.get("surface"), default_table="classifications"
                        ),
                        report_key_column=predictions.get("report_key_column"),
                        label_columns={
                            str(key): value if isinstance(value, str) else None
                            for key, value in labels.items()
                        },
                        invalid_records=predictions.get("invalid_records"),
                        unfinished_records=predictions.get("unfinished_records"),
                    ),
                )
            )

    return ComparatorIntake(
        schema_version=payload.get("schema_version", 0),
        status=payload.get("status"),
        evidence_layer=layer,
        model_identity=ModelIdentity(
            upstream_repo_id=model.get("upstream_repo_id"),
            upstream_revision=model.get("upstream_revision"),
            artifact=_artifact(model.get("artifact")),
            size_bytes=model.get("size_bytes"),
            quantization=model.get("quantization"),
            license=model.get("license"),
        ),
        runtime=RuntimeIdentity(
            engine=runtime.get("engine"),
            engine_version=runtime.get("engine_version"),
            engine_revision=runtime.get("engine_revision"),
            chat_template=ChatTemplateIdentity(
                mode=template.get("mode"),
                source=template.get("source"),
                artifact=_artifact(template.get("artifact")),
                applied=template.get("applied"),
            ),
            n_ctx=runtime.get("n_ctx"),
            n_gpu_layers=runtime.get("n_gpu_layers"),
            temperature=runtime.get("temperature"),
            top_k=runtime.get("top_k"),
            top_p=runtime.get("top_p"),
            seed=runtime.get("seed"),
            max_tokens=runtime.get("max_tokens"),
            hardware=runtime.get("hardware"),
            operating_system=runtime.get("operating_system"),
        ),
        prompt=PromptIdentity(
            prompt_id=prompt.get("id"),
            artifact=_artifact(prompt.get("artifact")),
            development_population=prompt.get("development_population"),
            reference_outcomes_inspected_during_selection=prompt.get(
                "reference_outcomes_inspected_during_selection"
            ),
            frozen_before_final_evaluation=prompt.get("frozen_before_final_evaluation"),
            stopping_rule=prompt.get("stopping_rule"),
            selection_history_artifact=_artifact(prompt.get("selection_history_artifact")),
        ),
        grammar=GrammarIdentity(
            mode=grammar.get("mode"),
            artifact=_artifact(grammar.get("artifact")),
            purpose=grammar.get("purpose"),
        ),
        key_contract=KeyContract(
            report_key_column=keys.get("report_key_column"),
            report_key_namespace=keys.get("report_key_namespace"),
            report_key_normalization=keys.get("report_key_normalization"),
            patient_key_column=keys.get("patient_key_column"),
            patient_key_namespace=keys.get("patient_key_namespace"),
            patient_key_semantics_confirmed=keys.get("patient_key_semantics_confirmed"),
        ),
        cohorts=tuple(cohorts),
        privacy_boundary=payload.get("privacy_boundary"),
    )


def load_intake(path: Path) -> ComparatorIntake:
    with path.expanduser().resolve(strict=True).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Intake contract must be a JSON object")
    return parse_intake(payload)


def _resolve_artifact(
    artifact: ArtifactIdentity, contract_path: Path, bundle_root: Path | None
) -> Path | None:
    if not isinstance(artifact.path, str) or not artifact.path.strip():
        return None
    path = Path(artifact.path).expanduser()
    if not path.is_absolute():
        base = bundle_root.expanduser() if bundle_root else contract_path.parent
        path = base / path
    return path.resolve()


def _key_digest(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in sorted(values.astype(str).tolist()):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _key_mapping_digest(keys: pd.Series, values: pd.Series) -> str:
    digest = hashlib.sha256()
    pairs = sorted(zip(keys.astype(str), values.astype(str), strict=True))
    for key, value in pairs:
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _required_text(
    issues: list[ValidationIssue], field: str, value: Any, *, allow_none: bool = False
) -> None:
    if not allow_none and (value is None or not isinstance(value, str) or not value.strip()):
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, field, "non-empty text is required"))


def _required_bool(issues: list[ValidationIssue], field: str, value: Any) -> None:
    if not isinstance(value, bool):
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, field, "boolean value is required"))


def _required_number(
    issues: list[ValidationIssue],
    field: str,
    value: Any,
    *,
    integer: bool = False,
    minimum: float = 0,
) -> None:
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected):
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, field, "numeric value is required"))
    elif value < minimum:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                field,
                f"value cannot be less than {minimum:g}",
            )
        )


def _validate_artifact_identity(
    issues: list[ValidationIssue], field: str, artifact: ArtifactIdentity, *, required: bool = True
) -> None:
    if required:
        _required_text(issues, f"{field}.path", artifact.path)
        _required_text(issues, f"{field}.sha256", artifact.sha256)
    if artifact.sha256 is not None and not _is_sha256(artifact.sha256):
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER, f"{field}.sha256", "must be a SHA-256 hex digest"
            )
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _validate_population(
    issues: list[ValidationIssue], cohort: CohortContract, prefix: str
) -> dict[str, Any]:
    population = cohort.population
    for name in (
        "source_records",
        "candidate_records",
        "included_records",
        "reference_complete_records",
        "prediction_expected_records",
    ):
        _required_number(
            issues, f"{prefix}.population.{name}", getattr(population, name), integer=True
        )
    exclusion_total = 0
    if not population.excluded_records_by_reason:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                f"{prefix}.population.excluded_records_by_reason",
                "at least one named reason is required; use an explicit zero-valued 'none' entry",
            )
        )
    for reason, count in population.excluded_records_by_reason.items():
        _required_text(issues, f"{prefix}.population.excluded_records_by_reason key", reason)
        _required_number(
            issues,
            f"{prefix}.population.excluded_records_by_reason.{reason}",
            count,
            integer=True,
        )
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            exclusion_total += count

    counts = (
        population.source_records,
        population.candidate_records,
        population.included_records,
        population.reference_complete_records,
        population.prediction_expected_records,
    )
    if all(isinstance(value, int) and not isinstance(value, bool) for value in counts):
        source, candidate, included, reference_complete, prediction_expected = counts
        if candidate > source:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.population",
                    "candidate_records cannot exceed source_records",
                )
            )
        if included + exclusion_total != candidate:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.population",
                    "included_records plus named exclusions must equal candidate_records",
                )
            )
        if reference_complete > included:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.population",
                    "reference_complete_records cannot exceed included_records",
                )
            )
        if prediction_expected != included:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.population.prediction_expected_records",
                    "the frozen prediction surface must cover every included report",
                )
            )
    return {
        "source_records": population.source_records,
        "candidate_records": population.candidate_records,
        "included_records": population.included_records,
        "excluded_records": exclusion_total,
        "reference_complete_records": population.reference_complete_records,
        "prediction_expected_records": population.prediction_expected_records,
        "candidate_minus_included": (
            population.candidate_records - population.included_records
            if isinstance(population.candidate_records, int)
            and isinstance(population.included_records, int)
            else None
        ),
    }


def _inspect_cohort_files(
    issues: list[ValidationIssue],
    cohort: CohortContract,
    prefix: str,
    contract_path: Path,
    bundle_root: Path | None,
    patient_key_column: str | None,
) -> dict[str, Any]:
    report_key = cohort.predictions.report_key_column
    manifest_report_key = report_key or ""
    manifest_patient_key = patient_key_column
    result: dict[str, Any] = {
        "files_checked": True,
        "manifest_records": None,
        "prediction_records": None,
        "manifest_duplicate_report_keys": None,
        "prediction_duplicate_report_keys": None,
        "missing_prediction_keys": None,
        "extra_prediction_keys": None,
        "exact_same_case_surface": False,
        "patient_grouping_ready": False,
        "report_key_set_sha256": None,
        "patient_key_missing_records": None,
        "report_to_patient_mapping_sha256": None,
        "complete_four_level_prediction_records": None,
        "incomplete_or_invalid_prediction_records": None,
    }
    manifest_path = _resolve_artifact(cohort.manifest.artifact, contract_path, bundle_root)
    prediction_path = _resolve_artifact(
        cohort.predictions.tabular.artifact, contract_path, bundle_root
    )
    if (
        manifest_path is None
        or prediction_path is None
        or not isinstance(report_key, str)
        or not report_key
        or not all(
            isinstance(value, str) and value for value in cohort.predictions.label_columns.values()
        )
    ):
        return result
    for field, path, artifact in (
        (f"{prefix}.manifest.artifact", manifest_path, cohort.manifest.artifact),
        (
            f"{prefix}.predictions.surface.artifact",
            prediction_path,
            cohort.predictions.tabular.artifact,
        ),
    ):
        if not path.exists():
            issues.append(ValidationIssue(IssueSeverity.BLOCKER, field, "file does not exist"))
            return result
        if _is_sha256(artifact.sha256) and sha256_file(path).lower() != artifact.sha256.lower():
            issues.append(
                ValidationIssue(IssueSeverity.BLOCKER, f"{field}.sha256", "file checksum mismatch")
            )

    manifest_columns = [manifest_report_key]
    if manifest_patient_key:
        manifest_columns.append(manifest_patient_key)
    try:
        manifest = load_table(manifest_path, manifest_columns, cohort.manifest.table)
        predictions = load_table(
            prediction_path,
            [report_key, *cohort.predictions.label_columns.values()],
            cohort.predictions.tabular.table,
        )
    except (KeyError, ValueError) as error:
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, prefix, str(error)))
        return result

    manifest_null = int(manifest[manifest_report_key].isna().sum())
    prediction_null = int(predictions[report_key].isna().sum())
    manifest_non_string = int(
        manifest[manifest_report_key].dropna().map(lambda value: not isinstance(value, str)).sum()
    )
    prediction_non_string = int(
        predictions[report_key].dropna().map(lambda value: not isinstance(value, str)).sum()
    )
    manifest_duplicates = int(manifest[manifest_report_key].duplicated(keep=False).sum())
    prediction_duplicates = int(predictions[report_key].duplicated(keep=False).sum())
    if manifest_null:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                f"{prefix}.manifest",
                f"manifest contains {manifest_null} missing report keys",
            )
        )
    if prediction_null:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                f"{prefix}.predictions",
                f"prediction surface contains {prediction_null} missing report keys",
            )
        )
    if manifest_non_string:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                f"{prefix}.manifest",
                f"manifest contains {manifest_non_string} non-string report keys",
            )
        )
    if prediction_non_string:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                f"{prefix}.predictions",
                f"prediction surface contains {prediction_non_string} non-string report keys",
            )
        )
    if manifest_duplicates:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                f"{prefix}.manifest",
                f"manifest contains {manifest_duplicates} rows with duplicate report keys",
            )
        )
    if prediction_duplicates:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                f"{prefix}.predictions",
                (
                    f"prediction surface contains {prediction_duplicates} rows with duplicate "
                    "report keys"
                ),
            )
        )

    manifest_keys = set(manifest.loc[manifest[manifest_report_key].notna(), manifest_report_key])
    prediction_keys = set(predictions.loc[predictions[report_key].notna(), report_key])
    missing = len(manifest_keys - prediction_keys)
    extra = len(prediction_keys - manifest_keys)
    if missing or extra:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                prefix,
                f"prediction key surface differs from manifest: {missing} missing, {extra} extra",
            )
        )

    included = cohort.population.included_records
    expected = cohort.population.prediction_expected_records
    if isinstance(included, int) and len(manifest) != included:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                f"{prefix}.population.included_records",
                f"declares {included}, but manifest has {len(manifest)} rows",
            )
        )
    if isinstance(expected, int) and len(predictions) != expected:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                f"{prefix}.population.prediction_expected_records",
                f"declares {expected}, but prediction surface has {len(predictions)} rows",
            )
        )

    patient_missing = None
    patient_mapping_sha256 = None
    patient_non_string = None
    if manifest_patient_key:
        patient_missing = int(manifest[manifest_patient_key].isna().sum())
        patient_non_string = int(
            manifest[manifest_patient_key]
            .dropna()
            .map(lambda value: not isinstance(value, str))
            .sum()
        )
        if patient_missing:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.manifest",
                    f"manifest contains {patient_missing} missing patient keys",
                )
            )
        if patient_non_string:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.manifest",
                    f"manifest contains {patient_non_string} non-string patient keys",
                )
            )
        if patient_missing == 0 and patient_non_string == 0 and manifest_non_string == 0:
            patient_mapping_sha256 = _key_mapping_digest(
                manifest[manifest_report_key], manifest[manifest_patient_key]
            )

    prediction_validity = pd.DataFrame(
        {
            canonical: pd.to_numeric(predictions[source], errors="coerce").isin([1, 2, 3, 4])
            for canonical, source in cohort.predictions.label_columns.items()
        }
    ).all(axis=1)
    incomplete_predictions = int((~prediction_validity).sum())
    declared_incomplete = (
        cohort.predictions.invalid_records + cohort.predictions.unfinished_records
        if isinstance(cohort.predictions.invalid_records, int)
        and not isinstance(cohort.predictions.invalid_records, bool)
        and isinstance(cohort.predictions.unfinished_records, int)
        and not isinstance(cohort.predictions.unfinished_records, bool)
        else None
    )
    if declared_incomplete is not None and declared_incomplete != incomplete_predictions:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                f"{prefix}.predictions",
                (
                    "invalid_records plus unfinished_records must equal rows lacking a complete "
                    "five-label four-level prediction"
                ),
            )
        )
    result.update(
        {
            "manifest_records": int(len(manifest)),
            "prediction_records": int(len(predictions)),
            "manifest_duplicate_report_keys": manifest_duplicates,
            "prediction_duplicate_report_keys": prediction_duplicates,
            "missing_prediction_keys": missing,
            "extra_prediction_keys": extra,
            "exact_same_case_surface": not any(
                [
                    manifest_null,
                    prediction_null,
                    manifest_non_string,
                    prediction_non_string,
                    manifest_duplicates,
                    prediction_duplicates,
                    missing,
                    extra,
                ]
            ),
            "patient_grouping_ready": bool(manifest_patient_key)
            and patient_missing == 0
            and patient_non_string == 0,
            "report_key_set_sha256": _key_digest(manifest[manifest_report_key].dropna()),
            "patient_key_missing_records": patient_missing,
            "report_to_patient_mapping_sha256": patient_mapping_sha256,
            "complete_four_level_prediction_records": int(prediction_validity.sum()),
            "incomplete_or_invalid_prediction_records": incomplete_predictions,
        }
    )
    return result


def _inspect_identity_artifacts(
    issues: list[ValidationIssue],
    intake: ComparatorIntake,
    contract_path: Path,
    bundle_root: Path | None,
) -> dict[str, Any]:
    artifacts = {
        "model_identity.artifact": intake.model_identity.artifact,
        "prompt.artifact": intake.prompt.artifact,
        "prompt.selection_history_artifact": intake.prompt.selection_history_artifact,
    }
    if intake.runtime.chat_template.applied:
        artifacts["runtime.chat_template.artifact"] = intake.runtime.chat_template.artifact
    if intake.grammar.mode != "none":
        artifacts["grammar.artifact"] = intake.grammar.artifact

    receipt: dict[str, Any] = {}
    for field, artifact in artifacts.items():
        path = _resolve_artifact(artifact, contract_path, bundle_root)
        details = {"present": False, "sha256_matches": False, "size_bytes": None}
        if path is None or not path.exists():
            issues.append(ValidationIssue(IssueSeverity.BLOCKER, field, "file does not exist"))
            receipt[field] = details
            continue
        observed_sha256 = sha256_file(path)
        details.update(
            {
                "present": True,
                "sha256_matches": bool(
                    _is_sha256(artifact.sha256)
                    and observed_sha256.lower() == artifact.sha256.lower()
                ),
                "size_bytes": path.stat().st_size,
            }
        )
        if not details["sha256_matches"]:
            issues.append(
                ValidationIssue(IssueSeverity.BLOCKER, f"{field}.sha256", "file checksum mismatch")
            )
        if field == "model_identity.artifact" and (
            isinstance(intake.model_identity.size_bytes, int)
            and path.stat().st_size != intake.model_identity.size_bytes
        ):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    "model_identity.size_bytes",
                    "declared byte size does not match the model artifact",
                )
            )
        receipt[field] = details
    return receipt


def validate_intake(
    contract_path: Path,
    *,
    bundle_root: Path | None = None,
    check_files: bool = False,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve(strict=True)
    intake = load_intake(contract_path)
    issues: list[ValidationIssue] = []
    if intake.schema_version != 2:
        issues.append(
            ValidationIssue(IssueSeverity.BLOCKER, "schema_version", "schema version 2 is required")
        )
    if intake.evidence_layer is None:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "evidence_layer",
                "must be one of the three preregistered evidence layers",
            )
        )
    _required_text(issues, "status", intake.status)
    if intake.status not in {"template_unreceipted", "draft", "frozen", "source_of_record"}:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "status",
                "must be template_unreceipted, draft, frozen, or source_of_record",
            )
        )
    elif intake.status not in {"frozen", "source_of_record"}:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "status",
                "draft and template contracts are not analysis-ready",
            )
        )
    model = intake.model_identity
    for field, value in (
        ("model_identity.upstream_repo_id", model.upstream_repo_id),
        ("model_identity.upstream_revision", model.upstream_revision),
        ("model_identity.quantization", model.quantization),
        ("model_identity.license", model.license),
    ):
        _required_text(issues, field, value)
    _required_number(issues, "model_identity.size_bytes", model.size_bytes, integer=True)
    _validate_artifact_identity(issues, "model_identity.artifact", model.artifact)

    runtime = intake.runtime
    for field, value in (
        ("runtime.engine", runtime.engine),
        ("runtime.engine_version", runtime.engine_version),
        ("runtime.engine_revision", runtime.engine_revision),
        ("runtime.hardware", runtime.hardware),
        ("runtime.operating_system", runtime.operating_system),
        ("runtime.chat_template.mode", runtime.chat_template.mode),
        ("runtime.chat_template.source", runtime.chat_template.source),
    ):
        _required_text(issues, field, value)
    _required_bool(issues, "runtime.chat_template.applied", runtime.chat_template.applied)
    if runtime.chat_template.applied:
        _validate_artifact_identity(
            issues, "runtime.chat_template.artifact", runtime.chat_template.artifact
        )
    for field, value, integer, minimum in (
        ("runtime.n_ctx", runtime.n_ctx, True, 1),
        ("runtime.n_gpu_layers", runtime.n_gpu_layers, True, -1),
        ("runtime.temperature", runtime.temperature, False, 0),
        ("runtime.top_k", runtime.top_k, True, 0),
        ("runtime.top_p", runtime.top_p, False, 0),
        ("runtime.seed", runtime.seed, True, 0),
        ("runtime.max_tokens", runtime.max_tokens, True, 1),
    ):
        _required_number(issues, field, value, integer=integer, minimum=minimum)
    if (
        isinstance(runtime.top_p, (int, float))
        and not isinstance(runtime.top_p, bool)
        and runtime.top_p > 1
    ):
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, "runtime.top_p", "must be at most 1"))

    prompt = intake.prompt
    _required_text(issues, "prompt.id", prompt.prompt_id)
    _validate_artifact_identity(issues, "prompt.artifact", prompt.artifact)
    _required_text(issues, "prompt.development_population", prompt.development_population)
    _required_bool(
        issues,
        "prompt.reference_outcomes_inspected_during_selection",
        prompt.reference_outcomes_inspected_during_selection,
    )
    _required_bool(
        issues, "prompt.frozen_before_final_evaluation", prompt.frozen_before_final_evaluation
    )
    _required_text(issues, "prompt.stopping_rule", prompt.stopping_rule)
    _validate_artifact_identity(
        issues, "prompt.selection_history_artifact", prompt.selection_history_artifact
    )

    grammar = intake.grammar
    if grammar.mode not in {"gbnf", "json_schema", "none"}:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "grammar.mode",
                "must be 'gbnf', 'json_schema', or 'none'",
            )
        )
    _required_text(issues, "grammar.purpose", grammar.purpose)
    _validate_artifact_identity(
        issues, "grammar.artifact", grammar.artifact, required=grammar.mode != "none"
    )

    keys = intake.key_contract
    for field, value in (
        ("key_contract.report_key_column", keys.report_key_column),
        ("key_contract.report_key_namespace", keys.report_key_namespace),
        ("key_contract.report_key_normalization", keys.report_key_normalization),
    ):
        _required_text(issues, field, value)
    if keys.report_key_normalization != "exact_string":
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "key_contract.report_key_normalization",
                "must be 'exact_string'; implicit key coercion is not allowed",
            )
        )
    if keys.patient_key_column is not None:
        _required_text(issues, "key_contract.patient_key_namespace", keys.patient_key_namespace)
        _required_bool(
            issues,
            "key_contract.patient_key_semantics_confirmed",
            keys.patient_key_semantics_confirmed,
        )
    elif keys.patient_key_semantics_confirmed:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "key_contract.patient_key_column",
                "a patient key column is required when patient semantics are confirmed",
            )
        )

    if not intake.cohorts:
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, "cohorts", "at least one is required"))
    cohort_ids = [cohort.cohort_id for cohort in intake.cohorts if cohort.cohort_id]
    if len(cohort_ids) != len(set(cohort_ids)):
        issues.append(
            ValidationIssue(IssueSeverity.BLOCKER, "cohorts", "cohort IDs must be unique")
        )

    population_receipts: dict[str, Any] = {}
    key_receipts: dict[str, Any] = {}
    for index, cohort in enumerate(intake.cohorts):
        prefix = f"cohorts[{index}]"
        cohort_id = cohort.cohort_id or f"unidentified-{index}"
        _required_text(issues, f"{prefix}.cohort_id", cohort.cohort_id)
        if cohort.role not in {"development", "evaluation", "descriptive_unlabeled"}:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.role",
                    "must be development, evaluation, or descriptive_unlabeled",
                )
            )
        _validate_artifact_identity(issues, f"{prefix}.manifest.artifact", cohort.manifest.artifact)
        population_receipts[cohort_id] = _validate_population(issues, cohort, prefix)
        predictions = cohort.predictions
        _validate_artifact_identity(
            issues, f"{prefix}.predictions.surface.artifact", predictions.tabular.artifact
        )
        _required_text(
            issues, f"{prefix}.predictions.report_key_column", predictions.report_key_column
        )
        if predictions.report_key_column != keys.report_key_column:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.predictions.report_key_column",
                    "must match key_contract.report_key_column",
                )
            )
        if set(predictions.label_columns) != set(DEFAULT_LABELS):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.predictions.label_columns",
                    "canonical label mappings must contain exactly the five study labels",
                )
            )
        for canonical, source in predictions.label_columns.items():
            _required_text(
                issues,
                f"{prefix}.predictions.label_columns.{canonical}",
                source,
            )
        source_columns = list(predictions.label_columns.values())
        if all(isinstance(value, str) for value in source_columns) and len(source_columns) != len(
            set(source_columns)
        ):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.predictions.label_columns",
                    "each canonical label must map to a distinct source column",
                )
            )
        _required_number(
            issues,
            f"{prefix}.predictions.invalid_records",
            predictions.invalid_records,
            integer=True,
        )
        _required_number(
            issues,
            f"{prefix}.predictions.unfinished_records",
            predictions.unfinished_records,
            integer=True,
        )
        key_receipts[cohort_id] = (
            _inspect_cohort_files(
                issues,
                cohort,
                prefix,
                contract_path,
                bundle_root,
                keys.patient_key_column,
            )
            if check_files
            else {
                "files_checked": False,
                "exact_same_case_surface": None,
                "patient_grouping_ready": None,
                "report_key_set_sha256": None,
                "report_to_patient_mapping_sha256": None,
            }
        )

    artifact_receipt = (
        _inspect_identity_artifacts(issues, intake, contract_path, bundle_root)
        if check_files
        else {"files_checked": False}
    )
    blockers = [issue for issue in issues if issue.severity == IssueSeverity.BLOCKER]
    return {
        "schema_version": 1,
        "contract_schema_version": intake.schema_version,
        "contract_sha256": sha256_file(contract_path),
        "evidence_layer": intake.evidence_layer.value if intake.evidence_layer else None,
        "status": intake.status,
        "ready_for_analysis": not blockers and check_files,
        "files_checked": check_files,
        "population_arithmetic": population_receipts,
        "artifact_validation": artifact_receipt,
        "key_validation": key_receipts,
        "issues": [{**asdict(issue), "severity": issue.severity.value} for issue in issues],
        "privacy_boundary": (
            "Aggregate validation counts and digests only; report and patient keys are not emitted."
        ),
    }


def validate_intake_to_directory(
    contract_path: Path,
    output_dir: Path,
    *,
    bundle_root: Path | None = None,
    check_files: bool = False,
) -> dict[str, Any]:
    result = validate_intake(contract_path, bundle_root=bundle_root, check_files=check_files)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "intake_validation.json", result)
    atomic_write_json(
        output_dir / "run_manifest.json",
        build_manifest(
            "intake-validate",
            [contract_path],
            {"check_files": check_files, "bundle_root_supplied": bundle_root is not None},
        ),
    )
    return result
