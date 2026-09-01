from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .intake import IssueSeverity, ValidationIssue
from .io import atomic_write_json
from .manifest import build_manifest, sha256_file


class SurfaceStatus(StrEnum):
    SOURCE_OF_RECORD = "source_of_record"
    COMPLETED_GOVERNED = "completed_governed"
    PLANNED = "planned"
    EXTERNAL_SUMMARY_PENDING_INTAKE = "external_summary_pending_intake"


class SurfaceRole(StrEnum):
    SUBMITTED = "submitted"
    REPRODUCTION = "reproduction"
    DEVELOPMENT = "development"
    HELD_OUT_EVALUATION = "held_out_evaluation"
    EXPLORATORY_SELECTION = "exploratory_selection"
    DESCRIPTIVE_UNLABELED = "descriptive_unlabeled"
    EXTERNAL_PENDING_INTAKE = "external_pending_intake"


class ContrastKind(StrEnum):
    REPRODUCTION = "reproduction"
    CONTROLLED_ABLATION = "controlled_ablation"
    MODEL_NATIVE_TASK_COMPARISON = "model_native_task_comparison"
    CONFIGURED_SYSTEM_COMPARISON = "configured_system_comparison"
    COHORT_STRATIFICATION = "cohort_stratification"
    DESCRIPTIVE_UNLABELED = "descriptive_unlabeled"


class ReferenceRequirement(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    PROHIBITED = "prohibited"


class DefinitionStatus(StrEnum):
    DEFINED = "defined"
    PENDING_PRODUCING_BUNDLE = "pending_producing_bundle"


class DesignFamilyStatus(StrEnum):
    INTERNAL_RECEIPTED = "internal_receipted"
    EXTERNAL_SUMMARY_PENDING_INTAKE = "external_summary_pending_intake"


@dataclass(frozen=True)
class FactorDefinition:
    factor_id: str | None
    scope: str | None
    allowed_values: tuple[str, ...]
    interpretation: str | None


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str | None
    level: str | None
    statistic: str | None
    reference_requirement: ReferenceRequirement | None
    unlabeled_allowed: bool | None
    definition_status: DefinitionStatus | None
    interpretation: str | None


@dataclass(frozen=True)
class PopulationComponent:
    cohort_id: str | None
    records: int | None


@dataclass(frozen=True)
class PopulationDefinition:
    records: int | None
    includes_development: bool | None
    components: tuple[PopulationComponent, ...]
    arithmetic_status: str | None


@dataclass(frozen=True)
class EvaluationSurface:
    surface_id: str | None
    status: SurfaceStatus | None
    role: SurfaceRole | None
    provenance_node_id: str | None
    artifact_revision: str | None
    factors: dict[str, str | None]
    population: PopulationDefinition
    metric_ids: tuple[str, ...]
    result_values_in_registry: bool | None


@dataclass(frozen=True)
class EvaluationContrast:
    contrast_id: str | None
    status: str | None
    kind: ContrastKind | None
    surface_a: str | None
    surface_b: str | None
    declared_changed_factors: tuple[str, ...]
    metric_ids: tuple[str, ...]
    causal_attribution_allowed: bool | None
    claim_scope: str | None


@dataclass(frozen=True)
class DesignFamily:
    family_id: str | None
    status: DesignFamilyStatus | None
    factor_levels: dict[str, tuple[str, ...]]
    cohort_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    intake_boundary: str | None


@dataclass(frozen=True)
class EvaluationSurfaceRegistry:
    schema_version: int
    status: str | None
    study_id: str | None
    task_semantics_id: str | None
    factor_definitions: tuple[FactorDefinition, ...]
    metric_definitions: tuple[MetricDefinition, ...]
    surfaces: tuple[EvaluationSurface, ...]
    contrasts: tuple[EvaluationContrast, ...]
    design_families: tuple[DesignFamily, ...]
    claim_boundary: str | None
    privacy_boundary: str | None


TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "study_id",
    "task_semantics_id",
    "factor_definitions",
    "metric_definitions",
    "surfaces",
    "contrasts",
    "design_families",
    "claim_boundary",
    "privacy_boundary",
}
FACTOR_FIELDS = {"factor_id", "scope", "allowed_values", "interpretation"}
METRIC_FIELDS = {
    "metric_id",
    "level",
    "statistic",
    "reference_requirement",
    "unlabeled_allowed",
    "definition_status",
    "interpretation",
}
SURFACE_FIELDS = {
    "surface_id",
    "status",
    "role",
    "provenance_node_id",
    "artifact_revision",
    "factors",
    "population",
    "metric_ids",
    "result_values_in_registry",
}
POPULATION_FIELDS = {"records", "includes_development", "components", "arithmetic_status"}
COMPONENT_FIELDS = {"cohort_id", "records"}
CONTRAST_FIELDS = {
    "contrast_id",
    "status",
    "kind",
    "surface_a",
    "surface_b",
    "declared_changed_factors",
    "metric_ids",
    "causal_attribution_allowed",
    "claim_scope",
}
DESIGN_FAMILY_FIELDS = {
    "family_id",
    "status",
    "factor_levels",
    "cohort_ids",
    "metric_ids",
    "intake_boundary",
}

REQUIRED_FACTOR_IDS = {
    "model_family",
    "quantization",
    "interface_mode",
    "task_semantics",
    "prompt_variant",
    "grammar_mode",
    "cohort",
    "reference",
    "selection_role",
    "evidence_layer",
}
CONFIGURATION_FACTORS = {
    "model_family",
    "quantization",
    "interface_mode",
    "task_semantics",
    "prompt_variant",
    "grammar_mode",
}
CONTROLLED_ABLATION_FACTORS = {
    "quantization",
    "interface_mode",
    "prompt_variant",
    "grammar_mode",
}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _enum_or_none(enum_type: type[StrEnum], value: Any) -> Any:
    try:
        return enum_type(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _reject_unknown_fields(
    issues: list[ValidationIssue], field: str, value: Any, allowed: set[str]
) -> None:
    if not isinstance(value, dict):
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, field, "object is required"))
        return
    for key in sorted(set(value) - allowed):
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                f"{field}.{key}" if field else key,
                "unrecognized field is prohibited in the evaluation-surface registry",
            )
        )


def _validate_shape(payload: dict[str, Any], issues: list[ValidationIssue]) -> None:
    _reject_unknown_fields(issues, "", payload, TOP_LEVEL_FIELDS)
    for block, allowed in (
        ("factor_definitions", FACTOR_FIELDS),
        ("metric_definitions", METRIC_FIELDS),
        ("surfaces", SURFACE_FIELDS),
        ("contrasts", CONTRAST_FIELDS),
        ("design_families", DESIGN_FAMILY_FIELDS),
    ):
        values = payload.get(block)
        if not isinstance(values, list):
            issues.append(ValidationIssue(IssueSeverity.BLOCKER, block, "array is required"))
            continue
        for index, value in enumerate(values):
            _reject_unknown_fields(issues, f"{block}[{index}]", value, allowed)
            if block == "surfaces" and isinstance(value, dict):
                _reject_unknown_fields(
                    issues,
                    f"surfaces[{index}].population",
                    value.get("population"),
                    POPULATION_FIELDS,
                )
                population = _mapping(value.get("population"))
                components = population.get("components")
                if not isinstance(components, list):
                    issues.append(
                        ValidationIssue(
                            IssueSeverity.BLOCKER,
                            f"surfaces[{index}].population.components",
                            "array is required",
                        )
                    )
                else:
                    for component_index, component in enumerate(components):
                        _reject_unknown_fields(
                            issues,
                            f"surfaces[{index}].population.components[{component_index}]",
                            component,
                            COMPONENT_FIELDS,
                        )


def parse_evaluation_surface_registry(payload: dict[str, Any]) -> EvaluationSurfaceRegistry:
    factors = tuple(
        FactorDefinition(
            factor_id=item.get("factor_id"),
            scope=item.get("scope"),
            allowed_values=_strings(item.get("allowed_values")),
            interpretation=item.get("interpretation"),
        )
        for raw in payload.get("factor_definitions", [])
        if (item := _mapping(raw))
    )
    metrics = tuple(
        MetricDefinition(
            metric_id=item.get("metric_id"),
            level=item.get("level"),
            statistic=item.get("statistic"),
            reference_requirement=_enum_or_none(
                ReferenceRequirement, item.get("reference_requirement")
            ),
            unlabeled_allowed=item.get("unlabeled_allowed"),
            definition_status=_enum_or_none(DefinitionStatus, item.get("definition_status")),
            interpretation=item.get("interpretation"),
        )
        for raw in payload.get("metric_definitions", [])
        if (item := _mapping(raw))
    )
    surfaces: list[EvaluationSurface] = []
    for raw in payload.get("surfaces", []):
        item = _mapping(raw)
        population = _mapping(item.get("population"))
        components = tuple(
            PopulationComponent(
                cohort_id=component.get("cohort_id"), records=component.get("records")
            )
            for raw_component in population.get("components", [])
            if (component := _mapping(raw_component))
        )
        surfaces.append(
            EvaluationSurface(
                surface_id=item.get("surface_id"),
                status=_enum_or_none(SurfaceStatus, item.get("status")),
                role=_enum_or_none(SurfaceRole, item.get("role")),
                provenance_node_id=item.get("provenance_node_id"),
                artifact_revision=item.get("artifact_revision"),
                factors={
                    key: value if isinstance(value, str) else None
                    for key, value in _mapping(item.get("factors")).items()
                },
                population=PopulationDefinition(
                    records=population.get("records"),
                    includes_development=population.get("includes_development"),
                    components=components,
                    arithmetic_status=population.get("arithmetic_status"),
                ),
                metric_ids=_strings(item.get("metric_ids")),
                result_values_in_registry=item.get("result_values_in_registry"),
            )
        )
    contrasts = tuple(
        EvaluationContrast(
            contrast_id=item.get("contrast_id"),
            status=item.get("status"),
            kind=_enum_or_none(ContrastKind, item.get("kind")),
            surface_a=item.get("surface_a"),
            surface_b=item.get("surface_b"),
            declared_changed_factors=_strings(item.get("declared_changed_factors")),
            metric_ids=_strings(item.get("metric_ids")),
            causal_attribution_allowed=item.get("causal_attribution_allowed"),
            claim_scope=item.get("claim_scope"),
        )
        for raw in payload.get("contrasts", [])
        if (item := _mapping(raw))
    )
    families = tuple(
        DesignFamily(
            family_id=item.get("family_id"),
            status=_enum_or_none(DesignFamilyStatus, item.get("status")),
            factor_levels={
                key: _strings(value) for key, value in _mapping(item.get("factor_levels")).items()
            },
            cohort_ids=_strings(item.get("cohort_ids")),
            metric_ids=_strings(item.get("metric_ids")),
            intake_boundary=item.get("intake_boundary"),
        )
        for raw in payload.get("design_families", [])
        if (item := _mapping(raw))
    )
    return EvaluationSurfaceRegistry(
        schema_version=payload.get("schema_version", 0),
        status=payload.get("status"),
        study_id=payload.get("study_id"),
        task_semantics_id=payload.get("task_semantics_id"),
        factor_definitions=factors,
        metric_definitions=metrics,
        surfaces=tuple(surfaces),
        contrasts=contrasts,
        design_families=families,
        claim_boundary=payload.get("claim_boundary"),
        privacy_boundary=payload.get("privacy_boundary"),
    )


def _duplicates(values: list[str]) -> set[str]:
    return {value for value in values if values.count(value) > 1}


def _require_text(issues: list[ValidationIssue], field: str, value: str | None) -> None:
    if not isinstance(value, str) or not value.strip():
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, field, "non-empty text is required"))


def _validate_registry(registry: EvaluationSurfaceRegistry, issues: list[ValidationIssue]) -> None:
    if registry.schema_version != 1:
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, "schema_version", "must be exactly 1"))
    for field, value in (
        ("status", registry.status),
        ("study_id", registry.study_id),
        ("task_semantics_id", registry.task_semantics_id),
        ("claim_boundary", registry.claim_boundary),
        ("privacy_boundary", registry.privacy_boundary),
    ):
        _require_text(issues, field, value)

    factor_ids = [factor.factor_id for factor in registry.factor_definitions if factor.factor_id]
    for duplicate in sorted(_duplicates(factor_ids)):
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER, f"factor_definitions[{duplicate}]", "duplicate factor_id"
            )
        )
    missing_factors = REQUIRED_FACTOR_IDS - set(factor_ids)
    for factor_id in sorted(missing_factors):
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "factor_definitions",
                f"required comparison axis is missing: {factor_id}",
            )
        )
    factor_map = {
        factor.factor_id: factor for factor in registry.factor_definitions if factor.factor_id
    }
    for index, factor in enumerate(registry.factor_definitions):
        _require_text(issues, f"factor_definitions[{index}].factor_id", factor.factor_id)
        _require_text(issues, f"factor_definitions[{index}].scope", factor.scope)
        _require_text(issues, f"factor_definitions[{index}].interpretation", factor.interpretation)
        if not factor.allowed_values:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"factor_definitions[{index}].allowed_values",
                    "at least one allowed value is required",
                )
            )
        if len(set(factor.allowed_values)) != len(factor.allowed_values):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"factor_definitions[{index}].allowed_values",
                    "allowed values must be unique",
                )
            )

    metric_ids = [metric.metric_id for metric in registry.metric_definitions if metric.metric_id]
    for duplicate in sorted(_duplicates(metric_ids)):
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER, f"metric_definitions[{duplicate}]", "duplicate metric_id"
            )
        )
    metric_map = {
        metric.metric_id: metric for metric in registry.metric_definitions if metric.metric_id
    }
    if "certainty_f1" in metric_map:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "metric_definitions[certainty_f1]",
                "ambiguous shorthand is prohibited; name the exact four-level "
                "statistic and formula",
            )
        )
    for index, metric in enumerate(registry.metric_definitions):
        for field, value in (
            ("metric_id", metric.metric_id),
            ("level", metric.level),
            ("statistic", metric.statistic),
            ("interpretation", metric.interpretation),
        ):
            _require_text(issues, f"metric_definitions[{index}].{field}", value)
        if metric.reference_requirement is None:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"metric_definitions[{index}].reference_requirement",
                    "recognized reference requirement is required",
                )
            )
        if metric.definition_status is None:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"metric_definitions[{index}].definition_status",
                    "recognized definition status is required",
                )
            )
        if not isinstance(metric.unlabeled_allowed, bool):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"metric_definitions[{index}].unlabeled_allowed",
                    "boolean is required",
                )
            )

    surface_ids = [surface.surface_id for surface in registry.surfaces if surface.surface_id]
    for duplicate in sorted(_duplicates(surface_ids)):
        issues.append(
            ValidationIssue(IssueSeverity.BLOCKER, f"surfaces[{duplicate}]", "duplicate surface_id")
        )
    surface_map = {
        surface.surface_id: surface for surface in registry.surfaces if surface.surface_id
    }
    for index, surface in enumerate(registry.surfaces):
        prefix = f"surfaces[{surface.surface_id or index}]"
        _require_text(issues, f"{prefix}.surface_id", surface.surface_id)
        _require_text(issues, f"{prefix}.provenance_node_id", surface.provenance_node_id)
        if surface.status is None:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER, f"{prefix}.status", "recognized status is required"
                )
            )
        if surface.role is None:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER, f"{prefix}.role", "recognized role is required"
                )
            )
        unknown_factor_ids = set(surface.factors) - set(factor_map)
        for factor_id in sorted(unknown_factor_ids):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER, f"{prefix}.factors.{factor_id}", "factor is not defined"
                )
            )
        for factor_id in sorted(REQUIRED_FACTOR_IDS - set(surface.factors)):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.factors",
                    f"required factor is missing: {factor_id}",
                )
            )
        for factor_id, value in surface.factors.items():
            factor = factor_map.get(factor_id)
            if value is not None and factor and value not in factor.allowed_values:
                issues.append(
                    ValidationIssue(
                        IssueSeverity.BLOCKER,
                        f"{prefix}.factors.{factor_id}",
                        f"value is not registered: {value}",
                    )
                )
        if surface.status != SurfaceStatus.EXTERNAL_SUMMARY_PENDING_INTAKE:
            for factor_id in sorted(REQUIRED_FACTOR_IDS):
                if not surface.factors.get(factor_id):
                    issues.append(
                        ValidationIssue(
                            IssueSeverity.BLOCKER,
                            f"{prefix}.factors.{factor_id}",
                            "non-external surfaces require a resolved factor value",
                        )
                    )
        if surface.result_values_in_registry is not False:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.result_values_in_registry",
                    "this public-safe design registry must not contain result values",
                )
            )
        if not surface.metric_ids:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER, f"{prefix}.metric_ids", "at least one metric is required"
                )
            )
        for metric_id in surface.metric_ids:
            if metric_id not in metric_map:
                issues.append(
                    ValidationIssue(
                        IssueSeverity.BLOCKER,
                        f"{prefix}.metric_ids",
                        f"unknown metric: {metric_id}",
                    )
                )
        records = surface.population.records
        if records is not None and (not isinstance(records, int) or records <= 0):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.population.records",
                    "positive integer or null is required",
                )
            )
        component_records = [component.records for component in surface.population.components]
        if any(
            value is None or not isinstance(value, int) or value <= 0 for value in component_records
        ):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.population.components",
                    "component records must be positive integers",
                )
            )
        if records is not None and component_records and records != sum(component_records):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.population.records",
                    "population arithmetic does not reconcile: "
                    f"{records} != {sum(component_records)}",
                )
            )
        if (
            surface.role == SurfaceRole.HELD_OUT_EVALUATION
            and surface.population.includes_development is not False
        ):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.population.includes_development",
                    "a held-out evaluation surface cannot include development cases",
                )
            )
        reference = surface.factors.get("reference")
        if surface.role == SurfaceRole.DESCRIPTIVE_UNLABELED and reference != "none":
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.factors.reference",
                    "unlabeled descriptive surfaces require reference=none",
                )
            )
        if reference == "none":
            for metric_id in surface.metric_ids:
                metric = metric_map.get(metric_id)
                if metric and (
                    metric.reference_requirement == ReferenceRequirement.REQUIRED
                    or metric.unlabeled_allowed is not True
                ):
                    issues.append(
                        ValidationIssue(
                            IssueSeverity.BLOCKER,
                            f"{prefix}.metric_ids[{metric_id}]",
                            "metric is not valid without a reference annotation",
                        )
                    )

    contrast_ids = [contrast.contrast_id for contrast in registry.contrasts if contrast.contrast_id]
    for duplicate in sorted(_duplicates(contrast_ids)):
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER, f"contrasts[{duplicate}]", "duplicate contrast_id"
            )
        )
    for index, contrast in enumerate(registry.contrasts):
        prefix = f"contrasts[{contrast.contrast_id or index}]"
        _require_text(issues, f"{prefix}.contrast_id", contrast.contrast_id)
        _require_text(issues, f"{prefix}.status", contrast.status)
        _require_text(issues, f"{prefix}.claim_scope", contrast.claim_scope)
        if contrast.kind is None:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER, f"{prefix}.kind", "recognized contrast kind is required"
                )
            )
            continue
        surface_a = surface_map.get(contrast.surface_a)
        surface_b = surface_map.get(contrast.surface_b)
        if surface_a is None or surface_b is None:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER, prefix, "both referenced surfaces must exist"
                )
            )
            continue
        changed = {
            factor_id
            for factor_id in REQUIRED_FACTOR_IDS
            if surface_a.factors.get(factor_id) != surface_b.factors.get(factor_id)
        }
        declared = set(contrast.declared_changed_factors)
        if changed != declared:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.declared_changed_factors",
                    f"declared {sorted(declared)} but surfaces differ on {sorted(changed)}",
                )
            )
        for metric_id in contrast.metric_ids:
            if (
                metric_id not in metric_map
                or metric_id not in surface_a.metric_ids
                or metric_id not in surface_b.metric_ids
            ):
                issues.append(
                    ValidationIssue(
                        IssueSeverity.BLOCKER,
                        f"{prefix}.metric_ids[{metric_id}]",
                        "contrast metrics must be registered on both surfaces",
                    )
                )
        if contrast.kind == ContrastKind.CONTROLLED_ABLATION and (
            len(changed) != 1 or not changed <= CONTROLLED_ABLATION_FACTORS
        ):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.declared_changed_factors",
                    "controlled ablation must change exactly one interface, prompt, "
                    "grammar, or quantization factor",
                )
            )
        if contrast.kind == ContrastKind.MODEL_NATIVE_TASK_COMPARISON:
            required_same = {"interface_mode", "task_semantics", "cohort", "reference"}
            if any(
                surface_a.factors.get(factor) != surface_b.factors.get(factor)
                for factor in required_same
            ):
                issues.append(
                    ValidationIssue(
                        IssueSeverity.BLOCKER,
                        prefix,
                        "model-native task comparison requires matched native interface, "
                        "task, cohort, and reference",
                    )
                )
            if surface_a.factors.get("interface_mode") != "model_native_chat":
                issues.append(
                    ValidationIssue(
                        IssueSeverity.BLOCKER, prefix, "both surfaces must use model_native_chat"
                    )
                )
        if contrast.kind == ContrastKind.REPRODUCTION and changed - {"evidence_layer"}:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    prefix,
                    "reproduction may differ only in evidence-layer provenance",
                )
            )
        if contrast.kind == ContrastKind.COHORT_STRATIFICATION and changed != {"cohort"}:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    prefix,
                    "cohort stratification must change only cohort",
                )
            )
        if (
            contrast.kind
            in {
                ContrastKind.MODEL_NATIVE_TASK_COMPARISON,
                ContrastKind.CONFIGURED_SYSTEM_COMPARISON,
            }
            and contrast.causal_attribution_allowed is not False
        ):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.causal_attribution_allowed",
                    "multi-axis model/system contrasts cannot attribute the effect to base weights",
                )
            )

    family_ids = [family.family_id for family in registry.design_families if family.family_id]
    for duplicate in sorted(_duplicates(family_ids)):
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER, f"design_families[{duplicate}]", "duplicate family_id"
            )
        )
    for index, family in enumerate(registry.design_families):
        prefix = f"design_families[{family.family_id or index}]"
        _require_text(issues, f"{prefix}.family_id", family.family_id)
        _require_text(issues, f"{prefix}.intake_boundary", family.intake_boundary)
        if family.status is None:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER, f"{prefix}.status", "recognized status is required"
                )
            )
        for factor_id, values in family.factor_levels.items():
            if factor_id not in factor_map:
                issues.append(
                    ValidationIssue(
                        IssueSeverity.BLOCKER,
                        f"{prefix}.factor_levels.{factor_id}",
                        "factor is not defined",
                    )
                )
                continue
            invalid = set(values) - set(factor_map[factor_id].allowed_values)
            if invalid:
                issues.append(
                    ValidationIssue(
                        IssueSeverity.BLOCKER,
                        f"{prefix}.factor_levels.{factor_id}",
                        f"unregistered values: {sorted(invalid)}",
                    )
                )
        for metric_id in family.metric_ids:
            if metric_id not in metric_map:
                issues.append(
                    ValidationIssue(
                        IssueSeverity.BLOCKER,
                        f"{prefix}.metric_ids",
                        f"unknown metric: {metric_id}",
                    )
                )


def validate_evaluation_surface_registry(contract_path: Path) -> dict[str, Any]:
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation-surface registry must be a JSON object")
    issues: list[ValidationIssue] = []
    _validate_shape(payload, issues)
    registry = parse_evaluation_surface_registry(payload)
    _validate_registry(registry, issues)
    blockers = [issue for issue in issues if issue.severity == IssueSeverity.BLOCKER]
    return {
        "schema_version": 1,
        "contract_schema_version": registry.schema_version,
        "contract_sha256": sha256_file(contract_path),
        "study_id": registry.study_id,
        "design_valid": not blockers,
        "analysis_started": False,
        "factor_count": len(registry.factor_definitions),
        "metric_count": len(registry.metric_definitions),
        "surface_count": len(registry.surfaces),
        "contrast_count": len(registry.contrasts),
        "design_family_count": len(registry.design_families),
        "issues": [{**asdict(issue), "severity": issue.severity.value} for issue in issues],
        "claim_boundary": (
            "This receipt validates comparison axes and contrast semantics only. It contains no "
            "performance value and does not admit an external result or producing configuration."
        ),
        "privacy_boundary": (
            "No report text, report key, patient key, case-level prediction, model weight, or "
            "private correspondence is read or emitted."
        ),
    }


def validate_evaluation_surface_registry_to_directory(
    contract_path: Path, output_dir: Path
) -> dict[str, Any]:
    result = validate_evaluation_surface_registry(contract_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "evaluation_surface_validation.json", result)
    atomic_write_json(
        output_dir / "run_manifest.json",
        build_manifest("evaluation-surface-validate", [contract_path], {}),
    )
    return result
