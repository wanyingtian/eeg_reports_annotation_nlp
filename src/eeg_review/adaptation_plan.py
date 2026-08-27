from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .intake import ArtifactIdentity, IssueSeverity, ValidationIssue
from .io import atomic_write_json
from .manifest import build_manifest, sha256_file


class AdaptationPlanStatus(StrEnum):
    PREREGISTERED_UNFROZEN = "preregistered_unfrozen"
    FROZEN_BEFORE_EVALUATION = "frozen_before_evaluation"
    RETIRED = "retired"


class SignalUse(StrEnum):
    DEVELOPMENT = "development"
    DESIGN_PRIOR = "design_prior"
    EVALUATION_ONLY = "evaluation_only"
    CONTEXT_ONLY_PROHIBITED_FOR_SELECTION = "context_only_prohibited_for_selection"


class AdaptationMethod(StrEnum):
    ITERATIVE_PROMPT_ENGINEERING = "iterative_prompt_engineering"
    GRAMMAR_CONSTRAINED_DECODING = "grammar_constrained_decoding"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    DETERMINISTIC_CONSISTENCY_CHECKS = "deterministic_consistency_checks"
    POST_HOC_CALIBRATION = "post_hoc_calibration"
    SOFT_PROMPT_TUNING = "soft_prompt_tuning"
    LORA = "lora"
    TEACHER_STUDENT_DISTILLATION = "teacher_student_distillation"


@dataclass(frozen=True)
class AdaptationSignal:
    signal_id: str | None
    signal_type: str | None
    use: SignalUse | None
    used_for_parameter_or_variant_selection: bool | None
    outcomes_inspected: bool | None
    artifact: ArtifactIdentity
    note: str | None


@dataclass(frozen=True)
class TaskAdapter:
    adapter_id: str | None
    route: str | None
    methods: tuple[AdaptationMethod, ...]
    parameter_update: str | None
    teacher_model_outputs_used: bool | None
    artifact: ArtifactIdentity


@dataclass(frozen=True)
class FreezeState:
    author_group_admitted: bool | None
    frozen_before_evaluation: bool | None
    selection_rule: str | None
    stopping_rule: str | None
    receipt: ArtifactIdentity


@dataclass(frozen=True)
class EvaluationBoundary:
    development_cohort_id: str | None
    evaluation_cohort_ids: tuple[str, ...]
    exact_same_case_required: bool | None
    patient_grouping_required_when_confirmed: bool | None
    primary_contrast: str | None
    attribution_scope: str | None
    medgemma_comparison_scope: str | None


@dataclass(frozen=True)
class AdaptationPlan:
    schema_version: int
    status: AdaptationPlanStatus | None
    plan_id: str | None
    proposed_evidence_layer_id: str | None
    base_evidence_layer: str | None
    scientific_question: str | None
    hypothesis: str | None
    task_adapter: TaskAdapter
    signals: tuple[AdaptationSignal, ...]
    freeze: FreezeState
    evaluation: EvaluationBoundary
    claim_boundary: str | None
    privacy_boundary: str | None


REQUIRED_METHODS = {
    AdaptationMethod.ITERATIVE_PROMPT_ENGINEERING,
    AdaptationMethod.GRAMMAR_CONSTRAINED_DECODING,
    AdaptationMethod.EVIDENCE_EXTRACTION,
    AdaptationMethod.DETERMINISTIC_CONSISTENCY_CHECKS,
    AdaptationMethod.POST_HOC_CALIBRATION,
}

PARAMETRIC_OR_TEACHER_METHODS = {
    AdaptationMethod.SOFT_PROMPT_TUNING,
    AdaptationMethod.LORA,
    AdaptationMethod.TEACHER_STUDENT_DISTILLATION,
}

REQUIRED_SIGNAL_USES = {
    "zoe_development_first_100_ra": SignalUse.DEVELOPMENT,
    "clinical_guidelines_and_annotation_schema": SignalUse.DESIGN_PRIOR,
    "reviewer_method_requests": SignalUse.DESIGN_PRIOR,
    "zoe_evaluation": SignalUse.EVALUATION_ONLY,
    "maria_evaluation": SignalUse.EVALUATION_ONLY,
    "medgemma_v5g_predictions": SignalUse.CONTEXT_ONLY_PROHIBITED_FOR_SELECTION,
    "medgemma_v5g_aggregate_results": SignalUse.CONTEXT_ONLY_PROHIBITED_FOR_SELECTION,
}

TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "plan_id",
    "proposed_evidence_layer_id",
    "base_evidence_layer",
    "scientific_question",
    "hypothesis",
    "task_adapter",
    "signals",
    "freeze",
    "evaluation",
    "claim_boundary",
    "privacy_boundary",
}

BLOCK_FIELDS = {
    "task_adapter": {
        "adapter_id",
        "route",
        "methods",
        "parameter_update",
        "teacher_model_outputs_used",
        "artifact",
    },
    "freeze": {
        "author_group_admitted",
        "frozen_before_evaluation",
        "selection_rule",
        "stopping_rule",
        "receipt",
    },
    "evaluation": {
        "development_cohort_id",
        "evaluation_cohort_ids",
        "exact_same_case_required",
        "patient_grouping_required_when_confirmed",
        "primary_contrast",
        "attribution_scope",
        "medgemma_comparison_scope",
    },
}

SIGNAL_FIELDS = {
    "signal_id",
    "signal_type",
    "use",
    "used_for_parameter_or_variant_selection",
    "outcomes_inspected",
    "artifact",
    "note",
}

ARTIFACT_FIELDS = {"path", "sha256"}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _artifact(value: Any) -> ArtifactIdentity:
    data = _mapping(value)
    return ArtifactIdentity(path=data.get("path"), sha256=data.get("sha256"))


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
                "unrecognized field is prohibited in the frozen scientific contract",
            )
        )


def _validate_payload_shape(issues: list[ValidationIssue], payload: dict[str, Any]) -> None:
    _reject_unknown_fields(issues, "", payload, TOP_LEVEL_FIELDS)
    for block, fields in BLOCK_FIELDS.items():
        _reject_unknown_fields(issues, block, payload.get(block), fields)
    signals = payload.get("signals")
    if not isinstance(signals, list):
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, "signals", "array is required"))
    else:
        for index, signal in enumerate(signals):
            _reject_unknown_fields(issues, f"signals[{index}]", signal, SIGNAL_FIELDS)
            if isinstance(signal, dict):
                _reject_unknown_fields(
                    issues,
                    f"signals[{index}].artifact",
                    signal.get("artifact"),
                    ARTIFACT_FIELDS,
                )
    for field, artifact in (
        ("task_adapter.artifact", _mapping(payload.get("task_adapter")).get("artifact")),
        ("freeze.receipt", _mapping(payload.get("freeze")).get("receipt")),
    ):
        _reject_unknown_fields(issues, field, artifact, ARTIFACT_FIELDS)


def parse_adaptation_plan(payload: dict[str, Any]) -> AdaptationPlan:
    adapter = _mapping(payload.get("task_adapter"))
    freeze = _mapping(payload.get("freeze"))
    evaluation = _mapping(payload.get("evaluation"))

    methods: list[AdaptationMethod] = []
    raw_methods = adapter.get("methods")
    if isinstance(raw_methods, list):
        for value in raw_methods:
            method = _enum_or_none(AdaptationMethod, value)
            if method is not None:
                methods.append(method)

    signals: list[AdaptationSignal] = []
    raw_signals = payload.get("signals")
    if isinstance(raw_signals, list):
        for raw_signal in raw_signals:
            signal = _mapping(raw_signal)
            signals.append(
                AdaptationSignal(
                    signal_id=signal.get("signal_id"),
                    signal_type=signal.get("signal_type"),
                    use=_enum_or_none(SignalUse, signal.get("use")),
                    used_for_parameter_or_variant_selection=signal.get(
                        "used_for_parameter_or_variant_selection"
                    ),
                    outcomes_inspected=signal.get("outcomes_inspected"),
                    artifact=_artifact(signal.get("artifact")),
                    note=signal.get("note"),
                )
            )

    raw_evaluation_cohorts = evaluation.get("evaluation_cohort_ids")
    evaluation_cohorts = (
        tuple(value for value in raw_evaluation_cohorts if isinstance(value, str))
        if isinstance(raw_evaluation_cohorts, list)
        else ()
    )

    return AdaptationPlan(
        schema_version=payload.get("schema_version", 0),
        status=_enum_or_none(AdaptationPlanStatus, payload.get("status")),
        plan_id=payload.get("plan_id"),
        proposed_evidence_layer_id=payload.get("proposed_evidence_layer_id"),
        base_evidence_layer=payload.get("base_evidence_layer"),
        scientific_question=payload.get("scientific_question"),
        hypothesis=payload.get("hypothesis"),
        task_adapter=TaskAdapter(
            adapter_id=adapter.get("adapter_id"),
            route=adapter.get("route"),
            methods=tuple(methods),
            parameter_update=adapter.get("parameter_update"),
            teacher_model_outputs_used=adapter.get("teacher_model_outputs_used"),
            artifact=_artifact(adapter.get("artifact")),
        ),
        signals=tuple(signals),
        freeze=FreezeState(
            author_group_admitted=freeze.get("author_group_admitted"),
            frozen_before_evaluation=freeze.get("frozen_before_evaluation"),
            selection_rule=freeze.get("selection_rule"),
            stopping_rule=freeze.get("stopping_rule"),
            receipt=_artifact(freeze.get("receipt")),
        ),
        evaluation=EvaluationBoundary(
            development_cohort_id=evaluation.get("development_cohort_id"),
            evaluation_cohort_ids=evaluation_cohorts,
            exact_same_case_required=evaluation.get("exact_same_case_required"),
            patient_grouping_required_when_confirmed=evaluation.get(
                "patient_grouping_required_when_confirmed"
            ),
            primary_contrast=evaluation.get("primary_contrast"),
            attribution_scope=evaluation.get("attribution_scope"),
            medgemma_comparison_scope=evaluation.get("medgemma_comparison_scope"),
        ),
        claim_boundary=payload.get("claim_boundary"),
        privacy_boundary=payload.get("privacy_boundary"),
    )


def _required_text(issues: list[ValidationIssue], field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, field, "non-empty text is required"))


def _required_bool(issues: list[ValidationIssue], field: str, value: Any) -> None:
    if not isinstance(value, bool):
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, field, "boolean value is required"))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _resolve_artifact(
    artifact: ArtifactIdentity, contract_path: Path, bundle_root: Path | None
) -> Path | None:
    if not isinstance(artifact.path, str) or not artifact.path.strip():
        return None
    path = Path(artifact.path).expanduser()
    if not path.is_absolute():
        path = (bundle_root if bundle_root else contract_path.parent) / path
    return path.resolve()


def _validate_artifact(
    issues: list[ValidationIssue],
    field: str,
    artifact: ArtifactIdentity,
    *,
    required: bool,
    contract_path: Path,
    bundle_root: Path | None,
    check_files: bool,
) -> dict[str, Any]:
    result = {"declared": False, "checksum_valid": False, "file_checked": False, "matches": None}
    if artifact.path is None and artifact.sha256 is None and not required:
        return result
    _required_text(issues, f"{field}.path", artifact.path)
    _required_text(issues, f"{field}.sha256", artifact.sha256)
    result["declared"] = bool(artifact.path and artifact.sha256)
    if artifact.sha256 is not None and not _is_sha256(artifact.sha256):
        issues.append(
            ValidationIssue(IssueSeverity.BLOCKER, f"{field}.sha256", "must be a SHA-256 digest")
        )
        return result
    result["checksum_valid"] = _is_sha256(artifact.sha256)
    if not check_files or not result["declared"]:
        return result
    path = _resolve_artifact(artifact, contract_path, bundle_root)
    result["file_checked"] = True
    if path is None or not path.exists():
        issues.append(ValidationIssue(IssueSeverity.BLOCKER, field, "file does not exist"))
        result["matches"] = False
    else:
        result["matches"] = sha256_file(path).lower() == str(artifact.sha256).lower()
        if not result["matches"]:
            issues.append(
                ValidationIssue(IssueSeverity.BLOCKER, f"{field}.sha256", "file checksum mismatch")
            )
    return result


def validate_adaptation_plan(
    contract_path: Path,
    *,
    bundle_root: Path | None = None,
    check_files: bool = False,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve(strict=True)
    with contract_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Adaptation plan must be a JSON object")
    plan = parse_adaptation_plan(payload)
    issues: list[ValidationIssue] = []
    _validate_payload_shape(issues, payload)

    if plan.schema_version != 1:
        issues.append(
            ValidationIssue(IssueSeverity.BLOCKER, "schema_version", "schema version 1 is required")
        )
    if plan.status is None:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "status",
                "must be preregistered_unfrozen, frozen_before_evaluation, or retired",
            )
        )
    for field, value in (
        ("plan_id", plan.plan_id),
        ("scientific_question", plan.scientific_question),
        ("hypothesis", plan.hypothesis),
        ("claim_boundary", plan.claim_boundary),
        ("privacy_boundary", plan.privacy_boundary),
    ):
        _required_text(issues, field, value)
    if plan.proposed_evidence_layer_id != "post_submission_mistral_adapted":
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "proposed_evidence_layer_id",
                "the bounded route must remain separately named post_submission_mistral_adapted",
            )
        )
    if plan.base_evidence_layer != "reproduced_mistral":
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "base_evidence_layer",
                "adapter-effect attribution requires reproduced_mistral as the same-model base",
            )
        )

    adapter = plan.task_adapter
    for field, value in (
        ("task_adapter.adapter_id", adapter.adapter_id),
        ("task_adapter.route", adapter.route),
        ("task_adapter.parameter_update", adapter.parameter_update),
    ):
        _required_text(issues, field, value)
    if adapter.route != "schema_guided_inference_time_and_post_hoc_calibration":
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "task_adapter.route",
                "the preregistered route is the non-parametric task layer plus "
                "post-hoc calibration",
            )
        )
    if adapter.parameter_update != "calibration_thresholds_only":
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "task_adapter.parameter_update",
                "Mistral weights must remain frozen; only calibration thresholds may be learned",
            )
        )
    _required_bool(
        issues, "task_adapter.teacher_model_outputs_used", adapter.teacher_model_outputs_used
    )
    if adapter.teacher_model_outputs_used:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "task_adapter.teacher_model_outputs_used",
                "teacher use is outside this route and requires a separately named "
                "distillation plan",
            )
        )
    methods = set(adapter.methods)
    raw_methods = _mapping(payload.get("task_adapter")).get("methods")
    if not isinstance(raw_methods, list):
        issues.append(
            ValidationIssue(IssueSeverity.BLOCKER, "task_adapter.methods", "array is required")
        )
    else:
        if len(raw_methods) != len(set(value for value in raw_methods if isinstance(value, str))):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    "task_adapter.methods",
                    "method IDs must be unique",
                )
            )
        allowed_method_ids = {method.value for method in AdaptationMethod}
        unknown_methods = sorted(
            repr(value)
            for value in raw_methods
            if not isinstance(value, str) or value not in allowed_method_ids
        )
        if unknown_methods:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    "task_adapter.methods",
                    f"unrecognized methods are prohibited: {unknown_methods}",
                )
            )
    if not REQUIRED_METHODS.issubset(methods):
        missing = sorted(method.value for method in REQUIRED_METHODS - methods)
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "task_adapter.methods",
                f"required methods are missing: {', '.join(missing)}",
            )
        )
    disallowed = methods & PARAMETRIC_OR_TEACHER_METHODS
    if disallowed:
        names = sorted(method.value for method in disallowed)
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "task_adapter.methods",
                f"methods require a separately named experiment: {', '.join(names)}",
            )
        )

    signal_ids = [signal.signal_id for signal in plan.signals if signal.signal_id]
    if len(signal_ids) != len(set(signal_ids)):
        issues.append(
            ValidationIssue(IssueSeverity.BLOCKER, "signals", "signal IDs must be unique")
        )
    signals_by_id = {signal.signal_id: signal for signal in plan.signals if signal.signal_id}
    for signal_id, required_use in REQUIRED_SIGNAL_USES.items():
        signal = signals_by_id.get(signal_id)
        if signal is None:
            issues.append(
                ValidationIssue(IssueSeverity.BLOCKER, f"signals[{signal_id}]", "is required")
            )
            continue
        if signal.use != required_use:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"signals[{signal_id}].use",
                    f"must be {required_use.value}",
                )
            )
    for index, signal in enumerate(plan.signals):
        prefix = f"signals[{index}]"
        _required_text(issues, f"{prefix}.signal_id", signal.signal_id)
        _required_text(issues, f"{prefix}.signal_type", signal.signal_type)
        _required_text(issues, f"{prefix}.note", signal.note)
        _required_bool(
            issues,
            f"{prefix}.used_for_parameter_or_variant_selection",
            signal.used_for_parameter_or_variant_selection,
        )
        _required_bool(issues, f"{prefix}.outcomes_inspected", signal.outcomes_inspected)
        if signal.use is None:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER, f"{prefix}.use", "bounded signal use is required"
                )
            )
        elif (
            signal.use
            in {
                SignalUse.EVALUATION_ONLY,
                SignalUse.CONTEXT_ONLY_PROHIBITED_FOR_SELECTION,
            }
            and signal.used_for_parameter_or_variant_selection
        ):
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"{prefix}.used_for_parameter_or_variant_selection",
                    f"{signal.use.value} signals cannot select the adapter or its thresholds",
                )
            )

    development = signals_by_id.get("zoe_development_first_100_ra")
    if development and development.outcomes_inspected is not True:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "signals[zoe_development_first_100_ra].outcomes_inspected",
                "the historical development labels must be acknowledged as inspected",
            )
        )
    for signal_id in ("zoe_evaluation", "maria_evaluation"):
        signal = signals_by_id.get(signal_id)
        if signal and signal.outcomes_inspected:
            issues.append(
                ValidationIssue(
                    IssueSeverity.BLOCKER,
                    f"signals[{signal_id}].outcomes_inspected",
                    "evaluation outcomes must remain uninspected until the adapter is frozen",
                )
            )

    freeze = plan.freeze
    for field, value in (
        ("freeze.selection_rule", freeze.selection_rule),
        ("freeze.stopping_rule", freeze.stopping_rule),
    ):
        _required_text(issues, field, value)
    _required_bool(issues, "freeze.author_group_admitted", freeze.author_group_admitted)
    _required_bool(issues, "freeze.frozen_before_evaluation", freeze.frozen_before_evaluation)
    frozen = plan.status == AdaptationPlanStatus.FROZEN_BEFORE_EVALUATION
    if frozen and not freeze.author_group_admitted:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "freeze.author_group_admitted",
                "final evaluation requires author-group admission of the frozen work package",
            )
        )
    if frozen and not freeze.frozen_before_evaluation:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "freeze.frozen_before_evaluation",
                "must be true for a frozen-before-evaluation plan",
            )
        )
    if not frozen and freeze.frozen_before_evaluation:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "freeze.frozen_before_evaluation",
                "cannot be true while the plan status remains unfrozen or retired",
            )
        )

    evaluation = plan.evaluation
    if evaluation.development_cohort_id != "zoe_development_first_100_ra":
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "evaluation.development_cohort_id",
                "must preserve the historical first-100 Zoe development boundary",
            )
        )
    if set(evaluation.evaluation_cohort_ids) != {"zoe_evaluation_1395", "maria_evaluation_499"}:
        issues.append(
            ValidationIssue(
                IssueSeverity.BLOCKER,
                "evaluation.evaluation_cohort_ids",
                "must name the Zoe 1,395 and Maria 499 evaluation surfaces; "
                "manifests fix membership",
            )
        )
    for field, value in (
        ("evaluation.exact_same_case_required", evaluation.exact_same_case_required),
        (
            "evaluation.patient_grouping_required_when_confirmed",
            evaluation.patient_grouping_required_when_confirmed,
        ),
    ):
        _required_bool(issues, field, value)
        if value is False:
            issues.append(ValidationIssue(IssueSeverity.BLOCKER, field, "must be true"))
    for field, value in (
        ("evaluation.primary_contrast", evaluation.primary_contrast),
        ("evaluation.attribution_scope", evaluation.attribution_scope),
        ("evaluation.medgemma_comparison_scope", evaluation.medgemma_comparison_scope),
    ):
        _required_text(issues, field, value)

    artifacts = {
        "task_adapter.artifact": _validate_artifact(
            issues,
            "task_adapter.artifact",
            adapter.artifact,
            required=frozen,
            contract_path=contract_path,
            bundle_root=bundle_root,
            check_files=check_files,
        ),
        "freeze.receipt": _validate_artifact(
            issues,
            "freeze.receipt",
            freeze.receipt,
            required=frozen,
            contract_path=contract_path,
            bundle_root=bundle_root,
            check_files=check_files,
        ),
    }
    for signal in plan.signals:
        if signal.artifact.path is not None or signal.artifact.sha256 is not None:
            artifacts[f"signals[{signal.signal_id}].artifact"] = _validate_artifact(
                issues,
                f"signals[{signal.signal_id}].artifact",
                signal.artifact,
                required=False,
                contract_path=contract_path,
                bundle_root=bundle_root,
                check_files=check_files,
            )

    blockers = [issue for issue in issues if issue.severity == IssueSeverity.BLOCKER]
    design_valid = not blockers
    ready_for_implementation = bool(
        design_valid and plan.status == AdaptationPlanStatus.PREREGISTERED_UNFROZEN
    )
    ready_for_evaluation = bool(design_valid and frozen and check_files)
    return {
        "schema_version": 1,
        "contract_schema_version": plan.schema_version,
        "contract_sha256": sha256_file(contract_path),
        "plan_id": plan.plan_id,
        "status": plan.status.value if plan.status else None,
        "proposed_evidence_layer_id": plan.proposed_evidence_layer_id,
        "base_evidence_layer": plan.base_evidence_layer,
        "design_valid": design_valid,
        "ready_for_implementation": ready_for_implementation,
        "ready_for_evaluation": ready_for_evaluation,
        "analysis_started": False,
        "signal_use_counts": {
            use.value: sum(signal.use == use for signal in plan.signals) for use in SignalUse
        },
        "method_ids": [method.value for method in plan.task_adapter.methods],
        "artifact_validation": artifacts,
        "issues": [{**asdict(issue), "severity": issue.severity.value} for issue in issues],
        "claim_boundary": (
            "This receipt validates a proposed or frozen adaptation design only. It contains no "
            "performance estimate and does not admit a fourth evidence layer into final analysis."
        ),
        "privacy_boundary": (
            "No report text, report key, patient key, case-level prediction, or model weight "
            "is emitted."
        ),
    }


def validate_adaptation_plan_to_directory(
    contract_path: Path,
    output_dir: Path,
    *,
    bundle_root: Path | None = None,
    check_files: bool = False,
) -> dict[str, Any]:
    result = validate_adaptation_plan(
        contract_path,
        bundle_root=bundle_root,
        check_files=check_files,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "adaptation_plan_validation.json", result)
    atomic_write_json(
        output_dir / "run_manifest.json",
        build_manifest(
            "adaptation-plan-validate",
            [contract_path],
            {"check_files": check_files, "bundle_root_supplied": bundle_root is not None},
        ),
    )
    return result
