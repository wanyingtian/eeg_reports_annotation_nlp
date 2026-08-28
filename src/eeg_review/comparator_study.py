from __future__ import annotations

import ast
import hashlib
import json
import platform
import shutil
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import atomic_write_json
from .manifest import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SOURCE = REPOSITORY_ROOT / "src/LLM_pipeline/pipeline.py"
MODEL_REGISTRY_SOURCE = REPOSITORY_ROOT / "src/LLM_pipeline/llm_models.py"


@dataclass(frozen=True)
class StudyIssue:
    severity: str
    field: str
    message: str


@dataclass(frozen=True)
class CohortSpec:
    cohort_id: str
    role: str
    input_path: str
    input_sha256: str
    candidate_records: int
    complete_records: int
    excluded_incomplete_records: int
    execute_records: int


@dataclass(frozen=True)
class ModelSpec:
    registry_name: str
    distribution_repo_id: str
    distribution_revision: str
    filename: str
    sha256: str
    size_bytes: int
    quantization: str


@dataclass(frozen=True)
class InterfaceSpec:
    mode: str
    prompt_sha256: str
    grammar_path: str
    grammar_sha256: str
    chat_template_applied: bool
    classification_only: bool
    temperature: float
    top_k: int
    top_p: float
    max_tokens: int


@dataclass(frozen=True)
class StudySpec:
    schema_version: int
    status: str
    study_id: str
    evidence_layer: str
    configuration_id: str
    model: ModelSpec
    interface: InterfaceSpec
    cohorts: tuple[CohortSpec, ...]
    external_configurations: tuple[dict[str, Any], ...]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _extract_string_assignment(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            break
        return value
    raise ValueError(f"Could not find a literal string assignment for {name} in {path}")


def _extract_model_registry(path: Path) -> dict[str, dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "MODEL_CONFIGS"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, dict):
            break
        return value
    raise ValueError(f"Could not read MODEL_CONFIGS from {path}")


def _parse_spec(payload: dict[str, Any]) -> StudySpec:
    configuration = payload["independent_configuration"]
    model = configuration["model"]
    interface = configuration["interface"]
    cohorts = tuple(
        CohortSpec(
            cohort_id=value["cohort_id"],
            role=value["role"],
            input_path=value["input"]["path"],
            input_sha256=value["input"]["sha256"],
            candidate_records=value["population"]["candidate_records"],
            complete_records=value["population"]["complete_records"],
            excluded_incomplete_records=value["population"]["excluded_incomplete_records"],
            execute_records=value["population"]["execute_records"],
        )
        for value in payload["cohorts"]
    )
    return StudySpec(
        schema_version=payload["schema_version"],
        status=payload["status"],
        study_id=payload["study_id"],
        evidence_layer=payload["evidence_layer"],
        configuration_id=configuration["configuration_id"],
        model=ModelSpec(
            registry_name=model["registry_name"],
            distribution_repo_id=model["distribution_repo_id"],
            distribution_revision=model["distribution_revision"],
            filename=model["filename"],
            sha256=model["sha256"],
            size_bytes=model["size_bytes"],
            quantization=model["quantization"],
        ),
        interface=InterfaceSpec(
            mode=interface["mode"],
            prompt_sha256=interface["prompt_sha256"],
            grammar_path=interface["grammar"]["path"],
            grammar_sha256=interface["grammar"]["sha256"],
            chat_template_applied=interface["chat_template_applied"],
            classification_only=interface["classification_only"],
            temperature=interface["sampling"]["temperature"],
            top_k=interface["sampling"]["top_k"],
            top_p=interface["sampling"]["top_p"],
            max_tokens=interface["sampling"]["max_tokens"],
        ),
        cohorts=cohorts,
        external_configurations=tuple(payload["external_configurations"]),
    )


def _git_receipt() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "worktree_clean": status.returncode == 0 and not status.stdout.strip(),
    }


def _database_receipt(path: Path) -> dict[str, Any]:
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(reports)")]
        rows = int(connection.execute('SELECT COUNT(*) FROM "reports"').fetchone()[0])
        duplicate_keys = int(
            connection.execute(
                'SELECT COUNT(*) FROM ('
                'SELECT "Hashed_ReportURN" FROM "reports" '
                'GROUP BY "Hashed_ReportURN" HAVING COUNT(*) > 1)'
            ).fetchone()[0]
        )
        labels = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]
        complete_predicate = " AND ".join(
            f'CAST("{label}" AS INTEGER) IN (1, 2, 3, 4)' for label in labels
        )
        complete_rows = int(
            connection.execute(
                f'SELECT COUNT(*) FROM "reports" WHERE {complete_predicate}'
            ).fetchone()[0]
        )
        cluster_summary: dict[str, int] | None = None
        if "Cluster code" in columns:
            nonempty = (
                '"Cluster code" IS NOT NULL '
                'AND TRIM(CAST("Cluster code" AS TEXT)) <> \'\''
            )
            cluster_summary = {
                "nonempty_records": int(
                    connection.execute(
                        f'SELECT COUNT(*) FROM "reports" WHERE {nonempty}'
                    ).fetchone()[0]
                ),
                "distinct_values": int(
                    connection.execute(
                        f'SELECT COUNT(DISTINCT CAST("Cluster code" AS TEXT)) '
                        f'FROM "reports" WHERE {nonempty}'
                    ).fetchone()[0]
                ),
            }
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "rows": rows,
        "complete_reference_rows": complete_rows,
        "columns": columns,
        "duplicate_report_keys": duplicate_keys,
        "cluster_code_cardinality_diagnostic": cluster_summary,
    }


def _model_receipt_matches(
    issues: list[StudyIssue],
    receipt: dict[str, Any],
    model: ModelSpec,
    prefix: str,
) -> None:
    source = receipt.get("model", receipt)
    expected = {
        "registry_name": model.registry_name,
        "repo_id": model.distribution_repo_id,
        "filename": model.filename,
        "requested_revision": model.distribution_revision,
        "sha256": model.sha256,
        "size_bytes": model.size_bytes,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            issues.append(
                StudyIssue("blocker", f"{prefix}.{key}", f"expected {value!r}")
            )


def validate_comparator_study(
    plan_path: Path,
    *,
    source_run: Path | None = None,
    receipt_dir: Path | None = None,
    check_local: bool = False,
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve(strict=True)
    payload = _load_json(plan_path)
    spec = _parse_spec(payload)
    issues: list[StudyIssue] = []

    if spec.schema_version != 1:
        issues.append(StudyIssue("blocker", "schema_version", "must be 1"))
    if spec.status != "preregistered_pre_inference":
        issues.append(
            StudyIssue("blocker", "status", "must remain preregistered_pre_inference before a run")
        )
    if spec.evidence_layer != "post_submission_medgemma":
        issues.append(
            StudyIssue("blocker", "evidence_layer", "must remain a separate MedGemma layer")
        )
    if not spec.configuration_id.startswith(
        "jbhi-02463/comparator/medgemma-27b-text-it/configuration/"
    ):
        issues.append(
            StudyIssue("blocker", "configuration_id", "must use the comparator namespace")
        )

    config_ids = [spec.configuration_id]
    for index, external in enumerate(spec.external_configurations):
        config_id = external.get("configuration_id")
        if not isinstance(config_id, str) or not config_id:
            issues.append(
                StudyIssue("blocker", f"external_configurations[{index}]", "missing identifier")
            )
            continue
        config_ids.append(config_id)
        if external.get("blocks_independent_execution") is not False:
            issues.append(
                StudyIssue(
                    "blocker",
                    f"external_configurations[{index}].blocks_independent_execution",
                    "an external example configuration must not silently block the independent run",
                )
            )
        if external.get("status") != "pending_exact_producing_bundle":
            issues.append(
                StudyIssue(
                    "blocker",
                    f"external_configurations[{index}].status",
                    "must remain explicitly pending until its exact bundle is received",
                )
            )
    if len(config_ids) != len(set(config_ids)):
        issues.append(StudyIssue("blocker", "configurations", "configuration IDs must be unique"))

    if spec.interface.mode != "matched_historical_raw_completion":
        issues.append(
            StudyIssue("blocker", "interface.mode", "the primary configuration must be matched")
        )
    if spec.interface.chat_template_applied:
        issues.append(
            StudyIssue(
                "blocker",
                "interface.chat_template_applied",
                "the matched historical interface does not apply an embedded chat template",
            )
        )
    if not spec.interface.classification_only:
        issues.append(
            StudyIssue(
                "blocker",
                "interface.classification_only",
                "this comparator row is prespecified as classification-only",
            )
        )
    if spec.interface.temperature != 0:
        issues.append(StudyIssue("blocker", "interface.sampling.temperature", "must be zero"))

    prompt_hash = _sha256_text(_extract_string_assignment(PIPELINE_SOURCE, "PROMPT_CLASSIFY"))
    if prompt_hash != spec.interface.prompt_sha256:
        issues.append(StudyIssue("blocker", "interface.prompt_sha256", "pipeline prompt drifted"))
    grammar_path = REPOSITORY_ROOT / spec.interface.grammar_path
    if not grammar_path.exists() or sha256_file(grammar_path) != spec.interface.grammar_sha256:
        issues.append(StudyIssue("blocker", "interface.grammar", "grammar is absent or changed"))

    registry = _extract_model_registry(MODEL_REGISTRY_SOURCE)
    registered = registry.get(spec.model.registry_name)
    if not isinstance(registered, dict):
        issues.append(StudyIssue("blocker", "model.registry_name", "model is not registered"))
    else:
        expected_registry = {
            "repo_id": spec.model.distribution_repo_id,
            "filename": spec.model.filename,
            "revision": spec.model.distribution_revision,
            "sha256": spec.model.sha256,
        }
        for key, value in expected_registry.items():
            if registered.get(key) != value:
                issues.append(
                    StudyIssue("blocker", f"model.registry.{key}", "plan and registry differ")
                )

    cohort_ids = [cohort.cohort_id for cohort in spec.cohorts]
    if len(cohort_ids) != len(set(cohort_ids)):
        issues.append(StudyIssue("blocker", "cohorts", "cohort IDs must be unique"))
    roles = {cohort.role for cohort in spec.cohorts}
    if roles != {"development_transport_check", "evaluation"}:
        issues.append(
            StudyIssue(
                "blocker",
                "cohorts.role",
                "one transport check and evaluation cohorts are required",
            )
        )
    for index, cohort in enumerate(spec.cohorts):
        prefix = f"cohorts[{index}].population"
        if cohort.complete_records + cohort.excluded_incomplete_records != cohort.candidate_records:
            issues.append(
                StudyIssue(
                    "blocker", prefix, "complete plus excluded must equal candidates"
                )
            )
        if cohort.execute_records != cohort.complete_records:
            issues.append(
                StudyIssue("blocker", prefix, "execution must use the frozen complete-case key set")
            )

    local_receipts: dict[str, Any] = {}
    database_receipts: dict[str, Any] = {}
    smoke_seconds: float | None = None
    if check_local:
        if source_run is None or receipt_dir is None:
            issues.append(
                StudyIssue("blocker", "local_check", "source_run and receipt_dir are required")
            )
        else:
            source_run = source_run.expanduser().resolve(strict=True)
            receipt_dir = receipt_dir.expanduser().resolve(strict=True)
            source_state = _load_json(source_run / "state.json")
            if source_state.get("status") != "completed":
                issues.append(
                    StudyIssue(
                        "blocker", "source_run.state", "source run is not complete"
                    )
                )
            for index, cohort in enumerate(spec.cohorts):
                path = source_run / cohort.input_path
                if not path.exists():
                    issues.append(
                        StudyIssue("blocker", f"cohorts[{index}].input", "governed input is absent")
                    )
                    continue
                receipt = _database_receipt(path)
                database_receipts[cohort.cohort_id] = receipt
                if receipt["sha256"] != cohort.input_sha256:
                    issues.append(
                        StudyIssue("blocker", f"cohorts[{index}].input", "input checksum differs")
                    )
                if receipt["rows"] != cohort.candidate_records:
                    issues.append(
                        StudyIssue("blocker", f"cohorts[{index}].input", "input row count differs")
                    )
                if receipt["complete_reference_rows"] != cohort.complete_records:
                    issues.append(
                        StudyIssue(
                            "blocker",
                            f"cohorts[{index}].population.complete_records",
                            "complete-case count differs from the frozen plan",
                        )
                    )
                if receipt["duplicate_report_keys"]:
                    issues.append(
                        StudyIssue("blocker", f"cohorts[{index}].input", "duplicate report keys")
                    )

            receipt_files = payload["local_readiness"]["receipt_files"]
            for receipt_id, relative in receipt_files.items():
                path = receipt_dir / relative
                if not path.exists():
                    issues.append(StudyIssue("blocker", f"receipts.{receipt_id}", "receipt absent"))
                    continue
                value = _load_json(path)
                local_receipts[receipt_id] = {
                    "path": path.name,
                    "sha256": sha256_file(path),
                }
                _model_receipt_matches(issues, value, spec.model, f"receipts.{receipt_id}.model")
                if receipt_id == "classification_smoke":
                    method = value.get("method", {})
                    expected_method = {
                        "prompt_sha256": spec.interface.prompt_sha256,
                        "grammar_sha256": spec.interface.grammar_sha256,
                        "temperature": spec.interface.temperature,
                        "top_k": spec.interface.top_k,
                        "top_p": spec.interface.top_p,
                        "max_tokens": spec.interface.max_tokens,
                        "chat_template_applied": spec.interface.chat_template_applied,
                    }
                    for key, expected in expected_method.items():
                        if method.get(key) != expected:
                            issues.append(
                                StudyIssue(
                                    "blocker",
                                    f"receipts.classification_smoke.method.{key}",
                                    "smoke and frozen interface differ",
                                )
                            )
                    smoke_seconds = value.get("timing_and_tokens", {}).get("elapsed_seconds")
                if receipt_id == "runtime_smoke":
                    runtime = value.get("runtime", {})
                    expected_runtime = payload["independent_configuration"]["runtime"]
                    if runtime.get("llama_cpp_python") != expected_runtime["engine_version"]:
                        issues.append(
                            StudyIssue(
                                "blocker",
                                "receipts.runtime_smoke",
                                "runtime version differs",
                            )
                        )
                    for key in ("n_ctx", "n_gpu_layers"):
                        if runtime.get("load_parameters", {}).get(key) != expected_runtime[key]:
                            issues.append(
                                StudyIssue(
                                    "blocker",
                                    f"receipts.runtime_smoke.{key}",
                                    "load setting differs",
                                )
                            )

            cache_path = (
                Path.home()
                / ".cache/huggingface/hub"
                / f"models--{spec.model.distribution_repo_id.replace('/', '--')}"
                / "blobs"
                / spec.model.sha256
            )
            if not cache_path.exists() or cache_path.stat().st_size != spec.model.size_bytes:
                issues.append(
                    StudyIssue(
                        "blocker", "model.cache", "validated model blob is absent"
                    )
                )
            else:
                local_receipts["model_cache"] = {
                    "present": True,
                    "size_bytes": cache_path.stat().st_size,
                    "sha256_from_validated_preload_receipt": spec.model.sha256,
                }

    execute_records = sum(cohort.execute_records for cohort in spec.cohorts)
    evaluation_records = sum(
        cohort.execute_records for cohort in spec.cohorts if cohort.role == "evaluation"
    )
    estimate = {
        "basis": "single classification-only compatibility probe; wall time will vary by report",
        "seconds_per_report": smoke_seconds,
        "planned_records_including_transport_check": execute_records,
        "evaluation_records": evaluation_records,
        "estimated_hours_including_transport_check": (
            smoke_seconds * execute_records / 3600 if smoke_seconds is not None else None
        ),
        "estimated_evaluation_hours": (
            smoke_seconds * evaluation_records / 3600 if smoke_seconds is not None else None
        ),
    }

    git = _git_receipt()
    if check_local and not git["worktree_clean"]:
        issues.append(
            StudyIssue(
                "blocker",
                "repository.worktree",
                "freeze and commit the implementation before starting inference",
            )
        )
    issues.extend(
        [
            StudyIssue(
                "decision_gate",
                "patient_grouping",
                "no confirmed patient key is available; report-level analysis may proceed "
                "with an explicit limitation, while patient-grouped inference remains gated",
            ),
            StudyIssue(
                "decision_gate",
                "manuscript_admission",
                "validated results and author-group agreement are required before manuscript use",
            ),
        ]
    )
    blockers = [issue for issue in issues if issue.severity == "blocker"]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": sha256_file(plan_path),
        "study_id": spec.study_id,
        "evidence_layer": spec.evidence_layer,
        "independent_configuration_id": spec.configuration_id,
        "external_configurations": list(spec.external_configurations),
        "ready_to_start_governed_inference": check_local and not blockers,
        "independent_execution_blocked_by_external_configuration": False,
        "ready_for_patient_grouped_primary_analysis": False,
        "ready_for_manuscript_claim": False,
        "local_checks_performed": check_local,
        "repository": git,
        "platform": {
            "system": platform.platform(),
            "machine": platform.machine(),
            "disk_free_bytes": shutil.disk_usage(REPOSITORY_ROOT).free,
        },
        "model_and_runtime_receipts": local_receipts,
        "governed_input_receipts": database_receipts,
        "resource_estimate": estimate,
        "population_arithmetic": {
            cohort.cohort_id: {
                "candidate": cohort.candidate_records,
                "complete": cohort.complete_records,
                "excluded_incomplete": cohort.excluded_incomplete_records,
                "execute": cohort.execute_records,
            }
            for cohort in spec.cohorts
        },
        "issues": [asdict(issue) for issue in issues],
        "privacy_boundary": (
            "This receipt contains aggregate counts, paths relative to governed storage, "
            "and hashes only; no report text or report/patient keys are emitted."
        ),
    }


def validate_comparator_study_to_directory(
    plan_path: Path,
    output_dir: Path,
    *,
    source_run: Path | None = None,
    receipt_dir: Path | None = None,
    check_local: bool = False,
) -> dict[str, Any]:
    result = validate_comparator_study(
        plan_path,
        source_run=source_run,
        receipt_dir=receipt_dir,
        check_local=check_local,
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "medgemma_study_readiness.json", result)
    return result
