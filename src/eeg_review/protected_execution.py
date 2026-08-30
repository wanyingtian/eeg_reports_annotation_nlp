"""Fail-closed authorization gate for protected comparator execution."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STUDY_ID = "jbhi-02463-post-submission-medgemma-native-interface-sensitivity-v1"
CONFIGURATION_ID = (
    "jbhi-02463/comparator/medgemma-27b-text-it/configuration/independent-native-interface-q2-v1"
)
PROTOCOL_ID = "H18-02728"
COHORTS = {"zoe_evaluation_1395": 1395, "maria_evaluation_499": 499}
AUTHORITY_ROLES = {"principal_investigator", "authorized_data_custodian", "approved_study_record"}


@dataclass(frozen=True)
class AuthorizationValidation:
    valid: bool
    blockers: tuple[str, ...]
    receipt_sha256: str


class ProtectedExecutionLocked(RuntimeError):
    """Raised before any governed execution path is opened."""

    def __init__(self, blockers: tuple[str, ...] | list[str]):
        self.blockers = tuple(blockers)
        super().__init__("protected execution locked: " + "; ".join(self.blockers))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_authorization_receipt(path: Path) -> AuthorizationValidation:
    """Validate documentary authorization without interpreting its legal sufficiency."""

    blockers: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return AuthorizationValidation(False, (f"unreadable receipt: {error}",), "")

    if payload.get("schema_version") != 1:
        blockers.append("schema_version must be 1")
    if payload.get("status") != "confirmed":
        blockers.append("status must be confirmed")
    if payload.get("study_id") != STUDY_ID:
        blockers.append("study_id mismatch")
    if payload.get("configuration_id") != CONFIGURATION_ID:
        blockers.append("configuration_id mismatch")
    if payload.get("protocol_identifier") != PROTOCOL_ID:
        blockers.append("protocol_identifier mismatch")
    if not _nonempty(payload.get("authorization_id")):
        blockers.append("authorization_id is required")
    if not _nonempty(payload.get("coverage_statement")):
        blockers.append("coverage_statement is required")

    authority = payload.get("authority")
    if not isinstance(authority, dict):
        blockers.append("authority object is required")
    else:
        if authority.get("role") not in AUTHORITY_ROLES:
            blockers.append("authority.role is not an accepted confirming role")
        if not _nonempty(authority.get("name_or_record")):
            blockers.append("authority.name_or_record is required")
        if not _nonempty(authority.get("confirmation_source")):
            blockers.append("authority.confirmation_source is required")
        source_hash = authority.get("confirmation_source_sha256")
        if not _sha256(source_hash):
            blockers.append("authority.confirmation_source_sha256 must be a SHA-256")
        try:
            confirmed_at = datetime.fromisoformat(
                str(authority.get("confirmed_at_utc", "")).replace("Z", "+00:00")
            )
            if confirmed_at.tzinfo is None:
                raise ValueError
        except ValueError:
            blockers.append(
                "authority.confirmed_at_utc must be a timezone-aware ISO-8601 timestamp"
            )

    scope = payload.get("scope")
    if not isinstance(scope, dict):
        blockers.append("scope object is required")
    else:
        if scope.get("operation") != "post_submission_model_inference_and_aggregate_analysis":
            blockers.append("scope.operation mismatch")
        if scope.get("cohorts") != COHORTS:
            blockers.append("scope.cohorts must exactly match the frozen protected cohorts")
        if scope.get("already_transferred_deidentified_reports") is not True:
            blockers.append("scope must explicitly cover already transferred de-identified reports")
        if scope.get("secondary_use_covered") is not True:
            blockers.append("scope must explicitly confirm secondary-use coverage")

    controls = payload.get("controls")
    expected_controls = {
        "keyed_outputs_remain_governed": True,
        "aggregate_release_requires_author_review": True,
        "weights_not_redistributed": True,
        "patient_grouped_claims_not_authorized_without_patient_key": True,
    }
    if not isinstance(controls, dict):
        blockers.append("controls object is required")
    else:
        for key, expected in expected_controls.items():
            if controls.get(key) is not expected:
                blockers.append(f"controls.{key} must be true")

    return AuthorizationValidation(not blockers, tuple(blockers), sha256_file(path))


def build_unlock_receipt(path: Path) -> dict[str, Any]:
    validation = validate_authorization_receipt(path)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "study_id": STUDY_ID,
        "configuration_id": CONFIGURATION_ID,
        "protected_evaluation_unlocked": validation.valid,
        "authorization_receipt_sha256": validation.receipt_sha256 or None,
        "blockers": list(validation.blockers),
        "cohorts": COHORTS,
        "boundaries": [
            "This validates a documentary execution gate; it is not an ethics or "
            "legal determination.",
            "Manuscript admission, interpretation, and aggregate release remain "
            "author-group decisions.",
            "Patient-grouped inference remains unavailable without a validated patient key.",
        ],
    }


def authorize_plan_before_governed_access(
    plan: dict[str, Any], authorization_path: Path | None
) -> AuthorizationValidation | None:
    """Validate a public plan's gate before callers resolve governed paths.

    The caller may read the public execution plan and the documentary authorization
    receipt before invoking this function. It must not resolve, inspect, or create a
    governed run directory until this function returns successfully.
    """

    gate = plan.get("authorization_gate")
    if not isinstance(gate, dict) or gate.get("required") is not True:
        return None
    if authorization_path is None:
        raise ProtectedExecutionLocked(["--authorization is required by the execution plan"])
    validation = validate_authorization_receipt(authorization_path)
    if not validation.valid:
        raise ProtectedExecutionLocked(validation.blockers)
    if gate.get("study_id") != STUDY_ID:
        raise ProtectedExecutionLocked(["execution-plan authorization study_id mismatch"])
    if gate.get("configuration_id") != CONFIGURATION_ID:
        raise ProtectedExecutionLocked(
            ["execution-plan authorization configuration_id mismatch"]
        )
    if gate.get("cohorts") != COHORTS:
        raise ProtectedExecutionLocked(
            ["execution-plan authorization cohorts differ from the frozen scope"]
        )
    if gate.get("protocol_identifier") != PROTOCOL_ID:
        raise ProtectedExecutionLocked(
            ["execution-plan authorization protocol_identifier mismatch"]
        )
    return validation


def validate_frozen_parent_receipts(plan: dict[str, Any], repository_root: Path) -> None:
    """Verify immutable public parents before a governed bundle is prepared."""

    parents = plan.get("frozen_parent_receipts")
    if not isinstance(parents, dict) or not parents:
        raise ProtectedExecutionLocked(["frozen_parent_receipts are required"])
    blockers: list[str] = []
    for parent_id, receipt in parents.items():
        if not isinstance(receipt, dict):
            blockers.append(f"frozen parent {parent_id} is not an object")
            continue
        relative = receipt.get("path")
        expected = receipt.get("sha256")
        if not _nonempty(relative) or not _sha256(expected):
            blockers.append(f"frozen parent {parent_id} lacks path or SHA-256")
            continue
        path = repository_root / str(relative)
        if not path.is_file():
            blockers.append(f"frozen parent {parent_id} is missing")
        elif sha256_file(path) != expected:
            blockers.append(f"frozen parent {parent_id} hash mismatch")
    if blockers:
        raise ProtectedExecutionLocked(blockers)


def validate_protected_job_binding(
    plan: dict[str, Any], job: dict[str, Any], authorization: AuthorizationValidation | None
) -> None:
    """Prevent a prepared protected job from drifting from its validated gate."""

    gate = plan.get("authorization_gate")
    if not isinstance(gate, dict) or gate.get("required") is not True:
        return
    if authorization is None or not authorization.valid:
        raise ProtectedExecutionLocked(["validated authorization is missing"])
    binding = job.get("protected_authorization")
    if not isinstance(binding, dict):
        raise ProtectedExecutionLocked(["prepared job lacks protected_authorization binding"])
    if binding.get("receipt_sha256") != authorization.receipt_sha256:
        raise ProtectedExecutionLocked(["prepared job authorization receipt hash mismatch"])
    if binding.get("study_id") != STUDY_ID:
        raise ProtectedExecutionLocked(["prepared job authorization study_id mismatch"])
    if binding.get("configuration_id") != CONFIGURATION_ID:
        raise ProtectedExecutionLocked(
            ["prepared job authorization configuration_id mismatch"]
        )
    cohorts = {item.get("cohort_id"): item.get("records") for item in job.get("cohorts", [])}
    if cohorts != COHORTS:
        raise ProtectedExecutionLocked(
            ["prepared job cohorts differ from the exact authorized populations"]
        )
    if job.get("frozen_parent_receipts") != plan.get("frozen_parent_receipts"):
        raise ProtectedExecutionLocked(["prepared job frozen parent receipts mismatch"])
    expected_stages = {f"{cohort_id}_inference" for cohort_id in COHORTS}
    inference = {
        item.get("stage"): item.get("command")
        for item in job.get("commands", [])
        if str(item.get("stage", "")).endswith("_inference")
    }
    if set(inference) != expected_stages:
        raise ProtectedExecutionLocked(["prepared job inference stages mismatch"])
    for stage, command in inference.items():
        if not isinstance(command, list):
            raise ProtectedExecutionLocked([f"{stage} command is not a list"])
        try:
            interface = command[command.index("--classification-interface") + 1]
        except (ValueError, IndexError):
            interface = None
        if interface != "native_chat":
            raise ProtectedExecutionLocked([f"{stage} is not bound to native_chat"])
        if "--classification-only" not in command:
            raise ProtectedExecutionLocked([f"{stage} is not classification-only"])
        if "--capture-classification-logprobs" in command:
            raise ProtectedExecutionLocked(
                [f"{stage} unexpectedly enables development probability capture"]
            )
