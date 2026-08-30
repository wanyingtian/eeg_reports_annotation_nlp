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
