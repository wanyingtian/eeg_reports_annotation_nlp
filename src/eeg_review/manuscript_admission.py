"""Typed admission gate for aggregate MedGemma native-interface evidence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

EVIDENCE_ID = "JBHI-02463-2026-MEDGEMMA-NATIVE-PROTECTED-RESULT-CANDIDATE"
CONFIGURATION_ID = (
    "jbhi-02463/comparator/medgemma-27b-text-it/configuration/"
    "independent-native-interface-q2-v1"
)
ACCEPTED_ROLES = {"corresponding_author", "author_group_record"}
DESTINATIONS = {"supplement", "reviewer_response"}


@dataclass(frozen=True)
class AdmissionValidation:
    valid: bool
    blockers: tuple[str, ...]
    approved_claim_ids: tuple[str, ...]
    approved_destinations: tuple[str, ...]


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_manuscript_admission(
    path: Path, *, candidate_sha256: str, required_claim_ids: set[str]
) -> AdmissionValidation:
    blockers: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return AdmissionValidation(False, (f"unreadable admission receipt: {error}",), (), ())

    if payload.get("schema_version") != 1:
        blockers.append("schema_version must be 1")
    if payload.get("status") != "approved":
        blockers.append("status must be approved")
    if payload.get("manuscript_id") != "JBHI-02463-2026":
        blockers.append("manuscript_id mismatch")
    if payload.get("evidence_id") != EVIDENCE_ID:
        blockers.append("evidence_id mismatch")
    if payload.get("configuration_id") != CONFIGURATION_ID:
        blockers.append("configuration_id mismatch")
    if payload.get("candidate_receipt_sha256") != candidate_sha256:
        blockers.append("candidate receipt hash mismatch")

    confirmation = payload.get("confirmation")
    if not isinstance(confirmation, dict):
        blockers.append("confirmation object is required")
    else:
        if confirmation.get("role") not in ACCEPTED_ROLES:
            blockers.append("confirmation.role is not an accepted admission role")
        for key in ["name_or_record", "source"]:
            if not _nonempty(confirmation.get(key)):
                blockers.append(f"confirmation.{key} is required")
        if not _sha256(confirmation.get("source_sha256")):
            blockers.append("confirmation.source_sha256 must be a SHA-256")
        try:
            confirmed_at = datetime.fromisoformat(
                str(confirmation.get("confirmed_at_utc", "")).replace("Z", "+00:00")
            )
            if confirmed_at.tzinfo is None:
                raise ValueError
        except ValueError:
            blockers.append(
                "confirmation.confirmed_at_utc must be a timezone-aware ISO-8601 timestamp"
            )

    decisions = payload.get("decisions")
    approved_claim_ids: tuple[str, ...] = ()
    approved_destinations: tuple[str, ...] = ()
    if not isinstance(decisions, dict):
        blockers.append("decisions object is required")
    else:
        if decisions.get("aggregate_release_approved") is not True:
            blockers.append("aggregate_release_approved must be true")
        destinations = decisions.get("approved_destinations")
        if not isinstance(destinations, list) or set(destinations) != DESTINATIONS:
            blockers.append("approved_destinations must be supplement and reviewer_response")
        else:
            approved_destinations = tuple(sorted(destinations))
        claim_ids = decisions.get("approved_claim_ids")
        if not isinstance(claim_ids, list) or len(claim_ids) != len(set(claim_ids)):
            blockers.append("approved_claim_ids must be a duplicate-free list")
        elif set(claim_ids) != required_claim_ids:
            blockers.append("approved_claim_ids must exactly cover the rendered primary claims")
        else:
            approved_claim_ids = tuple(sorted(claim_ids))
        for key in [
            "methods_language_approved",
            "results_language_approved",
            "reviewer_response_language_approved",
            "matched_historical_q2_preserved",
            "external_v5g_remains_separate",
            "report_level_nonindependence_limitation_retained",
        ]:
            if decisions.get(key) is not True:
                blockers.append(f"decisions.{key} must be true")
        if decisions.get("patient_grouped_claims_added") is not False:
            blockers.append("patient_grouped_claims_added must remain false without a patient key")

    distribution = payload.get("distribution")
    expected_distribution = {
        "aggregate_candidate": "author_approved_for_named_destinations",
        "case_level_products": "governed_not_distributed",
        "model_weights": "not_redistributed",
    }
    if distribution != expected_distribution:
        blockers.append("distribution states do not match the bounded admission contract")

    return AdmissionValidation(
        not blockers,
        tuple(blockers),
        approved_claim_ids,
        approved_destinations,
    )
