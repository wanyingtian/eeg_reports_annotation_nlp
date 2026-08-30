from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eeg_review.protected_execution import (
    ProtectedExecutionLocked,
    assert_governed_run_active,
    authorize_plan_before_governed_access,
    build_unlock_receipt,
    validate_authorization_receipt,
    validate_frozen_parent_receipts,
    validate_protected_job_binding,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "review/model-receipts/medgemma-native-protected-authorization.template.json"
STUDY_PLAN = (
    ROOT
    / "review/model-receipts/medgemma-native-protected-comparator.preregistered.json"
)
TIER_PLAN = (
    ROOT
    / "review/model-receipts/medgemma-native-protected-tiered-execution.preregistered.json"
)


def confirmed() -> dict:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "confirmed",
            "authorization_id": "documented-confirmation-1",
            "coverage_statement": (
                "The frozen comparator execution is covered by the current approval "
                "and data-use arrangement."
            ),
        }
    )
    payload["authority"] = {
        "role": "principal_investigator",
        "name_or_record": "confirmed-authority",
        "confirmation_source": "governed correspondence record",
        "confirmation_source_sha256": "a" * 64,
        "confirmed_at_utc": "2026-08-30T00:00:00Z",
    }
    payload["scope"]["already_transferred_deidentified_reports"] = True
    payload["scope"]["secondary_use_covered"] = True
    return payload


def write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_pending_template_fails_closed() -> None:
    result = validate_authorization_receipt(TEMPLATE)
    assert result.valid is False
    assert "status must be confirmed" in result.blockers


def test_exact_documented_confirmation_unlocks(tmp_path: Path) -> None:
    result = build_unlock_receipt(write(tmp_path, confirmed()))
    assert result["protected_evaluation_unlocked"] is True
    assert result["blockers"] == []


def test_authorship_is_not_an_authorizing_role(tmp_path: Path) -> None:
    payload = confirmed()
    payload["authority"]["role"] = "author"
    result = validate_authorization_receipt(write(tmp_path, payload))
    assert result.valid is False
    assert any("accepted confirming role" in blocker for blocker in result.blockers)


def test_authorized_study_researcher_can_document_existing_scope(tmp_path: Path) -> None:
    payload = confirmed()
    payload["authority"]["role"] = "authorized_study_researcher"
    result = validate_authorization_receipt(write(tmp_path, payload))
    assert result.valid is True


def test_cohort_scope_cannot_be_reduced_or_expanded_silently(tmp_path: Path) -> None:
    payload = confirmed()
    payload["scope"]["cohorts"]["zoe_evaluation_1395"] = 1394
    result = validate_authorization_receipt(write(tmp_path, payload))
    assert result.valid is False
    assert any("exactly match" in blocker for blocker in result.blockers)


def test_secondary_use_must_be_explicit(tmp_path: Path) -> None:
    payload = confirmed()
    payload["scope"]["secondary_use_covered"] = None
    result = validate_authorization_receipt(write(tmp_path, payload))
    assert result.valid is False
    assert any("secondary-use" in blocker for blocker in result.blockers)


def test_distribution_controls_cannot_be_relaxed(tmp_path: Path) -> None:
    payload = confirmed()
    payload["controls"]["keyed_outputs_remain_governed"] = False
    result = validate_authorization_receipt(write(tmp_path, payload))
    assert result.valid is False
    assert any("keyed_outputs_remain_governed" in blocker for blocker in result.blockers)


def test_configuration_cannot_drift(tmp_path: Path) -> None:
    payload = confirmed()
    payload["configuration_id"] += "-changed"
    result = validate_authorization_receipt(write(tmp_path, payload))
    assert result.valid is False
    assert "configuration_id mismatch" in result.blockers


def test_confirmation_source_hash_must_be_hex(tmp_path: Path) -> None:
    payload = confirmed()
    payload["authority"]["confirmation_source_sha256"] = "z" * 64
    result = validate_authorization_receipt(write(tmp_path, payload))
    assert result.valid is False
    assert any("SHA-256" in blocker for blocker in result.blockers)


def test_confirmation_timestamp_must_include_timezone(tmp_path: Path) -> None:
    payload = confirmed()
    payload["authority"]["confirmed_at_utc"] = "2026-08-30T00:00:00"
    result = validate_authorization_receipt(write(tmp_path, payload))
    assert result.valid is False
    assert any("timezone-aware" in blocker for blocker in result.blockers)


def test_public_plan_requires_authorization_before_governed_access() -> None:
    plan = json.loads(STUDY_PLAN.read_text(encoding="utf-8"))
    with pytest.raises(ProtectedExecutionLocked, match="--authorization"):
        authorize_plan_before_governed_access(plan, None)


def test_frozen_parent_receipts_match_repository() -> None:
    plan = json.loads(STUDY_PLAN.read_text(encoding="utf-8"))
    validate_frozen_parent_receipts(plan, ROOT)


def test_frozen_parent_drift_is_rejected() -> None:
    plan = json.loads(STUDY_PLAN.read_text(encoding="utf-8"))
    plan["frozen_parent_receipts"]["development_freeze"]["sha256"] = "0" * 64
    with pytest.raises(ProtectedExecutionLocked, match="hash mismatch"):
        validate_frozen_parent_receipts(plan, ROOT)


def protected_job(receipt_sha256: str, plan: dict) -> dict:
    commands = []
    for cohort_id in ["zoe_evaluation_1395", "maria_evaluation_499"]:
        commands.append(
            {
                "stage": f"{cohort_id}_inference",
                "command": [
                    "python",
                    "pipeline.py",
                    "--classification-only",
                    "--local-model-only",
                    "--classification-interface",
                    "native_chat",
                ],
            }
        )
    return {
        "study_id": plan["study_id"],
        "configuration_id": plan["configuration_id"],
        "cohorts": [
            {"cohort_id": cohort_id, "records": records}
            for cohort_id, records in plan["authorization_gate"]["cohorts"].items()
        ],
        "commands": commands,
        "protected_authorization": {
            "receipt_sha256": receipt_sha256,
            "study_id": plan["study_id"],
            "configuration_id": plan["configuration_id"],
        },
        "frozen_parent_receipts": plan["frozen_parent_receipts"],
    }


def test_frozen_job_binding_accepts_only_native_exact_cohorts(tmp_path: Path) -> None:
    plan = json.loads(TIER_PLAN.read_text(encoding="utf-8"))
    authorization_path = write(tmp_path, confirmed())
    authorization = authorize_plan_before_governed_access(plan, authorization_path)
    assert authorization is not None
    validate_protected_job_binding(
        plan, protected_job(authorization.receipt_sha256, plan), authorization
    )


def test_job_cannot_silently_switch_back_to_raw_completion(tmp_path: Path) -> None:
    plan = json.loads(TIER_PLAN.read_text(encoding="utf-8"))
    authorization_path = write(tmp_path, confirmed())
    authorization = authorize_plan_before_governed_access(plan, authorization_path)
    assert authorization is not None
    job = protected_job(authorization.receipt_sha256, plan)
    job["commands"][0]["command"][-1] = "raw_completion"
    with pytest.raises(ProtectedExecutionLocked, match="native_chat"):
        validate_protected_job_binding(plan, job, authorization)


def test_prepare_fails_on_authorization_before_nonexistent_governed_paths() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_medgemma_study.py"),
            "--plan",
            str(STUDY_PLAN),
            "--source-run",
            "/definitely/not/a/governed/source",
            "--output-dir",
            "/definitely/not/a/governed/output",
            "--authorization",
            str(TEMPLATE),
            "--acknowledge-governed-output",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "status must be confirmed" in result.stdout
    assert "FileNotFoundError" not in result.stderr


def test_runner_fails_on_authorization_before_nonexistent_governed_path() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_tiered_medgemma_study.py"),
            "dry-run",
            "--run-dir",
            "/definitely/not/a/governed/run",
            "--tier-plan",
            str(TIER_PLAN),
            "--authorization",
            str(TEMPLATE),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "status must be confirmed" in result.stdout
    assert "FileNotFoundError" not in result.stderr


def test_eclipse_marker_blocks_governed_run(tmp_path: Path) -> None:
    assert_governed_run_active(tmp_path)
    (tmp_path / "ECLIPSED.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtectedExecutionLocked, match="eclipsed"):
        assert_governed_run_active(tmp_path)
