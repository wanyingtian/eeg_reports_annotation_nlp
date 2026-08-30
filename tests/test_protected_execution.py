from __future__ import annotations

import json
from pathlib import Path

from eeg_review.protected_execution import build_unlock_receipt, validate_authorization_receipt

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "review/model-receipts/medgemma-native-protected-authorization.template.json"


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
