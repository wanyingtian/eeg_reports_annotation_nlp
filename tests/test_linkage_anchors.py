import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "anchor_inventory", ROOT / "scripts/audit_linkage_anchors.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_generic_placeholder_and_numeric_health_id_not_identity_anchor():
    summary, details = module.scan(["Patient ID: REDACTED. PHN: 1234567890."] * 2)
    assert all(v["distinct_tokens"] == 0 for v in summary.values())
    assert not any(details.values())


def test_repeated_opaque_key_is_candidate_not_confirmation():
    token = "0123456789abcdef0123456789abcdef"
    summary, details = module.scan([f"Patient ID: {token}", f"Patient_ID={token}"])
    assert summary["explicit_patient_opaque_key"]["tokens_repeated_across_reports"] == 1
    assert list(details["explicit_patient_opaque_key"].values()) == [[0, 1]]


def test_single_occurrence_or_repetition_in_one_report_is_not_cross_report_link():
    token = "0123456789abcdef0123456789abcdef"
    summary, _ = module.scan([f"Patient ID: {token}; Patient ID: {token}", "Normal EEG"])
    assert summary["explicit_patient_opaque_key"]["tokens_repeated_across_reports"] == 0


def test_unlabelled_uuid_is_not_promoted_to_patient_field():
    token = "01234567-89ab-cdef-0123-456789abcdef"
    summary, _ = module.scan([token, token])
    assert summary["opaque_hex_or_uuid"]["tokens_repeated_across_reports"] == 1
    assert summary["explicit_patient_opaque_key"]["distinct_tokens"] == 0
