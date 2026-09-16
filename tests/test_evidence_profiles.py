from __future__ import annotations

import json
from pathlib import Path

import pytest

from eeg_review.evidence_profiles import (
    ADJUDICATED_DECISION_SUPPORT,
    DECISION_CONDITIONED,
    INDEPENDENT_CATEGORY,
    NORMALIZED_PROVENANCE,
    ORIGINAL_THESIS_SUPPORT,
    VERBATIM_PROVENANCE,
    catalog,
    profile_selection_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


def test_committed_catalog_matches_typed_registry() -> None:
    committed = json.loads(
        (ROOT / "review/model-receipts/evidence-profile-catalog.v1.json").read_text()
    )
    assert committed == catalog()
    assert len(committed["generation_profiles"]) == 2
    assert len(committed["claim_profiles"]) == 5


def test_profiles_bind_generation_and_claim_separately() -> None:
    receipt = profile_selection_receipt(INDEPENDENT_CATEGORY, VERBATIM_PROVENANCE)
    assert receipt["classification_supplied_to_model"] is False
    assert receipt["human_adjudication_required"] is False
    assert len(receipt["generation_profile_sha256"]) == 64
    assert len(receipt["claim_profile_sha256"]) == 64


def test_original_thesis_claim_profile_preserves_its_generation_lineage() -> None:
    receipt = profile_selection_receipt(DECISION_CONDITIONED, ORIGINAL_THESIS_SUPPORT)
    assert receipt["classification_supplied_to_model"] is True
    with pytest.raises(ValueError, match="incompatible"):
        profile_selection_receipt(INDEPENDENT_CATEGORY, ORIGINAL_THESIS_SUPPORT)


def test_adjudicated_profile_cannot_hide_human_review_requirement() -> None:
    receipt = profile_selection_receipt(
        DECISION_CONDITIONED,
        ADJUDICATED_DECISION_SUPPORT,
    )
    assert receipt["human_adjudication_required"] is True
    assert "clinical correctness" in receipt["prohibited_claims"]


@pytest.mark.parametrize(
    "generation,claim",
    [
        ("unknown", NORMALIZED_PROVENANCE),
        (DECISION_CONDITIONED, "unknown"),
    ],
)
def test_unknown_profile_is_rejected(generation: str, claim: str) -> None:
    with pytest.raises(ValueError, match="unknown evidence"):
        profile_selection_receipt(generation, claim)

