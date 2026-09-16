#!/usr/bin/env python3
"""Verify the follow-up draft against the completed factorial result and firewall."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "review/model-receipts/evidence-schema-factorial.result.json"
DRAFT = ROOT / "review/FOLLOW_UP_METHODS_PAPER_DRAFT_2026-09-16.md"
SECTION = ROOT / "review/FOLLOW_UP_PAPER_DRAFT_SCHEMA_CONFOUND_SECTION_2026-09-16.md"
FIREWALL = ROOT / "review/JBHI_TO_FOLLOW_UP_CLAIM_FIREWALL_2026-09-16.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ratio(value: list[int]) -> str:
    return f"{value[0]}/{value[1]}"


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    draft = DRAFT.read_text(encoding="utf-8")
    section = SECTION.read_text(encoding="utf-8")
    firewall = FIREWALL.read_text(encoding="utf-8")

    labels = {
        "medgemma_decision_conditioned": "MedGemma, decision-conditioned",
        "medgemma_independent_category_evidence": "MedGemma, independent-category",
        "mistral_decision_conditioned": "Mistral, decision-conditioned",
        "mistral_independent_category_evidence": "Mistral, independent-category",
    }
    for cell_id, label in labels.items():
        cell = result["cells"][cell_id]
        expected = " | ".join(
            [
                label,
                ratio(cell["evidence_units"]),
                ratio(cell["units_with_unchanged_quote"]),
                ratio(cell["unchanged_segments"]),
            ]
        )
        require(expected in draft, f"follow-up table drift: {cell_id}")

    med = result["within_model_schema_effects_independent_minus_decision_conditioned"][
        "medgemma"
    ]
    mistral = result[
        "within_model_schema_effects_independent_minus_decision_conditioned"
    ]["mistral"]
    interaction = result[
        "descriptive_interaction_mistral_minus_medgemma_percentage_points"
    ]
    for value in (
        abs(med["evidence_coverage_percentage_points"]),
        mistral["evidence_coverage_percentage_points"],
        interaction["evidence_coverage"],
        interaction["any_unchanged_quote"],
    ):
        require(f"{value:g}" in draft or f"{value:g}" in section, f"missing effect {value:g}")

    require(
        "Not part of the current JBHI" in draft,
        "follow-up draft does not declare its submission boundary",
    )
    require(
        "What stays out of O8" in firewall,
        "claim firewall lacks an explicit exclusion section",
    )
    require(
        "does not require new prose or numbers in O8" in firewall,
        "factorial result has become an implicit O8 dependency",
    )
    require(
        "classification-only" in firewall and "model-family superiority" in firewall,
        "current-revision scientific boundary is incomplete",
    )
    require(
        "held-out explanation-quality claim" in section,
        "follow-up draft does not distinguish its development result from future claims",
    )
    print(
        json.dumps(
            {
                "factorial_cells": 4,
                "jbhi_dependency": False,
                "status": "verified",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
