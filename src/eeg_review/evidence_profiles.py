"""Typed evidence-generation and traceability-claim compatibility profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceGenerationProfile:
    profile_id: str
    lineage: str
    classification_supplied_to_model: bool
    schema_fields: tuple[str, ...]
    grammar_artifact: str
    intended_question: str
    prohibited_interpretations: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceClaimProfile:
    profile_id: str
    accepted_stages: tuple[str, ...]
    aggregation: str
    suitable_language: str
    human_adjudication_required: bool
    compatible_generation_profiles: tuple[str, ...]
    prohibited_claims: tuple[str, ...]


DECISION_CONDITIONED = "jbhi-02463/evidence-generation/decision-conditioned/v1"
INDEPENDENT_CATEGORY = "jbhi-02463/evidence-generation/independent-category/v1"

ORIGINAL_THESIS_SUPPORT = (
    "jbhi-02463/evidence-claim/original-thesis-source-support/v1"
)
VERBATIM_PROVENANCE = "jbhi-02463/evidence-claim/verbatim-provenance/v1"
NORMALIZED_PROVENANCE = "jbhi-02463/evidence-claim/normalized-provenance/v1"
RETRIEVAL_ASSISTED_REVIEW = (
    "jbhi-02463/evidence-claim/retrieval-assisted-review/v1"
)
ADJUDICATED_DECISION_SUPPORT = (
    "jbhi-02463/evidence-claim/adjudicated-decision-support/v1"
)


GENERATION_PROFILES = {
    DECISION_CONDITIONED: EvidenceGenerationProfile(
        profile_id=DECISION_CONDITIONED,
        lineage="Chris Tian thesis classify-then-extract pathway",
        classification_supplied_to_model=True,
        schema_fields=("decision", "reasons"),
        grammar_artifact="src/LLM_pipeline/result_grammar_exp.gbnf",
        intended_question=(
            "Which source phrases does the configured model return for a frozen "
            "four-level category decision?"
        ),
        prohibited_interpretations=(
            "hidden chain of thought",
            "causal faithfulness",
            "clinical correctness",
        ),
    ),
    INDEPENDENT_CATEGORY: EvidenceGenerationProfile(
        profile_id=INDEPENDENT_CATEGORY,
        lineage="post-submission independent category-evidence extension",
        classification_supplied_to_model=False,
        schema_fields=(
            "present_evidence",
            "absent_evidence",
            "qualification_evidence",
        ),
        grammar_artifact="src/LLM_pipeline/result_grammar_category_evidence.gbnf",
        intended_question=(
            "Which present, absent and qualifying source passages can the configured "
            "model identify without receiving a category decision?"
        ),
        prohibited_interpretations=(
            "classification performance",
            "hidden chain of thought",
            "causal faithfulness",
            "clinical correctness",
        ),
    ),
}


CLAIM_PROFILES = {
    ORIGINAL_THESIS_SUPPORT: EvidenceClaimProfile(
        profile_id=ORIGINAL_THESIS_SUPPORT,
        accepted_stages=(
            "normalized_substring",
            "fuzzy_token_match_70",
            "whole_report_semantic_similarity_0.70",
        ),
        aggregation="any phrase passes any declared stage",
        suitable_language="three-stage source support under the thesis profile",
        human_adjudication_required=False,
        compatible_generation_profiles=(DECISION_CONDITIONED,),
        prohibited_claims=(
            "unchanged quotation",
            "entailment",
            "clinical factuality",
        ),
    ),
    VERBATIM_PROVENANCE: EvidenceClaimProfile(
        profile_id=VERBATIM_PROVENANCE,
        accepted_stages=("unchanged_substring",),
        aggregation="report every segment and report-category denominator",
        suitable_language="unchanged source quotation",
        human_adjudication_required=False,
        compatible_generation_profiles=(DECISION_CONDITIONED, INDEPENDENT_CATEGORY),
        prohibited_claims=("relevance", "sufficiency", "entailment"),
    ),
    NORMALIZED_PROVENANCE: EvidenceClaimProfile(
        profile_id=NORMALIZED_PROVENANCE,
        accepted_stages=("unchanged_substring", "declared_lossless_normalization"),
        aggregation="report unchanged and normalized stages separately",
        suitable_language="source-equivalent after declared normalization",
        human_adjudication_required=False,
        compatible_generation_profiles=(DECISION_CONDITIONED, INDEPENDENT_CATEGORY),
        prohibited_claims=("verbatim identity", "relevance", "entailment"),
    ),
    RETRIEVAL_ASSISTED_REVIEW: EvidenceClaimProfile(
        profile_id=RETRIEVAL_ASSISTED_REVIEW,
        accepted_stages=(
            "unchanged_substring",
            "declared_lossless_normalization",
            "fuzzy_span_localization",
            "semantic_span_localization",
        ),
        aggregation="retain best span, score and method per generated segment",
        suitable_language="candidate source passage located for review",
        human_adjudication_required=False,
        compatible_generation_profiles=(DECISION_CONDITIONED, INDEPENDENT_CATEGORY),
        prohibited_claims=("decision support", "clinical factuality", "entailment"),
    ),
    ADJUDICATED_DECISION_SUPPORT: EvidenceClaimProfile(
        profile_id=ADJUDICATED_DECISION_SUPPORT,
        accepted_stages=(
            "source_localization",
            "blinded_relevance_review",
            "blinded_sufficiency_review",
        ),
        aggregation=(
            "preserve support, contradiction, qualification, insufficiency and "
            "abstention"
        ),
        suitable_language="independently reviewed decision-supporting evidence",
        human_adjudication_required=True,
        compatible_generation_profiles=(DECISION_CONDITIONED, INDEPENDENT_CATEGORY),
        prohibited_claims=("clinical correctness", "hidden model reasoning"),
    ),
}


def _profile_dict(value: Any) -> dict[str, Any]:
    # JSON round-trip converts tuple-valued contracts to their serialized list form,
    # so the in-memory catalog and committed JSON have exactly the same structure.
    return json.loads(json.dumps(asdict(value)))


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def catalog() -> dict[str, Any]:
    """Return a stable, public-safe catalog of every named profile."""
    return {
        "schema_version": 1,
        "generation_profiles": {
            key: _profile_dict(value) for key, value in GENERATION_PROFILES.items()
        },
        "claim_profiles": {
            key: _profile_dict(value) for key, value in CLAIM_PROFILES.items()
        },
        "selection_rule": (
            "Every new evidence run names one generation profile and one compatible "
            "claim profile; neither is inferred from a model name."
        ),
    }


def profile_selection_receipt(
    generation_profile_id: str,
    claim_profile_id: str,
) -> dict[str, Any]:
    """Validate and bind an explicit generation/claim-profile selection."""
    try:
        generation = GENERATION_PROFILES[generation_profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown evidence-generation profile: {generation_profile_id}") from exc
    try:
        claim = CLAIM_PROFILES[claim_profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown evidence-claim profile: {claim_profile_id}") from exc
    if generation_profile_id not in claim.compatible_generation_profiles:
        raise ValueError(
            f"claim profile {claim_profile_id} is incompatible with "
            f"generation profile {generation_profile_id}"
        )
    generation_dict = _profile_dict(generation)
    claim_dict = _profile_dict(claim)
    return {
        "schema_version": 1,
        "generation_profile_id": generation_profile_id,
        "generation_profile_sha256": _sha256_json(generation_dict),
        "claim_profile_id": claim_profile_id,
        "claim_profile_sha256": _sha256_json(claim_dict),
        "classification_supplied_to_model": generation.classification_supplied_to_model,
        "human_adjudication_required": claim.human_adjudication_required,
        "suitable_claim_language": claim.suitable_language,
        "prohibited_claims": list(claim.prohibited_claims),
    }
