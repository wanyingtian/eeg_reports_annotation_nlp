"""Inspect existing rules and prompt constraints, without new clinical labels.

The original rule classifier is called unchanged. Its default-negative result
is exposed as a fallback, never promoted to a validated semantic assessment.
This is not the ClinicalBERT-based alignment score selected in the thesis.
"""

from __future__ import annotations

from collections import Counter

from .evidence_extraction import JSON_KEYS
from .prompt_diagnostics import KEY
from .source_grounding import double_asterisk_source_candidates

POLICY_ID = "saved-evidence-alignment-diagnostic-v1"


def legacy_rule_trace(text: str) -> dict:
    # Import the actual inherited rules rather than copying or retuning them.
    from evidence_analysis.evidence_alignment import (
        NEGATION_PATTERNS,
        POSITIVE_KEYWORDS,
        classify_reason_rule,
    )

    if not isinstance(text, str):
        raise ValueError("rule diagnostic requires text")
    s = text.strip().lower()
    negatives = [p.pattern for p in NEGATION_PATTERNS if p.search(s)]
    positives = [p.pattern for p in POSITIVE_KEYWORDS if p.search(s)]
    explicit = [
        p for p in ["this eeg is abnormal", "this is an abnormal", "abnormal eeg"] if p in s
    ]
    if not s:
        basis, expected = "empty_unclear", 0
    elif explicit:
        basis, expected = "explicit_abnormal", 1
    elif negatives:
        basis, expected = "negative_pattern", -1
    elif positives:
        basis, expected = "positive_pattern", 1
    else:
        basis, expected = "no_match_default_negative", -1
    actual = classify_reason_rule(text)
    if actual != expected:
        raise ValueError("historical rule changed; diagnostic trace requires review")
    return {
        "historical_polarity": actual,
        "decision_basis": basis,
        "negative_pattern_matches": negatives,
        "positive_pattern_matches": positives,
        "explicit_abnormal_matches": explicit,
        "both_polarities_have_rule_matches": bool(negatives and (positives or explicit)),
        "semantic_alignment": "not_adjudicated",
    }


def prompt_consistency(levels: dict) -> dict:
    """Check both historical instructions; do not repair labels or infer truth."""
    if set(levels) != set(JSON_KEYS) or any(
        type(v) is not int or v not in {1, 2, 3, 4} for v in levels.values()
    ):
        raise ValueError("exactly five four-level decisions required")
    any_subtype = any(v >= 3 for k, v in levels.items() if k != "abnormality")
    overall = levels["abnormality"] >= 3
    return {
        "subtype_positive_overall_negative": any_subtype and not overall,
        "all_subtypes_negative_overall_positive": not any_subtype and overall,
    }


def diagnose_saved_evidence(packet: list[dict], first_sample_keys: list[str]) -> tuple[dict, list]:
    """Separate fixed first sample from error-enriched additions, never pool scores."""
    keys = [row[KEY] for row in packet]
    if not keys or any(not isinstance(k, str) or not k for k in keys):
        raise ValueError("missing diagnostic keys")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate diagnostic keys")
    if not first_sample_keys or first_sample_keys != keys[:len(first_sample_keys)]:
        raise ValueError("first evidence sample must be the manifest prefix")
    models = set(packet[0]["categories"]["abnormality"]["predictions"])
    consistency = {name: Counter() for name in sorted(models)}
    sample_counts = {name: Counter() for name in ["first_sample", "targeted_additions"]}
    basis_counts = {name: Counter() for name in sample_counts}
    marker_phrases = {name: set() for name in sample_counts}
    details = []
    for row in packet:
        cats = row["categories"]
        if set(cats) != set(JSON_KEYS):
            raise ValueError("diagnostic omitted a category")
        if any(set(c["predictions"]) != models for c in cats.values()):
            raise ValueError("diagnostic model sets differ")
        checks = {}
        for model in sorted(models):
            levels = {k: c["predictions"][model] for k, c in cats.items()}
            checks[model] = prompt_consistency(levels)
            consistency[model]["reports"] += 1
            consistency[model].update(checks[model])
        available = [c["evidence"] is not None for c in cats.values()]
        if any(available) != all(available):
            raise ValueError("partial explanation record")
        cells = {}
        group = "first_sample" if row[KEY] in first_sample_keys else "targeted_additions"
        counts = sample_counts[group]
        if row[KEY] in first_sample_keys and not all(available):
            raise ValueError("first sample evidence missing")
        if all(available):
            counts["reports"] += 1
            for category, cell in cats.items():
                evidence = cell["evidence"]
                reasons = evidence["reasons"]
                counts["decision_cells"] += 1
                if evidence["decision_copy_matches"] is not True or any(
                    not isinstance(r.get("original_reason"), str) for r in reasons
                ):
                    counts["withheld_cells"] += 1
                    cells[category] = {"status": "withheld_invalid_or_mismatched_evidence"}
                    continue
                # process_output.py serializes a reason list with semicolons.
                # Keep this cell-level rule unit distinct from individual phrases.
                trace = legacy_rule_trace("; ".join(r["original_reason"] for r in reasons))
                mismatch = trace["historical_polarity"] != (
                    1 if cell["predictions"]["medgemma_native_focal_v2"] >= 3 else -1
                )
                counts["traced_cells"] += 1
                counts["historical_rule_label_mismatches"] += mismatch
                counts["mixed_rule_signal_cells"] += trace["both_polarities_have_rule_matches"]
                basis_counts[group][trace["decision_basis"]] += 1
                candidate_details = []
                for index, reason in enumerate(reasons):
                    if reason["status"] != "unmatched_requires_review":
                        continue
                    counts["previously_unmatched_phrase_instances"] += 1
                    candidates = double_asterisk_source_candidates(
                        reason["original_reason"], row["Report"]
                    )
                    counts["double_asterisk_candidate_instances"] += bool(candidates)
                    if candidates:
                        marker_phrases[group].add((row[KEY], reason["original_reason"]))
                    candidate_details.append({
                        "reason_index": index,
                        "original_reason": reason["original_reason"],
                        "source_span_candidates": candidates,
                        "accepted_as_verbatim": False,
                        "semantic_support_confirmed": False,
                    })
                cells[category] = {
                    "legacy_rule_trace": trace,
                    "historical_rule_label_mismatch": mismatch,
                    "additional_source_candidates": candidate_details,
                    "status": "review_signal_only",
                }
        details.append({
            KEY: row[KEY],
            "prompt_constraint_checks": checks,
            "evidence_sample": group if all(available) else "not_generated",
            "cells": cells,
        })
    return {
        "policy_id": POLICY_ID,
        "records": len(packet),
        "prompt_constraints": {k: dict(v) for k, v in consistency.items()},
        "evidence_samples": {
            name: {
                **dict(counts),
                "double_asterisk_unique_report_phrases": len(marker_phrases[name]),
                "rule_decision_bases": dict(basis_counts[name]),
            }
            for name, counts in sample_counts.items()
        },
        "historical_classifier_changed": False,
        "clinicalbert_alignment_score_reproduced": False,
        "semantic_alignment_adjudicated": False,
        "source_quote_acceptance_changed": False,
        "classifications_changed": False,
        "model_inference_performed": False,
        "scope": "posthoc development diagnostics; not a prompt selection or causal measure",
    }, details
