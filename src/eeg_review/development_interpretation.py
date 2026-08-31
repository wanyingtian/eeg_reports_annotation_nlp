"""Read-only, exact-key interpretation of three frozen development versions.

Evidence roles are model suggestions. Source matching establishes text presence,
not that a quotation entails a label or explains a classifier's computation.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from . import category_evidence
from .alignment_diagnostics import prompt_consistency
from .evidence_extraction import classification_levels
from .logprob_adapter import JSON_KEY_TO_LABEL
from .prompt_diagnostics import KEY
from .source_grounding import double_asterisk_source_candidates, inspect_reason, text_sha

VERSIONS = ("v1", "v2", "v21")


def exact_index(frame, expected):
    keys = frame[KEY].tolist()
    if any(not isinstance(k, str) or not k.strip() for k in keys):
        raise ValueError("missing or invalid report key")
    if len(set(keys)) != len(keys) or keys != expected:
        raise ValueError("missing, duplicate, extra or reordered report keys")
    return frame.set_index(KEY)


def ground_independent(raw, report):
    result = {
        "schema_valid": False,
        "schema_error": None,
        "cells": {},
        "report_text_sha256": text_sha(report),
        "raw_response_sha256": text_sha(raw),
    }
    try:
        parsed = category_evidence.parse(raw)
    except (TypeError, ValueError) as exc:
        result["schema_error"] = str(exc)
        return result
    result["schema_valid"] = True
    for category, cell in parsed.items():
        result["cells"][category] = {}
        for role, phrases in cell.items():
            checked = []
            for phrase in phrases:
                item = {"original_phrase": phrase, **inspect_reason(phrase, report)}
                item["double_asterisk_candidates"] = (
                    double_asterisk_source_candidates(phrase, report)
                    if item["status"] == "unmatched_requires_review"
                    else []
                )
                checked.append(item)
            result["cells"][category][role] = checked
    return result


def interpret(reference, versions, evidence, evidence_keys):
    expected = reference[KEY].tolist()
    refs = exact_index(reference, expected)
    if set(versions) != set(VERSIONS):
        raise ValueError("all three frozen versions are required")
    if not evidence_keys or evidence_keys != expected[: len(evidence_keys)]:
        raise ValueError("evidence must retain the fixed manifest prefix")
    indexed = {name: exact_index(frame, expected) for name, frame in versions.items()}
    audits = exact_index(evidence, evidence_keys)
    comparisons = {
        parent: {label: Counter() for label in JSON_KEY_TO_LABEL.values()}
        for parent in ("v1", "v2")
    }
    consistency = {name: Counter() for name in VERSIONS}
    points = {
        name: {label: Counter(dict(tp=0, tn=0, fp=0, fn=0)) for label in JSON_KEY_TO_LABEL.values()}
        for name in VERSIONS
    }
    counts, statuses, by_role = (
        Counter(),
        Counter(),
        {r: Counter() for r in category_evidence.FIELDS},
    )
    packet = []
    for position, (key, ref) in enumerate(refs.iterrows()):
        report = ref["Report"]
        if not isinstance(report, str) or not report.strip():
            raise ValueError("missing report text")
        levels = {
            name: classification_levels(frame.at[key, "classifications"])
            for name, frame in indexed.items()
        }
        checks = {name: prompt_consistency(value) for name, value in levels.items()}
        for name, value in checks.items():
            consistency[name].update(value)
        grounded = None
        if key in audits.index:
            row = audits.loc[key]
            if row["fixed_classifications"] != indexed["v21"].at[key, "classifications"]:
                raise ValueError("independent evidence linkage drift")
            grounded = ground_independent(row["explanations"], report)
            counts["records"] += 1
            counts["invalid_schema_records"] += not grounded["schema_valid"]
        categories, changes = {}, {p: False for p in comparisons}
        for category, label in JSON_KEY_TO_LABEL.items():
            truth = ref[label]
            if pd.isna(truth) or truth not in {1, 2, 3, 4}:
                raise ValueError("invalid reference level")
            truth = int(truth)
            predictions = {name: values[category] for name, values in levels.items()}
            outcomes = {}
            for name, value in predictions.items():
                outcome = (
                    ("tp" if truth >= 3 else "fp") if value >= 3 else ("fn" if truth >= 3 else "tn")
                )
                points[name][label][outcome] += 1
                outcomes[name] = outcome
            transitions = {}
            for parent in comparisons:
                before, after = predictions[parent], predictions["v21"]
                was_correct, is_correct = (
                    (before >= 3) == (truth >= 3),
                    (after >= 3) == (truth >= 3),
                )
                repair, regression = not was_correct and is_correct, was_correct and not is_correct
                transition = {
                    "repair": repair,
                    "regression": regression,
                    "core_changed": (before >= 3) != (after >= 3),
                    "level_changed": before != after,
                    "confidence_only_change": before != after and (before >= 3) == (after >= 3),
                    "four_level_repair": before != truth and after == truth,
                    "four_level_regression": before == truth and after != truth,
                }
                comparisons[parent][label].update(transition)
                if transition["core_changed"]:
                    comparisons[parent][label]["changed_cells_with_independent_audit"] += (
                        grounded is not None
                    )
                    changes[parent] = True
                transitions[parent] = transition
            cell = grounded["cells"].get(category) if grounded else None
            flags = None
            if grounded is not None:
                counts["category_cells"] += 1
                if cell is not None:
                    counts["valid_role_lists"] += 3
                    counts["empty_role_lists"] += sum(not x for x in cell.values())
                    phrases = [x for group in cell.values() for x in group]
                    counts["nonempty_category_cells"] += bool(phrases)
                    counts["cells_with_literal_quotes"] += any(
                        x["accepted_as_verbatim"] for x in phrases
                    )
                    counts["cells_with_literal_or_whitespace_candidates"] += any(
                        x["accepted_as_verbatim"] or x["source_span_candidates"] for x in phrases
                    )
                    counts["cells_with_both_present_and_absent_lists"] += bool(
                        cell["present_evidence"] and cell["absent_evidence"]
                    )
                    flags = {
                        "present_list_with_negative_classification": bool(cell["present_evidence"])
                        and predictions["v21"] < 3,
                        "absent_list_with_positive_classification": bool(cell["absent_evidence"])
                        and predictions["v21"] >= 3,
                        "no_present_list_for_positive_classification": not cell["present_evidence"]
                        and predictions["v21"] >= 3,
                    }
                    counts.update(flags)
                    for role, group in cell.items():
                        for item in group:
                            statuses[item["status"]] += 1
                            by_role[role][item["status"]] += 1
                            counts["double_asterisk_candidate_instances"] += bool(
                                item["double_asterisk_candidates"]
                            )
            categories[category] = {
                "reference_level": truth,
                "predictions": predictions,
                "reference_outcomes": outcomes,
                "transitions": transitions,
                "independent_evidence": cell,
                "evidence_review_flags": flags,
                "semantic_alignment": "not_adjudicated",
                "causal_faithfulness": "not_measured",
            }
        if grounded:
            all_phrases = [
                x for cell in grounded["cells"].values() for group in cell.values() for x in group
            ]
            counts["all_empty_valid_records"] += grounded["schema_valid"] and not all_phrases
        packet.append(
            {
                KEY: key,
                "development_position": position,
                "Report": report,
                "report_text_sha256": text_sha(report),
                "categories": categories,
                "prompt_constraints": checks,
                "core_changed_from_parent": changes,
                "independent_audit": grounded,
            }
        )
    return {
        "records": len(packet),
        "versions": list(VERSIONS),
        "binary_counts": {
            name: {k: dict(v) for k, v in values.items()} for name, values in points.items()
        },
        "paired_changes": {
            parent: {k: dict(v) for k, v in values.items()}
            for parent, values in comparisons.items()
        },
        "changed_reports": {
            parent: sum(r["core_changed_from_parent"][parent] for r in packet)
            for parent in comparisons
        },
        "reports_with_any_binary_error": {
            name: sum(
                any(
                    c["reference_outcomes"][name] in {"fp", "fn"}
                    for c in row["categories"].values()
                )
                for row in packet
            )
            for name in VERSIONS
        },
        "prompt_constraints": {name: dict(value) for name, value in consistency.items()},
        "independent_evidence": {
            **dict(counts),
            "phrase_instances": sum(statuses.values()),
            "phrase_statuses": dict(statuses),
            "by_role": {role: dict(value) for role, value in by_role.items()},
        },
        "inference_performed": False,
        "classifications_changed": False,
        "source_acceptance_changed": False,
        "semantic_alignment_adjudicated": False,
        "clinicalbert_alignment_reproduced": False,
        "sample_scope": "fixed first-20 only; missing case evidence is not inferred or generated",
        "interpretation": (
            "posthoc development; repeated dependent category/phrase instances, "
            "not independent tests"
        ),
    }, packet
