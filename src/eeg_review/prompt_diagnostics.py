"""Join saved decisions and source evidence for review, not voting or retuning."""

from __future__ import annotations

import pandas as pd

from .evidence_extraction import classification_levels
from .logprob_adapter import JSON_KEY_TO_LABEL

KEY = "Hashed_ReportURN"


def keyed(frame):
    if frame[KEY].isna().any() or frame[KEY].duplicated().any():
        raise ValueError("missing or duplicate diagnostic keys")
    return frame.set_index(KEY)


def build_diagnostic_packet(reference, versions, evidence):
    refs = keyed(reference)
    predictions = {name: keyed(frame) for name, frame in versions.items()}
    if not {"medgemma_native_v1", "medgemma_native_focal_v2"}.issubset(predictions):
        raise ValueError("both MedGemma prompt versions are required")
    for frame in predictions.values():
        if set(frame.index) != set(refs.index):
            raise ValueError("comparison prediction key sets differ")
    observed = {row[KEY]: row for row in evidence}
    if len(observed) != len(evidence) or not set(observed).issubset(refs.index):
        raise ValueError("duplicate or out-of-population evidence")
    summary = {
        label: {
            name: 0
            for name in [
                "reports",
                "reference_positive",
                "v2_false_positives",
                "v2_false_negatives",
                "v2_errors_with_evidence",
                "v2_errors_without_evidence",
                "v1_to_v2_repairs",
                "v1_to_v2_regressions",
                "cross_model_core_disagreements",
            ]
        }
        for label in JSON_KEY_TO_LABEL.values()
    }
    packet = []
    for case_key, row in refs.iterrows():
        levels = {
            name: classification_levels(frame.at[case_key, "classifications"])
            for name, frame in predictions.items()
        }
        details = {}
        for category, label in JSON_KEY_TO_LABEL.items():
            truth = row[label]
            if pd.isna(truth) or truth not in {1, 2, 3, 4}:
                raise ValueError("missing or invalid four-level reference")
            truth = int(truth)
            old = levels["medgemma_native_v1"][category]
            current = levels["medgemma_native_focal_v2"][category]
            old_correct, correct = (old >= 3) == (truth >= 3), (current >= 3) == (truth >= 3)
            outcome = (
                ("true_positive" if current >= 3 else "true_negative")
                if correct
                else ("false_positive" if current >= 3 else "false_negative")
            )
            external = [
                value[category]
                for name, value in levels.items()
                if name not in {"medgemma_native_v1", "medgemma_native_focal_v2"}
            ]
            disagree = any((value >= 3) != (current >= 3) for value in external)
            supported = observed.get(case_key, {}).get("cells", {}).get(category)
            questions = []
            if not correct:
                questions += [
                    "Which passage supports the reference label, and which supports the model?",
                    "Does the cited text affirm, negate, or discuss another finding or context?",
                ]
            if old_correct and not correct:
                questions.append(
                    "What changed from v1 to v2? Preserve this regression alongside repairs."
                )
            if disagree:
                questions.append(
                    "Which evidence distinction separates the saved model calls? "
                    "Disagreement is not adjudication."
                )
            if supported is None and (not correct or disagree):
                questions.append(
                    "No explanation was generated in the fixed first-20 sample; do not infer one."
                )
            if supported and any(
                x["status"] == "unmatched_requires_review" for x in supported["reasons"]
            ):
                questions.append(
                    "Is unmatched wording paraphrased, composite or unsupported? Requires review."
                )
            details[category] = {
                "reference_level": truth,
                "predictions": {name: values[category] for name, values in levels.items()},
                "v2_reference_outcome": outcome,
                "evidence_available": supported is not None,
                "evidence": supported,
                "cross_model_core_disagreement": disagree,
                "review_questions": questions,
                "review_answers": None,
                "semantic_alignment": "not_adjudicated",
                "causal_faithfulness": "not_measured",
            }
            counts = summary[label]
            counts["reports"] += 1
            counts["reference_positive"] += truth >= 3
            counts["v2_false_positives"] += outcome == "false_positive"
            counts["v2_false_negatives"] += outcome == "false_negative"
            counts["v2_errors_with_evidence"] += not correct and supported is not None
            counts["v2_errors_without_evidence"] += not correct and supported is None
            counts["v1_to_v2_repairs"] += not old_correct and correct
            counts["v1_to_v2_regressions"] += old_correct and not correct
            counts["cross_model_core_disagreements"] += disagree
        packet.append({KEY: case_key, "Report": row["Report"], "categories": details})
    return {
        "records": len(packet),
        "evidence_records": len(observed),
        "model_versions": list(predictions),
        "by_category": summary,
        "review_questions_executed": False,
        "reference_is_adjudicated_truth": False,
        "purpose": "evidence-informed diagnosis, not voting, causal explanation or retuning",
    }, packet


def targeted_missing_evidence(packet):
    """Error-enriched follow-up: all focal false positives and v1-to-v2 regressions."""
    selected, covered, missing = [], [], []
    for row in packet:
        target = False
        for category, cell in row["categories"].items():
            values = cell["predictions"]
            truth = cell["reference_level"] >= 3
            before = values["medgemma_native_v1"] >= 3
            after = values["medgemma_native_focal_v2"] >= 3
            target |= (before == truth and after != truth) or (
                category == "focal_epileptiform_activity"
                and cell["v2_reference_outcome"] == "false_positive"
            )
        if target:
            selected.append(row[KEY])
            if all(c["evidence_available"] for c in row["categories"].values()):
                covered.append(row[KEY])
            else:
                missing.append(row[KEY])
    if len(selected) > 3:
        raise ValueError("targeted diagnostic exceeds the frozen three-report cap")
    return missing, {
        "selection": "all focal false positives and all v1-to-v2 core regressions",
        "selected_reports": len(selected),
        "already_with_evidence": len(covered),
        "new_evidence_reports": len(missing),
        "maximum_reports": 3,
        "error_enriched_posthoc": True,
        "new_classification_calls": 0,
        "prompt_or_model_changes": False,
    }
