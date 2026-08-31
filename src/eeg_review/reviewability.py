"""Read-only, post-hoc case preparation. No inference, relabeling or exclusions."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from itertools import combinations

import pandas as pd

from .audit import DEFAULT_LABELS
from .error_review import review_handle

KEY = "Hashed_ReportURN"
STRATA = (
    "medgemma_correct_mistral_wrong",
    "medgemma_wrong_mistral_correct",
    "both_false_negative",
    "both_false_positive",
    "both_correct_positive",
    "both_correct_negative",
)
POLICY = {
    "id": "jbhi-reviewability-v1",
    "sampling_seed": 20260718,
    "error_cap_per_label_stratum": 5,
    "control_cap_per_label_stratum": 2,
    "near_duplicate_rule": "casefolded Unicode word 5-shingle set Jaccard >= 0.80",
    "shingle_words": 5,
    "minimum_unique_shingles": 20,
    "jaccard_numerator": 4,
    "jaccard_denominator": 5,
    "cross_cohort_only": True,
    "label_blind_similarity": True,
    "automatic_exclusions": False,
    "clinical_adjudication": False,
    "model_inference": False,
}


def exact_frame(frame: pd.DataFrame, keys: list[str], *, labels: bool = True) -> pd.DataFrame:
    if KEY not in frame or frame[KEY].isna().any():
        raise ValueError("missing report keys")
    if (
        frame[KEY].duplicated().any()
        or not frame[KEY].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    ):
        raise ValueError("duplicate or invalid report keys")
    if len(keys) != len(set(keys)) or set(frame[KEY]) != set(keys):
        raise ValueError("exact report-key sets differ")
    result = frame.set_index(KEY).loc[keys].reset_index()
    if labels:
        for label in DEFAULT_LABELS:
            if label not in result or not result[label].isin([1, 2, 3, 4]).all():
                raise ValueError("invalid or missing four-level values")
    return result


def paired_packet(reference, medgemma, mistral, *, cohort: str, salt: str):
    """Population counts plus order-invariant, capped review preparation."""
    if not salt:
        raise ValueError("private handle salt is required")
    keys = sorted(reference[KEY].tolist())
    frames = [exact_frame(frame, keys) for frame in (reference, medgemma, mistral)]
    reference, medgemma, mistral = frames
    rows, summaries = [], {}
    for label in DEFAULT_LABELS:
        counts = Counter()
        strata_rows = {name: [] for name in STRATA}
        for index, key in enumerate(keys):
            ref, a, b = (int(frame.iloc[index][label]) for frame in frames)
            rpos, apos, bpos = ref > 2, a > 2, b > 2
            if apos != bpos:
                stratum = STRATA[0] if apos == rpos else STRATA[1]
            elif apos != rpos:
                stratum = STRATA[2] if rpos else STRATA[3]
            else:
                stratum = STRATA[4] if rpos else STRATA[5]
            counts[stratum] += 1
            handle = review_handle(f"{cohort}:{key}", salt)
            strata_rows[stratum].append(
                {
                    "case_handle": handle,
                    "cohort": cohort,
                    "label": label,
                    "stratum": stratum,
                    "reference_level": ref,
                    "medgemma_level": a,
                    "mistral_level": b,
                    "review_status": "pending",
                    "source_assessment": "",
                    "reference_ambiguity": "",
                    "clinical_salience": "",
                    "workflow_consequence": "",
                    "reviewer_role": "",
                    "review_notes": "",
                }
            )
        selected = {}
        for stratum in STRATA:
            cap = (
                POLICY["control_cap_per_label_stratum"]
                if stratum.startswith("both_correct")
                else POLICY["error_cap_per_label_stratum"]
            )

            # SHA ranking is deterministic across row order, Python versions and machines.
            def rank(row, label=label, stratum=stratum):
                value = f"{POLICY['sampling_seed']}:{label}:{stratum}:{row['case_handle']}"
                return hashlib.sha256(value.encode()).hexdigest()

            chosen = sorted(strata_rows[stratum], key=rank)[:cap]
            rows.extend(chosen)
            selected[stratum] = len(chosen)
        summaries[label] = {
            "eligible": {s: counts[s] for s in STRATA},
            "selected": selected,
            "medgemma_errors": counts[STRATA[1]] + counts[STRATA[2]] + counts[STRATA[3]],
            "mistral_errors": counts[STRATA[0]] + counts[STRATA[2]] + counts[STRATA[3]],
        }
        if sum(counts.values()) != len(keys):
            raise ValueError("population arithmetic failed")
    packet = pd.DataFrame(rows)
    summary = {
        "records": len(keys),
        "labels": summaries,
        "selected_label_case_rows": len(packet),
        "selected_unique_reports": int(packet.case_handle.nunique()),
        "patient_grouped": False,
        "sampling_unit": "report within label and stratum",
        "population_error_counts_not_sample_estimates": True,
        "review_completed": False,
    }
    return summary, packet


def shingles(text: str) -> set[tuple[str, ...]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("missing source text")
    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    width = POLICY["shingle_words"]
    return {tuple(words[i : i + width]) for i in range(len(words) - width + 1)}


def near_duplicate_pair(left, right, *, left_name: str, right_name: str, salt: str):
    """Exhaustive cross-cohort lexical check; never a patient-linkage surrogate."""
    if not salt:
        raise ValueError("private handle salt is required")
    prepared = []
    for frame, name in ((left, left_name), (right, right_name)):
        exact_frame(frame, sorted(frame[KEY].tolist()), labels=False)
        prepared.append(
            [
                (review_handle(f"{name}:{key}", salt), shingles(text))
                for key, text in zip(frame[KEY], frame.Report, strict=True)
            ]
        )
    flagged, eligible, pruned, intersections = [], 0, 0, 0
    minimum = POLICY["minimum_unique_shingles"]
    for a_handle, a in prepared[0]:
        for b_handle, b in prepared[1]:
            if min(len(a), len(b)) < minimum:
                continue
            eligible += 1
            # An exact upper bound on Jaccard, not an approximate candidate sampler.
            if min(len(a), len(b)) * 5 < max(len(a), len(b)) * 4:
                pruned += 1
                continue
            common = len(a & b)
            intersections += 1
            union = len(a) + len(b) - common
            if common * 5 >= union * 4:
                flagged.append(
                    {
                        "left_handle": a_handle,
                        "right_handle": b_handle,
                        "left_cohort": left_name,
                        "right_cohort": right_name,
                        "shared_shingles": common,
                        "union_shingles": union,
                        "jaccard": common / union,
                    }
                )
    counts = {
        "candidate_pairs": len(left) * len(right),
        "eligible_pairs": eligible,
        "short_text_pairs_not_assessed": len(left) * len(right) - eligible,
        "length_bound_pruned_pairs": pruned,
        "exact_intersections": intersections,
        "flagged_pairs": len(flagged),
        "flagged_left_reports": len({r["left_handle"] for r in flagged}),
        "flagged_right_reports": len({r["right_handle"] for r in flagged}),
    }
    assert eligible == pruned + intersections
    return counts, flagged


def cohort_pairs(cohorts):
    return list(combinations(sorted(cohorts), 2))
