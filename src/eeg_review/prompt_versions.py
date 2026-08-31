"""Explicit post-submission prompt versions; the historical prompt is unchanged."""

from __future__ import annotations

from .native_interface import sha256_text

HISTORICAL_PROMPT_VERSION = "historical-submitted"
MEDGEMMA_FOCAL_V2 = "medgemma-native-focal-disambiguation-v2"
MEDGEMMA_SCOPE_V21 = "medgemma-native-category-scope-v2.1"
PROMPT_VERSIONS = (HISTORICAL_PROMPT_VERSION, MEDGEMMA_FOCAL_V2, MEDGEMMA_SCOPE_V21)
HISTORICAL_PROMPT_SHA256 = "52198221d8330e9857b51a7ad99b017aa18836e1718b08dd0ae355820f5a5e69"
ANCHOR = "\n\n2. Generalized Epileptiform Activity:"
FOCAL_DISAMBIGUATION = (
    "\nFocal epileptiform clarification:\n"
    "- Judge findings recorded in this EEG, not the indication for the study or a history "
    "of seizures alone. Respect explicit negation and uncertainty in the report.\n"
    "- A sharply contoured waveform alone is not sufficient evidence of focal epileptiform "
    "activity. Focal slowing or attenuation, artifacts, and waveforms explicitly described "
    "as benign or non-epileptiform do not by themselves establish this category.\n"
    "- Generalized or bilaterally synchronous epileptiform discharges do not become focal "
    "solely because one region has greater amplitude or prominence. Retain focal "
    "epileptiform activity when the report also documents independent focal epileptiform "
    "discharges.\n"
    "- These are contextual distinctions, not keyword exclusions. Retain genuinely described "
    'focal spikes or sharp waves; do not require the literal word "epileptiform" when the '
    "report clearly describes focal epileptiform discharges.\n"
)
CATEGORY_SCOPE = (
    "\nCategory-scope cross-check:\n"
    "- Evaluate each category from the current EEG findings throughout the report, including "
    "the impression. A statement excluding epileptiform abnormalities excludes that type of "
    "activity; it does not by itself exclude focal or generalized non-epileptiform abnormalities "
    "or establish that the EEG is normal overall.\n"
    "- Evaluate the generalized non-epileptiform category independently. Consider broadly "
    "distributed background slowing and other non-epileptiform changes using the definition "
    "above. Distinguish current abnormal findings from descriptions explicitly attributed to "
    "normal state, activation, artifact or history. Do not infer abnormality from a frequency "
    "word alone.\n"
    "- Read qualifying and contrary statements together with positive findings. Apply negation "
    "only to the finding it describes. If the overall impression and the four subtype answers "
    "appear inconsistent, re-read the relevant category evidence before returning the five "
    "answers; do not resolve the inconsistency by simply flipping the overall answer. "
    "Keep the original four-level confidence meanings and constraints.\n"
)


def classification_prompt(base: str, version: str = HISTORICAL_PROMPT_VERSION) -> str:
    """Resolve a named change only against the exact historical source bytes."""
    if version == HISTORICAL_PROMPT_VERSION:
        return base
    if version not in {MEDGEMMA_FOCAL_V2, MEDGEMMA_SCOPE_V21}:
        raise ValueError("unknown classification prompt version")
    if sha256_text(base) != HISTORICAL_PROMPT_SHA256 or base.count(ANCHOR) != 1:
        raise ValueError("focal v2 requires the unchanged submitted classification prompt")
    focal = base.replace(ANCHOR, "\n" + FOCAL_DISAMBIGUATION + ANCHOR, 1)
    if version == MEDGEMMA_FOCAL_V2:
        return focal
    return focal.replace("\nQuestions:\n", "\n" + CATEGORY_SCOPE + "\nQuestions:\n", 1)


def scope_development_verdict(v1: dict, v2: dict, candidate: dict) -> dict:
    """Exploratory v2.1 rule, deliberately separate from v2's focal-only target."""
    # Reuse support/count checks, not the earlier experiment's selection rule.
    development_verdict(v1, v2)
    development_verdict(v1, candidate)
    errors = lambda row: row["fp"] + row["fn"]  # noqa: E731
    worsened = [
        label for label in candidate
        if errors(candidate[label]) > min(errors(v1[label]), errors(v2[label]))
    ]
    rare_detection_losses = [
        label for label in ["Focal Epi", "Gen Epi"]
        if candidate[label]["fn"] > min(v1[label]["fn"], v2[label]["fn"])
    ]
    scope_fn_reduction = any(
        candidate[label]["fn"] < v2[label]["fn"] for label in ["Gen Non-epi", "Abnormality"]
    )
    improved = sum(map(errors, candidate.values())) < sum(map(errors, v2.values()))
    supported = improved and scope_fn_reduction and not worsened and not rare_detection_losses
    return {
        "status": "development_rule_met" if supported else "development_rule_not_met",
        "fewer_total_category_errors_than_v2": improved,
        "scope_false_negative_reduction": scope_fn_reduction,
        "categories_worse_than_either_parent": worsened,
        "rare_epileptiform_detection_losses": rare_detection_losses,
        "protected_evaluation_authorized": False,
        "independent_confirmation": False,
        "interpretation": "Exploratory rule; not superiority, equivalence or clinical truth.",
    }


def prompt_row_identity(version: str, prompt: str) -> dict[str, str]:
    if version == HISTORICAL_PROMPT_VERSION:
        return {}  # Preserve historical CSV columns by default.
    return {
        "classification_prompt_version": version,
        "classification_prompt_sha256": sha256_text(prompt),
    }


def validate_prompt_resume(frame, version: str, prompt: str) -> None:
    """Never mix prompt versions, missing identities, or changed bytes on resume."""
    if frame.empty:
        return
    expected = {
        "classification_prompt_version": version,
        "classification_prompt_sha256": sha256_text(prompt),
    }
    has_identity = any(column in frame for column in expected)
    if version == HISTORICAL_PROMPT_VERSION and not has_identity:
        return
    for column, value in expected.items():
        if column not in frame or frame[column].isna().any() or set(frame[column]) != {value}:
            raise ValueError("resumed output has missing, mixed, or changed prompt identity")


def development_verdict(v1: dict, v2: dict) -> dict:
    """Frozen descriptive rule; never promotes a variant to protected evaluation."""
    from .logprob_adapter import JSON_KEY_TO_LABEL

    expected = set(JSON_KEY_TO_LABEL.values())
    if set(v1) != expected or set(v2) != expected:
        raise ValueError("all five labels are required for the development rule")
    for label in expected:
        for table in [v1, v2]:
            if any(table[label][key] < 0 for key in ["tp", "tn", "fp", "fn"]):
                raise ValueError("negative confusion counts")
            if sum(table[label][key] for key in ["tp", "tn", "fp", "fn"]) != 100:
                raise ValueError("the rule requires all 100 development cases")
        for fields in [("tp", "fn"), ("tn", "fp")]:
            if sum(v1[label][key] for key in fields) != sum(v2[label][key] for key in fields):
                raise ValueError("reference support changed between versions")
    a, b = v1["Focal Epi"], v2["Focal Epi"]
    other_regressions = [
        label
        for label in v1
        if label != "Focal Epi"
        and v2[label]["fp"] + v2[label]["fn"] > v1[label]["fp"] + v1[label]["fn"]
    ]
    supported = b["fp"] < a["fp"] and b["fn"] <= a["fn"] and not other_regressions
    return {
        "status": "development_rule_met" if supported else "development_rule_not_met",
        "focal_false_positive_change": b["fp"] - a["fp"],
        "focal_false_negative_change": b["fn"] - a["fn"],
        "other_categories_with_more_core_errors": other_regressions,
        "protected_evaluation_authorized": False,
        "independent_confirmation": False,
        "interpretation": "Descriptive development rule, not proof of superiority or equivalence.",
    }
