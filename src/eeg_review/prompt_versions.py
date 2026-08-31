"""Explicit post-submission prompt versions; the historical prompt is unchanged."""

from __future__ import annotations

from .native_interface import sha256_text

HISTORICAL_PROMPT_VERSION = "historical-submitted"
MEDGEMMA_FOCAL_V2 = "medgemma-native-focal-disambiguation-v2"
PROMPT_VERSIONS = (HISTORICAL_PROMPT_VERSION, MEDGEMMA_FOCAL_V2)
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


def classification_prompt(base: str, version: str = HISTORICAL_PROMPT_VERSION) -> str:
    """Resolve a named change only against the exact historical source bytes."""
    if version == HISTORICAL_PROMPT_VERSION:
        return base
    if version != MEDGEMMA_FOCAL_V2:
        raise ValueError("unknown classification prompt version")
    if sha256_text(base) != HISTORICAL_PROMPT_SHA256 or base.count(ANCHOR) != 1:
        raise ValueError("focal v2 requires the unchanged submitted classification prompt")
    return base.replace(ANCHOR, "\n" + FOCAL_DISAMBIGUATION + ANCHOR, 1)


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
