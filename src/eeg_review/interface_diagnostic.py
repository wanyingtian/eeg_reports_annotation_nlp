"""Bounded transport diagnostic; never a new cohort or configuration-selection rule."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress

POLICY = {
    "diagnostic_id": "jbhi-02463/medgemma/interface-mechanics/development-v1",
    "population": "existing first-100 Zoe development manifest only",
    "positions_zero_based": [0, 14, 28, 42, 56, 70, 84, 98],
    "selection": "fixed spaced positions, not chosen by predictions or reference outcomes",
    "arms": ["historical", "trim_only", "native_chat", "assembled", "assembled_original_stop"],
    "max_model_calls": 40,
    "automatic_expansion": False,
    "model_sha256": "b137aac80f2bcb1c1ed35bfe13387bc496eb18898d5f46425687604f0f714481",
    "prompt_sha256": "52198221d8330e9857b51a7ad99b017aa18836e1718b08dd0ae355820f5a5e69",
    "grammar_sha256": "5237e13988062538cda9c21906f1f4e1fc8b99498e2462ea69fe24bface35016",
    "template_sha256": "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4",
    "database_sha256": "7af5de9c8561d2c3347b89517fb245caf1fc8fb7001900582c6337e58142034e",
    "manifest_sha256": "dddbb46edf0ded6e468b01513a8034f205dcad1de4015d6ce97365d5bec8c051",
    "native_output_sha256": "f799362d9f6a22523c0ae8240b8ebd71a4cfe7a0187e5d65bd1ec051841bf633",
    "historical_output_sha256": "0e1695c64c735bf4d7f599e3f46aba1c4463bc184e53e6a8a49b07ba6612096e",
    "protected_evaluation": False,
    "reference_labels_used": False,
    "interpretation": (
        "token/transport mechanism and saved-output agreement, not accuracy estimation"
    ),
    "grammar_retained_all_arms": True,
    "frozen_study_outputs_modified": False,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def manual_gemma_prompt(payload: str, bos: str) -> str:
    """Mirror the verified one-user-turn template, not a universal Gemma serializer."""
    return (
        bos + "<start_of_turn>user\n" + payload.strip() + ("<end_of_turn>\n<start_of_turn>model\n")
    )


class CapturedCompletion(Exception):
    """Internal sentinel: stop immediately before inference."""


def capture_chat_request(model, messages, parameters):
    """Capture the real native handler's lower-level call, with zero generation."""
    original = model.create_completion
    captured = {}

    def intercept(*args, **kwargs):
        if args:
            raise ValueError("unexpected positional completion arguments")
        captured.update(kwargs)
        raise CapturedCompletion

    model.create_completion = intercept
    try:
        with suppress(CapturedCompletion):
            model.create_chat_completion(messages=messages, **parameters)
    finally:
        model.create_completion = original
    if not captured or not isinstance(captured.get("prompt"), list):
        raise ValueError("native handler did not expose an explicit token list")
    return captured


def validate_checkpoint(value, contract_sha256, case, arm):
    if value.get("contract_sha256") != contract_sha256:
        raise ValueError("checkpoint contract changed")
    if value.get("position") != case or value.get("arm") != arm:
        raise ValueError("checkpoint position/arm changed")
    expected = value.get("receipt_sha256")
    if expected != digest({k: v for k, v in value.items() if k != "receipt_sha256"}):
        raise ValueError("checkpoint contents changed")
    return value
