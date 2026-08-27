from __future__ import annotations

import math

from eeg_review.logprob_adapter import (
    JSON_KEY_TO_LABEL,
    extract_core_positive_probabilities,
)


def fake_logprobs(completion: str, *, prompt_length: int = 0) -> dict:
    tokens = list(completion)
    offsets = list(range(prompt_length, prompt_length + len(completion)))
    token_logprobs = [-0.01] * len(completion)
    top_logprobs: list[dict[str, float]] = [{} for _ in completion]
    scores = {"1": math.log(0.1), "2": math.log(0.2), "3": math.log(0.3), "4": math.log(0.4)}
    for key in JSON_KEY_TO_LABEL:
        marker = f'"{key}":'
        digit_index = completion.index(marker) + len(marker)
        top_logprobs[digit_index] = scores.copy()
        token_logprobs[digit_index] = scores[completion[digit_index]]
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": offsets,
    }


def test_extracts_core_positive_probability_for_all_five_decisions() -> None:
    completion = (
        '{"focal_epileptiform_activity":1,'
        '"generalized_epileptiform_activity":2,'
        '"focal_non_epileptiform_activity":3,'
        '"generalized_non_epileptiform_activity":4,'
        '"abnormality":4}'
    )

    result = extract_core_positive_probabilities(
        completion,
        fake_logprobs(completion, prompt_length=4_096),
    )

    assert set(result) == set(JSON_KEY_TO_LABEL.values())
    assert all(value is not None and abs(value - 0.7) < 1e-12 for value in result.values())


def test_missing_level_alternative_is_not_silently_renormalized() -> None:
    completion = (
        '{"focal_epileptiform_activity":1,'
        '"generalized_epileptiform_activity":2,'
        '"focal_non_epileptiform_activity":3,'
        '"generalized_non_epileptiform_activity":4,'
        '"abnormality":4}'
    )
    payload = fake_logprobs(completion)
    marker = '"abnormality":'
    digit_index = completion.index(marker) + len(marker)
    del payload["top_logprobs"][digit_index]["2"]

    result = extract_core_positive_probabilities(completion, payload)

    assert result["Abnormality"] is None
    assert result["Focal Epi"] is not None


def test_malformed_logprob_surface_returns_explicit_unavailable_values() -> None:
    result = extract_core_positive_probabilities("{}", {"tokens": []})

    assert result == {label: None for label in JSON_KEY_TO_LABEL.values()}
