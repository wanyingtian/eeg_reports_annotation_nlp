from __future__ import annotations

import math
import re
from typing import Any

JSON_KEY_TO_LABEL = {
    "focal_epileptiform_activity": "Focal Epi",
    "generalized_epileptiform_activity": "Gen Epi",
    "focal_non_epileptiform_activity": "Focal Non-epi",
    "generalized_non_epileptiform_activity": "Gen Non-epi",
    "abnormality": "Abnormality",
}

PROBABILITY_COLUMNS = tuple(f"Prob_{label}" for label in JSON_KEY_TO_LABEL.values())


def _level_from_token(token: Any) -> int | None:
    if not isinstance(token, str):
        return None
    match = re.fullmatch(r"\s*([1-4])\s*[,}]?\s*", token)
    return int(match.group(1)) if match else None


def _token_index_at_offset(offsets: list[Any], position: int) -> int | None:
    integer_offsets = [offset for offset in offsets if isinstance(offset, int)]
    if not integer_offsets:
        return None
    # llama-cpp-python reports completion offsets relative to prompt+completion
    # when echo=False. Normalize the first completion token to zero so matching
    # remains independent of prompt length.
    base = min(integer_offsets)
    normalized = [offset - base if isinstance(offset, int) else offset for offset in offsets]
    valid = [offset for offset in normalized if isinstance(offset, int) and offset <= position]
    if not valid:
        return None
    target = max(valid)
    return next(index for index, offset in enumerate(normalized) if offset == target)


def extract_core_positive_probabilities(
    completion: str,
    logprobs: dict[str, Any] | None,
) -> dict[str, float | None]:
    """Extract P(level in {3,4}) at each grammar-constrained decision token.

    The returned values are normalized only over explicit level-token
    alternatives 1--4. A category is unavailable unless all four alternatives
    were present in llama.cpp's recorded top-logprob surface. This prevents a
    truncated top-k response from being silently treated as a calibrated
    probability.
    """
    result = {label: None for label in JSON_KEY_TO_LABEL.values()}
    if not isinstance(logprobs, dict):
        return result
    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    top_logprobs = logprobs.get("top_logprobs")
    offsets = logprobs.get("text_offset")
    if not all(
        isinstance(value, list) for value in (tokens, token_logprobs, top_logprobs, offsets)
    ):
        return result
    if not (len(tokens) == len(token_logprobs) == len(top_logprobs) == len(offsets)):
        return result

    for json_key, label in JSON_KEY_TO_LABEL.items():
        match = re.search(rf'"{re.escape(json_key)}"\s*:\s*([1-4])', completion)
        if match is None:
            continue
        token_index = _token_index_at_offset(offsets, match.start(1))
        if token_index is None:
            continue
        alternatives = top_logprobs[token_index]
        if not isinstance(alternatives, dict):
            continue
        level_logprobs: dict[int, float] = {}
        for token, value in alternatives.items():
            level = _level_from_token(token)
            if (
                level is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                current = level_logprobs.get(level)
                level_logprobs[level] = (
                    float(value) if current is None else max(current, float(value))
                )

        chosen_level = int(match.group(1))
        chosen_logprob = token_logprobs[token_index]
        if (
            chosen_level not in level_logprobs
            and isinstance(chosen_logprob, (int, float))
            and not isinstance(chosen_logprob, bool)
        ):
            level_logprobs[chosen_level] = float(chosen_logprob)
        if set(level_logprobs) != {1, 2, 3, 4}:
            continue
        maximum = max(level_logprobs.values())
        weights = {level: math.exp(value - maximum) for level, value in level_logprobs.items()}
        denominator = sum(weights.values())
        if not math.isfinite(denominator) or denominator <= 0:
            continue
        probability = (weights[3] + weights[4]) / denominator
        result[label] = float(probability) if math.isfinite(probability) else None
    return result


def extract_binary_core_positive_probabilities(
    completion: str,
    logprobs: dict[str, Any] | None,
) -> dict[str, float | None]:
    """Extract P(core positive) from an explicitly binary 1/4 grammar surface.

    The binary adaptation mode uses level token 1 for core absent and level
    token 4 for core present so existing explanation and processing code keeps
    its polarity semantics. Values are unavailable unless both alternatives
    are explicit at the decision position.
    """
    result = {label: None for label in JSON_KEY_TO_LABEL.values()}
    if not isinstance(logprobs, dict):
        return result
    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    top_logprobs = logprobs.get("top_logprobs")
    offsets = logprobs.get("text_offset")
    if not all(
        isinstance(value, list) for value in (tokens, token_logprobs, top_logprobs, offsets)
    ):
        return result
    if not (len(tokens) == len(token_logprobs) == len(top_logprobs) == len(offsets)):
        return result

    for json_key, label in JSON_KEY_TO_LABEL.items():
        match = re.search(rf'"{re.escape(json_key)}"\s*:\s*([14])', completion)
        if match is None:
            continue
        token_index = _token_index_at_offset(offsets, match.start(1))
        if token_index is None:
            continue
        alternatives = top_logprobs[token_index]
        if not isinstance(alternatives, dict):
            continue
        binary_logprobs: dict[int, float] = {}
        for token, value in alternatives.items():
            level = _level_from_token(token)
            if (
                level in {1, 4}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                current = binary_logprobs.get(level)
                binary_logprobs[level] = (
                    float(value) if current is None else max(current, float(value))
                )
        chosen_level = int(match.group(1))
        chosen_logprob = token_logprobs[token_index]
        if (
            chosen_level not in binary_logprobs
            and isinstance(chosen_logprob, (int, float))
            and not isinstance(chosen_logprob, bool)
        ):
            binary_logprobs[chosen_level] = float(chosen_logprob)
        if set(binary_logprobs) != {1, 4}:
            continue
        maximum = max(binary_logprobs.values())
        absent_weight = math.exp(binary_logprobs[1] - maximum)
        present_weight = math.exp(binary_logprobs[4] - maximum)
        denominator = absent_weight + present_weight
        if math.isfinite(denominator) and denominator > 0:
            probability = present_weight / denominator
            result[label] = float(probability) if math.isfinite(probability) else None
    return result
