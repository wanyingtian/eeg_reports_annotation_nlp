"""Pure helpers for receipted model-native EEG task interfaces."""

from __future__ import annotations

import hashlib
from typing import Any

NATIVE_CHAT_INTERFACE_MODE = "native_chat"
RAW_COMPLETION_INTERFACE_MODE = "raw_completion"
REPORT_PLACEHOLDER = "{REPORT}"
CLASSIFICATION_PLACEHOLDER = "{CLASSIFICATION_JSON}"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def native_task_message_template(classification_prompt: str) -> str:
    """Return the frozen user-message layout without exposing a report."""
    return classification_prompt + "\n\n" + REPORT_PLACEHOLDER


def native_classification_messages(classification_prompt: str, report: str) -> list[dict[str, str]]:
    """Serialize one EEG task through the model-native chat interface."""
    return native_user_messages(classification_prompt + "\n\n" + report)


def native_user_messages(content: str) -> list[dict[str, str]]:
    """Wrap exact task bytes in one user turn without changing their content."""
    return [{"role": "user", "content": content}]


def explanation_input(
    explanation_prompt: str,
    report: str,
    classification_json: str,
) -> str:
    """Return the historical explanation payload shared by both interfaces."""
    return (
        explanation_prompt
        + "\n\n---\nEEG Report:\n"
        + report
        + "\n\nClassification JSON:\n"
        + classification_json
        + "\n"
    )


def explanation_task_message_template(explanation_prompt: str) -> str:
    """Return the report-free explanation layout for immutable receipts."""
    return explanation_input(
        explanation_prompt,
        REPORT_PLACEHOLDER,
        CLASSIFICATION_PLACEHOLDER,
    )


def native_explanation_messages(
    explanation_prompt: str,
    report: str,
    classification_json: str,
) -> list[dict[str, str]]:
    """Serialize evidence extraction without relaxing its output grammar."""
    return native_user_messages(
        explanation_input(explanation_prompt, report, classification_json)
    )


def embedded_chat_template_receipt(model: Any) -> dict[str, Any]:
    """Record the exact embedded template identity without report content."""
    template = str(getattr(model, "metadata", {}).get("tokenizer.chat_template", ""))
    if not template:
        raise ValueError("The selected model has no embedded tokenizer.chat_template")
    return {
        "source": "GGUF tokenizer.chat_template metadata",
        "sha256": sha256_text(template),
        "text": template,
    }
