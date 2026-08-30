"""Pure helpers for the preregistered MedGemma native-chat sensitivity."""

from __future__ import annotations

import hashlib
from typing import Any

NATIVE_CHAT_INTERFACE_MODE = "native_chat"
RAW_COMPLETION_INTERFACE_MODE = "raw_completion"
REPORT_PLACEHOLDER = "{REPORT}"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def native_task_message_template(classification_prompt: str) -> str:
    """Return the frozen user-message layout without exposing a report."""
    return classification_prompt + "\n\n" + REPORT_PLACEHOLDER


def native_classification_messages(classification_prompt: str, report: str) -> list[dict[str, str]]:
    """Serialize one EEG task through the model-native chat interface."""
    return [{"role": "user", "content": classification_prompt + "\n\n" + report}]


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
