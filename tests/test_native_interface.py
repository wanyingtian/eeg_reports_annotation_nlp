from __future__ import annotations

from eeg_review.native_interface import (
    REPORT_PLACEHOLDER,
    embedded_chat_template_receipt,
    native_classification_messages,
    native_task_message_template,
    sha256_text,
)


class FakeModel:
    metadata = {"tokenizer.chat_template": "{{ messages[0]['content'] }}"}


def test_native_message_has_one_user_turn_and_preserves_bytes() -> None:
    messages = native_classification_messages("PROMPT", "REPORT")
    assert messages == [{"role": "user", "content": "PROMPT\n\nREPORT"}]


def test_task_template_is_report_free_and_hashable() -> None:
    template = native_task_message_template("PROMPT")
    assert template == f"PROMPT\n\n{REPORT_PLACEHOLDER}"
    assert sha256_text(template) == sha256_text(f"PROMPT\n\n{REPORT_PLACEHOLDER}")


def test_embedded_template_receipt_keeps_exact_identity() -> None:
    receipt = embedded_chat_template_receipt(FakeModel())
    assert receipt["text"] == "{{ messages[0]['content'] }}"
    assert receipt["sha256"] == sha256_text(receipt["text"])
