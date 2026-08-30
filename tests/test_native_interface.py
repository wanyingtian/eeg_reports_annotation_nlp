from __future__ import annotations

from eeg_review.native_interface import (
    CLASSIFICATION_PLACEHOLDER,
    REPORT_PLACEHOLDER,
    embedded_chat_template_receipt,
    explanation_input,
    explanation_task_message_template,
    native_classification_messages,
    native_explanation_messages,
    native_task_message_template,
    native_user_messages,
    sha256_text,
)


class FakeModel:
    metadata = {"tokenizer.chat_template": "{{ messages[0]['content'] }}"}


def test_native_message_has_one_user_turn_and_preserves_bytes() -> None:
    messages = native_classification_messages("PROMPT", "REPORT")
    assert messages == [{"role": "user", "content": "PROMPT\n\nREPORT"}]
    assert native_user_messages("exact\nbytes") == [
        {"role": "user", "content": "exact\nbytes"}
    ]


def test_task_template_is_report_free_and_hashable() -> None:
    template = native_task_message_template("PROMPT")
    assert template == f"PROMPT\n\n{REPORT_PLACEHOLDER}"
    assert sha256_text(template) == sha256_text(f"PROMPT\n\n{REPORT_PLACEHOLDER}")


def test_embedded_template_receipt_keeps_exact_identity() -> None:
    receipt = embedded_chat_template_receipt(FakeModel())
    assert receipt["text"] == "{{ messages[0]['content'] }}"
    assert receipt["sha256"] == sha256_text(receipt["text"])


def test_native_explanation_changes_only_the_interface_envelope() -> None:
    raw = explanation_input("EXPLAIN", "REPORT", '{"answer": 4}')
    messages = native_explanation_messages("EXPLAIN", "REPORT", '{"answer": 4}')

    assert messages == [{"role": "user", "content": raw}]
    assert raw == (
        "EXPLAIN\n\n---\nEEG Report:\nREPORT"
        '\n\nClassification JSON:\n{"answer": 4}\n'
    )


def test_explanation_template_contains_no_case_content() -> None:
    template = explanation_task_message_template("EXPLAIN")
    assert REPORT_PLACEHOLDER in template
    assert CLASSIFICATION_PLACEHOLDER in template
    assert "patient text" not in template
