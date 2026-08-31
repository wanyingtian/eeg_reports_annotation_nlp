import pytest

from eeg_review.interface_diagnostic import (
    POLICY,
    capture_chat_request,
    digest,
    manual_gemma_prompt,
    validate_checkpoint,
)


def test_scope_is_small_fixed_development_only():
    assert len(set(POLICY["positions_zero_based"])) == 8
    assert all(0 <= x < 100 for x in POLICY["positions_zero_based"])
    assert len(POLICY["arms"]) * 8 == POLICY["max_model_calls"] == 40
    assert POLICY["protected_evaluation"] is False
    assert POLICY["reference_labels_used"] is False
    assert POLICY["automatic_expansion"] is False


def test_manual_template_has_one_bos_and_explicit_answer_turn():
    result = manual_gemma_prompt("\nINSTRUCTIONS\n\nREPORT\n ", "<bos>")
    assert (
        result
        == "<bos><start_of_turn>user\nINSTRUCTIONS\n\nREPORT<end_of_turn>\n<start_of_turn>model\n"
    )
    assert result.count("<bos>") == 1


def test_capture_stops_before_inference_and_restores_method():
    class Model:
        def create_completion(self, **kwargs):
            raise AssertionError("inference must not happen")

        def create_chat_completion(self, messages, **kwargs):
            return self.create_completion(prompt=[1, 9, 4], messages=messages, **kwargs)

    model = Model()
    original = model.create_completion
    result = capture_chat_request(model, [{"role": "user", "content": "synthetic"}], {})
    assert result["prompt"] == [1, 9, 4]
    assert model.create_completion == original


def test_capture_restores_method_after_failure():
    class Model:
        def create_completion(self, **kwargs):
            return None

        def create_chat_completion(self, **kwargs):
            raise ValueError("format rejected")

    model = Model()
    original = model.create_completion
    with pytest.raises(ValueError):
        capture_chat_request(model, [], {})
    assert model.create_completion == original


def test_resumption_rejects_changed_content_contract_or_key():
    value = {"contract_sha256": "contract", "position": 14, "arm": "assembled", "text": "{}"}
    value["receipt_sha256"] = digest(value)
    assert validate_checkpoint(value, "contract", 14, "assembled") == value
    for changed in [{**value, "text": "changed"}, {**value, "position": 15}]:
        with pytest.raises(ValueError):
            validate_checkpoint(changed, "contract", 14, "assembled")
    with pytest.raises(ValueError):
        validate_checkpoint(value, "other-contract", 14, "assembled")
