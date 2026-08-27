"""Exercise inference receipts without downloading a model or processing governed data."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pandas as pd
import pipeline


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str, **kwargs: object) -> dict:
        self.calls += 1
        if self.calls % 2:
            text = (
                '{"focal_epileptiform_activity":1,'
                '"generalized_epileptiform_activity":2,'
                '"focal_non_epileptiform_activity":3,'
                '"generalized_non_epileptiform_activity":4,'
                '"abnormality":4}'
            )
        else:
            text = '{"abnormality": {"decision": 1, "reasons": []}}'
        choice: dict[str, object] = {"text": text}
        if kwargs.get("logprobs"):
            scores = {
                "1": math.log(0.1),
                "2": math.log(0.2),
                "3": math.log(0.3),
                "4": math.log(0.4),
            }
            token_logprobs = [-0.01] * len(text)
            top_logprobs: list[dict[str, float]] = [{} for _ in text]
            for key in pipeline.JSON_KEY_TO_LABEL:
                marker = f'"{key}":'
                digit_index = text.index(marker) + len(marker)
                token_logprobs[digit_index] = scores[text[digit_index]]
                top_logprobs[digit_index] = scores.copy()
            choice["logprobs"] = {
                "tokens": list(text),
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
                "text_offset": list(range(len(prompt), len(prompt) + len(text))),
            }
        return {
            "choices": [choice],
            "usage": {"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26},
        }


class BinaryFakeModel(FakeModel):
    def __call__(self, prompt: str, **kwargs: object) -> dict:
        self.calls += 1
        if self.calls % 2:
            text = (
                '{"focal_epileptiform_activity":1,'
                '"generalized_epileptiform_activity":4,'
                '"focal_non_epileptiform_activity":1,'
                '"generalized_non_epileptiform_activity":4,'
                '"abnormality":4}'
            )
        else:
            text = '{"abnormality": {"decision": 1, "reasons": []}}'
        choice: dict[str, object] = {"text": text}
        if kwargs.get("logprobs"):
            scores = {"1": math.log(0.25), "4": math.log(0.75)}
            token_logprobs = [-0.01] * len(text)
            top_logprobs: list[dict[str, float]] = [{} for _ in text]
            for key in pipeline.JSON_KEY_TO_LABEL:
                marker = f'"{key}":'
                digit_index = text.index(marker) + len(marker)
                token_logprobs[digit_index] = scores[text[digit_index]]
                top_logprobs[digit_index] = scores.copy()
            choice["logprobs"] = {
                "tokens": list(text),
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
                "text_offset": list(range(len(prompt), len(prompt) + len(text))),
            }
        return {
            "choices": [choice],
            "usage": {"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26},
        }


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="eeg-inference-receipt-") as temporary:
        output = Path(temporary)
        results = output / "raw_fixture_mistral_1_v1_run1.csv"
        config = output / "config_fixture_mistral_v1.json"
        run_config = pipeline.RunConfig(
            outdir=output,
            dataset_path=repository / "data" / "zoe_reports_sample.db",
            dataset_id="fixture",
            model_name="mistral",
            capture_classification_logprobs=True,
        )
        model_receipt = {
            "registry_name": "fake",
            "repo_id": "fixture/fake",
            "filename": "fake.gguf",
            "huggingface_snapshot": "fixture",
            "sha256": "0" * 64,
            "size_bytes": 0,
            "load_parameters": {"n_ctx": 4096, "n_gpu_layers": 0, "verbose": False},
        }
        pipeline.run_pipeline(
            FakeModel(),
            model_receipt,
            pd.DataFrame([{"Hashed_ReportURN": "fixture-id", "Report": "Normal EEG."}]),
            pd.DataFrame(),
            None,
            None,
            results,
            config,
            run_config,
            flush_every=1,
        )
        result_table = pd.read_csv(results)
        if "Report" in result_table.columns:
            raise AssertionError("Inference output leaked report text")
        if abs(float(result_table.loc[0, "Prob_Abnormality"]) - 0.7) > 1e-12:
            raise AssertionError("Grammar-token probability receipt missing")
        receipt = json.loads(results.with_suffix(".run.json").read_text(encoding="utf-8"))
        if receipt["model"]["sha256"] != "0" * 64:
            raise AssertionError("Model receipt missing")
        if receipt["telemetry"]["classify_prompt_tokens"]["mean"] != 20.0:
            raise AssertionError("Token receipt missing")
        instrumentation = receipt["calibration_instrumentation"]
        if not instrumentation["enabled"]:
            raise AssertionError("Calibration instrumentation was not receipted")
        if instrumentation["completion_logprobs_requested"] != 64:
            raise AssertionError("Classification logprob depth missing")
        if instrumentation["available_records_by_label"]["Abnormality"] != 1:
            raise AssertionError("Probability availability count missing")

        binary_results = output / "binary_core_fixture.csv"
        binary_config = output / "binary_core_fixture.json"
        binary_run_config = pipeline.RunConfig(
            outdir=output,
            dataset_path=repository / "data" / "zoe_reports_sample.db",
            dataset_id="fixture-binary-core",
            model_name="mistral",
            capture_classification_logprobs=True,
            classification_mode=pipeline.BINARY_CORE_ADAPTER_MODE,
        )
        pipeline.run_pipeline(
            BinaryFakeModel(),
            model_receipt,
            pd.DataFrame([{"Hashed_ReportURN": "fixture-id", "Report": "Normal EEG."}]),
            pd.DataFrame(),
            None,
            None,
            binary_results,
            binary_config,
            binary_run_config,
            flush_every=1,
        )
        binary_table = pd.read_csv(binary_results)
        if abs(float(binary_table.loc[0, "Prob_Abnormality"]) - 0.75) > 1e-12:
            raise AssertionError("Binary-core probability receipt missing")
        if (
            binary_table.loc[0, "adaptation_classification_mode"]
            != pipeline.BINARY_CORE_ADAPTER_MODE
        ):
            raise AssertionError("Binary-core resume marker missing")
        binary_receipt = json.loads(
            binary_results.with_suffix(".run.json").read_text(encoding="utf-8")
        )
        binary_instrument = binary_receipt["calibration_instrumentation"]
        if binary_instrument["classification_mode"] != pipeline.BINARY_CORE_ADAPTER_MODE:
            raise AssertionError("Binary-core mode not receipted")
        if "alternatives {1,4}" not in binary_instrument["feature_definition"]:
            raise AssertionError("Binary-core feature definition not receipted")
    print("inference receipt smoke test passed")


if __name__ == "__main__":
    main()
