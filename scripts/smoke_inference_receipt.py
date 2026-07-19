"""Exercise inference receipts without downloading a model or processing governed data."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pipeline


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _prompt: str, **_kwargs: object) -> dict:
        self.calls += 1
        if self.calls % 2:
            text = '{"abnormality": 1}'
        else:
            text = '{"abnormality": {"decision": 1, "reasons": []}}'
        return {
            "choices": [{"text": text}],
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
        if "Report" in pd.read_csv(results).columns:
            raise AssertionError("Inference output leaked report text")
        receipt = json.loads(results.with_suffix(".run.json").read_text(encoding="utf-8"))
        if receipt["model"]["sha256"] != "0" * 64:
            raise AssertionError("Model receipt missing")
        if receipt["telemetry"]["classify_prompt_tokens"]["mean"] != 20.0:
            raise AssertionError("Token receipt missing")
    print("inference receipt smoke test passed")


if __name__ == "__main__":
    main()
