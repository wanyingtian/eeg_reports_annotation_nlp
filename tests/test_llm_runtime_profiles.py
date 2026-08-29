from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[1] / "src/LLM_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

# The unit tests exercise validation and resume contracts without loading model
# weights. Keep the developer-only test environment independent of Metal wheels.
if importlib.util.find_spec("llama_cpp") is None:
    llama_cpp = types.ModuleType("llama_cpp")
    llama_cpp_llama = types.ModuleType("llama_cpp.llama")

    class StubLlama:
        pass

    class StubLlamaGrammar:
        @classmethod
        def from_string(cls, _content: str):
            return cls()

    llama_cpp_llama.Llama = StubLlama
    llama_cpp_llama.LlamaGrammar = StubLlamaGrammar
    llama_cpp.llama = llama_cpp_llama
    sys.modules["llama_cpp"] = llama_cpp
    sys.modules["llama_cpp.llama"] = llama_cpp_llama

import llm_models  # noqa: E402

PIPELINE_SPEC = importlib.util.spec_from_file_location(
    "runtime_profile_pipeline", PIPELINE_ROOT / "pipeline.py"
)
assert PIPELINE_SPEC and PIPELINE_SPEC.loader
PIPELINE = importlib.util.module_from_spec(PIPELINE_SPEC)
sys.modules[PIPELINE_SPEC.name] = PIPELINE
PIPELINE_SPEC.loader.exec_module(PIPELINE)


def test_load_parameters_accept_full_metal_offload_profile() -> None:
    parameters = llm_models.validated_load_parameters(
        logits_all=False,
        load_overrides={
            "n_gpu_layers": -1,
            "n_batch": 1024,
            "n_ubatch": 512,
            "n_threads": 4,
            "n_threads_batch": 10,
            "flash_attn": True,
        },
    )

    assert parameters["n_gpu_layers"] == -1
    assert parameters["flash_attn"] is True
    assert parameters["n_batch"] == 1024
    assert parameters["n_ubatch"] == 512


def test_load_parameters_reject_unknown_and_incoherent_values() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        llm_models.validated_load_parameters(
            logits_all=False, load_overrides={"unreceipted_switch": True}
        )
    with pytest.raises(ValueError, match="n_ubatch cannot exceed n_batch"):
        llm_models.validated_load_parameters(
            logits_all=False,
            load_overrides={"n_batch": 256, "n_ubatch": 512},
        )


def test_resume_refuses_to_mix_runtime_profiles(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["Hashed_ReportURN", "runtime_profile_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Hashed_ReportURN": "case-1",
                "runtime_profile_id": "llama-cpp-python-default",
            }
        )

    with pytest.raises(ValueError, match="cannot mix runtime profiles"):
        PIPELINE.process_completed_csv(
            path,
            runtime_profile_id="llama-cpp-python-metal-full-offload-flash-v1",
        )


def test_legacy_resume_is_explicitly_labeled_default_runtime(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Hashed_ReportURN"])
        writer.writeheader()
        writer.writerow({"Hashed_ReportURN": "case-1"})

    frame, completed = PIPELINE.process_completed_csv(path)

    assert set(frame["runtime_profile_id"]) == {"llama-cpp-python-default"}
    assert completed == {"case-1"}
