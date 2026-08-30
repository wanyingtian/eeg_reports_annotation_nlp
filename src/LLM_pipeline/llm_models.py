# Copyright (c) 2025 Wanying Tian
# Licensed under the Apache-2.0 License (see LICENSE file in the project root for details).
#!/usr/bin/env python3
"""
Simple Model Configuration Module

Edit MODEL_CONFIGS to add/modify models.
Edit DEFAULT_PARAMS to change default loading parameters.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from llama_cpp.llama import Llama

# Default parameters for model loading
DEFAULT_PARAMS = {
    "n_ctx": 4096, # context window size, 4096 tokens are about 3000 words
    "n_gpu_layers": 30,
    "verbose": False,
}

ALLOWED_LOAD_OVERRIDES = {
    "n_ctx",
    "n_gpu_layers",
    "n_batch",
    "n_ubatch",
    "n_threads",
    "n_threads_batch",
    "flash_attn",
    "use_mmap",
    "use_mlock",
}

# Model configurations - edit this to add new models
MODEL_CONFIGS = {
    "mistral": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
        "revision": "3a6fbf4a41a1d52e415a4958cde6856d34b2db93",
        "sha256": "b85cdd596ddd76f3194047b9108a73c74d77ba04bef49255a50fc0cfbda83d32",
    },
    # Candidate contemporary comparators. These registry entries pin the public
    # Unsloth GGUF artifacts that match the filenames and approximate sizes in
    # the received author-team summaries. Vasily's producing hashes and prompt
    # bundle are still required before either can be called an exact
    # reproduction of his runs.
    "medgemma-27b-q2-candidate": {
        "repo_id": "unsloth/medgemma-27b-text-it-GGUF",
        "filename": "medgemma-27b-text-it-Q2_K.gguf",
        "revision": "334fbf6811c963d223f6ac107a459347353f068d",
        "sha256": "b137aac80f2bcb1c1ed35bfe13387bc496eb18898d5f46425687604f0f714481",
    },
    "medgemma-27b-q4-candidate": {
        "repo_id": "unsloth/medgemma-27b-text-it-GGUF",
        "filename": "medgemma-27b-text-it-Q4_K_S.gguf",
        "revision": "334fbf6811c963d223f6ac107a459347353f068d",
        "sha256": "1ad12d20c9e2ef61f74c0e952de589c93cb3dce17750f1fbfe0db4921616a5b1",
    },
    "deepseek": {
        "repo_id": "TheBloke/deepseek-llm-7b-base-GGUF",
        "filename": "deepseek-llm-7b-base.Q5_K_M.gguf",
    },
    "deepseek-chat": {
        "repo_id": "TheBloke/deepseek-llm-7B-chat-GGUF",
        "filename": "deepseek-llm-7b-chat.Q5_K_M.gguf",
    },
    "hermes-mistral": {
        "repo_id": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF",
        "filename": "Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf",
    },
    "hermes-llama2": {
        "repo_id": "TheBloke/Nous-Hermes-Llama-2-7B-GGUF",
        "filename": "nous-hermes-llama-2-7b.Q5_K_M.gguf",
    },
}

def get_available_models():
    """Return list of available model names."""
    return list(MODEL_CONFIGS.keys())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_load_parameters(
    *,
    logits_all: bool,
    load_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = {key: value for key, value in (load_overrides or {}).items() if value is not None}
    unexpected = sorted(set(overrides) - ALLOWED_LOAD_OVERRIDES)
    if unexpected:
        raise ValueError(f"Unsupported llama.cpp load overrides: {unexpected}")
    parameters = {**DEFAULT_PARAMS, **overrides, "logits_all": logits_all}
    for key in ("n_ctx", "n_batch", "n_ubatch"):
        if key in parameters and int(parameters[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(parameters["n_gpu_layers"]) < -1:
        raise ValueError("n_gpu_layers must be -1 or non-negative")
    if int(parameters.get("n_ubatch", parameters.get("n_batch", 512))) > int(
        parameters.get("n_batch", 512)
    ):
        raise ValueError("n_ubatch cannot exceed n_batch")
    return parameters


def resolve_model_artifact(
    model_name: str,
    *,
    logits_all: bool = False,
    load_overrides: dict[str, Any] | None = None,
    local_files_only: bool = False,
) -> tuple[Path, dict]:
    """Resolve a GGUF model and return its validated provenance.

    Governed runs pass ``local_files_only=True`` so model resolution fails
    closed when the pinned artifact is absent instead of contacting the Hub.
    Report data are never inputs to artifact resolution.
    """
    if model_name not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unsupported model '{model_name}'. Available: {available}")

    cfg = MODEL_CONFIGS[model_name]
    logging.info(f"Downloading model {model_name}...")
    model_path = Path(
        hf_hub_download(
            repo_id=cfg["repo_id"],
            filename=cfg["filename"],
            revision=cfg.get("revision"),
            local_files_only=local_files_only,
        )
    )
    model_sha256 = sha256_file(model_path)
    expected_sha256 = cfg.get("sha256")
    if expected_sha256 and model_sha256 != expected_sha256:
        raise ValueError(
            f"Model checksum mismatch for {model_name}: "
            f"expected {expected_sha256}, found {model_sha256}"
        )
    load_parameters = validated_load_parameters(
        logits_all=logits_all,
        load_overrides=load_overrides,
    )
    receipt = {
        "registry_name": model_name,
        "repo_id": cfg["repo_id"],
        "filename": cfg["filename"],
        "requested_revision": cfg.get("revision"),
        "huggingface_snapshot": model_path.parent.name,
        "sha256": model_sha256,
        "expected_sha256": expected_sha256,
        "size_bytes": model_path.stat().st_size,
        "load_parameters": load_parameters,
        "artifact_access": {
            "mode": "local_cache_only" if local_files_only else "hub_cache_allowed",
            "network_lookup_allowed": not local_files_only,
        },
    }
    return model_path, receipt


def download_model_with_receipt(
    model_name: str,
    *,
    logits_all: bool = False,
    load_overrides: dict[str, Any] | None = None,
    local_files_only: bool = False,
) -> tuple[Llama, dict]:
    """Download and load a GGUF model, returning immutable model provenance."""
    model_path, receipt = resolve_model_artifact(
        model_name,
        logits_all=logits_all,
        load_overrides=load_overrides,
        local_files_only=local_files_only,
    )
    logging.info("Loading model into llama.cpp...")
    started = time.perf_counter()
    model = Llama(model_path=str(model_path), **receipt["load_parameters"])
    receipt["model_load_elapsed_seconds"] = time.perf_counter() - started
    return model, receipt


def download_model(model_name: str) -> Llama:
    """Compatibility wrapper returning only the loaded model."""
    model, _receipt = download_model_with_receipt(model_name)
    return model
