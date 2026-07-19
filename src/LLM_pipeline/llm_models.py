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
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp.llama import Llama

# Default parameters for model loading
DEFAULT_PARAMS = {
    "n_ctx": 4096, # context window size, 4096 tokens are about 3000 words
    "n_gpu_layers": 30,
    "verbose": False,
}

# Model configurations - edit this to add new models
MODEL_CONFIGS = {
    "mistral": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
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


def download_model_with_receipt(model_name: str) -> tuple[Llama, dict]:
    """Download and load a GGUF model, returning immutable model provenance."""
    if model_name not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unsupported model '{model_name}'. Available: {available}")

    cfg = MODEL_CONFIGS[model_name]
    logging.info(f"Downloading model {model_name}...")
    model_path = Path(hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"]))
    receipt = {
        "registry_name": model_name,
        "repo_id": cfg["repo_id"],
        "filename": cfg["filename"],
        "huggingface_snapshot": model_path.parent.name,
        "sha256": sha256_file(model_path),
        "size_bytes": model_path.stat().st_size,
        "load_parameters": dict(DEFAULT_PARAMS),
    }
    logging.info("Loading model into llama.cpp...")
    return Llama(model_path=str(model_path), **DEFAULT_PARAMS), receipt


def download_model(model_name: str) -> Llama:
    """Compatibility wrapper returning only the loaded model."""
    model, _receipt = download_model_with_receipt(model_name)
    return model
