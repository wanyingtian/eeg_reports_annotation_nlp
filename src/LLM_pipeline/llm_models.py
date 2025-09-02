# Copyright (c) 2025 Wanying Tian
# Licensed under the Apache-2.0 License (see LICENSE file in the project root for details).
#!/usr/bin/env python3
"""
Simple Model Configuration Module

Edit MODEL_CONFIGS to add/modify models.
Edit DEFAULT_PARAMS to change default loading parameters.
"""

import logging
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

def download_model(model_name: str) -> Llama:
    """Download and load a GGUF model."""
    if model_name not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unsupported model '{model_name}'. Available: {available}")

    cfg = MODEL_CONFIGS[model_name]
    logging.info(f"Downloading model {model_name}...")
    model_path = hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"])
    logging.info("Loading model into llama.cpp...")
    return Llama(model_path=model_path, **DEFAULT_PARAMS)