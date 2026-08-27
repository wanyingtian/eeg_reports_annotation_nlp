"""Load a pinned GGUF and record a deterministic chat/grammar smoke receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from llama_cpp import Llama, LlamaGrammar

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SOURCE = REPOSITORY_ROOT / "src" / "LLM_pipeline"
sys.path.insert(0, str(PIPELINE_SOURCE))

from llm_models import DEFAULT_PARAMS, get_available_models, resolve_model_artifact  # noqa: E402

PROBE_PROMPT = (
    'Return a JSON object with exactly one key named "status" and the string value "ok". '
    "Do not include other text."
)
PROBE_GRAMMAR = r'''root ::= "{\"status\":\"ok\"}"'''


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a pinned GGUF and verify chat-template plus GBNF execution."
    )
    parser.add_argument("--model", choices=get_available_models(), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    model_path, model_receipt = resolve_model_artifact(args.model)
    load_started = time.perf_counter()
    model = Llama(model_path=str(model_path), **DEFAULT_PARAMS)
    load_elapsed = time.perf_counter() - load_started

    template = str(model.metadata.get("tokenizer.chat_template", ""))
    grammar = LlamaGrammar.from_string(PROBE_GRAMMAR)
    inference_started = time.perf_counter()
    response = model.create_chat_completion(
        messages=[{"role": "user", "content": PROBE_PROMPT}],
        grammar=grammar,
        temperature=0.0,
        max_tokens=32,
    )
    inference_elapsed = time.perf_counter() - inference_started
    content = response["choices"][0]["message"]["content"]
    if content != '{"status":"ok"}':
        raise ValueError(f"Unexpected grammar-constrained probe output: {content!r}")

    receipt = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "runtime-load-chat-template-gbnf-smoke-test",
        "model": model_receipt,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "llama_cpp_python": package_version("llama-cpp-python"),
            "load_parameters": dict(DEFAULT_PARAMS),
            "load_elapsed_seconds": load_elapsed,
            "inference_elapsed_seconds": inference_elapsed,
        },
        "probe": {
            "prompt_sha256": sha256_text(PROBE_PROMPT),
            "grammar_sha256": sha256_text(PROBE_GRAMMAR),
            "embedded_chat_template_sha256": sha256_text(template) if template else None,
            "output": content,
            "usage": response.get("usage", {}),
        },
    }
    destination = args.receipt.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
