"""Preload and validate a pinned GGUF without allocating it in llama.cpp."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SOURCE = REPOSITORY_ROOT / "src" / "LLM_pipeline"
sys.path.insert(0, str(PIPELINE_SOURCE))

from llm_models import get_available_models, resolve_model_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/cache a pinned GGUF and verify its SHA-256 receipt."
    )
    parser.add_argument("--model", choices=get_available_models(), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    _model_path, receipt = resolve_model_artifact(args.model)
    destination = args.receipt.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
