"""Run one private, classification-only EEG compatibility probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SOURCE = REPOSITORY_ROOT / "src" / "LLM_pipeline"
sys.path.insert(0, str(PIPELINE_SOURCE))

from llm_models import download_model_with_receipt, get_available_models  # noqa: E402
from pipeline import (  # noqa: E402
    PROMPT_CLASSIFY,
    fetch_reports,
    llm_json_with_receipt,
    load_gbnf,
    sha256_file,
    sha256_text,
)

EXPECTED_KEYS = {
    "focal_epileptiform_activity",
    "generalized_epileptiform_activity",
    "focal_non_epileptiform_activity",
    "generalized_non_epileptiform_activity",
    "abnormality",
}


def private_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one report through the preserved classification prompt and grammar. "
            "No report text or source identifier is written to the receipt."
        )
    )
    parser.add_argument("--model", choices=get_available_models(), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "zoe_reports_sample.db",
    )
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve(strict=True)
    source_id, report = next(fetch_reports(dataset))
    grammar_path = PIPELINE_SOURCE / "result_grammar.gbnf"
    grammar = load_gbnf(grammar_path)
    model, model_receipt = download_model_with_receipt(args.model)
    call = llm_json_with_receipt(
        model=model,
        prompt=PROMPT_CLASSIFY + "\n\n" + report,
        temperature=0.0,
        max_tokens=256,
        stop=None,
        grammar=grammar,
        top_k=40,
        top_p=0.95,
    )
    parsed = json.loads(call.text)
    if set(parsed) != EXPECTED_KEYS:
        raise ValueError(f"Unexpected classification keys: {sorted(parsed)}")
    if any(not isinstance(value, int) or value not in {1, 2, 3, 4} for value in parsed.values()):
        raise ValueError(f"Unexpected classification values: {parsed}")

    receipt = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "candidate-classification-only-historical-prompt-compatibility",
        "interpretation": (
            "Runtime compatibility only; not a Vasily v5g reproduction and not a study estimate."
        ),
        "model": model_receipt,
        "input": {
            "dataset_filename": dataset.name,
            "dataset_sha256": sha256_file(dataset),
            "source_id_sha256": private_hash(source_id),
            "report_text_sha256": private_hash(report),
            "report_whitespace_words": len(report.split()),
        },
        "method": {
            "prompt_sha256": sha256_text(PROMPT_CLASSIFY),
            "grammar_filename": grammar_path.name,
            "grammar_sha256": sha256_file(grammar_path),
            "temperature": 0.0,
            "top_k": 40,
            "top_p": 0.95,
            "max_tokens": 256,
            "chat_template_applied": False,
        },
        "classification": parsed,
        "timing_and_tokens": {
            "elapsed_seconds": call.elapsed_seconds,
            "prompt_tokens": call.prompt_tokens,
            "completion_tokens": call.completion_tokens,
            "total_tokens": call.total_tokens,
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
