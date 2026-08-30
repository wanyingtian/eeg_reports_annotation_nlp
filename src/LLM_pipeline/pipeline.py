# Copyright (c) 2025 Wanying Tian
# Licensed under the Apache-2.0 License (see LICENSE file in the project root for details).
# #!/usr/bin/env python3
"""
EEG Report Processing Pipeline (LLM-based)

- Loads EEG reports from a SQLite database.
- Runs a classification prompt and an explanation prompt against an LLM (llama.cpp).
- Writes results incrementally to versioned CSVs and emits a run-config summary.
- Resumes safely after crashes; uses atomic file writes to avoid corruption.
- Smart versioning system prevents duplicate work and manages configuration changes.

Usage (examples)
---------------

# Process with default sample dataset (zoe):
python pipeline.py --num-reports 10 --model mistral --dataset-id "zoe" 

# Resume from a previous CSV:
python pipeline.py --num-reports 10 --model mistral --dataset-id "zoe" --completed-csv /path/to/previous.csv 

# Process 10 reports with custom dataset identifier, custom datapath, and output directory:
python pipeline.py --num-reports 10 --model mistral --dataset-id "john_data"  --dataset-path /path/to/data.db --outdir ../../outputs/pipeline_output

# Greedy (as default, temp = 0)
python pipeline.py --num-reports 50 --model mistral --temperature 0
# Exploratory
python pipeline.py --num-reports 50 --model mistral --temperature 0.7 --top-k 40 --top-p 0.95

"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing as mp
import os
import platform
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Tuple

import pandas as pd
from llama_cpp.llama import Llama, LlamaGrammar
from llm_models import download_model_with_receipt, get_available_models

try:
    from eeg_review.logprob_adapter import (
        JSON_KEY_TO_LABEL,
        PROBABILITY_COLUMNS,
        extract_binary_core_positive_probabilities,
        extract_core_positive_probabilities,
    )
    from eeg_review.native_interface import (
        NATIVE_CHAT_INTERFACE_MODE,
        RAW_COMPLETION_INTERFACE_MODE,
        embedded_chat_template_receipt,
        native_classification_messages,
        native_task_message_template,
        sha256_text as native_sha256_text,
    )
except ModuleNotFoundError:
    # Preserve the historical `python src/LLM_pipeline/pipeline.py` entry point
    # for environments that installed requirements without installing the
    # repository package itself.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eeg_review.logprob_adapter import (
        JSON_KEY_TO_LABEL,
        PROBABILITY_COLUMNS,
        extract_binary_core_positive_probabilities,
        extract_core_positive_probabilities,
    )
    from eeg_review.native_interface import (
        NATIVE_CHAT_INTERFACE_MODE,
        RAW_COMPLETION_INTERFACE_MODE,
        embedded_chat_template_receipt,
        native_classification_messages,
        native_task_message_template,
        sha256_text as native_sha256_text,
    )

# --------------------------- Defaults / Constants --------------------------- #

# Resolve paths relative to the repo root (two levels up from this script)
BASE_DIR = Path(__file__).resolve().parent      # e.g., src/LLM_pipeline
REPO_ROOT = BASE_DIR.parents[1]                 # repo root

DEFAULT_OUTDIR = REPO_ROOT / "outputs/pipeline_output"
DEFAULT_DB = REPO_ROOT / "data/zoe_reports_sample.db"


# Model defaults
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 3000
DEFAULT_STOP: Optional[Iterable[str]] = None
CLASSIFICATION_LOGPROBS = 64
MAX_WORKER_RESTARTS = 3
HISTORICAL_FOUR_LEVEL_MODE = "historical_four_level"
BINARY_CORE_ADAPTER_MODE = "binary_core_certainty_adapter"

# Prompts (kept as provided)
PROMPT_CLASSIFY = r"""
Read the following EEG Report data carefully, then answer the following questions about the report. Use the provided definitions and examples as a guide for interpretation.
Remember to follow constraints.

Definitions and Examples:
1. Focal Epileptiform Activity:
Definition: Epileptiform discharges limited to a specific area, suggesting focal seizure activity.
Example: "Sharp waves in the right temporal region" or "focal spike activity seen in the left frontal lobe."

2. Generalized Epileptiform Activity:
Definition: Epileptiform discharges occurring simultaneously across both hemispheres, indicating generalized epilepsy.
Example: "Generalized spike-and-wave discharges at 3 Hz" or "bilateral synchronous polyspike bursts."

3. Focal Non-Epileptiform Activity:
Definition: Non-epileptic activity confined to a specific area, possibly indicating localized brain dysfunction.
Example: "Regional slowing in the left posterior quadrant" or "focal attenuation in the right frontal area."

4. Generalized Non-Epileptiform Activity:
Definition: Non-epileptic abnormalities broadly distributed over both hemispheres, suggesting systemic dysfunction.
Example: "Diffuse background slowing" or "generalized low-amplitude theta activity."

5. Abnormality:
Definition: Any deviation from normal EEG patterns, which could be epileptiform or non-epileptiform.
Example: "Abnormal interictal spikes in the temporal lobe," "persistent delta slowing in the frontal regions," or "asymmetric voltage attenuation."

Questions:
Is there Focal Epileptiform Activity present?
(options: 1 = Confident no, 2 = Low confidence no, 3 = Low confidence yes, 4 = Confident yes)
Is there Generalized Epileptiform Activity present?
(options: 1 = Confident no, 2 = Low confidence no, 3 = Low confidence yes, 4 = Confident yes)
Is there Focal Non-Epileptiform Activity present?
(options: 1 = Confident no, 2 = Low confidence no, 3 = Low confidence yes, 4 = Confident yes)
Is there Generalized Non-Epileptiform Activity present?
(options: 1 = Confident no, 2 = Low confidence no, 3 = Low confidence yes, 4 = Confident yes)
Is the EEG abnormal?
(options: 1 = Confident no, 2 = Low confidence no, 3 = Low confidence yes, 4 = Confident yes)

Constraints:
Err on the side of confident decisions. Use 1 or 4 whenever possible. Only use 2 or 3 if there is strong, unavoidable ambiguity.
Choose 2 or 3 sparingly, only when absolutely necessary.
If all of "Focal Epileptiform Activity," "Generalized Epileptiform Activity," "Focal Non-Epileptiform Activity," and "Generalized Non-Epileptiform Activity" are marked as normal (1 or 2), then "Abnormality" must also be marked as normal (1 or 2).
If any of "Focal Epileptiform Activity," "Generalized Epileptiform Activity," "Focal Non-Epileptiform Activity," or "Generalized Non-Epileptiform Activity" is marked as abnormal (3 or 4), then "Abnormality" must also be marked as abnormal (3 or 4).

Please provide the answers ONLY in the following JSON format:

{
  "focal_epileptiform_activity": "integer",
  "generalized_epileptiform_activity": "integer",
  "focal_non_epileptiform_activity": "integer",
  "generalized_non_epileptiform_activity": "integer",
  "abnormality": "integer"
}
Do not include any additional explanations or comments in the output.
"""

PROMPT_CLASSIFY_BINARY_CORE = (
    PROMPT_CLASSIFY.replace(
        "(options: 1 = Confident no, 2 = Low confidence no, 3 = Low confidence yes, "
        "4 = Confident yes)",
        "(binary options: 1 = No / core absent, 4 = Yes / core present)",
    )
    .replace(
        "Err on the side of confident decisions. Use 1 or 4 whenever possible. Only use 2 or 3 "
        "if there is strong, unavoidable ambiguity.\nChoose 2 or 3 sparingly, only when "
        "absolutely necessary.",
        "Make only the binary core decision at this stage. Certainty is assigned later by the "
        "preregistered token-probability mapping. Use only 1 or 4.",
    )
    .replace("marked as normal (1 or 2)", "marked as core absent (1)")
    .replace("must also be marked as normal (1 or 2)", "must also be marked as core absent (1)")
    .replace("marked as abnormal (3 or 4)", "marked as core present (4)")
    .replace("must also be marked as abnormal (3 or 4)", "must also be marked as core present (4)")
)

PROMPT_EXPLAIN = r"""
Read the following EEG Report and the corresponding classification output carefully. Your task is to generate a machine-readable JSON output that provides explanations for each classification by identifying and extracting verbatim phrases from the EEG report that contributed to each decision.

***Guidelines***
1. The output must strictly follow the JSON format provided below with no extra text.
2. Each category (Focal Epileptiform Activity, Generalized Epileptiform Activity, Focal Non-Epileptiform Activity, Generalized Non-Epileptiform Activity, and Abnormality) should include:
    - "decision": An integer taken from the classification output.
    - "reasons": A list of verbatim phrases from the EEG report that support the classification.
3.  Handle all quotation marks properly:
4.  Escape double quotes inside text (" → \") to prevent JSON parsing errors.
5. Preserve single quotes (') inside text as-is unless they cause formatting issues.
6. If the classification is:
    - 1 (Confident No) or 2 (Low Confidence No) → Extract phrases that indicate an absence of relevant findings.
    - 3 (Low Confidence Yes) or 4 (Confident Yes) → Extract phrases that explicitly support the presence of relevant findings.
7. DO NOT paraphrase or summarize. Only extract exact text from the EEG report.
8. If no relevant phrase is found in the report, return "No specific mention in the report."
9. Ensure the output is valid JSON with no extra text.

**Output Format**
json
{
  "focal_epileptiform_activity": {
    "decision": <integer>,
    "reasons": ["<escaped verbatim text>", "<escaped verbatim text>", ...]
  },
  "generalized_epileptiform_activity": {
    "decision": <integer>,
    "reasons": ["<escaped verbatim text>", "<escaped verbatim text>", ...]
  },
  "focal_non_epileptiform_activity": {
    "decision": <integer>,
    "reasons": ["<escaped verbatim text>", "<escaped verbatim text>", ...]
  },
  "generalized_non_epileptiform_activity": {
    "decision": <integer>,
    "reasons": ["<escaped verbatim text>", "<escaped verbatim text>", ...]
  },
  "abnormality": {
    "decision": <integer>,
    "reasons": ["<escaped verbatim text>", "<escaped verbatim text>", ...]
  }
}

Do not include any additional explanations, comments, or extraneous text outside of the required JSON format.

---

**Input Format:**
- This prompt will be followed by:
  1. The original EEG report.
  2. The classification output from the previous LLM response.

Process the input accordingly and generate the required structured JSON output.
"""

# ------------------------------- Data Classes ------------------------------ #

@dataclass(frozen=True)
class RunConfig:
    outdir: Path
    dataset_path: Path
    dataset_id: str
    model_name: str
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS
    stop: Optional[Iterable[str]] = DEFAULT_STOP
    comment: str = "LLM pipeline run"
    capture_classification_logprobs: bool = False
    classification_mode: str = HISTORICAL_FOUR_LEVEL_MODE
    run_explanations: bool = True
    runtime_profile_id: str = "llama-cpp-python-default"
    n_ctx: int | None = None
    n_gpu_layers: int | None = None
    n_batch: int | None = None
    n_ubatch: int | None = None
    n_threads: int | None = None
    n_threads_batch: int | None = None
    flash_attn: bool | None = None
    classification_interface: str = RAW_COMPLETION_INTERFACE_MODE


@dataclass(frozen=True)
class LLMCallReceipt:
    text: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    logprobs: dict[str, Any] | None


# ------------------------------ Logging Setup ------------------------------ #

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def git_receipt() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "worktree_dirty": None}


def summarize_telemetry(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    columns = [
        "report_whitespace_words",
        "classify_elapsed_seconds",
        "classify_prompt_tokens",
        "classify_completion_tokens",
        "explain_elapsed_seconds",
        "explain_prompt_tokens",
        "explain_completion_tokens",
    ]
    summary: dict[str, dict[str, float | int | None]] = {}
    for column in columns:
        source = frame[column] if column in frame.columns else pd.Series(dtype=float)
        values = pd.to_numeric(source, errors="coerce").dropna()
        if values.empty:
            summary[column] = {
                "count": 0,
                "mean": None,
                "median": None,
                "minimum": None,
                "maximum": None,
            }
        else:
            summary[column] = {
                "count": int(values.size),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
    return summary


# -------------------------- DB / File I/O Utilities ------------------------ #

def fetch_reports(db_path: Path) -> Generator[Tuple[str, str], None, None]:
    """
    Stream (Hashed_ReportURN, Report) rows from SQLite without mutating the source.
    The legacy ``Hashed ID`` column is selected as an alias in memory.
    """
    resolved = db_path.expanduser().resolve(strict=True)
    logging.info(f"Connecting read-only to DB: {resolved.name}")
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()

        # Inspect columns in 'reports' table
        cursor.execute("PRAGMA table_info(reports)")
        cols = {row[1] for row in cursor.fetchall()}

        if "Hashed_ReportURN" in cols:
            id_col = '"Hashed_ReportURN"'
            logging.info("Column already named 'Hashed_ReportURN'; proceeding.")
        elif "Hashed ID" in cols:
            id_col = '"Hashed ID"'
            logging.warning("Using legacy 'Hashed ID' column without altering the database.")
        else:
            raise RuntimeError("Neither 'Hashed_ReportURN' nor 'Hashed ID' found in 'reports' table.")

        cursor.execute(f'SELECT {id_col}, "Report" FROM reports')
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield str(row[0]), str(row[1])

    except sqlite3.Error as e:
        logging.error(f"SQLite error: {e}")
        raise
    finally:
        conn.close()


def atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write CSV atomically to prevent partial files on crash.
    Also removes the 'Report' column for privacy.
    """
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    # Remove Report column for privacy
    output_df = df.drop(columns=['Report'], errors='ignore')
    output_df.to_csv(tmp, index=False)
    tmp.replace(out_path)


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_dataset_name(dataset_path: Path) -> str:
    """Extract dataset name from file path (without extension)."""
    return dataset_path.stem


def get_user_input(prompt: str, valid_responses: list = None) -> str:
    """Get user input with validation."""
    while True:
        response = input(prompt).strip().lower()
        if valid_responses is None or response in valid_responses:
            return response
        print(f"Please enter one of: {', '.join(valid_responses)}")


def find_existing_files(outdir: Path, dataset_id: str, model: str, num_reports: Optional[int] = None, version: Optional[int]= None) -> list:
    """Find existing files matching the pattern raw_{dataset_id}_{model}_{num_reports}_v*_run*.csv"""
    if version and num_reports:
        pattern = f"raw_{dataset_id}_{model}_{num_reports}_v{version}_run*.csv"
    elif num_reports:
        pattern = f"raw_{dataset_id}_{model}_{num_reports}_v*_run*.csv"
    elif version:
        pattern = f"raw_{dataset_id}_{model}_*_v{version}_run*.csv"
    else:
        pattern = f"raw_{dataset_id}_{model}_*_v*_run*.csv"
    existing_files = list(outdir.glob(pattern))
    return sorted(existing_files)


def parse_filename(filename: str) -> dict:
    """Parse filename to extract version and run numbers."""
    pattern = r"raw_(.+)_(.+)_(\d+)_v(\d+)_run(\d+)\.csv"
    match = re.match(pattern, filename)
    if match:
        return {
            'dataset_id': match.group(1),
            'model': match.group(2), 
            'num_reports': int(match.group(3)),
            'version': int(match.group(4)),
            'run': int(match.group(5))
        }
    return {}


def determine_output_path(outdir: Path, dataset_id: str, model: str, num_reports: int) -> Tuple[Path, Path, Optional[Path]]:
    """
    Determine output paths with smart versioning logic.
    Prompts user to select from available previous files or start fresh.
    """
    ensure_outdir(outdir)
    existing_files = find_existing_files(outdir, dataset_id, model)

    # If no previous runs
    if not existing_files:
        results_path = outdir / f"raw_{dataset_id}_{model}_{num_reports}_v1_run1.csv"
        config_path = outdir / f"config_{dataset_id}_{model}_v1.json"
        return results_path, config_path, None

    print("\n Found previous runs:")
    for idx, f in enumerate(existing_files):
        print(f"  [{idx}] {f.name}")
    print(f"  [{len(existing_files)}] start fresh (no resume)")

    while True:
        try:
            choice = int(input("Select a file to resume from, using the same config, or choose 'start fresh': "))
            if 0 <= choice < len(existing_files):
                base_file = existing_files[choice]
                parsed = parse_filename(base_file.name)

                parsed_num_reports = parsed['num_reports']
                parsed_version = parsed['version']
                parsed_run = parsed['run']

                if num_reports == parsed_num_reports:
                    # Ask if user wants to overwrite
                    response = get_user_input(
                        "Overwrite the selected file? (yes/no): ",
                        ['yes', 'y', 'no', 'n']
                    )
                    if response in ['yes', 'y']:
                        results_path = base_file
                        config_path = outdir / f"config_{dataset_id}_{model}_v{parsed_version}.json"
                        return results_path, config_path, base_file
                    else:
                        new_run = parsed_run + 1
                        results_path = outdir / f"raw_{dataset_id}_{model}_{num_reports}_v{parsed_version}_run{new_run}.csv"
                        config_path = outdir / f"config_{dataset_id}_{model}_v{parsed_version}.json"
                        return results_path, config_path, base_file

                elif num_reports > parsed_num_reports:
                    # You’re requesting more reports than previously processed → new run
                    # Reuse version, find latest run number for this num_reports

                    latest = latest_file_csv(outdir, dataset_id, model, num_reports, parsed_version)
                    if latest:
                        latest_parsed = parse_filename(latest.name)
                        print(f" Latest file for {num_reports} reports at {parsed_version} is {latest.name}.")
                        new_run = latest_parsed['run'] + 1
                    else:
                        new_run = 1
                    results_path = outdir / f"raw_{dataset_id}_{model}_{num_reports}_v{parsed_version}_run{new_run}.csv"
                    config_path = outdir / f"config_{dataset_id}_{model}_v{parsed_version}.json"
                    return results_path, config_path, base_file
                else:
                    print(f" Requested {num_reports} reports, which is fewer than {parsed_num_reports} in the selected file.")
                    print("Please choose a file with the same or fewer reports, or start fresh.")
                    return determine_output_path(outdir, dataset_id, model, num_reports)



            elif choice == len(existing_files):
                # User chose the "Start fresh" entry — let them pick a config version to reuse, or None.
                all_raw = find_existing_files(outdir, dataset_id, model, None)

                # Collect unique versions from any existing raw files
                versions = sorted({
                    pf["version"]
                    for f in all_raw
                    for pf in [parse_filename(f.name)]
                    if pf
                })

                # Build a single-response menu (indices 0..N, where N == "None (start fresh)")
                print("\nAre you re-runing a previous config version? If so, select the config version that you reused, or choose None to start fresh:\n"
                "Please note: if you select previous versions, your current config should match the previous config to avoid confusion.\n")
                for idx, v in enumerate(versions):
                    print(f"  [{idx}] v{v}")
                none_index = len(versions)
                print(f"  [{none_index}] None (start fresh)")

                valid = [str(i) for i in range(none_index + 1)]
                sel = get_user_input("Your choice: ", valid)

                if int(sel) != none_index:  # Reuse selected version
                    selected_version = versions[int(sel)]
                    latest = latest_file_csv(outdir, dataset_id, model, num_reports, selected_version)
                    if latest:
                        latest_parsed = parse_filename(latest.name)
                        new_run = latest_parsed['run'] + 1
                    else:
                        new_run = 1

                    results_path = outdir / f"raw_{dataset_id}_{model}_{num_reports}_v{selected_version}_run{new_run}.csv"
                    # Keep the config filename for that version (create if missing later)
                    config_path  = outdir / f"config_{dataset_id}_{model}_v{selected_version}.json"
                    return results_path, config_path, None

                # None selected → start fresh with a new version
                new_version = (max(versions) + 1) if versions else 1
                results_path = outdir / f"raw_{dataset_id}_{model}_{num_reports}_v{new_version}_run1.csv"
                config_path  = outdir / f"config_{dataset_id}_{model}_v{new_version}.json"
                return results_path, config_path, None

  
        except (ValueError, IndexError):
            pass
        print("❌ Invalid input. Try again.")



def latest_file_csv(outdir: Path, dataset_id: str, model: str, num_reports: int, version:Optional[int]=None) -> Optional[Path]:
    """
    Return the latest CSV path if present, else None.
    """
    existing_files = find_existing_files(outdir, dataset_id, model, num_reports, version)
    return existing_files[-1] if existing_files else None


# ------------------------------- LLM Helpers ------------------------------- #

def load_gbnf(path: Path) -> LlamaGrammar:
    """
    Load a .gbnf grammar file.
    """
    if not path.exists():
        raise FileNotFoundError(f"GBNF not found: {path}")
    content = path.read_text()
    if not content.strip():
        raise ValueError(f"GBNF is empty: {path}")
    return LlamaGrammar.from_string(content)


def llm_json(
    model: Llama,
    prompt: str,
    temperature: float,
    max_tokens: int,
    stop: Optional[Iterable[str]],
    grammar: Optional[LlamaGrammar] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> str:
    """
    Invoke the LLM and return the raw text (expected to be JSON per grammar).
    """
    return llm_json_with_receipt(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        grammar=grammar,
        top_k=top_k,
        top_p=top_p,
    ).text


def llm_json_with_receipt(
    model: Llama,
    prompt: str,
    temperature: float,
    max_tokens: int,
    stop: Optional[Iterable[str]],
    grammar: Optional[LlamaGrammar] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    logprobs: int | None = None,
) -> LLMCallReceipt:
    """Invoke the LLM and retain timing and llama.cpp token accounting."""
    kwargs = {
        "grammar": grammar,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
    }
    # Only include sampling args if provided
    if top_k is not None:
        kwargs["top_k"] = top_k
    if top_p is not None:
        kwargs["top_p"] = top_p
    if logprobs is not None:
        kwargs["logprobs"] = logprobs

    started = time.perf_counter()
    resp = model(prompt, **kwargs)
    elapsed = time.perf_counter() - started
    usage = resp.get("usage", {})
    choice = resp["choices"][0]
    return LLMCallReceipt(
        text=choice["text"],
        elapsed_seconds=elapsed,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        logprobs=choice.get("logprobs"),
    )


def llm_chat_json_with_receipt(
    model: Llama,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    stop: Optional[Iterable[str]],
    grammar: Optional[LlamaGrammar] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> LLMCallReceipt:
    """Invoke the embedded model-native chat template and retain accounting."""
    kwargs: dict[str, Any] = {
        "messages": messages,
        "grammar": grammar,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
    }
    if top_k is not None:
        kwargs["top_k"] = top_k
    if top_p is not None:
        kwargs["top_p"] = top_p
    started = time.perf_counter()
    response = model.create_chat_completion(**kwargs)
    elapsed = time.perf_counter() - started
    usage = response.get("usage", {})
    choice = response["choices"][0]
    content = choice["message"]["content"]
    return LLMCallReceipt(
        text=content,
        elapsed_seconds=elapsed,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        logprobs=None,
    )



# ------------------------------ Core Pipeline ------------------------------ #

def process_completed_csv(
    path: Optional[Path],
    *,
    capture_classification_logprobs: bool = False,
    classification_mode: str = HISTORICAL_FOUR_LEVEL_MODE,
    run_explanations: bool = True,
    runtime_profile_id: str = "llama-cpp-python-default",
    classification_interface: str = RAW_COMPLETION_INTERFACE_MODE,
) -> Tuple[pd.DataFrame, set[str]]:
    """
    Load existing results to resume. Returns (df, set_of_hashed_ids).
    """
    cols = [
        "Hashed_ReportURN",
        "Report",
        "runtime_profile_id",
        "classification_interface_mode",
        "classifications",
        "explanations",
        "report_whitespace_words",
        "classify_elapsed_seconds",
        "classify_prompt_tokens",
        "classify_completion_tokens",
        "classify_total_tokens",
        "explain_elapsed_seconds",
        "explain_prompt_tokens",
        "explain_completion_tokens",
        "explain_total_tokens",
    ]
    if not run_explanations:
        cols.append("pipeline_execution_mode")
    if capture_classification_logprobs:
        cols.extend(PROBABILITY_COLUMNS)
    if classification_mode == BINARY_CORE_ADAPTER_MODE:
        cols.append("adaptation_classification_mode")
    base = pd.DataFrame(columns=cols)
    if not path:
        logging.info("No completed CSV supplied; starting fresh.")
        return base, set()
    if not path.exists():
        logging.warning(f"Completed CSV not found: {path}; starting fresh.")
        return base, set()

    try:
        df = pd.read_csv(path)
        if len(df) and "runtime_profile_id" not in df.columns:
            df["runtime_profile_id"] = "llama-cpp-python-default"
        if len(df) and "classification_interface_mode" not in df.columns:
            if classification_interface != RAW_COMPLETION_INTERFACE_MODE:
                raise ValueError(
                    "A native-chat run cannot resume an output lacking interface identity"
                )
            df["classification_interface_mode"] = RAW_COMPLETION_INTERFACE_MODE
        observed_interfaces = set(
            df.get("classification_interface_mode", pd.Series(dtype=str)).dropna()
        )
        if observed_interfaces and observed_interfaces != {classification_interface}:
            raise ValueError(
                "A resumed CSV cannot mix classification interfaces: "
                f"expected {classification_interface}, found {sorted(observed_interfaces)}"
            )
        observed_profiles = set(df.get("runtime_profile_id", pd.Series(dtype=str)).dropna())
        if observed_profiles and observed_profiles != {runtime_profile_id}:
            raise ValueError(
                "A resumed CSV cannot mix runtime profiles: "
                f"expected {runtime_profile_id}, found {sorted(observed_profiles)}"
            )
        execution_modes = (
            set(df["pipeline_execution_mode"].dropna().astype(str))
            if "pipeline_execution_mode" in df.columns
            else set()
        )
        expected_execution_mode = (
            "classification_and_explanations" if run_explanations else "classification_only"
        )
        if execution_modes and execution_modes != {expected_execution_mode}:
            raise ValueError(
                "A resumed CSV must carry the same classification/explanation execution mode"
            )
        if not execution_modes and not run_explanations and len(df):
            raise ValueError(
                "A classification-only run cannot resume an unmarked historical CSV"
            )
        # Normalize columns in case of prior version drift
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        if classification_mode == BINARY_CORE_ADAPTER_MODE:
            modes = set(df["adaptation_classification_mode"].dropna().astype(str))
            if modes != {BINARY_CORE_ADAPTER_MODE}:
                raise ValueError(
                    "Binary-core adaptation can resume only a CSV carrying the same mode marker"
                )
        completed = set(df["Hashed_ReportURN"].dropna().astype(str))
        logging.info(f"Loaded {len(completed)} completed reports from {path}")
        return df[cols].copy(), completed
    except ValueError:
        raise
    except Exception as e:
        logging.error(f"Failed to read completed CSV: {e}")
        return base, set()


def load_reports_df(
    dataset_path: Path, num_reports: int, exclude_hashes: set[str]
) -> pd.DataFrame:
    """
    Pull up to (num_reports - already_completed) reports not in exclude_hashes.
    If the requested total is already met, return 0 rows.
    """
    # How many NEW reports do we actually need?
    target = max(num_reports - len(exclude_hashes), 0)

    if target == 0:
        logging.info(
            f"No new reports needed: requested {num_reports}, already completed {len(exclude_hashes)}."
        )
        return pd.DataFrame(columns=["Hashed_ReportURN", "Report"])

    rows = []
    for hid, rep in fetch_reports(dataset_path):
        if str(hid) in exclude_hashes:
            continue
        # stop BEFORE appending if we've reached target
        if len(rows) >= target:
            break
        rows.append((str(hid), rep))

    df = pd.DataFrame(rows, columns=["Hashed_ReportURN", "Report"])
    logging.info(
        f"Loaded {len(df)} pending reports from {dataset_path} "
        f"(target {target}, requested {num_reports}, skipped {len(exclude_hashes)})"
    )
    return df


def run_pipeline(
    model: Llama,
    model_receipt: dict[str, Any],
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    grammar_classify: LlamaGrammar,
    grammar_explain: LlamaGrammar,
    out_results: Path,
    out_config: Path,
    cfg: RunConfig,
    flush_every: int = 5,
) -> pd.DataFrame:
    """
    Iterate over reports, call LLM, and append results. Flush to disk regularly.
    """
    start = time.time()
    execution_started_at_utc = datetime.now(UTC).isoformat()
    execution_git = git_receipt()
    if cfg.classification_mode not in {HISTORICAL_FOUR_LEVEL_MODE, BINARY_CORE_ADAPTER_MODE}:
        raise ValueError(f"Unsupported classification mode: {cfg.classification_mode}")
    if cfg.classification_mode == BINARY_CORE_ADAPTER_MODE and not (
        cfg.capture_classification_logprobs
    ):
        raise ValueError("Binary-core adaptation requires classification log-probability capture")
    if cfg.classification_interface not in {
        RAW_COMPLETION_INTERFACE_MODE,
        NATIVE_CHAT_INTERFACE_MODE,
    }:
        raise ValueError(f"Unsupported classification interface: {cfg.classification_interface}")
    if (
        cfg.classification_interface == NATIVE_CHAT_INTERFACE_MODE
        and cfg.capture_classification_logprobs
    ):
        raise ValueError("Native-chat sensitivity does not support log-probability capture")
    classification_prompt = (
        PROMPT_CLASSIFY_BINARY_CORE
        if cfg.classification_mode == BINARY_CORE_ADAPTER_MODE
        else PROMPT_CLASSIFY
    )
    logging.info(f"Starting pipeline on {len(df)} reports; existing {len(results_df)} completed.")

    for idx, row in df.iterrows():
        hashed_id = str(row["Hashed_ReportURN"])
        report = str(row["Report"])

        # 1) Classification
        if cfg.classification_interface == NATIVE_CHAT_INTERFACE_MODE:
            classification_call = llm_chat_json_with_receipt(
                model=model,
                messages=native_classification_messages(classification_prompt, report),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                stop=cfg.stop,
                grammar=grammar_classify,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )
        else:
            classify_prompt = classification_prompt + "\n\n" + report
            classification_call = llm_json_with_receipt(
                model=model,
                prompt=classify_prompt,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                stop=cfg.stop,
                grammar=grammar_classify,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                logprobs=(
                    CLASSIFICATION_LOGPROBS
                    if cfg.capture_classification_logprobs
                    else None
                ),
            )
        classifications = classification_call.text
        if cfg.capture_classification_logprobs:
            extractor = (
                extract_binary_core_positive_probabilities
                if cfg.classification_mode == BINARY_CORE_ADAPTER_MODE
                else extract_core_positive_probabilities
            )
            classification_probabilities = extractor(
                classifications,
                classification_call.logprobs,
            )
        else:
            classification_probabilities = {}

        # 2) Explanations (feed classification JSON verbatim), unless a
        # separately receipted comparator is classification-only.
        if cfg.run_explanations:
            explain_input = (
                PROMPT_EXPLAIN
                + "\n\n---\nEEG Report:\n"
                + report
                + "\n\nClassification JSON:\n"
                + classifications
                + "\n"
            )
            explanation_call = llm_json_with_receipt(
                model=model,
                prompt=explain_input,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                stop=cfg.stop,
                grammar=grammar_explain,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )
            explanations = explanation_call.text
        else:
            explanation_call = LLMCallReceipt(
                text="",
                elapsed_seconds=0.0,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                logprobs=None,
            )
            explanations = ""

        # Append row
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    [
                        {
                            "Hashed_ReportURN": hashed_id,
                            "runtime_profile_id": cfg.runtime_profile_id,
                            "classification_interface_mode": cfg.classification_interface,
                            "classifications": classifications,
                            "explanations": explanations,
                            "report_whitespace_words": len(report.split()),
                            "classify_elapsed_seconds": classification_call.elapsed_seconds,
                            "classify_prompt_tokens": classification_call.prompt_tokens,
                            "classify_completion_tokens": classification_call.completion_tokens,
                            "classify_total_tokens": classification_call.total_tokens,
                            "explain_elapsed_seconds": explanation_call.elapsed_seconds,
                            "explain_prompt_tokens": explanation_call.prompt_tokens,
                            "explain_completion_tokens": explanation_call.completion_tokens,
                            "explain_total_tokens": explanation_call.total_tokens,
                            **(
                                {"pipeline_execution_mode": "classification_only"}
                                if not cfg.run_explanations
                                else {}
                            ),
                            **{
                                f"Prob_{label}": probability
                                for label, probability in classification_probabilities.items()
                            },
                            **(
                                {"adaptation_classification_mode": cfg.classification_mode}
                                if cfg.classification_mode == BINARY_CORE_ADAPTER_MODE
                                else {}
                            ),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        # Periodic flush
        if (idx + 1) % flush_every == 0:
            logging.info(f"[{idx+1}/{len(df)}] Flushing results to {out_results.name}")
            atomic_write_csv(results_df, out_results)

    # Final write + config dump
    atomic_write_csv(results_df, out_results)
    elapsed = time.time() - start
    
    grammar_classify_path = BASE_DIR / (
        "result_grammar_binary_core.gbnf"
        if cfg.classification_mode == BINARY_CORE_ADAPTER_MODE
        else "result_grammar.gbnf"
    )
    grammar_explain_path = BASE_DIR / "result_grammar_exp.gbnf"
    dataset_path = cfg.dataset_path.expanduser().resolve(strict=True)

    # Versioned, audit-ready run receipt. The governed path itself is omitted.
    native_template = (
        embedded_chat_template_receipt(model)
        if cfg.classification_interface == NATIVE_CHAT_INTERFACE_MODE
        else None
    )
    task_message_template = native_task_message_template(classification_prompt)
    config_data = {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "execution_started_at_utc": execution_started_at_utc,
        "dataset": {
            "id": cfg.dataset_id,
            "filename": dataset_path.name,
            "sha256": sha256_file(dataset_path),
        },
        "model": model_receipt,
        "runtime_profile_id": cfg.runtime_profile_id,
        "sampling": {
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            "stop_sequences": list(cfg.stop) if cfg.stop else None,
        },
        "calibration_instrumentation": {
            "enabled": cfg.capture_classification_logprobs,
            "classification_mode": cfg.classification_mode,
            "completion_logprobs_requested": (
                CLASSIFICATION_LOGPROBS if cfg.capture_classification_logprobs else None
            ),
            "feature_definition": (
                (
                    "P(core present token 4) normalized over explicit binary grammar-token "
                    "alternatives {1,4}"
                    if cfg.classification_mode == BINARY_CORE_ADAPTER_MODE
                    else "P(level in {3,4}) normalized over explicit grammar-constrained "
                    "level-token alternatives {1,2,3,4}"
                )
                if cfg.capture_classification_logprobs
                else None
            ),
            "probability_columns": (
                list(PROBABILITY_COLUMNS) if cfg.capture_classification_logprobs else []
            ),
            "available_records_by_label": (
                {
                    label: int(
                        pd.to_numeric(results_df[f"Prob_{label}"], errors="coerce").notna().sum()
                    )
                    for label in JSON_KEY_TO_LABEL.values()
                }
                if cfg.capture_classification_logprobs
                else {}
            ),
            "interpretation": (
                "Model token probability for the grammar decision, not human certainty and not "
                "a calibrated probability until a development-only calibration receipt exists."
                if cfg.capture_classification_logprobs
                else "not captured"
            ),
        },
        "comment": cfg.comment,
        "reports_completed": len(results_df),
        # A resumed run can contain rows produced by several worker attempts.
        # Keep the legacy field for compatibility, but label its scope and
        # separately report the accumulated model-call time from all rows.
        "elapsed_seconds": elapsed,
        "attempt_elapsed_seconds": elapsed,
        "inference_seconds_total": float(
            pd.to_numeric(results_df["classify_elapsed_seconds"], errors="coerce").sum()
            + pd.to_numeric(results_df["explain_elapsed_seconds"], errors="coerce").sum()
        ),
        "prompts": {
            "classify": {
                "sha256": sha256_text(classification_prompt),
                "text": classification_prompt,
            },
            "explain": {"sha256": sha256_text(PROMPT_EXPLAIN), "text": PROMPT_EXPLAIN},
        },
        "grammars": {
            "classify": {
                "filename": grammar_classify_path.name,
                "sha256": sha256_file(grammar_classify_path),
            },
            "explain": {
                "filename": grammar_explain_path.name,
                "sha256": sha256_file(grammar_explain_path),
                "executed": cfg.run_explanations,
            },
        },
        "input_policy": {
            "report_field": "Report",
            "context_limit": model_receipt["load_parameters"].get("n_ctx"),
            "truncation": "none; context-limit errors are surfaced",
            "classification_interface_mode": cfg.classification_interface,
            "classification_chat_template": (
                "GGUF embedded tokenizer.chat_template applied to one user message"
                if cfg.classification_interface == NATIVE_CHAT_INTERFACE_MODE
                else "raw prompt concatenated with two newlines and report"
            ),
            "embedded_chat_template": native_template,
            "task_message_template": {
                "sha256": native_sha256_text(task_message_template),
                "text": task_message_template,
            },
            "explanation_chat_template": (
                "raw prompt plus report and classification JSON"
                if cfg.run_explanations
                else "not executed"
            ),
        },
        "execution_surface": {
            "classification": True,
            "explanations": cfg.run_explanations,
        },
        "telemetry": summarize_telemetry(results_df),
        "output": {"filename": out_results.name, "sha256": sha256_file(out_results)},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                "huggingface-hub": package_version("huggingface-hub"),
                "llama-cpp-python": package_version("llama-cpp-python"),
                "pandas": package_version("pandas"),
            },
            "git": execution_git,
            "receipt_write_git": git_receipt(),
            "slurm_job_id_present": bool(os.getenv("SLURM_JOB_ID")),
            "cuda_visible_devices_set": "CUDA_VISIBLE_DEVICES" in os.environ,
        },
        "provenance_limits": [
            "This receipt identifies the executed prompt but does not reconstruct historical prompt-development decisions.",
            "Dataset checksum and ID do not establish cohort eligibility, patient independence, or ethics coverage.",
        ],
    }

    run_receipt = out_results.with_suffix(".run.json")
    for destination in (out_config, run_receipt):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(config_data, stream, indent=2)
            stream.write("\n")
        temporary.replace(destination)
    
    logging.info(f"Saved results -> {out_results}")
    logging.info(f"Saved config  -> {out_config}")
    logging.info(f"Saved receipt -> {run_receipt}")
    logging.info(f"Elapsed: {elapsed:.2f}s")
    return results_df


# --------------------------- Crash-Resistant Runner ------------------------- #

def worker_target(
    resume_csv: Optional[Path],
    num_reports: int,
    dataset_path: Path,
    model_name: str,
    out_results: Path,
    out_config: Path,
    cfg: RunConfig,
    flush_every: int,
) -> None:
    """Run one supervised inference attempt in a spawn-safe worker process."""
    grammar_classify = load_gbnf(
        BASE_DIR
        / (
            "result_grammar_binary_core.gbnf"
            if cfg.classification_mode == BINARY_CORE_ADAPTER_MODE
            else "result_grammar.gbnf"
        )
    )
    grammar_explain = load_gbnf(BASE_DIR / "result_grammar_exp.gbnf")

    load_overrides = {
        "n_ctx": cfg.n_ctx,
        "n_gpu_layers": cfg.n_gpu_layers,
        "n_batch": cfg.n_batch,
        "n_ubatch": cfg.n_ubatch,
        "n_threads": cfg.n_threads,
        "n_threads_batch": cfg.n_threads_batch,
        "flash_attn": cfg.flash_attn,
    }
    model, model_receipt = download_model_with_receipt(
        model_name,
        logits_all=cfg.capture_classification_logprobs,
        load_overrides=load_overrides,
    )
    model_receipt["runtime_profile_id"] = cfg.runtime_profile_id

    # (Re)load completed and pending.
    prior_df, prior_hashes = process_completed_csv(
        resume_csv,
        capture_classification_logprobs=cfg.capture_classification_logprobs,
        classification_mode=cfg.classification_mode,
        run_explanations=cfg.run_explanations,
        runtime_profile_id=cfg.runtime_profile_id,
        classification_interface=cfg.classification_interface,
    )
    logging.info(f"Initial completed count: {len(prior_hashes)}")
    pending = load_reports_df(dataset_path, num_reports, prior_hashes)

    if len(pending) == 0:
        logging.info("No pending reports to process. Finalizing the existing result receipt.")

    run_pipeline(
        model=model,
        model_receipt=model_receipt,
        df=pending,
        results_df=prior_df,
        grammar_classify=grammar_classify,
        grammar_explain=grammar_explain,
        out_results=out_results,
        out_config=out_config,
        cfg=cfg,
        flush_every=flush_every,
    )


def manager(
    num_reports: int,
    completed_csv: Optional[Path],
    dataset_id: str,
    dataset_path: Path,
    model_name: str,
    cfg: RunConfig,
    output_csv: Optional[Path] = None,
    resume_output: bool = False,
    flush_every: int = 5,
) -> None:
    """
    Supervises the run. If a worker crashes, it restarts and resumes from the
    latest versioned CSV.
    """
    if flush_every < 1:
        raise ValueError("flush_every must be at least 1")

    # An explicit output path is the non-interactive contract used by the
    # resumable study supervisor. The legacy smart-versioning UI remains the
    # default for interactive/manual runs.
    if output_csv is not None:
        out_results = output_csv.expanduser().resolve()
        ensure_outdir(out_results.parent)
        out_config = out_results.with_suffix(".config.json")
        auto_completed_csv = out_results if resume_output and out_results.exists() else None
        if out_results.exists() and not (resume_output or completed_csv):
            raise FileExistsError(
                f"Explicit output already exists: {out_results}. "
                "Use --resume-output or --completed-csv to continue it."
            )
        version = None
    else:
        out_results, out_config, auto_completed_csv = determine_output_path(
            cfg.outdir, dataset_id, model_name, num_reports
        )
        version = parse_filename(out_results.name).get("version", 1)
    
    # Priority: explicit --completed-csv > auto-detected from versioning > None
    effective_completed_csv = completed_csv or auto_completed_csv
    
    if effective_completed_csv:
        logging.info(f"Using completed CSV for resume: {effective_completed_csv}")
    
    # # preload completed (support resume)
    # existing_df, completed_hashes = process_completed_csv(effective_completed_csv)
    # logging.info(f"Initial completed count: {len(completed_hashes)}")

    # run loop
    resume_path = effective_completed_csv
    crashes = 0
    while True:
        proc = mp.Process(
            target=worker_target,
            args=(
                resume_path,
                num_reports,
                dataset_path,
                model_name,
                out_results,
                out_config,
                cfg,
                flush_every,
            ),
        )
        proc.start()
        proc.join()

        if proc.exitcode == 0:
            logging.info("Pipeline completed successfully.")
            break

        crashes += 1
        logging.error(
            "Worker crashed (exit %s): failure %s of %s.",
            proc.exitcode,
            crashes,
            MAX_WORKER_RESTARTS,
        )
        # Find latest CSV in our naming scheme to resume
        if output_csv is not None:
            resume_path = out_results if out_results.exists() else effective_completed_csv
        else:
            latest = latest_file_csv(
                cfg.outdir, dataset_id, model_name, num_reports, version
            )
            resume_path = latest if latest else effective_completed_csv
        # Write crash breadcrumb
        crash_log = cfg.outdir / "crash_report.txt"
        with open(crash_log, "a") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"[{ts}] Crash #{crashes}, exit={proc.exitcode}, resume={resume_path}\n")

        if crashes >= MAX_WORKER_RESTARTS:
            raise RuntimeError(
                f"Worker failed {crashes} consecutive times; refusing an unbounded restart loop. "
                "Inspect the final traceback and crash_report.txt before resuming."
            )
        logging.info("Restarting worker with the latest available partial CSV.")


# ----------------------------------- CLI ----------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process EEG reports with an LLM.")
    p.add_argument("--num-reports", type=int, required=True, help="Required: Number of reports to run.")
    p.add_argument("--completed-csv", type=Path, default=None, help="Optional: Path to an existing results CSV to resume from.")
    p.add_argument("--dataset-id", type=str, default=None, help='Optional: Dataset identifier (e.g., "zoe", "johns_data"). If not provided, uses dataset filename.')
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DB, help="Optional: Path to the dataset SQLite file. If not provided, uses default sample dataset.")    
    p.add_argument(
        "--model",
        type=str,
        choices=get_available_models(),  # Dynamic model list
        default="mistral",
        help="Model to use (GGUF). If not provided, defaults to 'mistral'.",
    )
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Optional: Directory to write outputs. Defaults to ./outputs/pipeline_output")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Explicit non-interactive output CSV. Intended for supervised jobs; "
            "bypasses the legacy version-selection prompts."
        ),
    )
    p.add_argument(
        "--resume-output",
        action="store_true",
        help="Resume --output-csv in place when it already exists.",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=5,
        help="Atomically checkpoint after this many new reports (default: 5).",
    )
    p.add_argument(
        "--classification-only",
        action="store_true",
        help=(
            "Run the five-label classification stage without the explanation stage. "
            "This is intended for separately receipted comparator evaluations; the default "
            "historical pipeline remains unchanged."
        ),
    )
    p.add_argument("--comment", type=str, default="LLM pipeline run", help="Optional: comment to save in config.")
    p.add_argument(
        "--runtime-profile-id",
        default="llama-cpp-python-default",
        help="Stable identifier recorded for the llama.cpp execution profile.",
    )
    p.add_argument(
        "--classification-interface",
        choices=[RAW_COMPLETION_INTERFACE_MODE, NATIVE_CHAT_INTERFACE_MODE],
        default=RAW_COMPLETION_INTERFACE_MODE,
        help=(
            "Use historical raw completion or the separately preregistered embedded "
            "model-native chat interface."
        ),
    )
    p.add_argument("--n-ctx", type=int, default=None)
    p.add_argument("--n-gpu-layers", type=int, default=None)
    p.add_argument("--n-batch", type=int, default=None)
    p.add_argument("--n-ubatch", type=int, default=None)
    p.add_argument("--n-threads", type=int, default=None)
    p.add_argument("--n-threads-batch", type=int, default=None)
    p.add_argument(
        "--flash-attn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly enable or disable llama.cpp Flash Attention.",
    )
    # sampling controls
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Optional: Sampling temperature (0 for greedy). Defaults to 0.")
    p.add_argument("--top-k", dest="top_k", type=int, default=DEFAULT_TOP_K, help="Optional: Top-k sampling cutoff. Defaults to 40.")
    p.add_argument("--top-p", dest="top_p", type=float, default=DEFAULT_TOP_P, help="Optional: Top-p (nucleus) sampling threshold. Defaults to 0.95.")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Optional: Max new tokens to generate. Defaults to 3000.")
    p.add_argument(
        "--capture-classification-logprobs",
        action="store_true",
        help=(
            "Capture grammar-decision token probabilities for a separately governed "
            "development-only calibration run. Disabled for historical reproduction."
        ),
    )
    p.add_argument(
        "--classification-mode",
        choices=[HISTORICAL_FOUR_LEVEL_MODE, BINARY_CORE_ADAPTER_MODE],
        default=HISTORICAL_FOUR_LEVEL_MODE,
        help=(
            "Use the historical direct four-level interface or the preregistered binary-core "
            "interface for post-hoc certainty mapping. Historical mode is the default."
        ),
    )
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # Handle dataset path and ID logic
    if args.dataset_path:
        dataset_path = args.dataset_path
        # Use provided dataset-id, or fallback to dataset filename
        if args.dataset_id:
            dataset_id = args.dataset_id
        else:
            dataset_id = extract_dataset_name(dataset_path)
    

    logging.info(f"Using dataset: {dataset_path} with identifier: {dataset_id}")

    capture_classification_logprobs = (
        args.capture_classification_logprobs
        or args.classification_mode == BINARY_CORE_ADAPTER_MODE
    )
    cfg = RunConfig(
        outdir=args.outdir,
        dataset_path=dataset_path,
        dataset_id=dataset_id,
        model_name=args.model,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        comment=args.comment,
        capture_classification_logprobs=capture_classification_logprobs,
        classification_mode=args.classification_mode,
        run_explanations=not args.classification_only,
        runtime_profile_id=args.runtime_profile_id,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        n_threads=args.n_threads,
        n_threads_batch=args.n_threads_batch,
        flash_attn=args.flash_attn,
        classification_interface=args.classification_interface,
    )

    # Helpful env overrides recorded in config output
    os.environ["MODEL_OVERRIDE"] = args.model
    os.environ["DATASET_ID_OVERRIDE"] = dataset_id

    manager(
        num_reports=args.num_reports,
        completed_csv=args.completed_csv,
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        model_name=args.model,
        cfg=cfg,
        output_csv=args.output_csv,
        resume_output=args.resume_output,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
