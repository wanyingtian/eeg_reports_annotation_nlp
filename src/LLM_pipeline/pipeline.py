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
import json
import logging
import multiprocessing as mp
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download
from llama_cpp.llama import Llama, LlamaGrammar
from llm_models import download_model, get_available_models

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
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS
    stop: Optional[Iterable[str]] = DEFAULT_STOP
    comment: str = "LLM pipeline run"


# ------------------------------ Logging Setup ------------------------------ #

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# -------------------------- DB / File I/O Utilities ------------------------ #

def fetch_reports(db_path: Path) -> Generator[Tuple[str, str], None, None]:
    """
    Stream (Hashed_ReportURN, Report) rows from the SQLite database.
    If the DB still uses 'Hashed ID', rename it to 'Hashed_ReportURN' in-place.
    """
    logging.info(f"Connecting to DB: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        # Inspect columns in 'reports' table
        cursor.execute("PRAGMA table_info(reports)")
        cols = {row[1] for row in cursor.fetchall()}

        if "Hashed_ReportURN" in cols:
            id_col = '"Hashed_ReportURN"'
            logging.info("Column already named 'Hashed_ReportURN'; proceeding.")
        elif "Hashed ID" in cols:
            # Try to rename in place
            try:
                logging.info("Renaming 'Hashed ID' -> 'Hashed_ReportURN' in DB...")
                cursor.execute('ALTER TABLE reports RENAME COLUMN "Hashed ID" TO "Hashed_ReportURN";')
                conn.commit()
                id_col = '"Hashed_ReportURN"'
                logging.info("✓ Renamed and saved to the same database file.")
            except sqlite3.OperationalError as e:
                # Older SQLite might not support RENAME COLUMN; fall back to alias
                logging.warning(f"Could not rename column in-place ({e}). Proceeding with alias.")
                id_col = '"Hashed ID"'
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
                        f"Overwrite the selected file? (yes/no): ",
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
                        print(f"⚠️ Latest file for {num_reports} reports at {parsed_version} is {latest.name}.")
                        new_run = latest_parsed['run'] + 1
                    else:
                        new_run = 1
                    results_path = outdir / f"raw_{dataset_id}_{model}_{num_reports}_v{parsed_version}_run{new_run}.csv"
                    config_path = outdir / f"config_{dataset_id}_{model}_v{parsed_version}.json"
                    return results_path, config_path, base_file
                else:
                    print(f"⚠️ Requested {num_reports} reports, which is fewer than {parsed_num_reports} in the selected file.")
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
                print("\nAre you re-runing a previous config version? Select a config version to reuse, or choose None to start fresh:")
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

    resp = model(prompt, **kwargs)
    return resp["choices"][0]["text"]



# ------------------------------ Core Pipeline ------------------------------ #

def process_completed_csv(path: Optional[Path]) -> Tuple[pd.DataFrame, set[str]]:
    """
    Load existing results to resume. Returns (df, set_of_hashed_ids).
    """
    cols = ["Hashed_ReportURN", "Report", "classifications", "explanations"]
    base = pd.DataFrame(columns=cols)
    if not path:
        logging.info("No completed CSV supplied; starting fresh.")
        return base, set()
    if not path.exists():
        logging.warning(f"Completed CSV not found: {path}; starting fresh.")
        return base, set()

    try:
        df = pd.read_csv(path)
        # Normalize columns in case of prior version drift
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        completed = set(df["Hashed_ReportURN"].dropna().astype(str))
        logging.info(f"Loaded {len(completed)} completed reports from {path}")
        return df[cols].copy(), completed
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
    logging.info(f"Starting pipeline on {len(df)} reports; existing {len(results_df)} completed.")

    for idx, row in df.iterrows():
        hashed_id = str(row["Hashed_ReportURN"])
        report = str(row["Report"])

        # 1) Classification
        classify_prompt = PROMPT_CLASSIFY + "\n\n" + report
        classifications = llm_json(
            model=model,
            prompt=classify_prompt,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop,
            grammar=grammar_classify,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )

        # 2) Explanations (feed classification JSON verbatim)
        explain_input = (
            PROMPT_EXPLAIN
            + "\n\n---\nEEG Report:\n"
            + report
            + "\n\nClassification JSON:\n"
            + classifications
            + "\n"
        )
        explanations = llm_json(
            model=model,
            prompt=explain_input,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop,
            grammar=grammar_explain,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )

        # Append row
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    [
                        {
                            "Hashed_ReportURN": hashed_id,
                            "classifications": classifications,
                            "explanations": explanations,
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
    
    # Save config as JSON for better structure
    config_data = {
        "dataset_id": cfg.dataset_id,
        "dataset_path": str(cfg.dataset_path),
        "model": os.getenv('MODEL_OVERRIDE', ''),
        "temperature": cfg.temperature,
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
        "stop_sequences": list(cfg.stop) if cfg.stop else None,
        "comment": cfg.comment,
        # "reports_completed": len(results_df),
        # "elapsed_seconds": elapsed,
        # "elapsed_minutes": elapsed / 60,
        "prompts": {
            "classify": PROMPT_CLASSIFY,
            "explain": PROMPT_EXPLAIN
        },
        # "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    
    with open(out_config, "w") as f:
        json.dump(config_data, f, indent=2)
    
    logging.info(f"Saved results -> {out_results}")
    logging.info(f"Saved config  -> {out_config}")
    logging.info(f"Elapsed: {elapsed:.2f}s")
    return results_df


# --------------------------- Crash-Resistant Runner ------------------------- #

def manager(
    num_reports: int,
    completed_csv: Optional[Path],
    dataset_id: str,
    dataset_path: Path,
    model_name: str,
    cfg: RunConfig,
) -> None:
    """
    Supervises the run. If a worker crashes, it restarts and resumes from the
    latest versioned CSV.
    """
    # Determine output paths and whether to use existing file as completed CSV
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

    # inner worker target
    def worker_target(resume_csv: Optional[Path]) -> None:
        grammar_classify = load_gbnf(BASE_DIR / "result_grammar.gbnf")
        grammar_explain = load_gbnf(BASE_DIR / "result_grammar_exp.gbnf")
    
        model = download_model(model_name)

        # (Re)load completed and pending
        prior_df, prior_hashes = process_completed_csv(resume_csv)
        logging.info(f"Initial completed count: {len(prior_hashes)}")
        pending = load_reports_df(dataset_path, num_reports, prior_hashes)
        
        if len(pending) == 0:
            logging.info("No pending reports to process. All reports already completed.")
            return
            
        results_df = run_pipeline(
            model=model,
            df=pending,
            results_df=prior_df,
            grammar_classify=grammar_classify,
            grammar_explain=grammar_explain,
            out_results=out_results,
            out_config=out_config,
            cfg=cfg,
            flush_every=5,
        )

    # run loop
    resume_path = effective_completed_csv
    crashes = 0
    while True:
        proc = mp.Process(target=worker_target, args=(resume_path,))
        proc.start()
        proc.join()

        if proc.exitcode == 0:
            logging.info("Pipeline completed successfully.")
            break

        crashes += 1
        logging.error(f"Worker crashed (exit {proc.exitcode}). Restarting (attempt {crashes})...")
        # Find latest CSV in our naming scheme to resume
        latest = latest_file_csv(cfg.outdir, dataset_id, model_name, num_reports,version)
        resume_path = latest if latest else effective_completed_csv
        # Write crash breadcrumb
        crash_log = cfg.outdir / "crash_report.txt"
        with open(crash_log, "a") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"[{ts}] Crash #{crashes}, exit={proc.exitcode}, resume={resume_path}\n")


# ----------------------------------- CLI ----------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process EEG reports with an LLM.")
    p.add_argument("--num-reports", type=int, required=True, help="Number of reports to process.")
    p.add_argument("--completed-csv", type=Path, default=None, help="Path to an existing results CSV to resume from.")
    p.add_argument("--dataset-id", type=str, default=None, help='Dataset identifier (e.g., "zoe", "johns_data"). If not provided, uses dataset filename.')
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DB, help="Path to the dataset SQLite file.")    
    p.add_argument(
        "--model",
        type=str,
        choices=get_available_models(),  # Dynamic model list
        default="mistral",
        help="Model to use (GGUF).",
    )
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Directory to write outputs.")
    p.add_argument("--comment", type=str, default="LLM pipeline run", help="Run comment to save in config.")
    # sampling controls
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature (0 for greedy).")
    p.add_argument("--top-k", dest="top_k", type=int, default=DEFAULT_TOP_K, help="Top-k sampling cutoff.")
    p.add_argument("--top-p", dest="top_p", type=float, default=DEFAULT_TOP_P, help="Top-p (nucleus) sampling threshold.")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max new tokens to generate.")
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

    cfg = RunConfig(
        outdir=args.outdir,
        dataset_path=dataset_path,
        dataset_id=dataset_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        comment=args.comment,
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
    )


if __name__ == "__main__":
    main()