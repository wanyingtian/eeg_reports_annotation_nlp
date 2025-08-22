#!/usr/bin/env python3
"""
EEG Report Processing Pipeline (LLM-based)

- Loads EEG reports from a SQLite database.
- Runs a classification prompt and an explanation prompt against an LLM (llama.cpp).
- Writes results incrementally to versioned CSVs and emits a run-config summary.
- Resumes safely after crashes; uses atomic file writes to avoid corruption.

Usage (examples)
---------------
# Process 10 reports from Zoe, mistral model:
python pipeline.py --num-reports 10 --author zoe --model mistral

# Resume from a previous CSV:
python pipeline.py --num-reports 10 --completed-csv /path/to/previous.csv

# Point to custom DBs and output directory:
python pipeline.py --num-reports 10 --zoe-db ../../data/zoe_reports_10.db \
  --maria-db ../../data/maria_reports_10.db --outdir ../../outputs/pipeline_output
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

# --------------------------- Defaults / Constants --------------------------- #

DEFAULT_OUTDIR = Path("../../outputs/pipeline_output")
DEFAULT_ZOE_DB = Path("../../data/zoe_reports_10.db")
DEFAULT_MARIA_DB = Path("../../data/zoe_reports_10.db")  # TODO: replace with real Maria DB

# Model defaults
DEFAULT_TEMPERATURE = 0.0
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
    """Immutable run configuration."""
    outdir: Path
    zoe_db: Path
    maria_db: Path
    temperature: float = DEFAULT_TEMPERATURE
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
    Stream (Hashed ID, Report) rows from the SQLite database.
    """
    logging.info(f"Connecting to DB: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT "Hashed ID", "Report" FROM reports')
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield row[0], row[1]
    except sqlite3.Error as e:
        logging.error(f"SQLite error: {e}")
        raise
    finally:
        conn.close()


def atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write CSV atomically to prevent partial files on crash.
    """
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def next_versioned_paths(outdir: Path, model: str, author: str, num_reports: int) -> Tuple[Path, Path]:
    """
    Determine next version index for results/config files. Pattern:
    {model}_{author}_first_{N}_results_v{K}.csv
    {model}_{author}_config_first_{N}_v{K}.txt
    """
    ensure_outdir(outdir)
    k = 1
    while True:
        results = outdir / f"{model}_{author}_first_{num_reports}_results_v{k}.csv"
        if not results.exists():
            config = outdir / f"{model}_{author}_config_first_{num_reports}_v{k}.txt"
            return results, config
        k += 1


def latest_version_csv(outdir: Path, model: str, author: str, num_reports: int) -> Optional[Path]:
    """
    Return the latest CSV path if present, else None.
    """
    pattern = re.compile(fr"{re.escape(model)}_{re.escape(author)}_first_{num_reports}_results_v(\d+)\.csv$")
    max_k, latest = -1, None
    for p in outdir.glob(f"{model}_{author}_first_{num_reports}_results_v*.csv"):
        m = pattern.match(p.name)
        if m:
            k = int(m.group(1))
            if k > max_k:
                max_k, latest = k, p
    return latest


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


def download_model(model_name: str) -> Llama:
    """
    Download & load a GGUF model via huggingface_hub and llama.cpp.
    """
    model_configs = {
        "mistral": {
            "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            "filename": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
        },
        "deepseek": {
            "repo_id": "TheBloke/deepseek-llm-7b-base-GGUF",
            "filename": "deepseek-llm-7b-base.Q5_K_M.gguf",
        },
        "deepseek-coder": {
            "repo_id": "TheBloke/deepseek-coder-6.7B-instruct-GGUF",
            "filename": "deepseek-coder-6.7b-instruct.Q5_K_M.gguf",
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
    if model_name not in model_configs:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {list(model_configs.keys())}.")

    cfg = model_configs[model_name]
    logging.info(f"Downloading model {model_name}...")
    model_path = hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"])
    logging.info("Loading model into llama.cpp...")
    # n_gpu_layers is environment-dependent; keep conservative defaults.
    return Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=30)


def llm_json(
    model: Llama,
    prompt: str,
    temperature: float,
    max_tokens: int,
    stop: Optional[Iterable[str]],
    grammar: Optional[LlamaGrammar] = None,
) -> str:
    """
    Invoke the LLM and return the raw text (expected to be JSON per grammar).
    """
    resp = model(
        prompt,
        grammar=grammar,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    text = resp["choices"][0]["text"]
    return text


# ------------------------------ Core Pipeline ------------------------------ #

def process_completed_csv(path: Optional[Path]) -> Tuple[pd.DataFrame, set[str]]:
    """
    Load existing results to resume. Returns (df, set_of_hashed_ids).
    """
    cols = ["Hashed ID", "Report", "classifications", "explanations"]
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
        completed = set(df["Hashed ID"].dropna().astype(str))
        logging.info(f"Loaded {len(completed)} completed reports from {path}")
        return df[cols].copy(), completed
    except Exception as e:
        logging.error(f"Failed to read completed CSV: {e}")
        return base, set()


def load_reports_df(
    db_path: Path, num_reports: int, exclude_hashes: set[str]
) -> pd.DataFrame:
    """
    Pull up to num_reports (Hashed ID, Report) not present in exclude_hashes.
    """
    rows = []
    for i, (hid, rep) in enumerate(fetch_reports(db_path)):
        if len(rows) >= num_reports:
            break
        if str(hid) in exclude_hashes:
            continue
        rows.append((str(hid), rep))
    df = pd.DataFrame(rows, columns=["Hashed ID", "Report"])
    logging.info(f"Loaded {len(df)} pending reports from {db_path}")
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
        hashed_id = str(row["Hashed ID"])
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
        )

        # Append row
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    [
                        {
                            "Hashed ID": hashed_id,
                            "Report": report,
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
            # lightweight run progress
            Path(out_config).write_text(
                f"Temperature: {cfg.temperature}\n"
                f"Max Tokens: {cfg.max_tokens}\n"
                f"Reports completed (this run): {idx+1}\n"
            )
            atomic_write_csv(results_df, out_results)

    # Final write + config dump
    atomic_write_csv(results_df, out_results)
    elapsed = time.time() - start
    with open(out_config, "w") as f:
        f.write(
            "Run Configuration\n"
            "------------------\n"
            f"Author: {os.getenv('AUTHOR_OVERRIDE','')}\n"
            f"Temperature: {cfg.temperature}\n"
            f"Stop Sequences: {list(cfg.stop) if cfg.stop else None}\n"
            f"Max Tokens: {cfg.max_tokens}\n"
            f"Model: {os.getenv('MODEL_OVERRIDE','')}\n"
            f"Comment: {cfg.comment}\n"
            f"Reports completed (total rows): {len(results_df)}\n"
            f"Elapsed: {elapsed:.2f}s ({elapsed/60:.2f} min)\n"
            f"Prompt1: {PROMPT_CLASSIFY}\n"
            f"Prompt2: {PROMPT_EXPLAIN}\n"
        )
    logging.info(f"Saved results -> {out_results}")
    logging.info(f"Saved config  -> {out_config}")
    logging.info(f"Elapsed: {elapsed:.2f}s")
    return results_df


# --------------------------- Crash-Resistant Runner ------------------------- #

def manager(
    num_reports: int,
    completed_csv: Optional[Path],
    author: str,
    model_name: str,
    cfg: RunConfig,
) -> None:
    """
    Supervises the run. If a worker crashes, it restarts and resumes from the
    latest versioned CSV.
    """
    db_path = cfg.zoe_db if author == "zoe" else cfg.maria_db
    out_results, out_config = next_versioned_paths(cfg.outdir, model_name, author, num_reports)

    # preload completed (support resume)
    existing_df, completed_hashes = process_completed_csv(completed_csv)
    logging.info(f"Initial completed count: {len(completed_hashes)}")

    # inner worker target
    def worker_target(resume_csv: Optional[Path]) -> None:
        grammar_classify = load_gbnf(Path("result_grammar.gbnf"))
        grammar_explain = load_gbnf(Path("result_grammar_exp.gbnf"))
        model = download_model(model_name)

        # (Re)load completed and pending
        prior_df, prior_hashes = process_completed_csv(resume_csv)
        pending = load_reports_df(db_path, num_reports, prior_hashes)
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
    resume_path = completed_csv
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
        latest = latest_version_csv(cfg.outdir, model_name, author, num_reports)
        resume_path = latest if latest else None
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
    p.add_argument("--author", type=str, choices=["zoe", "maria"], default="zoe", help='Report author: "zoe" or "maria".')
    p.add_argument(
        "--model",
        type=str,
        choices=["mistral", "deepseek", "deepseek-coder", "deepseek-chat", "hermes-mistral", "hermes-llama2"],
        default="mistral",
        help="Model to use (GGUF).",
    )
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Directory to write outputs.")
    p.add_argument("--zoe-db", type=Path, default=DEFAULT_ZOE_DB, help="Path to Zoe reports SQLite DB.")
    p.add_argument("--maria-db", type=Path, default=DEFAULT_MARIA_DB, help="Path to Maria reports SQLite DB.")
    p.add_argument("--comment", type=str, default="testing run", help="Run comment to save in config.")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    cfg = RunConfig(
        outdir=args.outdir,
        zoe_db=args.zoe_db,
        maria_db=args.maria_db,
        comment=args.comment,
    )

    # Helpful env overrides recorded in config output (not required)
    os.environ["AUTHOR_OVERRIDE"] = args.author
    os.environ["MODEL_OVERRIDE"] = args.model

    manager(
        num_reports=args.num_reports,
        completed_csv=args.completed_csv,
        author=args.author,
        model_name=args.model,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
