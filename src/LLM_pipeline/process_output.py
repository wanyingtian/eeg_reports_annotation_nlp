# Copyright (c) 2025 Wanying Tian
# Licensed under the Apache-2.0 License (see LICENSE file in the project root for details).
# #!/usr/bin/env python3
"""
Process LLM output CSVs into clean Excel + SQLite artifacts.

- Reads a results CSV (with columns: Hashed ID, Report, classifications, explanations)
- Parses/repairs JSON in `classifications` (decisions only) and `explanations`
  (decisions + reasons)
- Writes two sheets to an Excel file and two tables to a SQLite DB
- Logs any bad JSON rows into errors_log.csv


Usage
-----
python process_output.py mistral_zoe_first_10_results_v1.csv \

optional arguments:
  --input-dir ../pipeline_output \
  --outdir processed_output\
  --num-reports 10 \
  -v
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

# ------------------------------ Logging ------------------------------------ #

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ------------------------------ Config ------------------------------------- #

@dataclass(frozen=True)
class IOConfig:
    input_csv: Path
    input_dir: Path
    outdir: Path
    num_reports: Optional[int] = None
    excel_name: Optional[str] = None  # default derived from input
    sqlite_name: Optional[str] = None # default derived from input

# ------------------------ Constants & Utilities ---------------------------- #


# Resolve paths relative to the repo root (two levels up from this script)
BASE_DIR = Path(__file__).resolve().parent      # e.g., src/LLM_pipeline
REPO_ROOT = BASE_DIR.parents[1]                 # repo root
DEFAULT_INPUT_DIR = REPO_ROOT / "outputs/pipeline_output"
DEFAULT_OUTDIR    = REPO_ROOT / "outputs/processed_output"


STANDARDIZED_KEYS: Dict[str, str] = {
    "focal_epileptiform_activity": "Focal Epi",
    "generalized_epileptiform_activity": "Gen Epi",
    "focal_non_epileptiform_activity": "Focal Non-epi",
    "generalized_non_epileptiform_activity": "Gen Non-epi",
    "abnormality": "Abnormality",
}
ALLOWED = {1, 2, 3, 4}

EXPECTED_CLASS_COLS = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def atomic_write_excel(path: Path, dfs: dict[str, pd.DataFrame]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with pd.ExcelWriter(tmp, engine="xlsxwriter") as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    tmp.replace(path)

def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

# ------------------------ JSON Repair / Parsing ---------------------------- #

def _extract_json_block(s: str) -> Optional[str]:
    if not s:
        return None
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return None
    return s[a : b + 1]

def _truncate_to_balanced(s: str) -> str:
    depth = 0
    last_ok = -1
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_ok = i
        if depth < 0:
            break
    return s[: last_ok + 1] if last_ok != -1 else s

def _light_repair(s: str) -> str:
    # Remove trailing commas before ] or }
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    # Strip non-printables except tab/newline
    s = "".join(ch for ch in s if ch == "\t" or ch == "\n" or 31 < ord(ch) < 127)
    return s

def targeted_json_repair(s: str) -> str:
    """
    Conservative repairs seen in typical LLM JSON drift.
    Does NOT replace all quotes globally (to avoid breaking valid JSON).
    """
    blk = _extract_json_block(s) or s
    blk = _truncate_to_balanced(blk)
    # Trailing commas
    blk = re.sub(r",(\s*[}\]])", r"\1", blk)
    # Undo invalid escapes / artifacts occasionally produced by models
    blk = re.sub(r"\\\\_", "_", blk)  # \\_ -> _
    blk = re.sub(r"\\_", "_", blk)    # \_  -> _
    blk = blk.replace("\\'", "'")     # \'  -> '
    # Stray doubled quotes near punctuation
    blk = re.sub(r'\.\s*""\s*([\]\}])', r'."\1', blk)
    # Balance quick fixes
    return _light_repair(blk)

def coerce_decision(x: Any) -> Optional[int]:
    if isinstance(x, int) and x in ALLOWED:
        return x
    if isinstance(x, str) and x.strip().isdigit():
        xi = int(x.strip())
        return xi if xi in ALLOWED else None
    return None

def clean_reasons(rsns: Any) -> str:
    if not isinstance(rsns, list):
        return "No Explanation Given"
    cleaned: list[str] = []
    for r in rsns:
        if r is None:
            continue
        s = str(r).strip()
        if s:
            cleaned.append(s)
    return "; ".join(cleaned) if cleaned else "No Explanation Given"

# ------------------------ Row Parsers (robust) ----------------------------- #

def parse_classifications(raw: str, hid: str, errors: list[dict[str, str]]) -> dict[str, Any]:
    """
    Parse classification JSON and output standardized decision columns only.
    Returns empty dict on failure (caller will fill columns later).
    """
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(targeted_json_repair(raw))
    except json.JSONDecodeError as e:
        logging.error("Classifications JSON decode failed for %s: %s", hid, e)
        errors.append({
            "Hashed ID": hid,
            "Error Type": "Classifications JSON Decode",
            "Error Message": str(e),
            "Problematic JSON": raw,
        })
        return {}

    out: dict[str, Any] = {}
    for key, label in STANDARDIZED_KEYS.items():
        out[label] = data.get(key, None)
    return out

def parse_explanations(raw: str, hid: str, errors: list[dict[str, str]]) -> dict[str, Any]:
    """
    Parse explanations JSON; extract decisions + reasons into standardized columns.
    Always returns all columns with defaults if missing.
    """
    # Initialize with defaults
    out: dict[str, Any] = {}
    for _, label in STANDARDIZED_KEYS.items():
        out[label] = None
        out[f"{label} Reasons"] = "No Explanation Given"

    if not raw or not raw.strip():
        return out

    try:
        data = json.loads(targeted_json_repair(raw))
    except json.JSONDecodeError as e:
        logging.error("Explanations JSON decode failed for %s: %s", hid, e)
        errors.append({
            "Hashed ID": hid,
            "Error Type": "Explanations JSON Decode",
            "Error Message": str(e),
            "Problematic JSON": raw,
        })
        return out

    if not isinstance(data, dict):
        return out

    for raw_key, label in STANDARDIZED_KEYS.items():
        entry = data.get(raw_key, {}) or {}
        if isinstance(entry, dict):
            out[label] = coerce_decision(entry.get("decision"))
            out[f"{label} Reasons"] = clean_reasons(entry.get("reasons", []))
        else:
            # tolerate legacy: bare ints instead of objects
            out[label] = coerce_decision(entry)

    return out

# ------------------------------- Core -------------------------------------- #

def process_file(cfg: IOConfig) -> None:
    ensure_dir(cfg.outdir)

    input_path = cfg.input_dir / cfg.input_csv
    if not input_path.exists():
        logging.error("Input CSV not found: %s", input_path)
        return

    # Load CSV
    try:
        df = pd.read_csv(input_path, nrows=cfg.num_reports)
    except Exception as e:
        logging.error("Failed to read CSV: %s", e)
        return

    req_cols = {"Hashed ID", "Report", "classifications", "explanations"}
    missing = req_cols - set(df.columns)
    if missing:
        logging.warning("Missing expected columns: %s", ", ".join(sorted(missing)))

    logging.info("Loaded %d rows from %s", len(df), input_path)

    # Parse rows
    error_rows: list[dict[str, str]] = []

    # Classifications (decisions only)
    if "classifications" in df.columns and df["classifications"].notna().any():
        class_dicts = df.apply(
            lambda row: parse_classifications(str(row.get("classifications", "")), str(row.get("Hashed ID", "")), error_rows),
            axis=1,
        )
        class_df = pd.DataFrame(list(class_dicts))
    else:
        logging.warning("Column 'classifications' missing or empty.")
        class_df = pd.DataFrame(columns=EXPECTED_CLASS_COLS)

    # Explanations (decisions + reasons)
    if "explanations" in df.columns and df["explanations"].notna().any():
        expl_dicts = df.apply(
            lambda row: parse_explanations(str(row.get("explanations", "")), str(row.get("Hashed ID", "")), error_rows),
            axis=1,
        )
        expl_df = pd.DataFrame(list(expl_dicts))
    else:
        logging.warning("Column 'explanations' missing or empty.")
        expl_df = pd.DataFrame(columns=[*EXPECTED_CLASS_COLS, *[f"{c} Reasons" for c in EXPECTED_CLASS_COLS]])

    # Ensure expected columns exist
    for col in EXPECTED_CLASS_COLS:
        if col not in class_df.columns:
            class_df[col] = pd.NA
        if col not in expl_df.columns:
            expl_df[col] = pd.NA
        rcol = f"{col} Reasons"
        if rcol not in expl_df.columns:
            expl_df[rcol] = pd.NA

    # Preserve Hashed ID and Report
    for base in (class_df, expl_df):
        base["Hashed ID"] = df.get("Hashed ID", pd.Series([pd.NA] * len(df)))
        base["Report"] = df.get("Report", pd.Series([pd.NA] * len(df)))

    # Quick previews
    logging.debug("Classifications head:\n%s", class_df.head())
    logging.debug("Explanations head:\n%s", expl_df.head())

    # Determine output names
    base_name = cfg.input_csv.stem
    excel_name = cfg.excel_name or f"processed_{base_name}.xlsx"
    sqlite_name = cfg.sqlite_name or f"processed_{base_name}.db"

    excel_path = cfg.outdir / excel_name
    sqlite_path = cfg.outdir / sqlite_name
    errors_path = cfg.outdir / "errors_log.csv"

    # Save Excel (atomic)
    try:
        atomic_write_excel(excel_path, {"classifications": class_df, "explanations": expl_df})
        logging.info("Saved Excel -> %s", excel_path)
    except Exception as e:
        logging.error("Failed to write Excel: %s", e)

    # Save SQLite
    try:
        with sqlite3.connect(sqlite_path) as conn:
            class_df.to_sql("classifications", conn, if_exists="replace", index=False)
            expl_df.to_sql("explanations", conn, if_exists="replace", index=False)
        logging.info("Saved SQLite -> %s", sqlite_path)
    except Exception as e:
        logging.error("Failed to write SQLite: %s", e)

    # Save errors (if any)
    if error_rows:
        err_df = pd.DataFrame(error_rows, columns=["Hashed ID", "Error Type", "Error Message", "Problematic JSON"])
        try:
            atomic_write_csv(errors_path, err_df)
            logging.info("Wrote errors -> %s (rows=%d)", errors_path, len(err_df))
        except Exception as e:
            logging.error("Failed to write errors CSV: %s", e)
    else:
        logging.info("No JSON errors detected.")

# --------------------------------- CLI ------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process EEG classification and explanation results.")
    p.add_argument("input_filename", help="Input CSV file name (e.g., 'mistral_zoe_first_200_results_v4.csv')")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
               help="Folder containing the input CSV, default: /outputs/pipeline_output")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
               help="Output directory for Excel/DB/errors, default: outputs/processed_output")

    p.add_argument("--num-reports", type=int, default=None, help="Limit number of rows to read")
    p.add_argument("--excel-name", type=str, default=None, help="Override Excel file name")
    p.add_argument("--sqlite-name", type=str, default=None, help="Override SQLite file name")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv)")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    cfg = IOConfig(
        input_csv=Path(args.input_filename),
        input_dir=args.input_dir,
        outdir=args.outdir,
        num_reports=args.num_reports,
        excel_name=args.excel_name,
        sqlite_name=args.sqlite_name,
    )

    process_file(cfg)

if __name__ == "__main__":
    main()
