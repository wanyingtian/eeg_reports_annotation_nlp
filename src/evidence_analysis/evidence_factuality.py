#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# © 2025 Wanying Tian
"""
Evidence Support Validation
----------------------------
Validates whether LLM-generated explanations for abnormal findings are 
grounded in the original radiology report text using three-stage matching:
  1. Substring search (exact match)
  2. Fuzzy token matching (threshold: 70/100)
  3. Semantic similarity via embeddings (threshold: 0.70 cosine)

Processes explanations from process_output.py and validates against the
reports database to quantify hallucination rates.
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Sequence

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

# Paths
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "evidence_analysis_results"
REPORTS_DB = REPO_ROOT / "data" / "zoe_reports_sample.db"

# Matching thresholds
FUZZY_THRESHOLD = 70      # token_sort_ratio (0-100)
SEMANTIC_THRESHOLD = 0.70  # cosine similarity (0-1)

# Schema from process_output.py
ID_COL = "Hashed_ReportURN"
REASON_COLS = (
    "Focal Epi Reasons",
    "Gen Epi Reasons",
    "Focal Non-epi Reasons",
    "Gen Non-epi Reasons",
    "Abnormality Reasons",
)
LABEL_COLS = tuple(c.replace(" Reasons", "") for c in REASON_COLS)
POSITIVE_LABELS = (3, 4)


@dataclass
class SupportResults:
    """Results from evidence support validation."""
    matched: list[dict[str, str]]
    unmatched: list[dict[str, str]]
    total_checked: int
    
    @property
    def total_matched(self) -> int:
        return len(self.matched)
    
    @property
    def total_unmatched(self) -> int:
        return len(self.unmatched)
    
    @property
    def match_rate(self) -> float:
        return 100.0 * self.total_matched / self.total_checked if self.total_checked else 0.0


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def normalize_text(text: object) -> str:
    """Normalize text for matching: lowercase, remove punctuation, collapse whitespace."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_phrases(text: str) -> list[str]:
    """Split explanation text into key phrases (semicolon-delimited)."""
    if not text:
        return []
    return [p.strip() for p in text.split(";") if p.strip()]


def load_explanations(path: Path) -> pd.DataFrame:
    """Load explanations table from .xlsx or .db file."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name="explanations")
    elif path.suffix.lower() in {".db", ".sqlite"}:
        with sqlite3.connect(path) as con:
            df = pd.read_sql_query('SELECT * FROM "explanations"', con)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (use .xlsx or .db)")
    
    logging.info(f"Loaded explanations: {len(df)} rows from {path.name}")
    return df


def load_reports(db_path: Path) -> pd.DataFrame:
    """Load report texts from database, handling legacy column names."""
    logging.info(f"Loading reports from {db_path.name}")
    
    with sqlite3.connect(db_path) as conn:
        # Check for column name (handle legacy 'Hashed ID')
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(reports)")
        cols = {row[1] for row in cursor.fetchall()}
        
        if "Hashed_ReportURN" in cols:
            id_col = "Hashed_ReportURN"
        elif "Hashed ID" in cols:
            id_col = "Hashed ID"
            logging.warning("Using legacy column name 'Hashed ID'")
        else:
            raise RuntimeError("No ID column found in reports table")
        
        df = pd.read_sql_query(
            f'SELECT "{id_col}" as {ID_COL}, "Report" as report_text FROM reports',
            conn
        )
    
    df[ID_COL] = df[ID_COL].astype(str)
    logging.info(f"Loaded {len(df)} reports")
    return df


def merge_reports(expl_df: pd.DataFrame, reports_df: pd.DataFrame) -> pd.DataFrame:
    """Merge report texts with explanations, preferring embedded reports."""
    df = expl_df.copy()
    
    # Use 'Report' column if present in explanations
    if "Report" in df.columns:
        df["report_text"] = df["Report"]
    else:
        df["report_text"] = pd.NA
    
    # Fill missing reports from database
    df[ID_COL] = df[ID_COL].astype(str)
    df = df.merge(reports_df, how="left", on=ID_COL, suffixes=("", "_db"))
    
    missing_mask = df["report_text"].isna() & df["report_text_db"].notna()
    df.loc[missing_mask, "report_text"] = df.loc[missing_mask, "report_text_db"]
    df.drop(columns=["report_text_db"], inplace=True)
    
    n_missing = df["report_text"].isna().sum()
    if n_missing:
        logging.warning(f"Missing report text for {n_missing} rows")
    
    return df


class SemanticMatcher:
    """Lazy-loaded semantic similarity matcher using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
    
    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
            logging.info(f"Loaded {self.model_name} on {device}")
    
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts to normalized embeddings."""
        self._load()
        return self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between normalized vectors."""
        return float(np.dot(a, b))


def check_phrase_support(
    phrase: str,
    report_text: str,
    report_vec: np.ndarray,
    matcher: SemanticMatcher
) -> bool:
    """
    Check if a phrase is supported by the report using three-stage matching:
    1. Substring match
    2. Fuzzy token matching
    3. Semantic similarity
    """
    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return False
    
    report_norm = normalize_text(report_text)
    
    # Stage 1: Exact substring
    if phrase_norm in report_norm:
        return True
    
    # Stage 2: Fuzzy matching
    if fuzz.token_sort_ratio(phrase_norm, report_norm) >= FUZZY_THRESHOLD:
        return True
    
    # Stage 3: Semantic similarity
    if SEMANTIC_THRESHOLD > 0:
        phrase_vec = matcher.encode([phrase_norm])[0]
        if matcher.cosine_similarity(phrase_vec, report_vec) >= SEMANTIC_THRESHOLD:
            return True
    
    return False


def validate_support(df: pd.DataFrame) -> SupportResults:
    """
    Validate whether explanations for positive labels are supported by reports.
    
    Returns SupportResults with matched/unmatched instances per (ID, Category, Reason).
    """
    # Verify schema
    required = [*REASON_COLS, *LABEL_COLS, ID_COL, "report_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    
    # Prepare text and embeddings
    df["_report_norm"] = df["report_text"].apply(normalize_text)
    texts = [t if t else " " for t in df["_report_norm"]]
    
    matcher = SemanticMatcher()
    report_vecs = matcher.encode(texts)
    
    matched = []
    unmatched = []
    
    for idx, row in df.iterrows():
        report_text = row["report_text"]
        report_vec = report_vecs[idx]
        hashed_id = str(row[ID_COL])
        
        for reason_col, label_col in zip(REASON_COLS, LABEL_COLS):
            label = row.get(label_col)
            
            # Only check positive labels
            if label not in POSITIVE_LABELS:
                continue
            
            reason = row.get(reason_col, "")
            if pd.isna(reason) or not str(reason).strip():
                unmatched.append({
                    ID_COL: hashed_id,
                    "Category": label_col,
                    "Reason": "(No explanation provided)"
                })
                continue
            
            reason_str = str(reason)
            phrases = extract_phrases(reason_str)
            
            # Check if ANY phrase is supported
            is_supported = any(
                check_phrase_support(phrase, report_text, report_vec, matcher)
                for phrase in phrases
            )
            
            record = {
                ID_COL: hashed_id,
                "Category": label_col,
                "Reason": reason_str
            }
            
            if is_supported:
                matched.append(record)
            else:
                unmatched.append(record)
    
    df.drop(columns=["_report_norm"], inplace=True, errors="ignore")
    
    return SupportResults(
        matched=matched,
        unmatched=unmatched,
        total_checked=len(matched) + len(unmatched)
    )


def write_results(results: SupportResults, processed_path: Path, output_dir: Path) -> None:
    """Write validation results to a single summary file."""
    output_file = output_dir / "evidence_factuality_stats.txt"
    
    lines = [
        "Evidence Factuality Validation Results",
        "=" * 60,
        f"Processed file: {processed_path}",
        f"Reports DB: {REPORTS_DB}",
        "",
        "Summary Statistics",
        "-" * 60,
        f"Total instances checked: {results.total_checked}",
        f"Traceable (matched): {results.total_matched}",
        f"Untraceable (unmatched): {results.total_unmatched}",
        f"Traceable rate: {results.match_rate:.2f}%",
        "",
        "",
        f"Traceable Instances ({results.total_matched})",
        "=" * 60,
        "ID | Category | Reason",
        "-" * 60,
    ]
    
    for r in results.matched:
        reason = r["Reason"].replace("\n", " ").strip()
        lines.append(f"{r[ID_COL]} | {r['Category']} | {reason}")
    
    lines.extend([
        "",
        "",
        f"Untraceable Instances ({results.total_unmatched})",
        "=" * 60,
        "ID | Category | Reason",
        "-" * 60,
    ])
    
    for r in results.unmatched:
        reason = r["Reason"].replace("\n", " ").strip()
        lines.append(f"{r[ID_COL]} | {r['Category']} | {reason}")
    
    output_file.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"Results saved to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate LLM explanation support in original reports"
    )
    parser.add_argument(
        "processed_path",
        type=Path,
        help="Path to processed output file (.db or .xlsx from process_output.py)"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    processed_path = args.processed_path.resolve()
    if not processed_path.exists():
        raise SystemExit(f"File not found: {processed_path}")
    
    DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)
    
    # Load and merge data
    expl_df = load_explanations(processed_path)
    reports_df = load_reports(REPORTS_DB)
    df = merge_reports(expl_df, reports_df)
    
    # Validate support
    results = validate_support(df)
    
    # Write results
    write_results(results, processed_path, DEFAULT_OUTDIR)
    
    logging.info(
        f"Validation complete: {results.match_rate:.1f}% support rate "
        f"({results.total_matched}/{results.total_checked})"
    )


if __name__ == "__main__":
    main()