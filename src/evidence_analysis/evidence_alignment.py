#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# © 2025 Wanying Tian
"""
Explanation-Label Alignment Analysis
-------------------------------------
Validates whether LLM-generated explanations align with ground truth labels
by checking the polarity (positive/negative) of the explanation text.

Two methods:
  1. Rule-based (regex patterns) - default, no training data required
  2. ClinicalBERT + Logistic Regression - requires labeled training data

For EEG reports, positive explanations should indicate abnormalities while
negative explanations should indicate normal findings.
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from pathlib import Path
from typing import Pattern

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Paths
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "evidence_analysis_results"

# Schema
CATEGORIES = (
    "Focal Epi",
    "Gen Epi", 
    "Focal Non-epi",
    "Gen Non-epi",
    "Abnormality",
)
REASON_COLS = tuple(f"{c} Reasons" for c in CATEGORIES)
ID_COL = "Hashed_ReportURN"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# =================== Rule-Based Method ===================

NEGATION_PATTERNS: list[Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bno (evidence|indication|suggestion|signs|features|findings)\b",
        r"\bno .* (discharges|transients|changes|abnormalities)\b",
        r"\bno\s+(?:\w+\s+){0,5}?(abnormalities|discharges|features)",
        r"\bno (specific )?mention\b",
        r"\bnot demonstrate\b",
        r"\bno .* (were|was) (noted|observed|seen|identified|detected)\b",
        r"\b(?<!ab)normal\b",
        r"\bthis eeg (can be interpreted as|was|is) (virtually )?normal\b",
        r"\bvirtually normal\b",
        r"\bwithin normal limits\b",
        r"\bbenign\b",
        r"\bno precise (electrophysiological )?diagnosis\b",
        r"\b(emg )?artifact(ual)?\b",
        r"\bmild degree of nonspecific encephalopathy\b",
        r"\bcan be consistent with underlying microstructural\b",
        r"\bconsistent with previous history of infarcts\b",
        r"\bobserved .* could be explained\b",
        r"\bstate of drowsiness alone\b",
        r"\bsuboptimal quality\b",
        r"\bdoes not demonstrate significant epileptiform\b",
        r"\bno significant epileptiform\b",
        r"\binterpreted as normal\b",
        r"\blikely within normal limits\b",
        r"\bbenign electrophysiological phenomenon\b",
        r"\bfeatures suggestive of.*benign\b",
        r"\bno (clear|definitive|associated)? .* (discharges|abnormalities|transients|features)?\b",
        r"\bno (lateralized|generalized|focal)? ?(or )?epileptiform abnormalities\b",
        r"\bno periodic complexes\b",
        r"\bnormal (eeg|record|background|rhythm)\b",
        r"\b(eeg|electroencephalogram|record) (was|is)? (normal|within normal limits)\b",
        r"\bno .* (captured|seen|recorded|were recorded|were found)\b",
        r"^nan$",
        r"^n/?a$",
        r"no explanation given"
    ]
]

POSITIVE_KEYWORDS: list[Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\babnormal\b",
        r"abnormal.*theta and delta waveforms",
        r"amplitude of (alpha|beta) frequencies",
        r"\basymmetr(y|ical findings)\b",
        r"background slowing",
        r"bitemporal dysrhythmia",
        r"cerebral dysfunction",
        r"consistent with.*(encephalopathy|epilepsy)",
        r"delta (and|&) theta (activity|waveforms)|delta range activity",
        r"diffuse suppression of electrical activity",
        r"disturbance of cerebral function",
        r"\bdysrhythmia\b",
        r"epileptic (features|tendency)",
        r"epileptiform (abnormalities|discharges|disturbance|transients)",
        r"focal seizure",
        r"frontal intermittent rhythmic delta activity",
        r"frontal slowing",
        r"frequent discharges",
        r"generalized disturbance( of cerebral function)?",
        r"generalized (epileptiform )?disturbance",
        r"generalized (delta|theta) activity",
        r"impairment of cerebral function",
        r"increased (amplitude|frequencies)",
        r"ischemic stroke",
        r"(left|right) hemisphere slowing",
        r"localized abnormality",
        r"microstructural change",
        r"(mild|moderate|severe) abnormality",
        r"moderate-to-severe degree of.*disturbance",
        r"parietal central regions",
        r"phase reversal",
        r"right posterior rhythmic activity",
        r"sharp (contoured features|transients|wave)",
        r"sharply contoured transients",
        r"slow wave (activity|discharges)",
        r"spike( and)? wave activity|spike discharges",
        r"suppression of background",
        r"temporal (dysrhythmia|slowing)",
        r"theta range activity",
        r"triphasic transients",
        r"wicket waves",
    ]
]


def classify_reason_rule(text: object) -> int:
    """
    Classify explanation polarity using regex rules.
    Returns: 1 (positive/abnormal), -1 (negative/normal), 0 (unclear)
    """
    s = str(text or "").strip().lower()
    if not s:
        return 0
    
    # Explicit abnormal statement
    if any(phrase in s for phrase in ["this eeg is abnormal", "this is an abnormal", "abnormal eeg"]):
        return 1
    
    # Check negation patterns first
    for pattern in NEGATION_PATTERNS:
        if pattern.search(s):
            return -1
    
    # Check positive keywords
    for pattern in POSITIVE_KEYWORDS:
        if pattern.search(s):
            return 1
    
    # Default to negative if unclear
    return -1


# =================== ClinicalBERT Method ===================

def get_bert_embedding(text: str, tokenizer, model, device) -> np.ndarray:
    """Extract [CLS] token embedding from ClinicalBERT."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    
    return cls_embedding.squeeze().cpu().numpy()


def classify_with_clinicalbert(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    tokenizer,
    model,
    device
) -> pd.DataFrame:
    """
    Train category-specific classifiers using ClinicalBERT embeddings + LR.
    Uses 200 samples for training.
    """
    from tqdm import tqdm
    from sklearn.linear_model import LogisticRegression
    
    for cat in CATEGORIES:
        reason_col = f"{cat} Reasons"
        polarity_col = f"{cat} Reason Polarity"
        
        test_df[polarity_col] = 0
        
        # Prepare training data (first 200 labeled samples) - can be adjusted as needed
        train_mask = train_df[reason_col].notna()
        train_reasons = train_df.loc[train_mask, reason_col][:200]
        y_train = train_df.loc[train_mask, cat][:200]
        
        if len(train_reasons) < 50:
            # the minimum training data size can be adjusted
            logging.warning(f"Insufficient training samples for {cat} ({len(train_reasons)} < 50)")
            continue
        
        # Prepare test data (samples after index 200)
        test_mask = test_df[reason_col].notna()
        test_reasons = test_df.loc[test_mask, reason_col]
        test_idx = test_reasons.index
        
        if test_reasons.empty:
            logging.warning(f"No test samples for {cat}")
            continue
        
        # Compute embeddings
        logging.info(f"Training classifier for {cat}...")
        X_train = np.vstack([
            get_bert_embedding(txt, tokenizer, model, device)
            for txt in tqdm(train_reasons, desc="Train embeddings")
        ])
        X_test = np.vstack([
            get_bert_embedding(txt, tokenizer, model, device)
            for txt in tqdm(test_reasons, desc="Test embeddings")
        ])
        
        # Train and predict
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test_df.loc[test_idx, polarity_col] = y_pred
    
    return test_df


# =================== Evaluation ===================

def map_label_to_polarity(value: object) -> int:
    """Map labels: 1/2 → -1 (negative), 3/4 → +1 (positive)."""
    try:
        v = int(value)
        return -1 if v in (1, 2) else (1 if v in (3, 4) else 0)
    except (ValueError, TypeError):
        return 0


def compute_alignment_stats(
    df: pd.DataFrame,
    method: str
) -> tuple[dict, dict | None, list[dict] | None]:
    """
    Compute polarity counts and optionally evaluate against labels.
    Returns: (polarity_counts, eval_stats, disagreements)
    """
    polarity_counts = {}
    eval_stats = None
    disagreements = None
    
    # Check if labels are present
    has_labels = all(cat in df.columns for cat in CATEGORIES)
    if has_labels:
        eval_stats = {}
        disagreements = []
    
    for cat in CATEGORIES:
        reason_col = f"{cat} Reasons"
        
        # Classify polarity
        if method == "rule":
            polarity = df[reason_col].apply(classify_reason_rule).astype(int)
        else:  # clinicalbert - polarity already computed
            polarity_col = f"{cat} Reason Polarity"
            if polarity_col not in df.columns:
                continue
            polarity = df[polarity_col].astype(int)  # Already in -1/1 format
        
        # Count polarities
        polarity_counts[cat] = {
            "positive": int((polarity == 1).sum()),
            "negative": int((polarity == -1).sum()),
            "unclear": int((polarity == 0).sum()),
        }
        
        # Evaluate against labels if present
        if has_labels:
            # Test data labels are always in 1/2/3/4 format - convert to -1/1 for comparison
            label_polarity = df[cat].apply(map_label_to_polarity).astype(int)
            labeled_mask = label_polarity != 0
            n_labeled = int(labeled_mask.sum())
            
            if n_labeled > 0:
                agreement = int((polarity[labeled_mask] == label_polarity[labeled_mask]).sum())
                eval_stats[cat] = {
                    "n_labeled": n_labeled,
                    "agree": agreement,
                    "disagree": n_labeled - agreement,
                }
                
                # Collect disagreements
                mismatch_mask = labeled_mask & (polarity != label_polarity)
                for idx in df.index[mismatch_mask]:
                    hashed_id = str(df.at[idx, ID_COL]) if ID_COL in df.columns else ""
                    explanation = df.at[idx, reason_col]
                    disagreements.append({
                        ID_COL: hashed_id,
                        "Category": cat,
                        "Explanation": "" if pd.isna(explanation) else str(explanation),
                        "Explanation_Polarity": int(polarity.at[idx]),
                        "Model_Decision_Polarity": int(label_polarity.at[idx]),
                    })
    
    return polarity_counts, eval_stats, disagreements


# =================== I/O ===================

def load_explanations(path: Path) -> pd.DataFrame:
    """Load explanations from .xlsx or .db file."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name="explanations")
    elif path.suffix.lower() in {".db", ".sqlite"}:
        with sqlite3.connect(path) as con:
            df = pd.read_sql_query('SELECT * FROM "explanations"', con)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    logging.info(f"Loaded {len(df)} rows from {path.name}")
    return df


def write_results(
    output_dir: Path,
    processed_path: Path,
    method: str,
    polarity_counts: dict,
    eval_stats: dict | None,
    disagreements: list[dict] | None,
) -> None:
    """Write alignment statistics to file."""
    lines = [
        "Explanation-Label Alignment Results",
        "=" * 60,
        f"Processed file: {processed_path}",
        f"Method: {method}",
        "",
        "Polarity Distribution by Category",
        "-" * 60,
    ]
    
    total_pos = total_neg = total_unclear = 0
    for cat in CATEGORIES:
        if cat not in polarity_counts:
            continue
        
        counts = polarity_counts[cat]
        n_total = sum(counts.values())
        total_pos += counts["positive"]
        total_neg += counts["negative"]
        total_unclear += counts["unclear"]
        
        pct = lambda k: 100.0 * counts[k] / n_total if n_total else 0.0
        lines.append(
            f"{cat}: N={n_total} | "
            f"+{counts['positive']} ({pct('positive'):.1f}%)  "
            f"-{counts['negative']} ({pct('negative'):.1f}%)  "
            f"?{counts['unclear']} ({pct('unclear'):.1f}%)"
        )
    
    total = total_pos + total_neg + total_unclear
    lines.extend([
        "",
        f"TOTAL: N={total} | +{total_pos}  -{total_neg}  ?{total_unclear}",
        "",
    ])
    
    # Evaluation against labels
    if eval_stats:
        lines.extend([
            "",
            "Agreement with Ground Truth Labels",
            "-" * 60,
            "(Label mapping: 1/2 → negative, 3/4 → positive)",
            "",
        ])
        
        total_labeled = total_agree = total_disagree = 0
        for cat in CATEGORIES:
            if cat not in eval_stats:
                continue
            
            stats = eval_stats[cat]
            total_labeled += stats["n_labeled"]
            total_agree += stats["agree"]
            total_disagree += stats["disagree"]
            
            acc = 100.0 * stats["agree"] / stats["n_labeled"] if stats["n_labeled"] else 0.0
            lines.append(
                f"{cat}: labeled={stats['n_labeled']}  "
                f"agree={stats['agree']}  disagree={stats['disagree']}  "
                f"accuracy={acc:.1f}%"
            )
        
        overall_acc = 100.0 * total_agree / total_labeled if total_labeled else 0.0
        lines.extend([
            "",
            f"OVERALL: labeled={total_labeled}  agree={total_agree}  "
            f"disagree={total_disagree}  accuracy={overall_acc:.1f}%",
            "",
        ])
        
        if disagreements:
            lines.extend([
                "",
                f"Disagreements ({len(disagreements)})",
                "-" * 60,
                "Hashed_ReportURN | Category | Explanation_Pol | Model_Decision_Pol | Explanation",
            ])
            for r in disagreements:
                hid = r.get(ID_COL, "")
                cat = r.get("Category", "")
                expl_pol = r.get("Explanation_Polarity", 0)
                mapped_pol = r.get("Model_Decision_Polarity", 0)
                exp = (r.get("Explanation", "") or "").replace("\n", " ").strip()
                
                # Format polarity as +/- for readability
                expl_str = f"+1" if expl_pol == 1 else (f"-1" if expl_pol == -1 else " 0")
                mapped_str = f"+1" if mapped_pol == 1 else (f"-1" if mapped_pol == -1 else " 0")
                
                lines.append(f"{hid} | {cat} | {expl_str} | {mapped_str} | {exp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "alignment_stats.txt"
    output_file.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"Results saved to {output_file}")


# =================== CLI ===================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate explanation-label alignment using polarity analysis"
    )
    parser.add_argument(
        "processed_path",
        type=Path,
        help="Path to processed output (.db or .xlsx from process_output.py)"
    )
    parser.add_argument(
        "--method",
        choices=["rule", "clinicalbert"],
        default="rule",
        help="Polarity classification method (default: rule)"
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        help="Path to labeled training data (required for clinicalbert method)"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    if not args.processed_path.exists():
        raise SystemExit(f"File not found: {args.processed_path}")
    
    # Load data
    df = load_explanations(args.processed_path)
    
    # Handle ClinicalBERT method
    if args.method == "clinicalbert":
        if not args.train_data:
            raise SystemExit(
                "ERROR: --train-data required for ClinicalBERT method.\n\n"
                "The ClinicalBERT method requires labeled training data for training category-specific classifiers.  "
                "This data is not included in the repository.\n\n"
                "To use this method, you must provide your own labeled training data, with columns like 'Abnormality Reason Polarity' "
                "with the --train-data flag."
            )
        
        if not args.train_data.exists():
            raise SystemExit(f"Training data not found: {args.train_data}")
        
        # Load ClinicalBERT

        
        logging.info("Loading ClinicalBERT model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
        
        train_df = load_explanations(args.train_data)
        df = classify_with_clinicalbert(df, train_df, tokenizer, model, device)
    
    # Compute alignment statistics
    polarity_counts, eval_stats, disagreements = compute_alignment_stats(df, args.method)
    
    # Write results
    write_results(
        DEFAULT_OUTDIR,
        args.processed_path,
        args.method,
        polarity_counts,
        eval_stats,
        disagreements,
    )
    
    if eval_stats:
        total_agree = sum(s["agree"] for s in eval_stats.values())
        total_labeled = sum(s["n_labeled"] for s in eval_stats.values())
        acc = 100.0 * total_agree / total_labeled if total_labeled else 0.0
        logging.info(f"Overall alignment accuracy: {acc:.1f}%")


if __name__ == "__main__":
    main()