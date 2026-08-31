#!/usr/bin/env python3
"""Resumable local similarity investigation, with no inferred patient map."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from eeg_review.baseline import _reports_digest
from eeg_review.io import atomic_write_csv, atomic_write_json, load_table
from eeg_review.linkage_diagnostic import (
    POLICY,
    REVIEW_PROMPT,
    REVIEW_SCHEMA,
    cosine_matrix,
    jaccard,
    normalized_text,
    select_candidates,
    shingle_set,
    validate_review,
)
from eeg_review.manifest import sha256_file
from eeg_review.protected_execution import assert_governed_run_active

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_comparison_review import verify_bundle  # noqa: E402

KEY = "Hashed_ReportURN"
COHORTS = {
    "development": ("development", "zoe_development_100.db", None, 100),
    "zoe_evaluation": ("zoe", "zoe_evaluation_1400.db", "zoe_evaluation_1395.db", 1395),
    "maria_evaluation": ("maria", "maria_evaluation_500.db", "maria_evaluation_499.db", 499),
}
LOAD = {"n_ctx": 4096, "n_gpu_layers": -1, "n_batch": 512, "n_ubatch": 128, "flash_attn": True}
SAMPLING = {"temperature": 0.0, "top_k": 40, "top_p": 0.95, "max_tokens": 512, "seed": 20260831}


def read(path):
    return json.loads(path.read_text())


def utc():
    return datetime.now(UTC).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def active(args):
    for root in [args.original_run, args.native_run, args.output_dir]:
        assert_governed_run_active(root)


def cache_vectors(source, cache):
    frame = load_table(source, [KEY, "Report"])
    if frame[KEY].isna().any() or frame[KEY].duplicated().any() or frame.Report.isna().any():
        raise ValueError("cache source keys/text invalid")
    manifest_path = cache / "embedding_cache.json"
    manifest = read(manifest_path)
    if (
        manifest["records"] != len(frame)
        or manifest["checkpoint"] != "bert-base-uncased"
        or manifest["report_digest"] != _reports_digest(frame.Report.tolist())
    ):
        raise ValueError("cached embedding text/order identity mismatch")
    names, arrays = [], []
    for start in range(0, len(frame), manifest["batch_size"]):
        end = min(len(frame), start + manifest["batch_size"])
        name = f"batch_{start:06d}_{end:06d}.npy"
        value = np.load(cache / name, allow_pickle=False)
        if value.shape != (end - start, 768) or not np.isfinite(value).all():
            raise ValueError("cached embedding shape/value mismatch")
        names.append(name)
        arrays.append(value)
    if manifest["completed_batches"] != names:
        raise ValueError("incomplete or reordered cache manifest")
    files = [source, manifest_path, *[cache / name for name in names]]
    return frame, np.concatenate(arrays), files


def source_intake(args):
    active(args)
    bundle = verify_bundle(args.native_run)
    frames, vectors, files = [], [], []
    for cohort, (cache_name, source_name, selected_name, expected) in COHORTS.items():
        source = args.original_run / "inputs" / source_name
        cache = args.original_run / "cache/bert" / cache_name
        frame, embedding, cache_files = cache_vectors(source, cache)
        selected = args.native_run / "inputs" / selected_name if selected_name else source
        fixed = load_table(selected, [KEY, "Report"])
        if len(fixed) != expected or fixed[KEY].isna().any() or fixed[KEY].duplicated().any():
            raise ValueError("fixed cohort mismatch")
        lookup = {str(key): i for i, key in enumerate(frame[KEY])}
        indices = [lookup[str(key)] for key in fixed[KEY]]
        if frame.iloc[indices].Report.tolist() != fixed.Report.tolist():
            raise ValueError("keyed cache and selected report texts disagree")
        fixed = fixed.copy()
        fixed[KEY] = fixed[KEY].astype(str)
        fixed["cohort"] = cohort
        frames.append(fixed)
        vectors.append(embedding[indices])
        files.extend([*cache_files, selected])
    all_records = pd.concat(frames, ignore_index=True)
    versions = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scipy", "scikit-learn", "llama-cpp-python"]
    }
    code = [
        Path(__file__),
        ROOT / "src/eeg_review/linkage_diagnostic.py",
        ROOT / "review/REPORT_LINKAGE_DIAGNOSTIC.md",
        ROOT / "src/eeg_review/baseline.py",
        ROOT / "src/eeg_review/io.py",
        ROOT / "src/eeg_review/protected_execution.py",
        ROOT / "src/eeg_review/manifest.py",
        ROOT / "src/LLM_pipeline/llm_models.py",
        ROOT / "scripts/prepare_comparison_review.py",
    ]
    receipt = {
        "diagnostic_id": POLICY["diagnostic_id"],
        "policy": POLICY,
        "prompt_sha256": digest(REVIEW_PROMPT),
        "grammar_schema_sha256": digest(REVIEW_SCHEMA),
        "model_registry": "medgemma-27b-q2-candidate",
        "load": LOAD,
        "sampling": SAMPLING,
        "source_bundle": bundle,
        "python": sys.version.split()[0],
        "packages": versions,
        "source_hashes": {str(p.resolve()): sha256_file(p) for p in sorted(set(files))},
        "code_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in code},
        "records": len(all_records),
        "reference_labels_loaded": False,
        "model_predictions_loaded": False,
        "cohorts": {k: v[3] for k, v in COHORTS.items()},
    }
    return all_records, np.concatenate(vectors), receipt


def seal_stage(root, stage, outputs):
    atomic_write_json(
        root / f"receipts/{stage}.json",
        {
            "finished_at": utc(),
            "freeze_sha256": sha256_file(root / "freeze.json"),
            "files": {str(p.relative_to(root)): sha256_file(p) for p in outputs},
        },
    )


def stage_done(root, stage):
    path = root / f"receipts/{stage}.json"
    if not path.exists():
        return False
    receipt = read(path)
    if receipt["freeze_sha256"] != sha256_file(root / "freeze.json"):
        raise ValueError("stage freeze changed")
    for name, sha in receipt["files"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or sha256_file(target) != sha:
            raise ValueError("stage output changed")
    return True


def prepare(args):
    frame, vectors, receipt = source_intake(args)
    root = args.output_dir
    freeze = root / "freeze.json"
    if freeze.exists():
        if read(freeze) != receipt:
            raise ValueError("frozen source, policy, implementation or runtime changed")
    else:
        atomic_write_json(freeze, receipt)
    if stage_done(root, "prepare"):
        return
    salt_file = root / "private-salt.txt"
    if not salt_file.exists():
        import secrets

        salt_file.write_text(secrets.token_hex(32))
    salt = salt_file.read_text()
    records = []
    for row in frame.to_dict("records"):
        row["case_handle"] = hashlib.sha256(
            f"{salt}:{row['cohort']}:{row[KEY]}".encode()
        ).hexdigest()[:20]
        records.append(row)
    atomic_write_json(root / "records.json", records)
    temporary = root / "embeddings.npy.tmp"
    with temporary.open("wb") as stream:
        np.save(stream, vectors, allow_pickle=False)
    temporary.replace(root / "embeddings.npy")
    seal_stage(root, "prepare", [root / "records.json", root / "embeddings.npy", salt_file])


def match(args):
    root = args.output_dir
    if not stage_done(root, "prepare"):
        raise ValueError("prepare first")
    if stage_done(root, "match"):
        return
    active(args)
    records = read(root / "records.json")
    texts = [r["Report"] for r in records]
    groups = [r["cohort"] for r in records]
    matrices = {"bert_cls": cosine_matrix(np.load(root / "embeddings.npy", allow_pickle=False))}
    feature_counts = {}
    for name, config in [
        ("word_tfidf", POLICY["word_tfidf"]),
        ("char_tfidf", POLICY["char_tfidf"]),
    ]:
        config = {**config, "ngram_range": tuple(config["ngram_range"]), "dtype": np.float32}
        vectorizer = TfidfVectorizer(**config)
        matrix = vectorizer.fit_transform(texts)
        matrices[name] = (matrix @ matrix.T).toarray()
        feature_counts[name] = len(vectorizer.vocabulary_)
    candidates, review, strata = select_candidates(matrices, groups)
    shingles = [shingle_set(t) for t in texts]
    normalized = [normalized_text(t) for t in texts]
    # Exact coincidences are retained independently of retrieval ranks.
    exact_pairs = set()
    for field in ["normalized_text", KEY]:
        seen = {}
        values = normalized if field == "normalized_text" else [r[KEY] for r in records]
        for i, value in enumerate(values):
            for j in seen.get(value, []):
                pair = (j, i)
                candidates.setdefault(pair, set()).add(f"exact:{field}")
                exact_pairs.add(pair)
            seen.setdefault(value, []).append(i)
    rows, queue = [], []
    for (i, j), reasons in sorted(candidates.items()):
        pair_id = digest([records[i]["case_handle"], records[j]["case_handle"]])[:20]
        row = {
            "pair_id": pair_id,
            "index_a": i,
            "index_b": j,
            "cohort_a": groups[i],
            "cohort_b": groups[j],
            "handle_a": records[i]["case_handle"],
            "handle_b": records[j]["case_handle"],
            **{k: float(v[i, j]) for k, v in matrices.items()},
            "shingle_jaccard": jaccard(shingles[i], shingles[j]),
            "exact_normalized_text": normalized[i] == normalized[j],
            "same_report_key": records[i][KEY] == records[j][KEY],
            "selection_reasons": sorted(reasons),
            "patient_link_status": "unresolved_no_authoritative_anchor",
        }
        rows.append(row)
        if (i, j) in review:
            queue.append({**row, "review_selection": sorted(review[(i, j)])})
    summary = {
        "records": len(records),
        "unordered_pairs_scored_per_method": len(records) * (len(records) - 1) // 2,
        "unique_candidates": len(rows),
        "llm_queue_pairs": len(queue),
        "exact_coincidence_pairs": len(exact_pairs),
        "feature_counts": feature_counts,
        "strata": strata,
        "patient_identity_confirmed_pairs": 0,
        "candidate_lexical_counts": {
            str(t): sum(r["shingle_jaccard"] >= t for r in rows)
            for t in POLICY["lexical_diagnostic_thresholds"]
        },
        "gap_closed": False,
        "reason": "No authoritative patient anchors available",
    }
    atomic_write_json(root / "candidates.json", rows)
    atomic_write_csv(root / "candidates.csv", pd.DataFrame(rows))
    atomic_write_json(root / "review-queue.json", queue)
    atomic_write_json(root / "matching-summary.json", summary)
    seal_stage(
        root,
        "match",
        [
            root / n
            for n in [
                "candidates.json",
                "candidates.csv",
                "review-queue.json",
                "matching-summary.json",
            ]
        ],
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "records",
                    "unordered_pairs_scored_per_method",
                    "unique_candidates",
                    "llm_queue_pairs",
                ]
            }
        ),
        flush=True,
    )


def block_network():
    def denied(*_args, **_kwargs):
        raise RuntimeError("outbound network disabled for governed matching review")

    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"]:
        os.environ[key] = "1"


def review_pairs(args):
    root = args.output_dir
    if not stage_done(root, "match"):
        raise ValueError("match first")
    queue, records = read(root / "review-queue.json"), read(root / "records.json")
    remaining = [q for q in queue if not stage_done(root, f"pair-{q['pair_id']}")]
    if not remaining:
        return
    block_network()
    sys.path.insert(0, str(ROOT / "src/LLM_pipeline"))
    from llama_cpp import LlamaGrammar
    from llm_models import download_model_with_receipt

    active(args)
    model, model_receipt = download_model_with_receipt(
        "medgemma-27b-q2-candidate", local_files_only=True, load_overrides=LOAD
    )
    template = model.metadata.get("tokenizer.chat_template")
    if not template:
        raise ValueError("no embedded native chat template")
    runtime = {
        "model": {k: v for k, v in model_receipt.items() if k != "model_load_elapsed_seconds"},
        "chat_template": template,
        "chat_template_sha256": digest(template),
        "prompt_sha256": digest(REVIEW_PROMPT),
        "schema_sha256": digest(REVIEW_SCHEMA),
        "sampling": SAMPLING,
        "platform": platform.platform(),
    }
    runtime_file = root / "review-runtime.json"
    if runtime_file.exists() and read(runtime_file) != runtime:
        raise ValueError("review model/interface/runtime changed on resume")
    atomic_write_json(runtime_file, runtime)
    grammar = LlamaGrammar.from_json_schema(json.dumps(REVIEW_SCHEMA), verbose=False)
    try:
        for number, item in enumerate(remaining):
            if args.limit is not None and number >= args.limit:
                break
            active(args)
            if (root / "PAUSE").exists():
                break
            a, b = records[item["index_a"]]["Report"], records[item["index_b"]]["Report"]
            message = REVIEW_PROMPT.format(a=a, b=b)
            started = time.perf_counter()
            result = {
                "pair_id": item["pair_id"],
                "review_selection": item["review_selection"],
                "message_sha256": digest(message),
                "runtime_sha256": sha256_file(runtime_file),
                "patient_identity_confirmed": False,
                "started_at": utc(),
            }
            try:
                tokens = len(model.tokenize(message.encode(), add_bos=True))
                result["unserialized_message_tokens"] = tokens
                if tokens + SAMPLING["max_tokens"] + 128 > LOAD["n_ctx"]:
                    result["status"] = "skipped_context_budget_no_truncation"
                else:
                    response = model.create_chat_completion(
                        messages=[{"role": "user", "content": message}], grammar=grammar, **SAMPLING
                    )
                    result["raw_response"] = response
                    output = json.loads(response["choices"][0]["message"]["content"])
                    result["output"] = output
                    result["validation"] = validate_review(output, a, b)
                    result["status"] = (
                        "completed"
                        if response["choices"][0]["finish_reason"] == "stop"
                        else "truncated_retained"
                    )
            except Exception as error:
                # Keep raw output if present; no output-dependent retries or silent deletion.
                result["status"] = "failed_retained"
                result["error_type"] = type(error).__name__
            result["elapsed_seconds"] = time.perf_counter() - started
            path = root / "pair-reviews" / f"{item['pair_id']}.json"
            atomic_write_json(path, result)
            seal_stage(root, f"pair-{item['pair_id']}", [path, runtime_file])
            print(
                json.dumps(
                    {
                        "phase": "review",
                        "completed": len(queue) - len(remaining) + number + 1,
                        "total": len(queue),
                        "status": result["status"],
                        "seconds": round(result["elapsed_seconds"], 2),
                    }
                ),
                flush=True,
            )
    finally:
        model.close()


def finalize(args):
    root = args.output_dir
    if not stage_done(root, "match"):
        raise ValueError("verified match stage required")
    queue, records = read(root / "review-queue.json"), read(root / "records.json")
    if not all(stage_done(root, f"pair-{q['pair_id']}") for q in queue):
        return False
    if stage_done(root, "finalize"):
        return True
    results = [read(root / "pair-reviews" / f"{q['pair_id']}.json") for q in queue]
    reviewed = [r for r in results if "output" in r and "validation" in r]
    summary = {
        **read(root / "matching-summary.json"),
        "finished_at": utc(),
        "review_statuses": dict(Counter(r["status"] for r in results)),
        "model_relationships_unvalidated": dict(
            Counter(r["output"]["relationship"] for r in reviewed)
        ),
        "all_nonempty_quotes_verbatim_pairs": sum(
            r["validation"]["all_nonempty_quotes_verbatim"] for r in reviewed
        ),
        "possible_link_with_bilateral_quotes_pairs": sum(
            r["validation"]["specific_link_candidate_with_bilateral_source_quotes"]
            for r in reviewed
        ),
        "patient_identity_confirmed_pairs": 0,
        "validated_patient_map_created": False,
        "patient_linkage_precision_recall_estimable": False,
        "independent_patient_anchors_available": False,
        "classification_results_changed": False,
        "inference_seconds": sum(r["elapsed_seconds"] for r in results),
        "review_is_not_clinical_or_identity_adjudication": True,
    }
    atomic_write_json(root / "summary.json", summary)
    parts = [
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Private report-linkage candidates</title><style>body{max-width:1100px;margin:2em auto;
padding:1em;font:16px/1.5 Georgia}pre{white-space:pre-wrap;overflow-wrap:anywhere;
background:#f5f5f5;padding:1em}.pair{display:grid;grid-template-columns:1fr 1fr;gap:1em}
article{border-top:2px solid #aaa}
</style><h1>Private report-linkage candidates</h1><p>Governed, local review only. Do not email.
These are similarity candidates, NOT confirmed patient links. No patient map or clinical
adjudication has been created. Model quotes are checked for literal presence, not for truth
of the proposed relationship. Hash-selected controls are not known different-patient cases.</p>"""
    ]
    for q, result in zip(queue, results, strict=True):
        a, b = records[q["index_a"]], records[q["index_b"]]
        parts.append(
            f"<article><h2>{html.escape(q['pair_id'])}</h2><p>"
            + html.escape(", ".join(q["review_selection"]))
            + "</p><div class='pair'><pre>"
            + html.escape(a["Report"])
            + "</pre><pre>"
            + html.escape(b["Report"])
            + "</pre></div>"
            + "<details><summary>Unverified model review and source-quote checks</summary><pre>"
            + html.escape(json.dumps(result, indent=2))
            + "</pre></details></article>"
        )
    parts.append("</html>")
    path = root / "review.html"
    temporary = root / "review.html.tmp"
    temporary.write_text("".join(parts))
    temporary.replace(path)
    seal_stage(root, "finalize", [root / "summary.json", path])
    print(json.dumps({k: v for k, v in summary.items() if k != "strata"}), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["all", "prepare", "match", "review", "finalize", "status"]
    )
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-governed-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit", type=int, help="Operational per-call cap; does not change frozen queue"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    for name in ["original_run", "native_run", "output_dir"]:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    governed = (ROOT / "data/governed/analysis-runs").resolve()
    if not args.output_dir.is_relative_to(governed) or args.output_dir == governed:
        raise ValueError("dedicated governed analysis directory required")
    if not args.acknowledge_governed_output:
        raise ValueError("governed output acknowledgement required")
    active(args)
    if args.phase == "status":
        queue_file = args.output_dir / "review-queue.json"
        queue = read(queue_file) if queue_file.exists() else []
        print(
            json.dumps(
                {
                    "queued": len(queue),
                    "completed": sum(
                        stage_done(args.output_dir, f"pair-{q['pair_id']}") for q in queue
                    ),
                    "finalized": stage_done(args.output_dir, "finalize") if queue else False,
                }
            )
        )
        return
    if args.dry_run:
        _, _, receipt = source_intake(args)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "writes": False,
                    "records": receipt["records"],
                    "cache_identity_verified": True,
                    "cohorts": receipt["cohorts"],
                    "llm_pair_cap": POLICY["max_llm_pairs"],
                }
            )
        )
        return
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (args.output_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args)  # Recheck all source/cache/code digests before every resumed phase.
        if args.phase in {"all", "match"}:
            match(args)
        if args.phase in {"all", "review"}:
            review_pairs(args)
        if args.phase in {"all", "finalize"}:
            active(args)
            finalize(args)
        active(args)
        # Reverify the producing bundle and every input even after successful work.
        if source_intake(args)[2] != read(args.output_dir / "freeze.json"):
            raise ValueError("source changed during diagnostic")


if __name__ == "__main__":
    main()
