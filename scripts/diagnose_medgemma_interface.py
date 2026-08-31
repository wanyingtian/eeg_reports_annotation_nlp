#!/usr/bin/env python3
"""Small, local, resumable input-interface replay; no protected evaluation."""

from __future__ import annotations

import argparse
import fcntl
import html
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from eeg_review.evidence_extraction import JSON_KEYS, classification_levels
from eeg_review.interface_diagnostic import (
    POLICY,
    capture_chat_request,
    digest,
    manual_gemma_prompt,
    validate_checkpoint,
)
from eeg_review.io import atomic_write_json, load_table
from eeg_review.manifest import sha256_file
from eeg_review.native_interface import native_user_messages, sha256_text
from eeg_review.protected_execution import assert_governed_run_active

ROOT = Path(__file__).resolve().parents[1]
KEY = "Hashed_ReportURN"
PIPELINE = ROOT / "src/LLM_pipeline"
SOURCE_FILES = [
    Path(__file__),
    ROOT / "src/eeg_review/interface_diagnostic.py",
    ROOT / "src/eeg_review/native_interface.py",
    PIPELINE / "llm_models.py",
    ROOT / "review/MEDGEMMA_INTERFACE_MECHANICS_DIAGNOSTIC.md",
]


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(UTC).isoformat()


def active(args):
    for root in [args.native_run, args.historical_run, args.output_dir]:
        assert_governed_run_active(root)


def block_network():
    def denied(*_args, **_kwargs):
        raise RuntimeError("outbound network disabled for local interface replay")

    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    for name in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"]:
        os.environ[name] = "1"


def intake(args):
    active(args)
    native = args.native_run / "products/zoe_development_native_100"
    historical = args.historical_run / "products/zoe_development_transport_100"
    paths = {
        "database": args.native_run / "inputs/zoe_development_native_100.db",
        "manifest": args.native_run / "manifests/zoe_development_native_100.csv",
        "native_output": native / "raw.csv",
        "historical_output": historical / "raw.csv",
        "grammar": PIPELINE / "result_grammar.gbnf",
    }
    for name, path in paths.items():
        if sha256_file(path) != POLICY[name + "_sha256"]:
            raise ValueError(f"frozen {name} changed")
    nr, hr = read(native / "raw.run.json"), read(historical / "raw.run.json")
    for field in ["sampling", "execution_surface"]:
        if nr[field] != hr[field]:
            raise ValueError(f"parents differ in {field}")
    for field in ["sha256", "load_parameters"]:
        if nr["model"][field] != hr["model"][field]:
            raise ValueError(f"parent models differ in {field}")
    prompt = nr["prompts"]["classify"]["text"]
    if prompt != hr["prompts"]["classify"]["text"]:
        raise ValueError("parent classification prompt differs")
    if sha256_text(prompt) != POLICY["prompt_sha256"]:
        raise ValueError("frozen prompt changed")
    template = nr["input_policy"]["embedded_chat_template"]["text"]
    if sha256_text(template) != POLICY["template_sha256"]:
        raise ValueError("frozen chat template changed")
    versions = {name: importlib.metadata.version(name) for name in nr["environment"]["packages"]}
    if versions != nr["environment"]["packages"]:
        raise ValueError("runtime packages differ from the native run")
    manifest = pd.read_csv(paths["manifest"], usecols=[KEY])
    frame = load_table(paths["database"], [KEY, "Report"])
    parents = {
        arm: pd.read_csv(paths[arm + "_output"], usecols=[KEY, "classifications"])
        for arm in ["native", "historical"]
    }
    for value in [manifest, frame, *parents.values()]:
        if len(value) != 100 or value[KEY].isna().any() or value[KEY].duplicated().any():
            raise ValueError("invalid development key population")
        if set(value[KEY]) != set(manifest[KEY]):
            raise ValueError("development key set differs")
    frame = frame.set_index(KEY).loc[manifest[KEY]].reset_index()
    for arm, value in parents.items():
        frame[arm] = value.set_index(KEY).loc[manifest[KEY], "classifications"].to_list()
        for item in frame[arm]:
            classification_levels(item)
    paths.update(
        {
            "native_receipt": native / "raw.run.json",
            "historical_receipt": historical / "raw.run.json",
        }
    )
    contract = {
        "policy": POLICY,
        "inputs": {name: {"path": str(p), "sha256": sha256_file(p)} for name, p in paths.items()},
        "code": {str(p.relative_to(ROOT)): sha256_file(p) for p in SOURCE_FILES},
        "versions": versions,
        "sampling": nr["sampling"],
        "model": {k: nr["model"][k] for k in ["sha256", "load_parameters", "runtime_profile_id"]},
    }
    return frame, prompt, template, contract


def prepare(args):
    if args.output_dir.exists():
        raise FileExistsError("use run/status to resume an existing diagnostic")
    frame, _prompt, _template, contract = intake(args)
    args.output_dir.mkdir(parents=True, mode=0o700)
    (args.output_dir / "calls").mkdir(mode=0o700)
    atomic_write_json(args.output_dir / "contract.json", contract)
    atomic_write_json(
        args.output_dir / "execution.json",
        {
            "created_at_utc": now(),
            "contract_sha256": digest(contract),
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "authority": "Steven requested a bounded same-report local interface investigation",
            "governed_output": True,
            "new_evaluation_or_configuration_selection": False,
        },
    )
    publish(args, frame, contract)


def checkpoints(args, contract):
    values = []
    expected = {
        f"{p:03d}-{a}.json": (p, a) for p in POLICY["positions_zero_based"] for a in POLICY["arms"]
    }
    for path in sorted((args.output_dir / "calls").glob("*.json")):
        if path.name not in expected:
            raise ValueError("unexpected diagnostic checkpoint")
        p, a = expected[path.name]
        values.append(validate_checkpoint(read(path), digest(contract), p, a))
    return values


def paired_description(frame):
    counts = {
        key: {
            "ordinal_changes": 0,
            "binary_changes": 0,
            "historical_present_native_absent": 0,
            "historical_absent_native_present": 0,
        }
        for key in JSON_KEYS
    }
    changed_reports = 0
    for _, row in frame.iterrows():
        old, new = classification_levels(row.historical), classification_levels(row.native)
        changed_reports += old != new
        for key in JSON_KEYS:
            c = counts[key]
            c["ordinal_changes"] += old[key] != new[key]
            c["binary_changes"] += (old[key] >= 3) != (new[key] >= 3)
            c["historical_present_native_absent"] += old[key] >= 3 and new[key] < 3
            c["historical_absent_native_present"] += old[key] < 3 and new[key] >= 3
    return {
        "reports": 100,
        "reports_with_any_ordinal_change": changed_reports,
        "categories": counts,
        "reference_labels_used": False,
    }


def publish(args, frame, contract):
    active(args)
    calls = checkpoints(args, contract)
    lookup = {(c["position"], c["arm"]): c for c in calls}
    comparisons = {}
    for left, right in [
        ("native_chat", "assembled"),
        ("assembled", "assembled_original_stop"),
        ("historical", "trim_only"),
        ("historical", "native_chat"),
    ]:
        pairs = [
            (lookup[p, left], lookup[p, right])
            for p in POLICY["positions_zero_based"]
            if (p, left) in lookup and (p, right) in lookup
        ]
        valid = [(a, b) for a, b in pairs if a["levels"] is not None and b["levels"] is not None]
        comparisons[left + "_vs_" + right] = {
            "completed_pairs": len(pairs),
            "valid_pairs": len(valid),
            "same_output_text": sum(a["text"] == b["text"] for a, b in valid),
            "same_five_labels": sum(a["levels"] == b["levels"] for a, b in valid),
            "same_input_tokens": sum(
                a["input_token_ids"] == b["input_token_ids"] for a, b in pairs
            ),
        }
    saved_matches = {}
    for arm, parent in [("historical", "historical"), ("native_chat", "native")]:
        completed = [c for c in calls if c["arm"] == arm]
        saved_matches[arm] = {
            "completed": len(completed),
            "same_five_labels": sum(
                c["levels"] == classification_levels(frame.iloc[c["position"]][parent])
                for c in completed
            ),
        }
    result = {
        "diagnostic_id": POLICY["diagnostic_id"],
        "updated_at_utc": now(),
        "contract_sha256": digest(contract),
        "completed_calls": len(calls),
        "target_calls": POLICY["max_model_calls"],
        "status": "completed" if len(calls) == POLICY["max_model_calls"] else "incomplete",
        "invalid_outputs": sum(c["levels"] is None for c in calls),
        "comparisons": comparisons,
        "saved_parent_replay": saved_matches,
        "saved_development_disagreements": paired_description(frame),
        "new_accuracy_estimate": False,
        "protected_evaluation": False,
        "publication_admission": "not_requested_by_this_diagnostic",
        "call_receipts": {Path(c["file"]).name: c["receipt_sha256"] for c in calls},
    }
    atomic_write_json(args.output_dir / "summary.json", result)
    render_cases(args, frame, lookup)
    print(
        json.dumps(
            {k: result[k] for k in ["status", "completed_calls", "target_calls", "invalid_outputs"]}
        ),
        flush=True,
    )
    return result


def render_cases(args, frame, lookup):
    escape = lambda value: html.escape(str(value))  # noqa: E731
    body = [
        "<!doctype html><meta charset='utf-8'><title>Governed interface comparison</title>",
        "<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:2em auto}"
        "table{border-collapse:collapse;width:100%}td,th{padding:.4em;border:1px solid #bbb}"
        "pre{white-space:pre-wrap;background:#f4f5f6;padding:1em}details{margin:1em 0}"
        ".diff{background:#fff0d8}</style>",
        "<h1>Same report, different interface</h1><p>GOVERNED LOCAL CASE MATERIAL. "
        "Not for email, Git or publication. No reference labels are shown; "
        "a change does not by itself mean an improvement.</p>",
        "<p>All 100 saved development pairs are below. New replay is limited to the "
        "eight fixed positions. Each model label is 1/2 absent or 3/4 present.</p>",
    ]
    for position, row in frame.iterrows():
        old, new = classification_levels(row.historical), classification_levels(row.native)
        body.append(
            f"<details><summary>Development position {position + 1}: "
            f"{'labels differ' if old != new else 'labels agree'}</summary>"
        )
        body.append(
            "<pre>" + escape(row.Report) + "</pre><table><tr><th>Category</th>"
            "<th>Saved historical format</th><th>Saved native format</th></tr>"
        )
        for key in JSON_KEYS:
            cls = " class='diff'" if old[key] != new[key] else ""
            body.append(
                f"<tr{cls}><td>{escape(key)}</td><td>{old[key]}</td><td>{new[key]}</td></tr>"
            )
        body.append("</table>")
        for arm in POLICY["arms"]:
            if (position, arm) not in lookup:
                continue
            c = lookup[position, arm]
            body.append(
                f"<h3>Replay: {escape(arm)}</h3><pre>{escape(c['text'])}</pre>"
                f"<details><summary>Exact input and receipt</summary><pre>"
                f"{escape(c['prompt_text'])}</pre><p>Input tokens: {len(c['input_token_ids'])}; "
                f"receipt {escape(c['receipt_sha256'])}</p></details>"
            )
        body.append("</details>")
    target = args.output_dir / "same-report-comparison.html"
    temporary = target.with_suffix(".html.tmp")
    temporary.write_text("\n".join(body))
    temporary.replace(target)


def run(args):
    frame, prompt, template, contract = intake(args)
    if read(args.output_dir / "contract.json") != contract:
        raise ValueError("diagnostic contract changed; do not mix runs")
    existing = {(c["position"], c["arm"]) for c in checkpoints(args, contract)}
    if len(existing) == POLICY["max_model_calls"]:
        return publish(args, frame, contract)
    block_network()
    sys.path.insert(0, str(PIPELINE))
    from llama_cpp import LlamaGrammar
    from llm_models import download_model_with_receipt

    load = {
        k: v
        for k, v in contract["model"]["load_parameters"].items()
        if k not in {"logits_all", "verbose"}
    }
    model, receipt = download_model_with_receipt(
        "medgemma-27b-q2-candidate", local_files_only=True, load_overrides=load
    )
    if receipt["sha256"] != POLICY["model_sha256"]:
        raise ValueError("model artifact changed")
    if model.metadata.get("tokenizer.chat_template") != template:
        raise ValueError("loaded chat template differs")
    if not model._model.add_bos_token() or model._model.add_eos_token():
        raise ValueError("unexpected tokenizer BOS/EOS policy")
    bos = model._model.token_get_text(model.token_bos())
    grammar_text = (PIPELINE / "result_grammar.gbnf").read_text()
    sampling = {k: contract["sampling"][k] for k in ["temperature", "top_k", "top_p", "max_tokens"]}
    sampling["stop"] = contract["sampling"]["stop_sequences"]
    for position in POLICY["positions_zero_based"]:
        row = frame.iloc[position]
        payload = prompt + "\n\n" + str(row.Report)
        manual = manual_gemma_prompt(payload, bos)
        manual_tokens = model.tokenize(manual.encode(), add_bos=False, special=True)
        captured = capture_chat_request(model, native_user_messages(payload), sampling)
        if captured["prompt"] != manual_tokens or manual_tokens[:2] == [model.token_bos()] * 2:
            raise ValueError("manual prompt is not the exact native token stream")
        for arm in POLICY["arms"]:
            if (position, arm) in existing:
                continue
            active(args)
            if (args.output_dir / "PAUSE").exists():
                return publish(args, frame, contract)
            grammar = LlamaGrammar.from_string(grammar_text, verbose=False)
            if arm in {"historical", "trim_only"}:
                text_input = payload if arm == "historical" else payload.strip()
                tokens = [model.token_bos()] + model.tokenize(
                    text_input.encode(), add_bos=False, special=True
                )
                kwargs = {**sampling, "prompt": text_input, "grammar": grammar}
            else:
                text_input, tokens = manual, manual_tokens
                kwargs = {**captured, "prompt": tokens, "grammar": grammar}
                if arm == "assembled_original_stop":
                    kwargs.update(stop=None, stopping_criteria=None)
            if len(tokens) + sampling["max_tokens"] > load["n_ctx"]:
                raise ValueError("diagnostic input exceeds context; no silent truncation")
            model.reset()
            started = time.perf_counter()
            if arm == "native_chat":
                response = model.create_chat_completion(
                    messages=native_user_messages(payload), grammar=grammar, **sampling
                )
                text = response["choices"][0]["message"]["content"]
            else:
                response = model.create_completion(**kwargs)
                text = response["choices"][0]["text"]
            elapsed = time.perf_counter() - started
            actual_tokens = model.input_ids[: len(tokens)].tolist()
            tokens_verified = actual_tokens == tokens and response["usage"]["prompt_tokens"] == len(
                tokens
            )
            try:
                levels = classification_levels(text)
            except ValueError:
                levels = None
            safe_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in {"prompt", "grammar", "stopping_criteria"}
            }
            name = f"{position:03d}-{arm}.json"
            value = {
                "file": name,
                "contract_sha256": digest(contract),
                "position": position,
                "report_key": str(row[KEY]),
                "arm": arm,
                "created_at_utc": now(),
                "prompt_text": text_input,
                "input_token_ids": tokens,
                "input_tokens_sha256": digest(tokens),
                "actual_input_tokens_verified": tokens_verified,
                "actual_input_token_ids": actual_tokens,
                "effective_parameters": safe_kwargs,
                "native_eos_criterion": kwargs.get("stopping_criteria") is not None,
                "grammar_sha256": POLICY["grammar_sha256"],
                "model_sha256": receipt["sha256"],
                "text": text,
                "levels": levels,
                "usage": response["usage"],
                "finish_reason": response["choices"][0]["finish_reason"],
                "elapsed_seconds": elapsed,
            }
            value["receipt_sha256"] = digest(value)
            atomic_write_json(args.output_dir / "calls" / name, value)
            publish(args, frame, contract)
            if not tokens_verified:
                raise ValueError("actual input differs; discrepant output retained; stopping")
    model.close()
    return publish(args, frame, contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "status"])
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--historical-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    for name in ["native_run", "historical_run", "output_dir"]:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    governed = ROOT / "data/governed"
    if not args.output_dir.is_relative_to(governed):
        raise ValueError("diagnostic outputs must remain in the ignored governed workspace")
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run":
        with (args.output_dir / ".run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)
    else:
        frame, _p, _t, contract = intake(args)
        if read(args.output_dir / "contract.json") != contract:
            raise ValueError("diagnostic contract changed")
        publish(args, frame, contract)


if __name__ == "__main__":
    main()
