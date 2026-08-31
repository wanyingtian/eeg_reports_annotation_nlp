"""Additive literal-source checks, not semantic or clinical validation.

Only exact, nonblank substrings can be released as verified quotes. Normalized
matches are diagnostic hints and never change that acceptance rule. Original
outputs and decisions are preserved; this module does not run a model.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from typing import Any

from .evidence_extraction import FALLBACK_EVIDENCE, JSON_KEYS, classification_levels

POLICY_ID = "literal-source-span-v1"
TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact(text: str) -> str:
    return " ".join(text.split())


def typography(text: str) -> str:
    return compact(unicodedata.normalize("NFKC", text).translate(TYPOGRAPHY))


def whitespace_source_candidates(reason: str, report: str) -> list[dict]:
    """Locate source text without claiming the generated phrase was verbatim.

    Only whitespace is collapsed. Each returned quotation is sliced from the
    original report and carries original offsets; ambiguous occurrences remain
    multiple candidates, never a silently chosen location.
    """
    characters, offsets = [], []
    for index, char in enumerate(report):
        if char.isspace():
            if characters and characters[-1] != " ":
                characters.append(" ")
                offsets.append([index, index + 1])
            elif characters:
                offsets[-1][1] = index + 1
        else:
            characters.append(char)
            offsets.append([index, index + 1])
    normalized, needle = "".join(characters).rstrip(), compact(reason)
    if not needle:
        return []
    candidates, start = [], 0
    while (index := normalized.find(needle, start)) >= 0:
        a, b = offsets[index][0], offsets[index + len(needle) - 1][1]
        quote = report[a:b]
        if compact(quote) != needle:
            raise ValueError("source-offset normalization did not round-trip")
        candidates.append({"start": a, "end": b, "source_quote": quote})
        start = index + 1
    return candidates


def double_asterisk_source_candidates(reason: str, report: str) -> list[dict]:
    """Diagnostic only: elide source ** tokens, then collapse whitespace.

    This does not assume those tokens are actually Markdown or meaningless.
    Every candidate retains original offsets and text for inspection. Case,
    words, negation, numbers and every other punctuation character must agree.
    It is deliberately separate from the frozen literal-source-span-v1 policy.
    """
    if not isinstance(reason, str) or not reason.strip():
        return []
    characters, offsets = [], []
    index = 0
    while index < len(report):
        if report.startswith("**", index):
            index += 2
            continue
        char = report[index]
        if char.isspace():
            if characters and characters[-1] != " ":
                characters.append(" ")
                offsets.append([index, index + 1])
            elif characters:
                offsets[-1][1] = index + 1
        else:
            characters.append(char)
            offsets.append([index, index + 1])
        index += 1
    normalized, needle = "".join(characters).rstrip(), compact(reason)
    candidates, start = [], 0
    while (index := normalized.find(needle, start)) >= 0:
        a, b = offsets[index][0], offsets[index + len(needle) - 1][1]
        quote = report[a:b]
        if compact(quote.replace("**", "")) != needle:
            raise ValueError("double-asterisk source offsets did not round-trip")
        if "**" in quote:
            candidates.append({"start": a, "end": b, "source_quote": quote})
        start = index + 1
    return candidates


def inspect_reason(reason: Any, report: str) -> dict:
    result = {"accepted_as_verbatim": False, "source_spans": [], "source_span_candidates": []}
    if not isinstance(reason, str):
        return {**result, "status": "invalid_type"}
    if not reason.strip():
        return {**result, "status": "blank"}
    if reason == FALLBACK_EVIDENCE:
        return {**result, "status": "declared_no_evidence"}
    positions, start = [], 0
    while (offset := report.find(reason, start)) >= 0:
        positions.append({"start": offset, "end": offset + len(reason)})
        start = offset + 1  # Preserve multiple and overlapping matches.
    if positions:
        return {
            **result,
            "accepted_as_verbatim": True,
            "status": "exact",
            "source_spans": positions,
        }
    for status, transform in [
        ("casefold_only", str.casefold),
        ("whitespace_only", compact),
        ("typography_normalized_only", typography),
        ("typography_and_casefold_only", lambda value: typography(value).casefold()),
    ]:
        if transform(reason) in transform(report):
            candidates = (
                whitespace_source_candidates(reason, report) if status == "whitespace_only" else []
            )
            return {**result, "status": status, "source_span_candidates": candidates}
    return {**result, "status": "unmatched_requires_review"}


def _unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON field")
        output[key] = value
    return output


def inspect_grounding(raw: str, *, report: str, fixed: str) -> dict:
    """Keep raw response and fixed labels; attach a fail-closed quote-only view.

    Spans use zero-based Python Unicode code-point offsets with exclusive ends,
    against the exact source text bound by report_text_sha256. A span establishes
    text presence only, never that the text supports the assigned category.
    """
    decisions = classification_levels(fixed)
    result = {
        "policy_id": POLICY_ID,
        "report_text_sha256": text_sha(report),
        "raw_response": raw,
        "raw_response_sha256": text_sha(raw),
        "fixed_classifications": decisions,
        "schema_valid": False,
        "schema_error": None,
        "cells": {},
    }
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(parsed, dict) or set(parsed) != set(JSON_KEYS):
            raise ValueError("expected exactly five category objects")
        for cell in parsed.values():
            if not isinstance(cell, dict) or set(cell) != {"decision", "reasons"}:
                raise ValueError("wrong category fields")
            if type(cell["decision"]) is not int or cell["decision"] not in {1, 2, 3, 4}:
                raise ValueError("invalid four-level decision")
            if not isinstance(cell["reasons"], list):
                raise ValueError("reasons must be a list")
        result["schema_valid"] = True
    except (ValueError, TypeError) as exc:
        result["schema_error"] = str(exc)
        for key in JSON_KEYS:
            result["cells"][key] = {
                "decision": decisions[key],
                "decision_copy_matches": None,
                "status": "abstain_invalid_schema",
                "reasons": [],
                "verified_quotes": [],
            }
        return result
    for key in JSON_KEYS:
        cell = parsed[key]
        checked = [
            {"original_reason": value, **inspect_reason(value, report)} for value in cell["reasons"]
        ]
        matches = cell["decision"] == decisions[key]
        quotes = (
            [
                {"text": item["original_reason"], "source_spans": item["source_spans"]}
                for item in checked
                if item["accepted_as_verbatim"]
            ]
            if matches
            else []
        )
        if not matches:
            status = "abstain_decision_mismatch"
        elif not quotes:
            status = "abstain_no_verified_quote"
        elif all(item["status"] in {"exact", "declared_no_evidence"} for item in checked):
            status = "verified_quotes_only"
        else:
            status = "partial_verified_quotes"
        result["cells"][key] = {
            "decision": decisions[key],
            "generated_decision": cell["decision"],
            "decision_copy_matches": matches,
            "status": status,
            "reasons": checked,
            "verified_quotes": quotes,
        }
    return result


def aggregate_grounding(records: list[dict]) -> dict:
    """Aggregate all outputs, including rejected/abstained cells; no text or keys."""
    statuses, cell_statuses = Counter(), Counter()
    labels, complete_records, mismatch_count = {}, 0, 0
    recoverable_cells, unresolved_records = 0, 0
    for key in JSON_KEYS:
        labels[key] = {"cells": 0, "cells_with_verified_quotes": 0, "reason_statuses": Counter()}
    for record in records:
        unresolved_records += any(
            reason["status"] == "unmatched_requires_review"
            for cell in record["cells"].values()
            for reason in cell["reasons"]
        )
        complete_records += all(cell["verified_quotes"] for cell in record["cells"].values())
        for key, cell in record["cells"].items():
            recoverable_cells += cell["decision_copy_matches"] is True and any(
                item["accepted_as_verbatim"] or item["source_span_candidates"]
                for item in cell["reasons"]
            )
            labels[key]["cells"] += 1
            labels[key]["cells_with_verified_quotes"] += bool(cell["verified_quotes"])
            cell_statuses[cell["status"]] += 1
            mismatch_count += cell["decision_copy_matches"] is False
            for item in cell["reasons"]:
                statuses[item["status"]] += 1
                labels[key]["reason_statuses"][item["status"]] += 1
    nonfallback = sum(
        value
        for key, value in statuses.items()
        if key not in {"declared_no_evidence", "blank", "invalid_type"}
    )
    verified_cells = sum(x["cells_with_verified_quotes"] for x in labels.values())
    return {
        "records": len(records),
        "decision_cells": 5 * len(records),
        "invalid_schema_records": sum(not x["schema_valid"] for x in records),
        "decision_copy_mismatches": mismatch_count,
        "reason_statuses": dict(statuses),
        "cell_statuses": dict(cell_statuses),
        "nonfallback_nonblank_phrases": nonfallback,
        "literal_phrase_fraction": statuses["exact"] / nonfallback if nonfallback else None,
        "cells_with_verified_quotes": verified_cells,
        "cells_abstaining": 5 * len(records) - verified_cells,
        "cells_with_literal_or_whitespace_source_candidate": recoverable_cells,
        "source_locatable_phrases_whitespace_diagnostic": statuses["exact"]
        + statuses["whitespace_only"],
        "records_with_unmatched_phrases": unresolved_records,
        "records_with_verified_quotes_for_all_five_categories": complete_records,
        "by_category": {
            key: {**value, "reason_statuses": dict(value["reason_statuses"])}
            for key, value in labels.items()
        },
        "interpretation": [
            "Exact source presence, not clinical validity, entailment or causal faithfulness.",
            "Normalized-only matches remain unverified and are not repaired or accepted.",
            "Whitespace-only candidates retain generated wording and locate exact source slices.",
            "Abstention changes the evidence display, never the fixed classification.",
            "All record/cell denominators are retained; no rejected record is dropped.",
            "Phrase counts are repeated, dependent observations, not independent samples.",
            "An unmatched phrase is not by itself evidence of factual hallucination.",
        ],
    }
