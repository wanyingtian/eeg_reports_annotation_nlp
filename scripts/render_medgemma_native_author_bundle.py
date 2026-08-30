#!/usr/bin/env python3
"""Render deterministic aggregate-only authoring fragments from a native result receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from eeg_review.manuscript_admission import validate_manuscript_admission

EVIDENCE_ID = "JBHI-02463-2026-MEDGEMMA-NATIVE-PROTECTED-RESULT-CANDIDATE"
CONFIGURATION_ID = (
    "jbhi-02463/comparator/medgemma-27b-text-it/configuration/"
    "independent-native-interface-q2-v1"
)
COHORTS = [
    ("zoe_evaluation_1395", "Zoe", 1395),
    ("maria_evaluation_499", "Maria", 499),
]
COMPARATORS = [
    ("submitted_mistral", "Submitted Mistral"),
    ("reproduced_mistral", "Reproduced Mistral"),
]
LABELS = ["Abnormality", "Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi"]
EFFECTS = [
    "core_accuracy_difference",
    "certainty_adjusted_accuracy_difference",
    "false_negative_rate_difference",
]
ALL_COMPARATORS = [name for name, _label in COMPARATORS] + ["second_annotator"]
FORBIDDEN_FIELDS = {"Hashed_ReportURN", "Report", "report_text", "patient_key"}
FILES = {
    "table": "medgemma_native_core_comparisons.tex",
    "methods": "medgemma_native_methods.tex",
    "results": "medgemma_native_results.tex",
    "reviewer": "medgemma_native_reviewer_response.md",
    "ledger": "medgemma_native_claim_ledger.csv",
    "receipt": "bundle-receipt.json",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def number(value: Any) -> str:
    return f"{float(value):.3f}"


def p_value(value: Any) -> str:
    number_value = float(value)
    if number_value < 0.001:
        mantissa, exponent = f"{number_value:.2e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    return f"{number_value:.3f}"


def primary_claim_ids(payload: dict[str, Any]) -> set[str]:
    return {
        claim["claim_id"]
        for claim in payload.get("claim_candidates", [])
        if claim.get("comparator") in {name for name, _label in COMPARATORS}
        and claim.get("effect") == "core_accuracy_difference"
    }


def nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in nested_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in nested_keys(child)}
    return set()


def interval_direction(interval: dict[str, Any]) -> str:
    low = float(interval["low"])
    high = float(interval["high"])
    if low > 0:
        return "higher"
    if high < 0:
        return "lower"
    return "interval_includes_zero"


def validate_candidate(payload: dict[str, Any]) -> set[str]:
    if payload.get("evidence_id") != EVIDENCE_ID:
        raise ValueError("candidate evidence_id mismatch")
    if payload.get("configuration_id") != CONFIGURATION_ID:
        raise ValueError("candidate configuration_id mismatch")
    if payload.get("status") != "completed_validated_author_review_candidate":
        raise ValueError("candidate is not completed and validated")
    if payload.get("manuscript_admission") != "proposed_not_admitted":
        raise ValueError("candidate must remain proposed_not_admitted at intake")
    privacy = payload.get("privacy", {})
    if privacy != {
        "public_safe_aggregate": True,
        "case_level_content_included": False,
        "case_identifiers_included": False,
    }:
        raise ValueError("candidate privacy contract mismatch")
    claims = payload.get("claim_candidates")
    if not isinstance(claims, list) or len(claims) != 90:
        raise ValueError("candidate must contain all 90 prespecified claim rows")
    claim_ids = [claim.get("claim_id") for claim in claims]
    if any(not isinstance(claim_id, str) for claim_id in claim_ids):
        raise ValueError("candidate contains an invalid claim identifier")
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("candidate claim identifiers are not unique")
    if FORBIDDEN_FIELDS & nested_keys(payload):
        raise ValueError("candidate contains forbidden case-level fields")
    source_hashes = set(payload.get("source_aggregate_sha256", {}).values())
    expected_claims = {
        (cohort_id, comparator, label, effect)
        for cohort_id, _cohort_label, _records in COHORTS
        for comparator in ALL_COMPARATORS
        for label in LABELS
        for effect in EFFECTS
    }
    observed_claims: set[tuple[str, str, str, str]] = set()
    for claim in claims:
        identity = (
            claim.get("cohort_id"),
            claim.get("comparator"),
            claim.get("label"),
            claim.get("effect"),
        )
        observed_claims.add(identity)
        cohort_id, comparator, label, effect = identity
        expected_id = f"native-protected/{cohort_id}/{comparator}/{label}/{effect}"
        if claim.get("claim_id") != expected_id:
            raise ValueError(f"claim identity mismatch: {claim.get('claim_id')}")
        comparison = payload.get("comparisons", {}).get(cohort_id, {}).get(comparator, {})
        item = comparison.get("labels", {}).get(label, {})
        expected_estimate = item.get("effects_a_minus_b", {}).get(effect)
        expected_interval = item.get("paired_confidence_intervals_95", {}).get(effect)
        if claim.get("estimate_a_minus_b") != expected_estimate:
            raise ValueError(f"claim estimate mismatch: {expected_id}")
        if claim.get("ci_95") != expected_interval:
            raise ValueError(f"claim interval mismatch: {expected_id}")
        if not isinstance(expected_interval, dict) or claim.get(
            "direction_by_interval"
        ) != interval_direction(expected_interval):
            raise ValueError(f"claim interval direction mismatch: {expected_id}")
        if claim.get("interval_unit") != "report" or claim.get(
            "bootstrap_iterations"
        ) != 2000:
            raise ValueError(f"claim inference contract mismatch: {expected_id}")
        if claim.get("status") != "candidate_author_review_not_admitted":
            raise ValueError(f"claim admission status mismatch: {expected_id}")
        if claim.get("source_sha256") not in source_hashes:
            raise ValueError(f"claim source hash is unreceipted: {expected_id}")
    if observed_claims != expected_claims:
        raise ValueError("candidate claim factorial does not match the prespecified design")
    primary = primary_claim_ids(payload)
    if len(primary) != 20:
        raise ValueError("candidate must contain all 20 primary core-accuracy claims")
    return primary


def render_table(payload: dict[str, Any], mode: str) -> str:
    qualifier = (
        "Author-working; not admitted. "
        if mode == "author_working"
        else "Author-approved aggregate. "
    )
    lines = [
        "% Deterministically generated; do not edit by hand.",
        f"% Admission mode: {mode}.",
        r"{\renewcommand{\arraystretch}{0.82}",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{table*}[!t]",
        r"\scriptsize",
        r"\centering",
        (
            r"\caption{" + latex_escape(qualifier)
            + r"MedGemma-27B Q2\_K native-interface comparison. Differences are "
            + r"MedGemma minus the named Mistral comparator on exactly aligned reports. "
            + r"Intervals are paired 95\% report-bootstrap intervals; "
            + r"$p_{\mathrm{Holm}}$ is the Holm-adjusted exact report-level McNemar value.}"
        ),
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        (
            r"\textbf{Cohort} & \textbf{Comparator} & \textbf{Label} & "
            r"\textbf{MedGemma} & \textbf{Mistral} & "
            r"\textbf{Difference (95\% CI)} & \textbf{$p_{\mathrm{Holm}}$} \\"
        ),
        r"\midrule",
    ]
    rows = 0
    for cohort_id, cohort_label, expected in COHORTS:
        first_cohort = True
        for comparator_id, comparator_label in COMPARATORS:
            comparison = payload["comparisons"][cohort_id][comparator_id]
            if comparison.get("matched_records") != expected:
                raise ValueError(f"{cohort_id}/{comparator_id}: matched population mismatch")
            first_comparator = True
            for label in LABELS:
                item = comparison["labels"][label]
                interval = item["paired_confidence_intervals_95"][
                    "core_accuracy_difference"
                ]
                effects = item["effects_a_minus_b"]
                point_a = item["model_a_point_estimates"]["core_accuracy"]
                point_b = item["model_b_point_estimates"]["core_accuracy"]
                adjusted_p = item["discordant_correctness"]["core_accuracy"][
                    "multiplicity_adjusted_p_value"
                ]
                lines.append(
                    f"{cohort_label if first_cohort else ''} & "
                    f"{comparator_label if first_comparator else ''} & "
                    f"{latex_escape(label)} & {number(point_a)} & {number(point_b)} & "
                    f"{number(effects['core_accuracy_difference'])} "
                    f"({number(interval['low'])}, {number(interval['high'])}) & "
                    f"{p_value(adjusted_p)} \\\\"
                )
                first_cohort = False
                first_comparator = False
                rows += 1
            lines.append(r"\addlinespace[2pt]")
        lines.append(r"\midrule")
    if rows != 20:
        raise ValueError(f"expected 20 primary table rows, found {rows}")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\label{tab:medgemma_native_protected_comparison}",
            r"\end{table*}",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def render_tex_paragraph(text: str, mode: str) -> str:
    return "\n".join(
        [
            "% Deterministically generated; do not edit by hand.",
            f"% Admission mode: {mode}.",
            latex_escape(text),
            "",
        ]
    )


def render_reviewer_response(payload: dict[str, Any], mode: str) -> str:
    status = (
        "Author-working candidate; not admitted."
        if mode == "author_working"
        else "Approved for the reviewer response by the admission receipt."
    )
    return "\n".join(
        [
            f"> **Status:** {status}",
            "",
            payload["authoring_candidates"]["reviewer_response"],
            "",
            (
                "The associated table retains all 20 prespecified core-accuracy comparisons; "
                "the full aggregate ledger retains 90 effect rows including null and "
                "unfavorable outcomes. Patient-grouped inference was not added."
            ),
            "",
        ]
    )


def render_ledger(payload: dict[str, Any]) -> str:
    columns = [
        "claim_id",
        "cohort_id",
        "comparator",
        "label",
        "effect",
        "estimate_a_minus_b",
        "ci_low",
        "ci_high",
        "direction_by_interval",
        "interval_unit",
        "bootstrap_iterations",
        "source_sha256",
        "status",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for claim in sorted(payload["claim_candidates"], key=lambda item: item["claim_id"]):
        writer.writerow(
            {
                **{key: claim.get(key) for key in columns},
                "ci_low": claim["ci_95"]["low"],
                "ci_high": claim["ci_95"]["high"],
            }
        )
    return stream.getvalue()


def build_bundle(
    payload: dict[str, Any],
    candidate_sha256: str,
    admission_path: Path | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    required_claims = validate_candidate(payload)
    admission_sha256 = None
    admission = None
    mode = "author_working"
    if admission_path is not None:
        admission = validate_manuscript_admission(
            admission_path,
            candidate_sha256=candidate_sha256,
            required_claim_ids=required_claims,
        )
        if not admission.valid:
            raise ValueError("manuscript admission blocked: " + "; ".join(admission.blockers))
        admission_sha256 = sha256_file(admission_path)
        mode = "admitted_for_named_destinations"

    outputs = {
        FILES["table"]: render_table(payload, mode),
        FILES["methods"]: render_tex_paragraph(
            payload["authoring_candidates"]["methods"], mode
        ),
        FILES["results"]: render_tex_paragraph(
            payload["authoring_candidates"]["results"], mode
        ),
        FILES["reviewer"]: render_reviewer_response(payload, mode),
        FILES["ledger"]: render_ledger(payload),
    }
    receipt = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "configuration_id": CONFIGURATION_ID,
        "mode": mode,
        "candidate_receipt_sha256": candidate_sha256,
        "admission_receipt_sha256": admission_sha256,
        "approved_destinations": (
            list(admission.approved_destinations) if admission is not None else []
        ),
        "primary_claims_rendered": len(required_claims),
        "full_claim_ledger_rows": len(payload["claim_candidates"]),
        "source_aggregate_sha256": payload["source_aggregate_sha256"],
        "files": {
            filename: {
                "sha256": sha256_bytes(content.encode("utf-8")),
                "size_bytes": len(content.encode("utf-8")),
            }
            for filename, content in sorted(outputs.items())
        },
        "boundaries": [
            "No case-level content or identifier is present in this bundle.",
            "Patient-grouped inference is not represented.",
            "The completed matched-historical Q2 result remains a separate evidence layer.",
            "The external v5g configuration remains separate pending exact intake.",
            (
                "Author-working fragments are not manuscript-admitted."
                if admission is None
                else (
                    "Admission is limited to the destinations and claims in the "
                    "hash-bound receipt."
                )
            ),
        ],
    }
    outputs[FILES["receipt"]] = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    return outputs, receipt


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    candidate_path = args.candidate.expanduser().resolve(strict=True)
    admission_path = (
        args.admission.expanduser().resolve(strict=True) if args.admission else None
    )
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    outputs, receipt = build_bundle(payload, sha256_file(candidate_path), admission_path)
    output_dir = args.output_dir.expanduser().absolute()
    if args.check:
        stale = [
            filename
            for filename, content in outputs.items()
            if not (output_dir / filename).is_file()
            or (output_dir / filename).read_text(encoding="utf-8") != content
        ]
        if stale:
            raise SystemExit("stale authoring bundle files: " + ", ".join(stale))
        print(json.dumps({"verified": True, "mode": receipt["mode"]}, indent=2))
        return
    for filename, content in outputs.items():
        atomic_text(output_dir / filename, content)
    print(
        json.dumps(
            {
                "rendered": True,
                "mode": receipt["mode"],
                "output_dir": str(output_dir),
                "files": sorted(outputs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
