from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .adaptation_plan import (
    AdaptationPlanStatus,
    parse_adaptation_plan,
    validate_adaptation_plan,
)
from .audit import DEFAULT_LABELS
from .io import (
    atomic_write_csv,
    atomic_write_json,
    load_table,
    quote_identifier,
    sqlite_columns,
    sqlite_connection_readonly,
)
from .manifest import build_manifest, sha256_file

DEVELOPMENT_COHORT_ID = "zoe_development_first_100_ra"
DEVELOPMENT_RECORDS = 100
MANIFEST_FILENAME = "zoe-development-first-100-ra.manifest.csv"
RECEIPT_FILENAME = "zoe-development-first-100-ra.manifest.receipt.json"


def _digest_keys(keys: list[str], *, ordered: bool) -> str:
    values = keys if ordered else sorted(keys)
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_development_frame(
    reference_path: Path,
    *,
    table: str,
    id_column: str,
    labels: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [id_column, *labels]
    if reference_path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        available = set(sqlite_columns(reference_path, table))
        missing = sorted(set(columns) - available)
        if missing:
            raise ValueError(f"Missing columns in {table}: {missing}")
        selection = ", ".join(quote_identifier(column) for column in columns)
        with sqlite_connection_readonly(reference_path) as connection:
            frame = pd.read_sql_query(
                f"SELECT {selection} FROM {quote_identifier(table)} ORDER BY rowid",
                connection,
            )
        ordering_semantics = "sqlite_rowid_ascending"
    else:
        frame = load_table(reference_path, columns, table)
        ordering_semantics = "file_row_order"
    if len(frame) != DEVELOPMENT_RECORDS:
        raise ValueError(
            f"Development snapshot must contain exactly {DEVELOPMENT_RECORDS} rows; "
            f"found {len(frame)}"
        )

    raw_keys = frame[id_column].astype("string")
    if raw_keys.isna().any():
        raise ValueError("Development snapshot contains missing report keys")
    keys = raw_keys.astype(str)
    if (keys.str.strip() == "").any():
        raise ValueError("Development snapshot contains blank report keys")
    if (keys != keys.str.strip()).any():
        raise ValueError("Development snapshot contains report keys with boundary whitespace")
    if keys.duplicated().any():
        raise ValueError("Development snapshot contains duplicate report keys")

    label_counts: dict[str, dict[str, int]] = {}
    for label in labels:
        numeric = pd.to_numeric(frame[label], errors="coerce")
        valid = numeric.isin([1, 2, 3, 4])
        if not valid.all():
            raise ValueError(f"Development snapshot contains invalid or missing {label} labels")
        label_counts[label] = {
            str(level): int((numeric == level).sum()) for level in (1, 2, 3, 4)
        }

    normalized = frame.copy()
    normalized[id_column] = keys
    return normalized, {
        "records": len(normalized),
        "ordered_key_sha256": _digest_keys(keys.tolist(), ordered=True),
        "key_set_sha256": _digest_keys(keys.tolist(), ordered=False),
        "duplicate_keys": 0,
        "missing_keys": 0,
        "complete_labels": True,
        "label_counts": label_counts,
        "ordering_semantics": ordering_semantics,
    }


def create_development_manifest(
    reference_path: Path,
    output_dir: Path,
    *,
    table: str = "reports",
    id_column: str = "Hashed_ReportURN",
    labels: list[str] | None = None,
    acknowledge_governed_output: bool = False,
) -> dict[str, Any]:
    """Create an immutable keyed manifest and non-row-level receipt in governed storage."""
    if not acknowledge_governed_output:
        raise ValueError("Explicit acknowledgement of governed keyed output is required")
    labels = labels or DEFAULT_LABELS
    if labels != DEFAULT_LABELS:
        raise ValueError("The development manifest requires all five canonical labels in order")

    reference_path = reference_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path = output_dir / MANIFEST_FILENAME
    receipt_path = output_dir / RECEIPT_FILENAME
    if manifest_path.exists() or receipt_path.exists():
        raise FileExistsError("Development manifest outputs are immutable and already exist")

    frame, summary = _validated_development_frame(
        reference_path,
        table=table,
        id_column=id_column,
        labels=labels,
    )
    atomic_write_csv(manifest_path, frame[[id_column]])
    manifest_path.chmod(0o600)

    run_manifest = build_manifest(
        "development-manifest-create",
        [reference_path],
        {
            "cohort_id": DEVELOPMENT_COHORT_ID,
            "table": table,
            "id_column": id_column,
            "ordering_semantics": summary["ordering_semantics"],
            "labels": labels,
            "expected_records": DEVELOPMENT_RECORDS,
        },
        privacy_boundary=(
            "The keyed CSV is governed and must remain in authorized storage; this receipt "
            "contains only counts and cryptographic identities, with no report text or key."
        ),
    )
    receipt = {
        "schema_version": 1,
        "receipt_type": "immutable_development_manifest",
        "cohort_id": DEVELOPMENT_COHORT_ID,
        "source": {
            "path": reference_path.name,
            "sha256": sha256_file(reference_path),
            "table": table,
            "id_column": id_column,
            "ordering_semantics": summary["ordering_semantics"],
        },
        "manifest": {
            "path": manifest_path.name,
            "sha256": sha256_file(manifest_path),
            "id_column": id_column,
        },
        **summary,
        "run_manifest": run_manifest,
        "ready_for_development_plan_binding": True,
        "ready_for_evaluation": False,
    }
    atomic_write_json(receipt_path, receipt)
    receipt_path.chmod(0o600)
    return receipt


def _relative_artifact_path(artifact_path: Path, plan_path: Path) -> str:
    return os.path.relpath(artifact_path, start=plan_path.parent)


def prepare_adaptation_execution_plan(
    preregistered_plan_path: Path,
    development_reference_path: Path,
    development_manifest_path: Path,
    development_manifest_receipt_path: Path,
    output_plan_path: Path,
    *,
    reference_table: str = "reports",
    id_column: str = "Hashed_ReportURN",
    labels: list[str] | None = None,
    acknowledge_governed_output: bool = False,
) -> dict[str, Any]:
    """Bind immutable development artifacts into an unfrozen governed execution plan."""
    if not acknowledge_governed_output:
        raise ValueError("Explicit acknowledgement of governed execution output is required")
    labels = labels or DEFAULT_LABELS
    if labels != DEFAULT_LABELS:
        raise ValueError("The execution plan requires all five canonical labels in order")

    preregistered_plan_path = preregistered_plan_path.expanduser().resolve(strict=True)
    development_reference_path = development_reference_path.expanduser().resolve(strict=True)
    development_manifest_path = development_manifest_path.expanduser().resolve(strict=True)
    development_manifest_receipt_path = (
        development_manifest_receipt_path.expanduser().resolve(strict=True)
    )
    output_plan_path = output_plan_path.expanduser().resolve()
    output_plan_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    preparation_receipt_path = output_plan_path.with_suffix(".preparation.receipt.json")
    if output_plan_path.exists() or preparation_receipt_path.exists():
        raise FileExistsError("Execution-plan outputs are immutable and already exist")

    plan_validation = validate_adaptation_plan(preregistered_plan_path)
    if not plan_validation["design_valid"]:
        raise ValueError("Preregistered adaptation plan is invalid")
    with preregistered_plan_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    plan = parse_adaptation_plan(payload)
    if plan.status != AdaptationPlanStatus.PREREGISTERED_UNFROZEN:
        raise ValueError("Only a preregistered-unfrozen plan may be prepared for development")

    with development_manifest_receipt_path.open(encoding="utf-8") as stream:
        manifest_receipt = json.load(stream)
    if not isinstance(manifest_receipt, dict):
        raise ValueError("Development manifest receipt must be a JSON object")
    if manifest_receipt.get("receipt_type") != "immutable_development_manifest":
        raise ValueError("Development manifest receipt has the wrong receipt type")
    if manifest_receipt.get("cohort_id") != DEVELOPMENT_COHORT_ID:
        raise ValueError("Development manifest receipt has the wrong cohort identity")
    if manifest_receipt.get("records") != DEVELOPMENT_RECORDS:
        raise ValueError("Development manifest receipt has the wrong record count")
    if manifest_receipt.get("complete_labels") is not True:
        raise ValueError("Development manifest receipt does not attest complete labels")
    if manifest_receipt.get("ready_for_development_plan_binding") is not True:
        raise ValueError("Development manifest receipt is not ready for plan binding")

    source_identity = manifest_receipt.get("source")
    manifest_identity = manifest_receipt.get("manifest")
    if not isinstance(source_identity, dict) or not isinstance(manifest_identity, dict):
        raise ValueError("Development manifest receipt lacks source or manifest identity")
    reference_sha256 = sha256_file(development_reference_path)
    manifest_sha256 = sha256_file(development_manifest_path)
    if source_identity.get("sha256") != reference_sha256:
        raise ValueError("Development reference checksum does not match its manifest receipt")
    if manifest_identity.get("sha256") != manifest_sha256:
        raise ValueError("Development manifest checksum does not match its receipt")
    if source_identity.get("table") != reference_table:
        raise ValueError("Development reference table does not match its manifest receipt")
    if source_identity.get("id_column") != id_column:
        raise ValueError("Development reference key column does not match its manifest receipt")
    if source_identity.get("ordering_semantics") not in {
        "sqlite_rowid_ascending",
        "file_row_order",
    }:
        raise ValueError("Development reference ordering semantics are not explicit")
    if manifest_identity.get("id_column") != id_column:
        raise ValueError("Development manifest key column does not match its receipt")

    _, live_summary = _validated_development_frame(
        development_reference_path,
        table=reference_table,
        id_column=id_column,
        labels=labels,
    )
    if source_identity.get("ordering_semantics") != live_summary["ordering_semantics"]:
        raise ValueError("Development reference ordering semantics do not match the receipt")
    manifest = load_table(development_manifest_path, [id_column], "manifest")
    manifest_keys = manifest[id_column].astype("string")
    if manifest_keys.isna().any() or (manifest_keys.astype(str).str.strip() == "").any():
        raise ValueError("Development manifest contains missing or blank report keys")
    manifest_key_values = manifest_keys.astype(str).tolist()
    if len(manifest_key_values) != DEVELOPMENT_RECORDS:
        raise ValueError("Development manifest must contain exactly 100 report keys")
    if len(set(manifest_key_values)) != DEVELOPMENT_RECORDS:
        raise ValueError("Development manifest contains duplicate report keys")
    ordered_digest = _digest_keys(manifest_key_values, ordered=True)
    set_digest = _digest_keys(manifest_key_values, ordered=False)
    if ordered_digest != manifest_receipt.get("ordered_key_sha256"):
        raise ValueError("Development manifest ordered-key digest does not match its receipt")
    if set_digest != manifest_receipt.get("key_set_sha256"):
        raise ValueError("Development manifest key-set digest does not match its receipt")
    if ordered_digest != live_summary["ordered_key_sha256"]:
        raise ValueError("Development manifest order does not match the reference snapshot")

    reference_artifact = {
        "path": _relative_artifact_path(development_reference_path, output_plan_path),
        "sha256": reference_sha256,
    }
    manifest_artifact = {
        "path": _relative_artifact_path(development_manifest_path, output_plan_path),
        "sha256": manifest_sha256,
    }
    development_signal_matches = 0
    for signal in payload.get("signals", []):
        if signal.get("signal_id") == DEVELOPMENT_COHORT_ID:
            signal["artifact"] = reference_artifact
            development_signal_matches += 1
    if development_signal_matches != 1:
        raise ValueError("Plan must contain exactly one fixed Zoe development signal")
    payload["certainty_mapping"]["development_manifest"] = manifest_artifact

    atomic_write_json(output_plan_path, payload)
    output_plan_path.chmod(0o600)
    bound_validation = validate_adaptation_plan(
        output_plan_path,
        bundle_root=output_plan_path.parent,
        check_files=True,
    )
    if not bound_validation["design_valid"] or not bound_validation["ready_for_implementation"]:
        output_plan_path.unlink(missing_ok=True)
        raise ValueError("Bound execution plan failed adaptation-plan validation")
    if bound_validation["ready_for_evaluation"]:
        output_plan_path.unlink(missing_ok=True)
        raise ValueError("Development preparation must not produce an evaluation-ready plan")

    receipt = {
        "schema_version": 1,
        "receipt_type": "adaptation_development_execution_preparation",
        "base_plan": {
            "path": preregistered_plan_path.name,
            "sha256": sha256_file(preregistered_plan_path),
        },
        "execution_plan": {
            "path": output_plan_path.name,
            "sha256": sha256_file(output_plan_path),
        },
        "development_reference_sha256": reference_sha256,
        "development_manifest_sha256": manifest_sha256,
        "development_manifest_receipt_sha256": sha256_file(
            development_manifest_receipt_path
        ),
        "cohort_id": DEVELOPMENT_COHORT_ID,
        "records": DEVELOPMENT_RECORDS,
        "status": AdaptationPlanStatus.PREREGISTERED_UNFROZEN.value,
        "ready_for_development_inference_after_author_decisions": True,
        "ready_for_evaluation": False,
        "privacy_boundary": (
            "Execution plan and receipts remain governed. Report keys and reference labels "
            "are not copied into either JSON artifact."
        ),
    }
    atomic_write_json(preparation_receipt_path, receipt)
    preparation_receipt_path.chmod(0o600)
    return receipt
