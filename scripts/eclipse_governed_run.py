#!/usr/bin/env python3
"""Fail closed a governed run without deleting its reviewable local record."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from eeg_review.protected_execution import ECLIPSE_MARKER


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve(strict=True)
    job = run_dir / "job.json"
    if not job.is_file():
        raise FileNotFoundError(f"Governed job receipt is absent: {job}")
    marker = run_dir / ECLIPSE_MARKER
    if marker.exists():
        print(marker.read_text(encoding="utf-8"), end="")
        return

    payload = {
        "schema_version": 1,
        "status": "eclipsed_governance_hold",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "actor": args.actor,
        "reason": args.reason,
        "job_sha256": sha256_file(job),
        "effects": {
            "further_execution_allowed": False,
            "analysis_allowed": False,
            "result_release_allowed": False,
            "local_record_preserved_pending_retention_decision": True,
        },
        "boundary": (
            "This marker withdraws the run from active scientific use. It does not delete "
            "governed files; deletion or archival requires a separate explicit decision."
        ),
    }
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(marker)
    marker.chmod(0o400)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
