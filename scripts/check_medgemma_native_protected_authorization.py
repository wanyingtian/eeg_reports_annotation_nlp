#!/usr/bin/env python3
"""Check the protected-evaluation authorization gate without running inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_review.protected_execution import build_unlock_receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = build_unlock_receipt(args.authorization)
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
        args.output.chmod(0o600)
    print(rendered, end="")
    if not receipt["protected_evaluation_unlocked"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
