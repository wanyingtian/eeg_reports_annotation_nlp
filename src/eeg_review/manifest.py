from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty() -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_manifest(command: str, inputs: list[Path], parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "git_revision": git_revision(),
        "git_worktree_dirty": git_worktree_dirty(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(
            [
                "numpy",
                "pandas",
                "scikit-learn",
                "torch",
                "transformers",
                "llama-cpp-python",
            ]
        ),
        "inputs": [
            {
                "name": path.name,
                "sha256": sha256_file(path.expanduser().resolve(strict=True)),
            }
            for path in inputs
        ],
        "parameters": parameters,
        "privacy_boundary": "aggregate outputs only; no report text or row identifiers emitted",
    }
