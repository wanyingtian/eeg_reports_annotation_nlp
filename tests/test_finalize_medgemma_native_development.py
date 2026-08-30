from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/finalize_medgemma_native_development.py"
SPEC = importlib.util.spec_from_file_location("native_finalize", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    keys = [f"key-{index:03d}" for index in range(100)]
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, [{"Hashed_ReportURN": key} for key in keys])
    raw = tmp_path / "raw.csv"
    rows = []
    for index, key in enumerate(keys):
        values = {label: 1 for label in MODULE.LABELS}
        values[MODULE.LABELS[index % len(MODULE.LABELS)]] = 4
        rows.append(
            {
                "Hashed_ReportURN": key,
                "classification_interface_mode": "native_chat",
                "classifications": json.dumps(values),
            }
        )
    write_csv(raw, rows)
    receipt = tmp_path / "raw.run.json"
    payload = {
        "model": {"sha256": MODULE.EXPECTED["model"]},
        "prompts": {"classify": {"sha256": MODULE.EXPECTED["prompt"]}},
        "grammars": {"classify": {"sha256": MODULE.EXPECTED["grammar"]}},
        "input_policy": {
            "classification_interface_mode": "native_chat",
            "embedded_chat_template": {"sha256": MODULE.EXPECTED["chat_template"]},
            "task_message_template": {"sha256": MODULE.EXPECTED["task_message"]},
        },
        "reports_completed": 100,
        "output": {"sha256": MODULE.sha256_file(raw)},
        "environment": {"git": {"revision": "a" * 40, "worktree_dirty": False}},
    }
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    return raw, receipt, manifest


def test_complete_nonconstant_singleton_is_selected(tmp_path: Path) -> None:
    result = MODULE.finalize(*fixture(tmp_path))
    assert result["selected_for_freeze"] is True
    assert result["blockers"] == []
    assert result["population"]["full_pattern_cardinality"] == 5


def test_duplicate_or_constant_output_is_rejected(tmp_path: Path) -> None:
    raw, receipt, manifest = fixture(tmp_path)
    rows = list(csv.DictReader(raw.open(encoding="utf-8")))
    rows[1]["Hashed_ReportURN"] = rows[0]["Hashed_ReportURN"]
    constant = json.dumps({label: 1 for label in MODULE.LABELS})
    for row in rows:
        row["classifications"] = constant
    write_csv(raw, rows)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["output"]["sha256"] = MODULE.sha256_file(raw)
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    result = MODULE.finalize(raw, receipt, manifest)
    assert result["selected_for_freeze"] is False
    assert "duplicate output keys" in result["blockers"]
    assert "degenerate constant five-label output pattern" in result["blockers"]
