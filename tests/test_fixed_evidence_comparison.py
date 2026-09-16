from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from eeg_review.evidence_extraction import JSON_KEYS
from eeg_review.fixed_evidence_comparison import (
    load_paired_evidence_surface,
    summarize_paired_evidence,
)


def classification(level=1):
    return json.dumps(dict.fromkeys(JSON_KEYS, level))


def explanation(level=1, reason="normal background"):
    return json.dumps(
        {key: {"decision": level, "reasons": [reason]} for key in JSON_KEYS}
    )


def fixture(tmp_path):
    dataset = tmp_path / "reports.db"
    reports = pd.DataFrame(
        {
            "Hashed_ReportURN": ["a", "b"],
            "Report": ["normal background", "abnormal slowing"],
        }
    )
    with sqlite3.connect(dataset) as connection:
        reports.to_sql("reports", connection, index=False)
    manifest = tmp_path / "manifest.csv"
    reports[["Hashed_ReportURN"]].to_csv(manifest, index=False)
    stream = pd.DataFrame(
        {
            "Hashed_ReportURN": ["a", "b"],
            "fixed_classifications": [classification(), classification()],
            "explanations": [explanation(), explanation(reason="slowing")],
        }
    )
    path = tmp_path / "evidence.csv"
    stream.to_csv(path, index=False)
    return dataset, manifest, path


def test_exact_paired_summary_contains_no_case_content(tmp_path):
    dataset, manifest, path = fixture(tmp_path)
    reports, streams = load_paired_evidence_surface(
        dataset=dataset,
        manifest=manifest,
        streams={"medgemma": (path, "fixed_classifications")},
    )
    summary = summarize_paired_evidence(reports, streams)
    serialized = json.dumps(summary)
    assert summary["records"] == 2
    assert summary["streams"]["medgemma"]["verified_exact_segments"] == 10
    assert "normal background" not in serialized
    assert "abnormal slowing" not in serialized
    assert '"a"' not in serialized and '"b"' not in serialized


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "decision", "order"])
def test_rejects_unpaired_or_changed_evidence(tmp_path, mutation):
    dataset, manifest, path = fixture(tmp_path)
    frame = pd.read_csv(path)
    if mutation == "missing":
        frame = frame.iloc[:1]
    elif mutation == "duplicate":
        frame.loc[1, "Hashed_ReportURN"] = "a"
    elif mutation == "decision":
        parsed = json.loads(frame.loc[0, "explanations"])
        parsed[JSON_KEYS[0]]["decision"] = 4
        frame.loc[0, "explanations"] = json.dumps(parsed)
    else:
        frame = frame.iloc[::-1]
        pd.DataFrame({"Hashed_ReportURN": ["b", "a"]}).to_csv(manifest, index=False)
    frame.to_csv(path, index=False)
    if mutation in {"missing", "duplicate"}:
        with pytest.raises(ValueError):
            load_paired_evidence_surface(
                dataset=dataset,
                manifest=manifest,
                streams={"stream": (path, "fixed_classifications")},
            )
        return
    reports, streams = load_paired_evidence_surface(
        dataset=dataset,
        manifest=manifest,
        streams={"stream": (path, "fixed_classifications")},
    )
    if mutation == "decision":
        with pytest.raises(ValueError, match="does not copy"):
            summarize_paired_evidence(reports, streams)
    else:
        assert reports["Hashed_ReportURN"].tolist() == ["b", "a"]
        assert streams["stream"]["Hashed_ReportURN"].tolist() == ["b", "a"]
