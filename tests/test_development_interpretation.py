import copy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from eeg_review import category_evidence
from eeg_review.development_interpretation import KEY, ground_independent, interpret
from eeg_review.evidence_extraction import JSON_KEYS
from eeg_review.logprob_adapter import JSON_KEY_TO_LABEL


def fixture():
    keys = ["synthetic-a", "synthetic-b"]
    reference = pd.DataFrame(
        {
            KEY: keys,
            "Report": ["A source\npassage."] * 2,
            **{label: [1, 4] for label in JSON_KEY_TO_LABEL.values()},
        }
    )
    raw = [json.dumps(dict.fromkeys(JSON_KEYS, 1)), json.dumps(dict.fromkeys(JSON_KEYS, 4))]
    versions = {
        name: pd.DataFrame({KEY: keys, "classifications": raw}) for name in ("v1", "v2", "v21")
    }
    payload = {key: {role: [] for role in category_evidence.FIELDS} for key in JSON_KEYS}
    payload[JSON_KEYS[0]]["present_evidence"] = ["A source passage."]
    evidence = pd.DataFrame(
        {KEY: keys[:1], "fixed_classifications": raw[:1], "explanations": [json.dumps(payload)]}
    )
    return reference, versions, evidence, keys[:1]


def test_preserves_empty_missing_and_whitespace_without_changing_decisions():
    args = fixture()
    before = copy.deepcopy(args[1])
    summary, packet = interpret(*args)
    evidence = summary["independent_evidence"]
    assert evidence["records"] == 1 and evidence["category_cells"] == 5
    assert evidence["empty_role_lists"] == 14 and evidence["nonempty_category_cells"] == 1
    assert evidence["phrase_statuses"] == {"whitespace_only": 1}
    assert evidence["cells_with_literal_quotes"] == 0
    assert evidence["cells_with_literal_or_whitespace_candidates"] == 1
    assert evidence["present_list_with_negative_classification"] == 1
    assert packet[1]["independent_audit"] is None
    assert packet[0]["categories"][JSON_KEYS[0]]["semantic_alignment"] == "not_adjudicated"
    for name in before:
        pd.testing.assert_frame_equal(args[1][name], before[name])


def test_all_empty_is_not_successful_evidence_or_a_normal_label():
    reference, versions, evidence, keys = fixture()
    payload = {key: {role: [] for role in category_evidence.FIELDS} for key in JSON_KEYS}
    evidence.loc[0, "explanations"] = json.dumps(payload)
    summary, _ = interpret(reference, versions, evidence, keys)
    assert summary["independent_evidence"]["all_empty_valid_records"] == 1
    assert summary["independent_evidence"]["phrase_instances"] == 0
    assert summary["independent_evidence"]["empty_role_lists"] == 15


@pytest.mark.parametrize(
    "mutation", ["parent", "duplicate", "missing", "order", "reference", "prefix", "linkage"]
)
def test_join_rejects_population_or_parent_drift(mutation):
    reference, versions, evidence, keys = fixture()
    if mutation == "parent":
        del versions["v1"]
    elif mutation == "duplicate":
        versions["v1"].loc[1, KEY] = "synthetic-a"
    elif mutation == "missing":
        versions["v2"].loc[0, KEY] = None
    elif mutation == "order":
        versions["v21"] = versions["v21"].iloc[::-1]
    elif mutation == "reference":
        reference.loc[0, "Abnormality"] = 0
    elif mutation == "prefix":
        keys = ["synthetic-b"]
    else:
        evidence.loc[0, "fixed_classifications"] = json.dumps(dict.fromkeys(JSON_KEYS, 4))
    with pytest.raises(ValueError):
        interpret(reference, versions, evidence, keys)


def test_unchanged_totals_do_not_hide_case_exchanges():
    reference, versions, evidence, keys = fixture()
    # Both parents: miss the positive; candidate: repair that miss, add a false positive.
    for name in ("v1", "v2"):
        row = json.loads(versions[name].loc[1, "classifications"])
        row["abnormality"] = 1
        versions[name].loc[1, "classifications"] = json.dumps(row)
    row = json.loads(versions["v21"].loc[0, "classifications"])
    row["abnormality"] = 4
    versions["v21"].loc[0, "classifications"] = json.dumps(row)
    evidence.loc[0, "fixed_classifications"] = json.dumps(row)
    summary, _ = interpret(reference, versions, evidence, keys)
    c = summary["paired_changes"]["v2"]["Abnormality"]
    assert c["repair"] == c["regression"] == 1 and c["core_changed"] == 2
    assert c["changed_cells_with_independent_audit"] == 1
    assert summary["binary_counts"]["v2"]["Abnormality"]["fn"] == 1
    assert summary["binary_counts"]["v21"]["Abnormality"]["fp"] == 1


def test_invalid_audit_preserves_denominator_and_does_not_synthesize_evidence():
    args = fixture()
    args[2].loc[0, "explanations"] = "{}"
    summary, packet = interpret(*args)
    assert summary["independent_evidence"]["invalid_schema_records"] == 1
    assert summary["independent_evidence"]["category_cells"] == 5
    assert packet[0]["categories"][JSON_KEYS[0]]["independent_evidence"] is None


def test_marker_diagnostic_does_not_upgrade_literal_acceptance():
    raw = json.dumps(
        {
            key: {
                "present_evidence": ["A source passage."],
                "absent_evidence": [],
                "qualification_evidence": [],
            }
            for key in JSON_KEYS
        }
    )
    result = ground_independent(raw, "A **source** passage.")
    phrase = result["cells"][JSON_KEYS[0]]["present_evidence"][0]
    assert phrase["double_asterisk_candidates"]
    assert phrase["accepted_as_verbatim"] is False


def test_source_manifest_cannot_omit_required_files_or_change_hash(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/audit_medgemma_v21.py"
    spec = importlib.util.spec_from_file_location("v21audit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "state.json").write_text(json.dumps({"status": "completed"}))
    (tmp_path / "job.json").write_text(json.dumps({"study_id": module.STUDY}))
    path = tmp_path / "final-scientific-manifest.json"
    path.write_text(
        json.dumps({"files": [{"path": "job.json", "sha256": module.sha(tmp_path / "job.json")}]})
    )
    with pytest.raises(ValueError, match="omitted"):
        module.validate_source(tmp_path)
    payload = json.loads(path.read_text())
    payload["files"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="changed"):
        module.validate_source(tmp_path)
