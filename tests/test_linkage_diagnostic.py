import importlib.util
import json
import socket
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from eeg_review.baseline import _reports_digest
from eeg_review.linkage_diagnostic import (
    POLICY,
    cosine_matrix,
    jaccard,
    normalized_text,
    select_candidates,
    shingle_set,
    stratum_indices,
    validate_review,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "linkage_runner", ROOT / "scripts/run_linkage_diagnostic.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def output(**changes):
    return {
        "relationship": "possible_longitudinal_link",
        "support_a": "prior study",
        "support_b": "prior study",
        "contradiction_a": "",
        "contradiction_b": "",
        "rationale": "Needs an independent anchor.",
        **changes,
    }


def test_citations_cannot_confirm_patient():
    result = validate_review(output(), "The prior study was reviewed.", "See prior study.")
    assert result["specific_link_candidate_with_bilateral_source_quotes"]
    assert not result["patient_identity_confirmed"]
    assert result["patient_link_status"] == "unresolved_no_authoritative_anchor"


def test_hallucinated_quote_is_retained_but_fails_support():
    result = validate_review(output(support_b="invented detail"), "prior study", "prior study")
    assert not result["all_nonempty_quotes_verbatim"]
    assert not result["specific_link_candidate_with_bilateral_source_quotes"]


def test_empty_quotes_cannot_supply_bilateral_support():
    result = validate_review(output(support_a="", support_b=""), "same template", "same template")
    assert result["all_nonempty_quotes_verbatim"]
    assert not result["specific_link_candidate_with_bilateral_source_quotes"]


@pytest.mark.parametrize(
    "changed",
    [
        {"patient_id": "invented"},
        {"relationship": "confirmed_patient"},
        {"support_a": "x" * 241},
        {"support_b": None},
        {"rationale": "x" * 601},
    ],
)
def test_schema_expansion_or_bad_fields_rejected(changed):
    with pytest.raises(ValueError):
        validate_review(output(**changed), "prior study", "prior study")


def test_exhaustive_six_strata_have_no_self_or_duplicate_pairs():
    groups = ["d"] * 3 + ["z"] * 4 + ["m"] * 5
    pairs = [
        tuple(sorted((int(i), int(j))))
        for _, a, b in stratum_indices(groups)
        for i, j in zip(a, b, strict=True)
    ]
    assert len(pairs) == len(set(pairs)) == 12 * 11 // 2
    assert all(i != j for i, j in pairs)


def test_selection_is_deterministic_and_budgeted_with_ties():
    groups = ["d"] * 3 + ["z"] * 4 + ["m"] * 5
    matrices = {k: np.ones((12, 12)) for k in ["bert_cls", "word_tfidf", "char_tfidf"]}
    first = select_candidates(matrices, groups)
    assert first == select_candidates(dict(reversed(list(matrices.items()))), groups)
    candidates, reviews, strata = first
    assert len(reviews) <= POLICY["max_llm_pairs"]
    assert set(reviews) <= set(candidates)
    assert sum(s["pairs_scored"] for s in strata) == 66
    assert all(i != j for i, j in candidates)


def test_nearest_neighbor_is_not_an_identity_decision():
    matrix = np.eye(4)
    candidates, review, _ = select_candidates({"only": matrix}, ["x"] * 4)
    assert candidates and review  # Even no positive similarity produces a neighbour.
    assert not POLICY["patient_links_inferred"]
    assert not POLICY["trial_until_full_linkage"]


@pytest.mark.parametrize("vectors", [np.zeros((2, 3)), np.array([[np.nan, 1], [1, 1]])])
def test_bad_vectors_rejected(vectors):
    with pytest.raises(ValueError):
        cosine_matrix(vectors)


def test_cosine_and_lexical_rules():
    assert cosine_matrix(np.eye(3))[0, 1] == 0
    assert normalized_text("  NORMAL\nEEG ") == "normal eeg"
    assert jaccard(set(), set()) == 0
    assert jaccard(shingle_set("a b c d e f"), shingle_set("a b c d e f")) == 1


def test_stage_output_and_freeze_tampering_rejected(tmp_path):
    (tmp_path / "freeze.json").write_text("{}")
    target = tmp_path / "output.json"
    target.write_text("{}")
    runner.seal_stage(tmp_path, "sample", [target])
    assert runner.stage_done(tmp_path, "sample")
    target.write_text('{"changed": true}')
    with pytest.raises(ValueError):
        runner.stage_done(tmp_path, "sample")
    target.write_text("{}")
    (tmp_path / "freeze.json").write_text('{"changed": true}')
    with pytest.raises(ValueError):
        runner.stage_done(tmp_path, "sample")


def test_cache_checks_full_order_and_text(tmp_path):
    db = tmp_path / "source.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE reports (Hashed_ReportURN TEXT, Report TEXT)")
        c.executemany("INSERT INTO reports VALUES (?,?)", [("a", "first"), ("b", "second")])
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "batch_000000_000002.npy", np.ones((2, 768)))
    manifest = {
        "records": 2,
        "checkpoint": "bert-base-uncased",
        "batch_size": 2,
        "report_digest": _reports_digest(["first", "second"]),
        "completed_batches": ["batch_000000_000002.npy"],
    }
    (cache / "embedding_cache.json").write_text(json.dumps(manifest))
    frame, matrix, _ = runner.cache_vectors(db, cache)
    assert len(frame) == 2 and matrix.shape == (2, 768)
    manifest["report_digest"] = _reports_digest(["second", "first"])
    (cache / "embedding_cache.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.cache_vectors(db, cache)


def test_review_cannot_finalize_without_matching_receipt(tmp_path):
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="verified match"):
        runner.finalize(SimpleNamespace(output_dir=tmp_path))


def test_outbound_connection_block(monkeypatch):
    # Restore socket methods after the test even though the runner blocks process-wide.
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket.socket, "connect_ex", socket.socket.connect_ex)
    for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"]:
        monkeypatch.setenv(key, "0")
    runner.block_network()
    with pytest.raises(RuntimeError, match="network disabled"):
        socket.create_connection(("example.invalid", 443))
