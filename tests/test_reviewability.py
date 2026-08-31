from __future__ import annotations

import pandas as pd
import pytest

from eeg_review.audit import DEFAULT_LABELS
from eeg_review.reviewability import KEY, exact_frame, near_duplicate_pair, paired_packet, shingles


def frame(values):
    return pd.DataFrame(
        {
            KEY: [f"synthetic-{i}" for i in range(len(values))],
            **{label: values for label in DEFAULT_LABELS},
        }
    )


def test_all_strata_and_error_arithmetic():
    ref, a, b = frame([4, 4, 4, 1, 4, 1]), frame([4, 1, 1, 4, 4, 1]), frame([1, 4, 1, 4, 4, 1])
    summary, packet = paired_packet(ref, a, b, cohort="synthetic", salt="private")
    for row in summary["labels"].values():
        assert set(row["eligible"].values()) == {1}
        assert row["medgemma_errors"] == row["mistral_errors"] == 3
    assert summary["selected_unique_reports"] == 6
    assert "synthetic-0" not in packet.to_csv(index=False)
    assert "Report" not in packet and "patient" not in packet


def test_sampling_order_invariance_and_caps():
    ref, a, b = frame([4] * 40), frame([4] * 40), frame([1] * 40)
    one = paired_packet(ref, a, b, cohort="synthetic", salt="private")
    two = paired_packet(
        ref.iloc[::-1], a.sample(frac=1, random_state=2), b, cohort="synthetic", salt="private"
    )
    assert one[0] == two[0]
    pd.testing.assert_frame_equal(one[1], two[1])
    assert len(one[1]) == 5 * len(DEFAULT_LABELS)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra", "null", "invalid"])
def test_exact_alignment_fails_closed(mutation):
    original = frame([1, 4, 2])
    bad = original.copy()
    if mutation == "duplicate":
        bad.loc[1, KEY] = bad.loc[0, KEY]
    elif mutation == "missing":
        bad = bad.iloc[:2]
    elif mutation == "extra":
        bad.loc[3] = ["extra", *([1] * len(DEFAULT_LABELS))]
    elif mutation == "null":
        bad.loc[1, KEY] = None
    else:
        bad.loc[1, DEFAULT_LABELS[0]] = 0
    with pytest.raises(ValueError):
        exact_frame(bad, original[KEY].tolist())


def test_near_duplicate_exact_casefold_and_short_text_boundaries():
    words = " ".join(f"word{i}" for i in range(40))
    left = pd.DataFrame({KEY: ["source-a", "short"], "Report": [words, "too short"]})
    right = pd.DataFrame({KEY: ["source-b"], "Report": [words.upper()]})
    summary, pairs = near_duplicate_pair(left, right, left_name="dev", right_name="eval", salt="s")
    assert summary["flagged_pairs"] == 1 and pairs[0]["jaccard"] == 1
    assert summary["short_text_pairs_not_assessed"] == 1
    assert "source-a" not in str(pairs)
    assert shingles("no focal slowing is seen today") != shingles("focal slowing is seen today")


def test_near_duplicate_nonmatch_and_length_pruning():
    left = pd.DataFrame({KEY: ["a"], "Report": [" ".join(f"a{i}" for i in range(30))]})
    right = pd.DataFrame(
        {
            KEY: ["b", "c"],
            "Report": [" ".join(f"b{i}" for i in range(30)), " ".join(f"c{i}" for i in range(100))],
        }
    )
    summary, pairs = near_duplicate_pair(left, right, left_name="dev", right_name="eval", salt="s")
    assert pairs == []
    assert summary["length_bound_pruned_pairs"] == 1
    assert summary["exact_intersections"] == 1


@pytest.mark.parametrize("text", [None, "", "   "])
def test_missing_text_not_silently_dropped(text):
    with pytest.raises(ValueError, match="missing source"):
        shingles(text)
