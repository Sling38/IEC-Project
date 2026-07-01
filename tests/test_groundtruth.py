"""Tests for the ground-truth cases and outcome-label taxonomy.

These load the committed curated CSV (no network) and check that it is internally
consistent — every label is known and every score matches its label — plus a couple
of spot checks on documented outcomes.

Run with:  python -m pytest tests/  (or: python tests/test_groundtruth.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from marketfit.groundtruth import (
    GroundTruthLoader,
    OUTCOME_LABELS,
    label_for,
    score_for,
)
from marketfit.groundtruth.cases import REQUIRED_COLUMNS


def test_label_taxonomy():
    # The four-rung taxonomy maps onto the 1-5 entry-viability score.
    assert score_for("Strong Success") == 5
    assert score_for("Withdrew") == 1
    assert label_for("Struggled").is_success is False
    assert label_for("Moderate Success").is_success is True
    # Scores are unique so a label round-trips from its score.
    scores = [label.score for label in OUTCOME_LABELS.values()]
    assert len(scores) == len(set(scores))
    print("labels OK ->", {name: label.score for name, label in OUTCOME_LABELS.items()})


def test_cases_load_and_validate():
    df = GroundTruthLoader().load()
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) >= 6                       # Checkpoint 1 asks for 4-6 cases
    # hs_code keeps its leading zero (read as string).
    assert (df["hs_code"] == "0901").all()
    # Every label is valid and its score is consistent (validate() enforces this).
    assert set(df["outcome_label"]).issubset(OUTCOME_LABELS.keys())
    for _, row in df.iterrows():
        assert int(row["outcome_score"]) == score_for(row["outcome_label"])
    print("cases OK -> rows:", len(df))


def test_documented_outcomes():
    df = GroundTruthLoader().load().set_index("case_id")
    # Australia is the canonical Starbucks withdrawal; Japan the canonical success.
    assert df.loc["SBUX-AUS-2000", "outcome_label"] == "Withdrew"
    assert df.loc["SBUX-JPN-1996", "outcome_label"] == "Strong Success"
    # Every case carries a source URL for #Factual evidence.
    assert df["source_url"].str.startswith("http").all()
    # All four outcome labels appear at least once (a balanced validation set).
    assert set(df["outcome_label"]) == set(OUTCOME_LABELS.keys())
    print("outcomes OK -> labels present:", sorted(set(df["outcome_label"])))


if __name__ == "__main__":
    test_label_taxonomy()
    test_cases_load_and_validate()
    test_documented_outcomes()
    print("\nAll ground-truth tests passed.")
