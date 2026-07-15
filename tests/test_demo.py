"""Offline tests for the demo-UI helper layer (marketfit.demo)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketfit.demo import assess_market, find_case, signals_for
from marketfit.groundtruth import GroundTruthLoader

CASES = GroundTruthLoader().load()


def test_find_case_matches_and_misses():
    hit = find_case(CASES, "ita", "0901")  # case-insensitive ISO-3
    assert hit is not None and hit["case_id"] == "SBUX-ITA-2018"
    assert find_case(CASES, "ITA", "9999") is None   # wrong product
    assert find_case(CASES, "USA", "0901") is None   # uncurated market
    print("find_case OK ->", hit["case_id"])


def test_signals_for_fixtures_hit_and_miss():
    sig, notes = signals_for("JPN", "0901", source="fixtures")
    assert sig is not None and sig["macro"] and not notes
    missing, notes = signals_for("USA", "0901", source="fixtures")
    assert missing is None and notes and "not in the bundled fixtures" in notes[0]
    print("signals_for OK -> JPN keys:", sorted(sig))


def test_assess_market_with_ground_truth():
    sig, _ = signals_for("ITA", "0901", source="fixtures")
    a = assess_market("ita", "0901", sig, cases=CASES)
    assert a.country_iso3 == "ITA"
    assert 1 <= a.result.score <= 5
    assert a.has_ground_truth and a.ground_truth["case_id"] == "SBUX-ITA-2018"
    print("assess_market OK -> ITA predicted", a.result.score,
          "actual", int(a.ground_truth["outcome_score"]))


def test_assess_market_without_ground_truth_or_cases():
    sig, _ = signals_for("KOR", "0901", source="fixtures")
    a = assess_market("KOR", "8703", sig, cases=CASES)  # cars, not curated
    assert not a.has_ground_truth
    b = assess_market("KOR", "0901", sig)               # no cases supplied
    assert not b.has_ground_truth
    assert a.result.score == b.result.score             # same signals, same score


def test_streamlit_app_compiles():
    import py_compile

    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    py_compile.compile(str(app), doraise=True)
    print("streamlit app compiles OK")


if __name__ == "__main__":
    test_find_case_matches_and_misses()
    test_signals_for_fixtures_hit_and_miss()
    test_assess_market_with_ground_truth()
    test_assess_market_without_ground_truth_or_cases()
    test_streamlit_app_compiles()
    print("\nAll demo tests passed.")
