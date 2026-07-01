"""Offline tests for the entry-viability scoring model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from marketfit.features import FeatureBuilder
from marketfit.scoring import DEFAULT_WEIGHTS, MarketFitScorer

BUILDER = FeatureBuilder()


def _fv(macro, imports=None, demand=None, iso="TST"):
    return BUILDER.from_signals(iso, "0901", macro,
                                product_import_usd=imports, demand_interest=demand)


STRONG_MACRO = {
    "SP.POP.TOTL": 1_000_000_000, "NY.GDP.PCAP.CD": 60_000,
    "NY.GDP.MKTP.KD.ZG": 6.0, "FP.CPI.TOTL.ZG": 2.0,
    "NE.TRD.GNFS.ZS": 100.0, "IT.NET.USER.ZS": 95.0,
}
WEAK_MACRO = {
    "SP.POP.TOTL": 800_000, "NY.GDP.PCAP.CD": 1_200,
    "NY.GDP.MKTP.KD.ZG": -1.5, "FP.CPI.TOTL.ZG": 18.0,
    "NE.TRD.GNFS.ZS": 22.0, "IT.NET.USER.ZS": 12.0,
}


def test_score_bounds_and_ordering():
    scorer = MarketFitScorer()
    strong = scorer.score(_fv(STRONG_MACRO, imports=5e9, demand=90))
    weak = scorer.score(_fv(WEAK_MACRO, imports=2e6, demand=5))
    assert 1 <= weak.score <= strong.score <= 5
    assert 0.0 <= weak.composite <= strong.composite <= 1.0
    assert strong.is_success and not weak.is_success
    print("scoring OK -> strong:", strong.score, "weak:", weak.score)


def test_contributions_sum_to_composite():
    scorer = MarketFitScorer()
    result = scorer.score(_fv(STRONG_MACRO, imports=5e9, demand=90))
    assert result.contributions
    assert abs(sum(result.contributions.values()) - result.composite) < 1e-9
    # Used weights (renormalized) sum to 1 when all features present.
    assert abs(sum(result.used_weights.values()) - 1.0) < 1e-9


def test_missing_feature_renormalizes_weights():
    scorer = MarketFitScorer()
    # Drop the trade + demand signals; weights should renormalize over the rest.
    result = scorer.score(_fv(STRONG_MACRO))
    assert "existing_trade" in result.missing
    assert "consumer_demand" in result.missing
    assert abs(sum(result.used_weights.values()) - 1.0) < 1e-9
    # A strong macro profile should still read as a success without the extras.
    assert result.is_success


def test_top_drivers_and_gaps():
    scorer = MarketFitScorer()
    result = scorer.score(_fv(STRONG_MACRO, imports=5e9, demand=90))
    drivers = result.top_drivers(3)
    assert len(drivers) == 3
    # Drivers are sorted descending by contribution.
    assert drivers[0][1] >= drivers[-1][1]


def test_calibrate_threshold_improves_separation():
    scorer = MarketFitScorer(success_threshold=0.99)  # deliberately bad
    labeled = [
        (_fv(STRONG_MACRO, imports=5e9, demand=90), True),
        (_fv(WEAK_MACRO, imports=2e6, demand=5), False),
    ]
    t = scorer.calibrate_threshold(labeled)
    assert 0.0 <= t <= 1.0
    # After calibration the two cases separate correctly.
    assert scorer.score(labeled[0][0]).is_success
    assert not scorer.score(labeled[1][0]).is_success


def test_rejects_unknown_weight_key():
    with pytest.raises(ValueError):
        MarketFitScorer(weights={"not_a_feature": 1.0})


def test_default_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


if __name__ == "__main__":
    test_score_bounds_and_ordering()
    test_contributions_sum_to_composite()
    test_missing_feature_renormalizes_weights()
    test_top_drivers_and_gaps()
    test_calibrate_threshold_improves_separation()
    test_default_weights_sum_to_one()
    print("\nAll scoring tests passed.")
