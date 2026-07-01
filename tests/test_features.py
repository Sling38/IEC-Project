"""Offline tests for feature engineering."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from marketfit.features import FEATURE_NAMES, FeatureBuilder
from marketfit.features.engineering import _inflation_score, _log_lin


def test_all_features_normalized_in_unit_interval():
    builder = FeatureBuilder()
    macro = {
        "SP.POP.TOTL": 60_000_000,
        "NY.GDP.PCAP.CD": 35_000,
        "NY.GDP.MKTP.KD.ZG": 2.0,
        "FP.CPI.TOTL.ZG": 2.0,
        "NE.TRD.GNFS.ZS": 60.0,
        "IT.NET.USER.ZS": 85.0,
    }
    fv = builder.from_signals("ITA", "0901", macro,
                              product_import_usd=1_600_000_000, demand_interest=40)
    assert set(fv.values) == set(FEATURE_NAMES)
    assert all(0.0 <= v <= 1.0 for v in fv.values.values())
    assert fv.missing == ()
    # Ideal ~2% inflation should score near the top of price_stability.
    assert fv.values["price_stability"] > 0.9
    print("features OK ->", {k: round(v, 2) for k, v in fv.values.items()})


def test_missing_signals_are_tracked_not_faked():
    builder = FeatureBuilder()
    fv = builder.from_signals("XYZ", "0901", macro={"SP.POP.TOTL": 10_000_000})
    # Only market_size computable; everything else reported missing.
    assert "market_size" in fv.values
    assert "consumer_demand" in fv.missing
    assert "existing_trade" in fv.missing
    assert len(fv.values) == 1
    print("missing OK -> missing count:", len(fv.missing))


def test_monotonic_transforms():
    # Bigger market / richer country -> higher normalized value.
    assert _log_lin(1e9, 5e5, 1.4e9) > _log_lin(1e6, 5e5, 1.4e9)
    # Runaway and negative inflation both score worse than ~2%.
    assert _inflation_score(2.0) > _inflation_score(15.0)
    assert _inflation_score(2.0) > _inflation_score(-5.0)


def test_snapshot_to_macro_roundtrip():
    snap = pd.DataFrame(
        {
            "indicator_code": ["SP.POP.TOTL", "NY.GDP.PCAP.CD"],
            "indicator": ["Population, total", "GDP per capita"],
            "year": [2022, 2022],
            "value": [60_000_000.0, 35_000.0],
        }
    )
    macro = FeatureBuilder.snapshot_to_macro(snap)
    assert macro["SP.POP.TOTL"] == 60_000_000.0
    assert FeatureBuilder.snapshot_to_macro(pd.DataFrame()) == {}


if __name__ == "__main__":
    test_all_features_normalized_in_unit_interval()
    test_missing_signals_are_tracked_not_faked()
    test_monotonic_transforms()
    test_snapshot_to_macro_roundtrip()
    print("\nAll feature tests passed.")
