"""Sensitivity analysis: what actually moves the score?

Two questions this answers, both of which turned out to matter for how we describe
the system (final report §4.6, §5):

1. **How much of the score is the country, the product, and the company?** Six of the
   eight features are pure country attributes; only ``existing_trade`` depends on the
   product (HS code) and only ``consumer_demand`` depends on the company (the Google
   Trends brand keyword). We report the weight split and show the score range obtained
   by sweeping the brand signal across its full 0–100 range with everything else fixed.

2. **Is the trade feature calibrated beyond coffee?** ``FeatureBuilder.import_value_range``
   is a hand-set $1M–$10B window. This sweeps import volumes across orders of magnitude
   to show where the feature floors and saturates.

Usage::

    python scripts/experiment_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketfit.features import FeatureBuilder
from marketfit.scoring import DEFAULT_WEIGHTS, MarketFitScorer
from marketfit.validation import load_signal_fixtures

# Which input each feature actually varies with.
PROVENANCE = {
    "market_size": ("country", "population"),
    "purchasing_power": ("country", "GDP per capita"),
    "growth": ("country", "GDP growth"),
    "price_stability": ("country", "inflation"),
    "openness": ("country", "trade % of GDP"),
    "connectivity": ("country", "internet %"),
    "existing_trade": ("product", "imports for the HS code"),
    "consumer_demand": ("company", "Trends brand keyword"),
}

# Illustrative import volumes spanning well outside the coffee-scale window.
VOLUME_PROBES = [
    ("artisanal / niche good", 5e5),
    ("specialty tea", 2e7),
    ("coffee, HS 0901 (calibrated)", 1.3e9),
    ("crude petroleum", 1.2e11),
    ("all goods, TOTAL", 7.5e11),
]


def weight_split() -> None:
    """Print each feature's provenance and the country/product/company weight split."""
    print("1. What each feature varies with\n")
    print(f"{'feature':22}{'varies with':12}{'source':26}{'weight':>7}")
    print("-" * 67)
    totals = {"country": 0.0, "product": 0.0, "company": 0.0}
    for feature, weight in DEFAULT_WEIGHTS.items():
        kind, source = PROVENANCE[feature]
        totals[kind] += weight
        print(f"{feature:22}{kind:12}{source:26}{weight:>7.2f}")
    print("-" * 67)
    print(f"country-only {totals['country']:.0%}   product {totals['product']:.0%}   "
          f"company {totals['company']:.0%}")


def brand_sensitivity(iso3: str = "JPN", hs_code: str = "0901") -> None:
    """Sweep the brand signal with country and product held fixed."""
    builder, scorer = FeatureBuilder(), MarketFitScorer()
    macro = load_signal_fixtures()[iso3]["macro"]
    imports = load_signal_fixtures()[iso3]["product_import_usd"]

    print(f"\n\n2. Same country ({iso3}) + same product (HS {hs_code}), brand varied\n")
    print(f"{'Trends interest':>18}{'composite':>12}{'score':>8}")
    print("-" * 38)
    scores = []
    for interest in (0, 10, 25, 50, 70, 100):
        fv = builder.from_signals(iso3, hs_code, macro, imports, interest)
        result = scorer.score(fv)
        scores.append(result.score)
        print(f"{interest:>18}{result.composite:>12.3f}{result.score:>8}")
    print("-" * 38)
    print(f"An unknown brand and a globally dominant one differ by "
          f"{max(scores) - min(scores)} point(s) on the 1-5 scale.")


def volume_calibration(iso3: str = "JPN") -> None:
    """Show where the trade feature floors and saturates outside its calibrated range."""
    builder = FeatureBuilder()
    macro = load_signal_fixtures()[iso3]["macro"]

    print(f"\n\n3. Trade-feature calibration (import_value_range="
          f"{builder.import_value_range})\n")
    print(f"{'product category':30}{'imports (USD)':>18}{'existing_trade':>16}")
    print("-" * 64)
    for label, volume in VOLUME_PROBES:
        fv = builder.from_signals(iso3, "x", macro, volume, 50)
        value = fv.values["existing_trade"]
        note = "  <- floored" if value <= 0.0 else ("  <- saturated" if value >= 1.0 else "")
        print(f"{label:30}{volume:>18,.0f}{value:>16.2f}{note}")
    print("\nOutside roughly $1M-$10B the feature is a constant and carries no signal,")
    print("so it effectively drops out of the model for other product categories.")


def main() -> int:
    weight_split()
    brand_sensitivity()
    volume_calibration()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
