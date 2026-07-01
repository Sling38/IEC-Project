"""Feature engineering for the MarketFit scoring model.

Turns the three ingested signals for a ``(product, country)`` pair into a small,
interpretable, normalized feature vector that the scorer
(:mod:`marketfit.scoring.model`) consumes:

* **macro-fit** features from the World Bank snapshot (size, wealth, growth,
  price stability, openness, connectivity), and
* **trade-history** / **demand** features from Comtrade/BACI and Google Trends.

Every feature is mapped to ``[0, 1]`` where higher = more favorable for entry.
Because the ground-truth set is tiny (a handful of cases), we deliberately use
fixed, documented reference ranges rather than fitting scalers on the data — this
keeps the features stable and the scores explainable instead of overfit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import pandas as pd

# World Bank indicator codes we read out of the macro snapshot.
GDP_PER_CAPITA = "NY.GDP.PCAP.CD"
GDP_GROWTH = "NY.GDP.MKTP.KD.ZG"
POPULATION = "SP.POP.TOTL"
INFLATION = "FP.CPI.TOTL.ZG"
TRADE_PCT_GDP = "NE.TRD.GNFS.ZS"
INTERNET_PCT = "IT.NET.USER.ZS"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _lin(value: float, lo: float, hi: float) -> float:
    """Min-max normalize ``value`` into [0, 1] against the range ``[lo, hi]``."""
    if hi == lo:
        return 0.0
    return _clamp01((value - lo) / (hi - lo))


def _log_lin(value: float, lo: float, hi: float) -> float:
    """Like :func:`_lin` but on a log10 scale (for size/value quantities)."""
    if value <= 0:
        return 0.0
    return _lin(math.log10(value), math.log10(lo), math.log10(hi))


def _inflation_score(pct: float) -> float:
    """Favorable when inflation is low/stable. ~2% is ideal, penalize both tails.

    Deflation and runaway inflation are both bad; the sweet spot is ~1-3%.
    """
    ideal = 2.0
    # Distance from ideal, normalized so ~12pp away -> 0.
    return _clamp01(1.0 - abs(pct - ideal) / 12.0)


@dataclass(frozen=True)
class FeatureVector:
    """Normalized [0, 1] features for one ``(product, country)`` pair.

    ``values`` holds the named features; ``raw`` keeps the pre-normalization
    inputs for transparency in reports/rationale. ``missing`` lists features that
    could not be computed (so the scorer can renormalize its weights).
    """

    country_iso3: str
    hs_code: str
    values: Dict[str, float]
    raw: Dict[str, Optional[float]] = field(default_factory=dict)
    missing: tuple = ()

    def as_series(self) -> pd.Series:
        return pd.Series(self.values, name=f"{self.country_iso3}:{self.hs_code}")


# Feature names, kept as a stable ordered contract with the scorer.
FEATURE_NAMES = (
    "market_size",       # population
    "purchasing_power",  # GDP per capita
    "growth",            # GDP growth %
    "price_stability",   # inverse inflation risk
    "openness",          # trade % of GDP
    "connectivity",      # internet penetration
    "existing_trade",    # imports of the product (category demand/competition)
    "consumer_demand",   # Google Trends search interest
)


@dataclass
class FeatureBuilder:
    """Builds a :class:`FeatureVector` from already-fetched signal inputs.

    The builder is intentionally decoupled from the network clients: callers pass
    in the macro snapshot, the product import value, and the demand score. This
    keeps feature engineering deterministic and unit-testable offline, and lets
    the same code run against live pulls or cached fixtures.
    """

    # Reference ranges (documented, domain-chosen) for normalization.
    population_range: tuple = (5e5, 1.4e9)        # 0.5M .. 1.4B people
    gdp_per_capita_range: tuple = (1_000.0, 70_000.0)
    growth_range: tuple = (-2.0, 8.0)             # annual %
    trade_pct_range: tuple = (20.0, 120.0)        # trade % of GDP
    internet_range: tuple = (10.0, 95.0)          # % population
    import_value_range: tuple = (1e6, 1e10)       # product imports, USD

    def from_signals(
        self,
        country_iso3: str,
        hs_code: str,
        macro: Mapping[str, float],
        product_import_usd: Optional[float] = None,
        demand_interest: Optional[float] = None,
    ) -> FeatureVector:
        """Assemble a feature vector.

        Parameters
        ----------
        macro:
            Mapping of World Bank indicator code -> value (e.g. the dict form of
            :meth:`WorldBankClient.latest_snapshot`). Missing keys are tolerated.
        product_import_usd:
            Target country's import value (USD) of the product/HS category — a
            proxy for existing category demand and competitive intensity.
        demand_interest:
            Google Trends mean interest (0-100) for the brand/term in-market.
        """
        raw: Dict[str, Optional[float]] = {}
        values: Dict[str, float] = {}
        missing = []

        def put(name: str, raw_value: Optional[float], norm) -> None:
            raw[name] = raw_value
            if raw_value is None:
                missing.append(name)
            else:
                values[name] = _clamp01(float(norm(raw_value)))

        put("market_size", macro.get(POPULATION),
            lambda v: _log_lin(v, *self.population_range))
        put("purchasing_power", macro.get(GDP_PER_CAPITA),
            lambda v: _log_lin(v, *self.gdp_per_capita_range))
        put("growth", macro.get(GDP_GROWTH),
            lambda v: _lin(v, *self.growth_range))
        put("price_stability", macro.get(INFLATION), _inflation_score)
        put("openness", macro.get(TRADE_PCT_GDP),
            lambda v: _lin(v, *self.trade_pct_range))
        put("connectivity", macro.get(INTERNET_PCT),
            lambda v: _lin(v, *self.internet_range))
        put("existing_trade", product_import_usd,
            lambda v: _log_lin(v, *self.import_value_range))
        put("consumer_demand", demand_interest, lambda v: _clamp01(v / 100.0))

        return FeatureVector(
            country_iso3=country_iso3,
            hs_code=str(hs_code),
            values=values,
            raw=raw,
            missing=tuple(missing),
        )

    @staticmethod
    def snapshot_to_macro(snapshot: pd.DataFrame) -> Dict[str, float]:
        """Convert a :meth:`WorldBankClient.latest_snapshot` frame to a code->value dict."""
        if snapshot is None or snapshot.empty:
            return {}
        return dict(zip(snapshot["indicator_code"], snapshot["value"]))
