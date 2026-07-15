"""Data assembly behind the demo UI.

The Streamlit app asks one question: *for this (product, market) pair, what do we
predict, and how does that compare to ground truth if we have it?* This module
answers it with plain functions so the flow is testable without a browser:

    signals_for(...)   -> the three input signals (fixtures or live pulls)
    assess_market(...) -> FeatureVector + ScoreResult (+ ground-truth row if any)

Live mode reuses the cached ingestion clients and *degrades per signal*: if a feed
is unreachable, the signal is simply absent and the scorer renormalizes its weights
(the same missing-feature behavior the model was built with). Every degradation is
reported in ``notes`` so the UI can say what happened rather than fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from marketfit.features import FeatureBuilder, FeatureVector
from marketfit.scoring import MarketFitScorer
from marketfit.scoring.model import ScoreResult
from marketfit.validation.harness import load_signal_fixtures


def find_case(
    cases: pd.DataFrame, country_iso3: str, hs_code: str
) -> Optional[pd.Series]:
    """Return the ground-truth case row for ``(country, product)``, if curated.

    Matches on ISO-3 (case-insensitive) and HS code. Returns ``None`` when the
    pair isn't in the curated set — the demo then shows the prediction alone.
    """
    if cases is None or cases.empty:
        return None
    hits = cases[
        (cases["country_iso3"].str.upper() == country_iso3.upper())
        & (cases["hs_code"].astype(str) == str(hs_code))
    ]
    if hits.empty:
        return None
    return hits.iloc[0]


def live_signals(
    country_iso3: str,
    hs_code: str,
    keyword: Optional[str] = None,
    comtrade_reporter_m49: Optional[int] = None,
    trade_year: int = 2022,
) -> Tuple[Dict, List[str]]:
    """Pull the three signals from the live (cached) ingestion clients.

    Returns ``(signals, notes)`` where ``signals`` matches the fixture schema
    (``macro`` / ``product_import_usd`` / ``demand_interest``) and ``notes`` lists
    any signal that had to be skipped and why. Never raises for a feed outage —
    a missing signal is legitimate input for the scorer.
    """
    signals: Dict = {"macro": {}}
    notes: List[str] = []

    try:
        from marketfit.ingestion import WorldBankClient

        snapshot = WorldBankClient().latest_snapshot(country_iso3)
        signals["macro"] = FeatureBuilder.snapshot_to_macro(snapshot)
        if not signals["macro"]:
            notes.append(f"World Bank returned no indicators for {country_iso3!r}.")
    except Exception as exc:  # feed outage -> degrade, don't fail the demo
        notes.append(f"World Bank unavailable ({exc.__class__.__name__}); macro signal skipped.")

    if comtrade_reporter_m49 is not None:
        try:
            from marketfit.ingestion import ComtradeClient

            flows = ComtradeClient().get_trade(
                reporter=comtrade_reporter_m49,
                partner=0,
                hs_code=hs_code,
                period=trade_year,
                flow="M",
            )
            if not flows.empty and "trade_value_usd" in flows.columns:
                signals["product_import_usd"] = float(flows["trade_value_usd"].sum())
            else:
                notes.append("Comtrade returned no rows; trade signal skipped.")
        except Exception as exc:
            notes.append(f"Comtrade unavailable ({exc.__class__.__name__}); trade signal skipped.")
    else:
        notes.append("No Comtrade M49 reporter code for this market; trade signal skipped.")

    if keyword:
        try:
            from marketfit.ingestion import GoogleTrendsClient, iso3_to_geo

            geo = iso3_to_geo(country_iso3)
            demand = GoogleTrendsClient().demand_snapshot(keyword, geo=geo)
            if not demand.empty:
                signals["demand_interest"] = float(demand["mean_interest"].iloc[0])
            else:
                notes.append(f"Google Trends returned no series for {keyword!r} in {geo or 'world'}.")
        except Exception as exc:
            notes.append(f"Google Trends unavailable ({exc.__class__.__name__}); demand signal skipped.")
    else:
        notes.append("No Trends keyword given; demand signal skipped.")

    return signals, notes


def signals_for(
    country_iso3: str,
    hs_code: str,
    source: str = "fixtures",
    keyword: Optional[str] = None,
    comtrade_reporter_m49: Optional[int] = None,
) -> Tuple[Optional[Dict], List[str]]:
    """Resolve signals from the requested ``source`` (``"fixtures"`` or ``"live"``).

    Fixture mode returns ``(None, notes)`` when the country isn't in the bundled
    fixture set; live mode always returns a (possibly partial) signals dict.
    """
    if source == "live":
        return live_signals(
            country_iso3,
            hs_code,
            keyword=keyword,
            comtrade_reporter_m49=comtrade_reporter_m49,
        )
    fixtures = load_signal_fixtures()
    sig = fixtures.get(country_iso3.upper())
    if sig is None:
        return None, [
            f"{country_iso3.upper()!r} is not in the bundled fixtures "
            f"({', '.join(sorted(fixtures))}). Switch to live mode for other markets."
        ]
    return sig, []


@dataclass(frozen=True)
class MarketAssessment:
    """Everything the demo shows for one (product, market) query."""

    country_iso3: str
    hs_code: str
    features: FeatureVector
    result: ScoreResult
    signals: Dict
    notes: Tuple[str, ...] = ()
    ground_truth: Optional[pd.Series] = field(default=None, compare=False)

    @property
    def has_ground_truth(self) -> bool:
        return self.ground_truth is not None


def assess_market(
    country_iso3: str,
    hs_code: str,
    signals: Dict,
    cases: Optional[pd.DataFrame] = None,
    scorer: Optional[MarketFitScorer] = None,
    notes: Tuple[str, ...] = (),
) -> MarketAssessment:
    """Build features from ``signals``, score them, and attach ground truth if any."""
    builder = FeatureBuilder()
    features = builder.from_signals(
        country_iso3=country_iso3.upper(),
        hs_code=str(hs_code),
        macro=signals.get("macro", {}),
        product_import_usd=signals.get("product_import_usd"),
        demand_interest=signals.get("demand_interest"),
    )
    result = (scorer or MarketFitScorer()).score(features)
    case = find_case(cases, country_iso3, hs_code) if cases is not None else None
    return MarketAssessment(
        country_iso3=country_iso3.upper(),
        hs_code=str(hs_code),
        features=features,
        result=result,
        signals=signals,
        notes=tuple(notes),
        ground_truth=case,
    )
