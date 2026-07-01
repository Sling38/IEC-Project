"""World Bank World Development Indicators (WDI) ingestion.

The World Bank Indicators API is free and requires no key:

    https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}?format=json

We pull a curated basket of macro indicators that describe a country's market
attractiveness (size, wealth, openness, connectivity) and return them as a tidy
:class:`pandas.DataFrame`, plus a single normalized "macro snapshot" per country.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

from marketfit.ingestion.cache import JsonCache

API_ROOT = "https://api.worldbank.org/v2"

# Curated indicators relevant to market-entry viability.
# code -> human-readable label
DEFAULT_INDICATORS: Dict[str, str] = {
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "SP.POP.TOTL": "Population, total",
    "SP.URB.TOTL.IN.ZS": "Urban population (% of total)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "NE.TRD.GNFS.ZS": "Trade (% of GDP)",
    "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
}


@dataclass
class WorldBankClient:
    """Client for the World Bank Indicators API with on-disk caching.

    Parameters
    ----------
    indicators:
        Mapping of indicator code -> label. Defaults to :data:`DEFAULT_INDICATORS`.
    timeout:
        Per-request timeout in seconds.
    cache_ttl_seconds:
        Freshness window for cached responses. ``None`` keeps entries indefinitely.
    """

    indicators: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_INDICATORS))
    timeout: float = 30.0
    cache_ttl_seconds: Optional[float] = None
    session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        self._cache = JsonCache("worldbank", ttl_seconds=self.cache_ttl_seconds)

    # -- low-level fetch ---------------------------------------------------

    def _fetch_indicator(
        self, country: str, indicator: str, start_year: int, end_year: int
    ) -> List[dict]:
        """Return the raw list of yearly observations for one indicator."""
        url = f"{API_ROOT}/country/{country}/indicator/{indicator}"
        params = {
            "format": "json",
            "date": f"{start_year}:{end_year}",
            "per_page": "1000",
        }
        cache_key = f"{url}?{sorted(params.items())}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        body = resp.json()
        # The API returns [metadata, observations]; observations may be None.
        observations = body[1] if isinstance(body, list) and len(body) > 1 else []
        observations = observations or []
        self._cache.set(cache_key, observations)
        return observations

    # -- public API --------------------------------------------------------

    def fetch_indicators(
        self,
        country: str,
        start_year: int = 2000,
        end_year: int = 2023,
        indicators: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Fetch a tidy time series of indicators for ``country`` (ISO-3 code).

        Returns a long-format DataFrame with columns:
        ``country_iso3, country, indicator_code, indicator, year, value``.
        """
        codes = list(indicators) if indicators is not None else list(self.indicators)
        rows: List[dict] = []
        for code in codes:
            for obs in self._fetch_indicator(country, code, start_year, end_year):
                if obs.get("value") is None:
                    continue
                rows.append(
                    {
                        "country_iso3": obs["countryiso3code"] or country.upper(),
                        "country": obs["country"]["value"],
                        "indicator_code": code,
                        "indicator": self.indicators.get(code, code),
                        "year": int(obs["date"]),
                        "value": float(obs["value"]),
                    }
                )
        columns = [
            "country_iso3",
            "country",
            "indicator_code",
            "indicator",
            "year",
            "value",
        ]
        return pd.DataFrame(rows, columns=columns).sort_values(
            ["indicator_code", "year"]
        ).reset_index(drop=True)

    def latest_snapshot(
        self,
        country: str,
        start_year: int = 2000,
        end_year: int = 2023,
    ) -> pd.DataFrame:
        """One row per indicator holding the most recent non-null value.

        This is the normalized "macro snapshot" the scoring agent consumes.
        Columns: ``indicator_code, indicator, year, value``.
        """
        df = self.fetch_indicators(country, start_year, end_year)
        if df.empty:
            return df
        latest = (
            df.sort_values("year")
            .groupby("indicator_code", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        return latest[["indicator_code", "indicator", "year", "value"]]
