"""UN Comtrade trade-flow ingestion (live API).

Uses the current UN Comtrade v1 API:

    https://comtradeapi.un.org/data/v1/get/{typeCode}/{freqCode}/{clCode}

The free "preview" tier works without a key but is capped (~500 rows / limited
calls). A subscription key (env var ``COMTRADE_API_KEY`` or the ``api_key`` arg)
lifts those limits and is sent via the ``Ocp-Apim-Subscription-Key`` header.

All responses are cached locally (see :mod:`marketfit.ingestion.cache`) because
the free tier is rate limited. For large historical bulk pulls prefer the BACI
loader (:mod:`marketfit.ingestion.baci`), which is built on cleaned Comtrade data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import pandas as pd
import requests

from marketfit.ingestion.cache import JsonCache

# Keyed (subscription) endpoint and the free, rate-limited preview endpoint.
API_ROOT = "https://comtradeapi.un.org/data/v1/get"
PREVIEW_ROOT = "https://comtradeapi.un.org/public/v1/preview"

# Columns we keep from the (very wide) Comtrade response, renamed to tidy names.
_COLUMN_MAP = {
    "refYear": "year",
    "reporterCode": "reporter_code",
    "reporterISO": "reporter_iso3",
    "reporterDesc": "reporter",
    "partnerCode": "partner_code",
    "partnerISO": "partner_iso3",
    "partnerDesc": "partner",
    "flowCode": "flow_code",
    "flowDesc": "flow",
    "cmdCode": "hs_code",
    "cmdDesc": "commodity",
    "primaryValue": "trade_value_usd",
    "netWgt": "net_weight_kg",
    "qty": "quantity",
}


@dataclass
class ComtradeClient:
    """Client for the UN Comtrade v1 API with local caching.

    Parameters
    ----------
    api_key:
        Subscription key. Falls back to the ``COMTRADE_API_KEY`` env var.
        ``None`` uses the rate-limited free preview tier.
    timeout:
        Per-request timeout in seconds.
    cache_ttl_seconds:
        Freshness window for cached responses. ``None`` keeps entries indefinitely.
    """

    api_key: Optional[str] = None
    timeout: float = 60.0
    cache_ttl_seconds: Optional[float] = None
    session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("COMTRADE_API_KEY")
        self._cache = JsonCache("comtrade", ttl_seconds=self.cache_ttl_seconds)

    # -- low-level fetch ---------------------------------------------------

    def _get(self, path: str, params: dict) -> List[dict]:
        # With a subscription key use the full endpoint; otherwise fall back to
        # the free preview endpoint (rate limited, ~500 rows).
        root = API_ROOT if self.api_key else PREVIEW_ROOT
        url = f"{root}/{path}"
        # Cache key includes params but NOT the secret key.
        cache_key = f"{url}?{sorted(params.items())}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        headers = {}
        query = dict(params)
        if self.api_key:
            headers["Ocp-Apim-Subscription-Key"] = self.api_key

        resp = self.session.get(
            url, params=query, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json().get("data", []) or []
        self._cache.set(cache_key, data)
        return data

    # -- public API --------------------------------------------------------

    def get_trade(
        self,
        reporter: Union[str, int] = "all",
        partner: Union[str, int] = "0",
        hs_code: str = "TOTAL",
        period: Union[str, int, List[Union[str, int]]] = "2022",
        flow: str = "M",
        frequency: str = "A",
    ) -> pd.DataFrame:
        """Fetch trade flows and return a tidy DataFrame.

        Parameters
        ----------
        reporter:
            Reporter country M49 code (e.g. ``842`` for USA) or ``"all"``.
        partner:
            Partner country M49 code. ``"0"`` means World (aggregate).
        hs_code:
            HS commodity code(s), e.g. ``"0901"`` (coffee) or ``"TOTAL"``.
        period:
            Year(s), e.g. ``2022`` or ``[2020, 2021, 2022]``.
        flow:
            ``"M"`` imports, ``"X"`` exports (Comtrade ``flowCode``).
        frequency:
            ``"A"`` annual or ``"M"`` monthly.

        Returns a DataFrame with the columns in :data:`_COLUMN_MAP` values.
        """
        if isinstance(period, (list, tuple)):
            period_str = ",".join(str(p) for p in period)
        else:
            period_str = str(period)

        params = {
            "reporterCode": str(reporter),
            "partnerCode": str(partner),
            "cmdCode": str(hs_code),
            "flowCode": flow,
            "period": period_str,
        }
        # typeCode=C (commodities), freqCode, clCode=HS
        raw = self._get(f"C/{frequency}/HS", params)
        return self._normalize(raw)

    @staticmethod
    def _normalize(raw: List[dict]) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame(columns=list(_COLUMN_MAP.values()))
        df = pd.DataFrame(raw)
        keep = {src: dst for src, dst in _COLUMN_MAP.items() if src in df.columns}
        df = df[list(keep)].rename(columns=keep)
        if "trade_value_usd" in df.columns:
            df = df.sort_values("trade_value_usd", ascending=False)
        return df.reset_index(drop=True)
