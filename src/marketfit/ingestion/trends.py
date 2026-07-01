"""Google Trends demand-signal ingestion (via pytrends).

Google Trends is MarketFit's third input signal: real-time **consumer demand**.
For a given search term (a product/brand) and geography we pull the normalized
"interest over time" series and reduce it to a single demand snapshot per market.

pytrends wraps an *unofficial* Google Trends endpoint that is rate limited and can
break unexpectedly (HTTP 429s, schema changes). So, exactly like the Comtrade and
World Bank clients, every response is cached on disk (see
:mod:`marketfit.ingestion.cache`) — a re-run hits Google only once per unique query
and a later service hiccup never blocks downstream work.

Geographies use Google Trends' ISO-2 ``geo`` codes (``"IT"``, ``"JP"``); the empty
string means worldwide. :func:`iso3_to_geo` maps the ISO-3 codes used elsewhere in
the pipeline (and in the ground-truth cases) to those geo codes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Union

import pandas as pd

from marketfit.ingestion.cache import JsonCache

# ISO-3 (used across the pipeline / ground truth) -> ISO-2 (Google Trends geo).
# Covers the validation-case countries plus a few common markets; extend as needed.
_ISO3_TO_ISO2 = {
    "USA": "US", "ITA": "IT", "JPN": "JP", "CHN": "CN", "AUS": "AU",
    "KOR": "KR", "IND": "IN", "VNM": "VN", "GBR": "GB", "DEU": "DE",
    "FRA": "FR", "ESP": "ES", "BRA": "BR", "CAN": "CA", "MEX": "MX",
}


def iso3_to_geo(iso3: str) -> str:
    """Map an ISO-3 code to a Google Trends ``geo`` (ISO-2).

    Returns ``""`` (worldwide) for unknown codes so a lookup miss degrades to a
    global pull rather than raising.
    """
    return _ISO3_TO_ISO2.get(iso3.upper(), "")


def _iot_cache_key(keywords: Iterable[str], geo: str, timeframe: str) -> str:
    """Stable cache key for an interest-over-time query (order-independent JSON)."""
    return json.dumps(
        {
            "method": "interest_over_time",
            "keywords": list(keywords),
            "geo": geo,
            "timeframe": timeframe,
        },
        sort_keys=True,
    )


def _ibr_cache_key(keyword: str, resolution: str, geo: str, timeframe: str) -> str:
    """Stable cache key for an interest-by-region query."""
    return json.dumps(
        {
            "method": "interest_by_region",
            "keyword": keyword,
            "resolution": resolution,
            "geo": geo,
            "timeframe": timeframe,
        },
        sort_keys=True,
    )


def _num(value) -> float:
    """Coerce a Google Trends interest value to float; unparseable -> 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class GoogleTrendsClient:
    """Client for Google Trends (pytrends) with on-disk caching.

    Parameters
    ----------
    hl:
        pytrends host language, e.g. ``"en-US"``.
    tz:
        Timezone offset in minutes passed to pytrends (360 = US CST, its default).
    timeframe:
        Default Google Trends timeframe string, e.g. ``"today 5-y"`` or
        ``"2015-01-01 2020-12-31"``.
    cache_ttl_seconds:
        Freshness window for cached responses. ``None`` keeps entries indefinitely.
    """

    hl: str = "en-US"
    tz: int = 360
    timeframe: str = "today 5-y"
    cache_ttl_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        self._cache = JsonCache("trends", ttl_seconds=self.cache_ttl_seconds)
        self._pytrends = None  # lazily constructed only on a cache miss

    # -- low-level fetch ---------------------------------------------------

    def _client(self):
        """Lazily build a pytrends ``TrendReq`` (imported only when we hit the net)."""
        if self._pytrends is None:
            try:
                from pytrends.request import TrendReq
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise ImportError(
                    "pytrends is required for live Google Trends pulls. "
                    "Install it with `pip install pytrends`."
                ) from exc
            self._pytrends = TrendReq(hl=self.hl, tz=self.tz)
        return self._pytrends

    def _interest_over_time_records(
        self, keywords: List[str], geo: str, timeframe: str
    ) -> List[dict]:
        key = _iot_cache_key(keywords, geo, timeframe)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        client = self._client()
        client.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop="")
        records = _frame_to_records(client.interest_over_time(), index_name="date")
        self._cache.set(key, records)
        return records

    def _interest_by_region_records(
        self, keyword: str, resolution: str, geo: str, timeframe: str
    ) -> List[dict]:
        key = _ibr_cache_key(keyword, resolution, geo, timeframe)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        client = self._client()
        client.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop="")
        frame = client.interest_by_region(resolution=resolution, inc_low_vol=True)
        records = _frame_to_records(frame, index_name="geoName")
        self._cache.set(key, records)
        return records

    # -- public API --------------------------------------------------------

    def interest_over_time(
        self,
        keywords: Union[str, Iterable[str]],
        geo: str = "",
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """Normalized search interest over time for one or more terms.

        Parameters
        ----------
        keywords:
            A single term or an iterable of terms (Google Trends compares up to 5).
        geo:
            Google Trends ISO-2 geo (``"IT"``); ``""`` means worldwide. Pass an
            ISO-3 code through :func:`iso3_to_geo` first.
        timeframe:
            Override the client's default timeframe for this call.

        Returns a tidy long DataFrame:
        ``keyword, geo, date, interest, is_partial`` (interest is 0-100).
        """
        kw_list = [keywords] if isinstance(keywords, str) else list(keywords)
        tf = timeframe or self.timeframe
        records = self._interest_over_time_records(kw_list, geo, tf)

        rows: List[dict] = []
        for rec in records:
            is_partial = bool(rec.get("isPartial", False))
            for kw in kw_list:
                if kw not in rec:
                    continue
                rows.append(
                    {
                        "keyword": kw,
                        "geo": geo or "WORLD",
                        "date": rec.get("date"),
                        "interest": _num(rec.get(kw)),
                        "is_partial": is_partial,
                    }
                )
        columns = ["keyword", "geo", "date", "interest", "is_partial"]
        df = pd.DataFrame(rows, columns=columns)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["keyword", "date"]).reset_index(drop=True)
        return df

    def demand_snapshot(
        self,
        keywords: Union[str, Iterable[str]],
        geo: str = "",
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """Reduce the interest series to one demand row per (keyword, geo).

        This is the normalized "consumer demand" signal the scoring agent consumes,
        parallel to the World Bank macro snapshot. Columns:
        ``keyword, geo, mean_interest, latest_interest, peak_interest, n_points``.
        """
        series = self.interest_over_time(keywords, geo=geo, timeframe=timeframe)
        columns = [
            "keyword", "geo", "mean_interest", "latest_interest",
            "peak_interest", "n_points",
        ]
        if series.empty:
            return pd.DataFrame(columns=columns)

        rows: List[dict] = []
        for (kw, g), sub in series.groupby(["keyword", "geo"], sort=False):
            sub = sub.sort_values("date")
            rows.append(
                {
                    "keyword": kw,
                    "geo": g,
                    "mean_interest": round(float(sub["interest"].mean()), 2),
                    "latest_interest": float(sub["interest"].iloc[-1]),
                    "peak_interest": float(sub["interest"].max()),
                    "n_points": int(len(sub)),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def interest_by_region(
        self,
        keyword: str,
        resolution: str = "COUNTRY",
        geo: str = "",
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """Where interest in ``keyword`` concentrates, as a tidy frame.

        With ``resolution="COUNTRY"`` and ``geo=""`` this ranks countries by demand
        for the term — a cheap way to compare candidate markets. Columns:
        ``keyword, region, interest``.
        """
        tf = timeframe or self.timeframe
        records = self._interest_by_region_records(keyword, resolution, geo, tf)
        rows = [
            {
                "keyword": keyword,
                "region": rec.get("geoName"),
                "interest": _num(rec.get(keyword)),
            }
            for rec in records
        ]
        df = pd.DataFrame(rows, columns=["keyword", "region", "interest"])
        return df.sort_values("interest", ascending=False).reset_index(drop=True)


def _frame_to_records(frame: pd.DataFrame, index_name: str) -> List[dict]:
    """Serialize a pytrends DataFrame to JSON-safe records for the cache.

    The index (dates or region names) is promoted to a column named ``index_name``
    and any Timestamps are stringified so the payload round-trips through JSON.
    """
    if frame is None or frame.empty:
        return []
    out = frame.reset_index()
    if index_name not in out.columns and out.columns.size:
        out = out.rename(columns={out.columns[0]: index_name})
    if index_name in out.columns:
        out[index_name] = out[index_name].astype(str)
    return out.to_dict("records")
