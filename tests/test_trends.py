"""Offline tests for the Google Trends ingestion module.

Google Trends is an unofficial, rate-limited endpoint, so these tests never touch
the network: they pre-seed the on-disk cache with a realistic pytrends payload and
exercise the normalization + snapshot logic. pytrends itself is only imported on a
cache miss, so it need not be installed to run these.

Run with:  python -m pytest tests/  (or: python tests/test_trends.py)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketfit.ingestion import GoogleTrendsClient, iso3_to_geo
from marketfit.ingestion.cache import JsonCache
from marketfit.ingestion.trends import _iot_cache_key


def _seeded_client(tmp_dir: Path) -> GoogleTrendsClient:
    """Client whose cache lives in ``tmp_dir`` and is pre-seeded with the fixture.

    Using a temp cache keeps the test hermetic — it never touches ``data/cache`` and
    so can't be served back to a real demo pull that happens to share a query key.
    """
    client = GoogleTrendsClient(timeframe="today 5-y")
    client._cache = JsonCache("trends", cache_dir=tmp_dir)
    client._cache.set(_iot_cache_key(["Starbucks"], "IT", "today 5-y"), _IOT_RECORDS)
    return client


# Shape of a pytrends interest_over_time() frame after reset_index -> records.
_IOT_RECORDS = [
    {"date": "2021-01-01", "Starbucks": 55, "isPartial": False},
    {"date": "2021-02-01", "Starbucks": 60, "isPartial": False},
    {"date": "2021-03-01", "Starbucks": 80, "isPartial": False},
    {"date": "2021-04-01", "Starbucks": 70, "isPartial": True},
]


def test_iso3_to_geo():
    assert iso3_to_geo("ITA") == "IT"
    assert iso3_to_geo("jpn") == "JP"          # case-insensitive
    assert iso3_to_geo("XXX") == ""            # unknown -> worldwide, not an error
    print("iso3_to_geo OK -> ITA:", iso3_to_geo("ITA"))


def test_interest_over_time_from_cache(tmp_path: Path):
    client = _seeded_client(tmp_path)  # cache pre-seeded, so no network call happens

    df = client.interest_over_time("Starbucks", geo="IT")
    assert list(df.columns) == ["keyword", "geo", "date", "interest", "is_partial"]
    assert len(df) == 4
    assert df["geo"].iloc[0] == "IT"
    # Sorted by date; last row is the partial April point.
    assert bool(df["is_partial"].iloc[-1]) is True
    assert df["interest"].iloc[0] == 55.0
    print("interest_over_time OK -> rows:", len(df))


def test_demand_snapshot_from_cache(tmp_path: Path):
    client = _seeded_client(tmp_path)

    snap = client.demand_snapshot("Starbucks", geo="IT")
    assert list(snap.columns) == [
        "keyword", "geo", "mean_interest", "latest_interest",
        "peak_interest", "n_points",
    ]
    assert len(snap) == 1
    row = snap.iloc[0]
    assert row["keyword"] == "Starbucks"
    assert row["mean_interest"] == round((55 + 60 + 80 + 70) / 4, 2)
    assert row["latest_interest"] == 70.0    # most recent date
    assert row["peak_interest"] == 80.0
    assert row["n_points"] == 4
    print("demand_snapshot OK -> mean interest:", row["mean_interest"])


if __name__ == "__main__":
    test_iso3_to_geo()
    with tempfile.TemporaryDirectory() as d:
        test_interest_over_time_from_cache(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_demand_snapshot_from_cache(Path(d))
    print("\nAll Google Trends tests passed.")
