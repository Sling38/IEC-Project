"""Offline tests for the ingestion pipeline.

These exercise the normalization logic without hitting the network by:
  * pre-seeding the on-disk cache with realistic API payloads, and
  * writing tiny synthetic BACI CSVs to a temp directory.

Run with:  python -m pytest tests/  (or: python tests/test_ingestion.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from marketfit.ingestion import BaciLoader, ComtradeClient, WorldBankClient


# --- World Bank -----------------------------------------------------------

# Shape of a real World Bank API response: [metadata, [observations]].
_WB_POP_OBS = [
    {
        "indicator": {"id": "SP.POP.TOTL", "value": "Population, total"},
        "country": {"id": "IT", "value": "Italy"},
        "countryiso3code": "ITA",
        "date": "2021",
        "value": 59109668,
    },
    {
        "indicator": {"id": "SP.POP.TOTL", "value": "Population, total"},
        "country": {"id": "IT", "value": "Italy"},
        "countryiso3code": "ITA",
        "date": "2022",
        "value": 58940425,
    },
]


def test_world_bank_normalization_from_cache():
    wb = WorldBankClient(indicators={"SP.POP.TOTL": "Population, total"})
    # Seed the cache so no network call happens.
    url = "https://api.worldbank.org/v2/country/ITA/indicator/SP.POP.TOTL"
    params = {"format": "json", "date": "2000:2023", "per_page": "1000"}
    key = f"{url}?{sorted(params.items())}"
    wb._cache.set(key, _WB_POP_OBS)

    df = wb.fetch_indicators("ITA")
    assert list(df.columns) == [
        "country_iso3", "country", "indicator_code", "indicator", "year", "value",
    ]
    assert len(df) == 2
    assert df["country"].iloc[0] == "Italy"

    snap = wb.latest_snapshot("ITA")
    # latest_snapshot keeps the most recent year per indicator.
    assert len(snap) == 1
    assert snap["year"].iloc[0] == 2022
    assert snap["value"].iloc[0] == 58940425
    print("world_bank OK -> latest population 2022:", snap["value"].iloc[0])


# --- Comtrade -------------------------------------------------------------

_COMTRADE_RAW = [
    {
        "refYear": 2022, "reporterCode": 380, "reporterISO": "ITA",
        "reporterDesc": "Italy", "partnerCode": 0, "partnerISO": "W00",
        "partnerDesc": "World", "flowCode": "M", "flowDesc": "Import",
        "cmdCode": "0901", "cmdDesc": "Coffee", "primaryValue": 1600000000.0,
        "netWgt": 500000000.0, "qty": 500000000.0,
    },
]


def test_comtrade_normalization():
    df = ComtradeClient._normalize(_COMTRADE_RAW)
    assert df.loc[0, "reporter"] == "Italy"
    assert df.loc[0, "hs_code"] == "0901"
    assert df.loc[0, "trade_value_usd"] == 1600000000.0
    # Empty input yields a well-formed empty frame, not an error.
    assert ComtradeClient._normalize([]).empty
    print("comtrade OK -> Italy coffee imports 2022:", df.loc[0, "trade_value_usd"])


# --- BACI -----------------------------------------------------------------

def test_baci_loader(tmp_path: Path):
    baci_dir = tmp_path / "baci"
    baci_dir.mkdir()

    # Minimal synthetic BACI year file: exporter 76 (BRA) -> importer 380 (ITA), coffee.
    (baci_dir / "BACI_HS92_Y2022_V202401.csv").write_text(
        "t,i,j,k,v,q\n"
        "2022,76,380,090111,1500000.0,300000.0\n"
        "2022,704,380,090111,200000.0,40000.0\n"
    )
    (baci_dir / "country_codes_V202401.csv").write_text(
        "country_code,country_name,iso_3digit_alpha\n"
        "76,Brazil,BRA\n704,Viet Nam,VNM\n380,Italy,ITA\n"
    )
    (baci_dir / "product_codes_HS92_V202401.csv").write_text(
        "code,description\n090111,Coffee; not roasted\n"
    )

    loader = BaciLoader(baci_dir=baci_dir)
    df = loader.load_year(2022, hs_prefix="0901")
    assert len(df) == 2
    top = df.iloc[0]
    assert top["exporter_iso3"] == "BRA"
    assert top["importer_iso3"] == "ITA"
    # Value stored in thousands -> multiplied to USD.
    assert top["trade_value_usd"] == 1500000.0 * 1000
    assert top["commodity"] == "Coffee; not roasted"
    print("baci OK -> top coffee exporter to Italy:", top["exporter"])


if __name__ == "__main__":
    import tempfile

    test_world_bank_normalization_from_cache()
    test_comtrade_normalization()
    with tempfile.TemporaryDirectory() as d:
        test_baci_loader(Path(d))
    print("\nAll ingestion tests passed.")
