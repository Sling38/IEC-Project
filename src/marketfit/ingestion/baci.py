"""BACI bulk trade-file ingestion (CEPII, offline).

BACI is CEPII's cleaned, reconciled version of UN Comtrade, distributed as yearly
CSV files. It has no rate limits and gives consistent bilateral flows, which makes
it the better source for large historical pulls than the live Comtrade API.

Download: https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37

Expected layout (HS92 release, for example)::

    data/baci/
        BACI_HS92_Y2022_V202401.csv   # t,i,j,k,v,q
        country_codes_V202401.csv     # country_code,country_name,iso_3digit_alpha,...
        product_codes_HS92_V202401.csv# code,description

Raw BACI columns: ``t`` year, ``i`` exporter, ``j`` importer, ``k`` HS6 product,
``v`` value (thousands US$), ``q`` quantity (metric tons). This loader joins the
numeric country codes to ISO-3 and returns tidy, human-readable frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_BACI_DIR = Path(__file__).resolve().parents[3] / "data" / "baci"


@dataclass
class BaciLoader:
    """Loads and normalizes BACI yearly trade files from a local directory.

    Parameters
    ----------
    baci_dir:
        Directory holding the BACI CSVs and code lookups. Defaults to
        ``<repo>/data/baci``.
    """

    baci_dir: Path = DEFAULT_BACI_DIR

    def __post_init__(self) -> None:
        self.baci_dir = Path(self.baci_dir)
        self._country_codes: Optional[pd.DataFrame] = None
        self._product_codes: Optional[pd.DataFrame] = None

    # -- code lookups ------------------------------------------------------

    def _find(self, pattern: str) -> Path:
        matches = sorted(self.baci_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No file matching {pattern!r} in {self.baci_dir}. "
                "Download the BACI release from CEPII and unzip it there."
            )
        return matches[-1]  # latest version by name

    def country_codes(self) -> pd.DataFrame:
        """Return the country-code lookup (``country_code`` -> ISO-3 / name)."""
        if self._country_codes is None:
            path = self._find("country_codes*.csv")
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            self._country_codes = df
        return self._country_codes

    def product_codes(self) -> pd.DataFrame:
        """Return the HS6 product-code lookup (``code`` -> ``description``)."""
        if self._product_codes is None:
            path = self._find("product_codes*.csv")
            df = pd.read_csv(path, dtype={"code": str})
            df.columns = [c.strip().lower() for c in df.columns]
            self._product_codes = df
        return self._product_codes

    # -- main loader -------------------------------------------------------

    def load_year(self, year: int, hs_prefix: Optional[str] = None) -> pd.DataFrame:
        """Load one BACI year, joined to ISO-3 codes and product descriptions.

        Parameters
        ----------
        year:
            Trade year to load (must have a matching ``BACI_*_Y{year}_*.csv``).
        hs_prefix:
            Optional HS-code prefix filter, e.g. ``"09"`` (coffee/tea/spices) or
            ``"0901"`` (coffee). Enables the "HS-code trace" the pipeline needs.

        Returns tidy columns: ``year, exporter_iso3, exporter, importer_iso3,
        importer, hs_code, commodity, trade_value_usd, quantity_ton``.
        """
        path = self._find(f"BACI_*_Y{year}_*.csv")
        df = pd.read_csv(path, dtype={"k": str})
        # Standard BACI schema: t,i,j,k,v,q
        df = df.rename(
            columns={
                "t": "year",
                "i": "exporter_code",
                "j": "importer_code",
                "k": "hs_code",
                "v": "trade_value_kusd",
                "q": "quantity_ton",
            }
        )

        if hs_prefix:
            df = df[df["hs_code"].str.startswith(str(hs_prefix))]

        # BACI values are in thousands of US$.
        df["trade_value_usd"] = pd.to_numeric(df["trade_value_kusd"], errors="coerce") * 1000
        df["quantity_ton"] = pd.to_numeric(df["quantity_ton"], errors="coerce")

        cc = self.country_codes()
        iso_col = _pick(cc, ["iso_3digit_alpha", "iso3", "iso_3"])
        code_col = _pick(cc, ["country_code", "code"])
        name_col = _pick(cc, ["country_name", "name", "country_name_full"])
        lut = cc[[code_col, iso_col, name_col]].rename(
            columns={code_col: "_code", iso_col: "_iso", name_col: "_name"}
        )

        df = df.merge(
            lut.rename(columns={"_code": "exporter_code", "_iso": "exporter_iso3", "_name": "exporter"}),
            on="exporter_code",
            how="left",
        ).merge(
            lut.rename(columns={"_code": "importer_code", "_iso": "importer_iso3", "_name": "importer"}),
            on="importer_code",
            how="left",
        )

        pc = self.product_codes()
        pdesc_col = _pick(pc, ["description", "desc"])
        df = df.merge(
            pc[["code", pdesc_col]].rename(columns={"code": "hs_code", pdesc_col: "commodity"}),
            on="hs_code",
            how="left",
        )

        cols = [
            "year",
            "exporter_iso3",
            "exporter",
            "importer_iso3",
            "importer",
            "hs_code",
            "commodity",
            "trade_value_usd",
            "quantity_ton",
        ]
        return df[cols].sort_values("trade_value_usd", ascending=False).reset_index(drop=True)


def _pick(df: pd.DataFrame, candidates: list) -> str:
    """Return the first candidate column present in ``df`` (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")
