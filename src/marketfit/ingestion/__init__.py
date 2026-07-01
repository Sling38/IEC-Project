"""Data ingestion pipeline for MarketFit.

Pulls and normalizes the three input signals for a (product, market) pair:

* :mod:`marketfit.ingestion.comtrade`   — UN Comtrade trade flows (live API).
* :mod:`marketfit.ingestion.baci`       — BACI bulk trade files (CEPII, offline).
* :mod:`marketfit.ingestion.world_bank` — World Bank World Development Indicators.
* :mod:`marketfit.ingestion.trends`     — Google Trends consumer-demand signal.

All network clients cache responses locally (see :mod:`marketfit.ingestion.cache`)
so that UN Comtrade / World Bank / Google Trends rate limits do not block work.
"""

from marketfit.ingestion.comtrade import ComtradeClient
from marketfit.ingestion.world_bank import WorldBankClient
from marketfit.ingestion.baci import BaciLoader
from marketfit.ingestion.trends import GoogleTrendsClient, iso3_to_geo

__all__ = [
    "ComtradeClient",
    "WorldBankClient",
    "BaciLoader",
    "GoogleTrendsClient",
    "iso3_to_geo",
]
