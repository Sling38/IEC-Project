"""Data ingestion pipeline for MarketFit.

Pulls and normalizes the macro/trade input signals for a (product, market) pair:

* :mod:`marketfit.ingestion.comtrade` — UN Comtrade trade flows (live API).
* :mod:`marketfit.ingestion.baci`     — BACI bulk trade files (CEPII, offline).
* :mod:`marketfit.ingestion.world_bank` — World Bank World Development Indicators.

All network clients cache responses locally (see :mod:`marketfit.ingestion.cache`)
so that UN Comtrade / World Bank rate limits do not block downstream work.
"""

from marketfit.ingestion.comtrade import ComtradeClient
from marketfit.ingestion.world_bank import WorldBankClient
from marketfit.ingestion.baci import BaciLoader

__all__ = ["ComtradeClient", "WorldBankClient", "BaciLoader"]
