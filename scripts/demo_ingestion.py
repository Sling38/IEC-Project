"""Hello-World demo for the MarketFit ingestion pipeline.

Pulls a macro snapshot from the World Bank and (if reachable) a trade flow from
UN Comtrade for a given (product, market) pair, then prints tidy tables.

Usage::

    python scripts/demo_ingestion.py --country ITA --hs 0901 --reporter 380

``0901`` = coffee, ``380`` = Italy (M49). World Bank uses ISO-3 (``ITA``).
The World Bank pull needs no API key; Comtrade uses the free preview tier unless
``COMTRADE_API_KEY`` is set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from marketfit.ingestion import ComtradeClient, WorldBankClient

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def main() -> int:
    parser = argparse.ArgumentParser(description="MarketFit ingestion demo")
    parser.add_argument("--country", default="ITA", help="ISO-3 country code (World Bank)")
    parser.add_argument("--reporter", default="380", help="Comtrade reporter M49 code")
    parser.add_argument("--hs", default="0901", help="HS commodity code")
    parser.add_argument("--year", default="2022", help="Trade year")
    parser.add_argument("--skip-comtrade", action="store_true", help="World Bank only")
    args = parser.parse_args()

    print(f"\n=== World Bank macro snapshot: {args.country} ===")
    try:
        wb = WorldBankClient()
        snapshot = wb.latest_snapshot(args.country)
        if snapshot.empty:
            print("No World Bank data returned (check the ISO-3 code / connectivity).")
        else:
            print(snapshot.to_string(index=False))
    except Exception as exc:  # network hiccup — demo should not hard-fail
        print(f"World Bank fetch failed ({exc.__class__.__name__}): {exc}")

    if args.skip_comtrade:
        return 0

    print(f"\n=== Comtrade imports: reporter={args.reporter} hs={args.hs} {args.year} ===")
    try:
        ct = ComtradeClient()
        trade = ct.get_trade(
            reporter=args.reporter,
            partner="0",
            hs_code=args.hs,
            period=args.year,
            flow="M",
        )
        if trade.empty:
            print("No Comtrade rows (free tier may be rate-limited; set COMTRADE_API_KEY).")
        else:
            cols = [c for c in ["year", "reporter", "partner", "commodity", "trade_value_usd"] if c in trade.columns]
            print(trade[cols].head(10).to_string(index=False))
    except Exception as exc:  # network / rate-limit — demo should not hard-fail
        print(f"Comtrade fetch failed ({exc.__class__.__name__}): {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
