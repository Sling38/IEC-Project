"""Hello-World demo for MarketFit's demand signal + ground-truth cases (Samuel).

Prints the curated ground-truth market-entry cases and the outcome-label taxonomy,
then pulls a Google Trends demand snapshot for a search term in a target market.
The Trends pull uses the free (unofficial) endpoint and caches its response; a
network hiccup or missing pytrends degrades gracefully instead of hard-failing.

Usage::

    python scripts/demo_trends.py                       # cases + Trends for Italy
    python scripts/demo_trends.py --country JPN --keyword Starbucks
    python scripts/demo_trends.py --skip-trends         # ground truth only (offline)

``--country`` is an ISO-3 code; it is mapped to a Google Trends geo automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from marketfit.groundtruth import GroundTruthLoader, OUTCOME_LABELS
from marketfit.ingestion import GoogleTrendsClient, iso3_to_geo

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def main() -> int:
    parser = argparse.ArgumentParser(description="MarketFit demand + ground-truth demo")
    parser.add_argument("--country", default="ITA", help="ISO-3 country code")
    parser.add_argument("--keyword", default="Starbucks", help="Google Trends search term")
    parser.add_argument("--skip-trends", action="store_true", help="Ground truth only")
    args = parser.parse_args()

    print("\n=== Outcome-label taxonomy (1-5 entry-viability) ===")
    for label in sorted(OUTCOME_LABELS.values(), key=lambda x: -x.score):
        print(f"  {label.score}  {label.name:<16} — {label.description}")

    print("\n=== Ground-truth market-entry cases ===")
    cases = GroundTruthLoader().load()
    cols = ["case_id", "country", "entry_year", "outcome_label", "outcome_score"]
    print(cases[cols].to_string(index=False))

    if args.skip_trends:
        return 0

    geo = iso3_to_geo(args.country)
    print(f"\n=== Google Trends demand snapshot: '{args.keyword}' geo={geo or 'WORLD'} ===")
    try:
        trends = GoogleTrendsClient()
        snapshot = trends.demand_snapshot(args.keyword, geo=geo)
        if snapshot.empty:
            print("No Trends data returned (free endpoint may be rate-limited).")
        else:
            print(snapshot.to_string(index=False))
    except Exception as exc:  # network / rate-limit / missing pytrends — never hard-fail
        print(f"Google Trends fetch failed ({exc.__class__.__name__}): {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
