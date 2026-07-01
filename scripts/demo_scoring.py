"""Demo: feature engineering + scoring over the ground-truth cases.

Builds features for each curated Starbucks market-entry case from bundled offline
signal fixtures (see ``data/fixtures/country_signals.json``), scores each into a
1-5 entry-viability score, and compares against the documented outcome — a
preview of the Checkpoint-2 validation loop.

Usage::

    python scripts/demo_scoring.py            # score all ground-truth cases
    python scripts/demo_scoring.py --calibrate  # also tune the success threshold

Live runs would replace the fixtures with WorldBankClient / Comtrade / Trends pulls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketfit.features import FeatureBuilder
from marketfit.groundtruth import GroundTruthLoader, score_for
from marketfit.groundtruth.labels import label_for
from marketfit.scoring import MarketFitScorer

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "country_signals.json"


def build_features_for_cases(cases, signals, builder):
    """Yield (case_row, FeatureVector) for every case with fixture signals."""
    for _, row in cases.iterrows():
        iso3 = row["country_iso3"]
        sig = signals.get(iso3)
        if sig is None:
            continue
        fv = builder.from_signals(
            country_iso3=iso3,
            hs_code=row["hs_code"],
            macro=sig["macro"],
            product_import_usd=sig.get("product_import_usd"),
            demand_interest=sig.get("demand_interest"),
        )
        yield row, fv


def main() -> int:
    parser = argparse.ArgumentParser(description="MarketFit scoring demo")
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrate the success threshold on the ground truth first")
    args = parser.parse_args()

    cases = GroundTruthLoader().load()
    signals = json.loads(FIXTURES.read_text())
    builder = FeatureBuilder()
    scorer = MarketFitScorer()

    paired = list(build_features_for_cases(cases, signals, builder))

    if args.calibrate:
        labeled = [(fv, label_for(row["outcome_label"]).is_success) for row, fv in paired]
        t = scorer.calibrate_threshold(labeled)
        print(f"Calibrated success threshold -> {t:.3f}\n")

    header = f"{'case':16} {'pred':>4} {'actual':>6} {'label':17} {'bucket':>8} {'top drivers'}"
    print(header)
    print("-" * len(header))

    score_hits = bucket_hits = 0
    for row, fv in paired:
        result = scorer.score(fv)
        actual_score = int(row["outcome_score"])
        actual_success = label_for(row["outcome_label"]).is_success
        score_hits += abs(result.score - actual_score) <= 1
        bucket_hits += result.is_success == actual_success
        drivers = ", ".join(f"{k}({v:.2f})" for k, v in result.top_drivers(2))
        bucket = "success" if result.is_success else "struggle"
        print(f"{row['case_id']:16} {result.score:>4} {actual_score:>6} "
              f"{row['outcome_label']:17} {bucket:>8} {drivers}")

    n = len(paired)
    print("-" * len(header))
    print(f"Cases scored: {n}")
    print(f"Score within +/-1 of actual: {score_hits}/{n} ({score_hits/n:.0%})")
    print(f"Success/struggle bucket correct: {bucket_hits}/{n} ({bucket_hits/n:.0%})")
    print("\nNote: illustrative offline fixtures; the point is the pipeline, not the accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
