"""Experiment: does a competitive-intensity feature fix the two failure cases?

Our error analysis (final report §4.4) diagnoses both misses — Vietnam and Australia —
as the model rewarding big, rich, open markets with no feature able to push such a market
*down*. The proposed fix was a competitive-intensity / category-maturity feature proxied
by **per-capita product imports**, inverse-scored: a market that already consumes a lot of
the category is saturated, so entry is harder.

This script implements that feature and sweeps its weight against the ground-truth cases.
It is a **negative result** — the feature degrades the model — and this script exists so
that claim is reproducible rather than asserted (see final report §4.5).

Usage::

    python scripts/experiment_competition.py
    python scripts/experiment_competition.py --weights 0.1,0.2,0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketfit.features import FeatureBuilder
from marketfit.features.engineering import _log_lin
from marketfit.groundtruth import GroundTruthLoader
from marketfit.groundtruth.labels import label_for
from marketfit.scoring import MarketFitScorer
from marketfit.validation import bucket_metrics, load_signal_fixtures, score_metrics

# Per-capita import range (USD/person/year) used to normalize the competition proxy.
# Spans the observed spread: India ~$0.08 to Italy ~$27.
PERCAP_RANGE = (0.05, 30.0)


def competition_score(product_import_usd: float, population: float) -> Tuple[float, float]:
    """Return ``(per_capita_usd, feature_value)`` for the competition proxy.

    The feature is **inverse**: a high per-capita import level is read as a mature,
    saturated category (harder to enter), so it maps toward 0.
    """
    per_capita = product_import_usd / population
    return per_capita, 1.0 - _log_lin(per_capita, *PERCAP_RANGE)


def run(weight: float) -> Tuple[List[dict], object, object]:
    """Score every case with the competition feature blended in at ``weight``."""
    cases = GroundTruthLoader().load()
    signals = load_signal_fixtures()
    builder, scorer = FeatureBuilder(), MarketFitScorer()

    rows: List[dict] = []
    for _, row in cases.iterrows():
        sig = signals.get(row["country_iso3"])
        if not sig:
            continue
        fv = builder.from_signals(
            row["country_iso3"], row["hs_code"], sig.get("macro", {}),
            sig.get("product_import_usd"), sig.get("demand_interest"),
        )
        result = scorer.score(fv)
        per_capita, comp = competition_score(
            sig["product_import_usd"], sig["macro"]["SP.POP.TOTL"]
        )
        # Blend: the existing composite keeps (1 - weight) of the mass.
        composite = result.composite * (1 - weight) + comp * weight
        rows.append({
            "case_id": row["case_id"],
            "outcome_label": row["outcome_label"],
            "actual": int(row["outcome_score"]),
            "baseline_pred": result.score,
            "pred": int(round(1 + 4 * max(0.0, min(1.0, composite)))),
            "actual_success": label_for(row["outcome_label"]).is_success,
            "pred_success": composite >= scorer.success_threshold,
            "per_capita": per_capita,
            "competition": comp,
        })

    score = score_metrics([r["pred"] for r in rows], [r["actual"] for r in rows])
    bucket = bucket_metrics([r["pred_success"] for r in rows], [r["actual_success"] for r in rows])
    return rows, score, bucket


def main() -> int:
    ap = argparse.ArgumentParser(description="Competitive-intensity feature experiment")
    ap.add_argument("--weights", default="0.15,0.25,0.35",
                    help="Comma-separated feature weights to sweep")
    args = ap.parse_args()

    print("Competitive-intensity feature (per-capita imports, inverse-scored)\n")
    print(f"{'weight':>8}{'MAE':>8}{'within-1':>11}{'bucket acc':>13}")
    print("-" * 40)
    base_rows, base_score, base_bucket = run(0.0)
    print(f"{'none':>8}{base_score.mae:>8.2f}{base_score.within_one_accuracy:>10.0%}"
          f"{base_bucket.accuracy:>13.0%}")
    for w in [float(x) for x in args.weights.split(",")]:
        _, s, b = run(w)
        print(f"{w:>8.2f}{s.mae:>8.2f}{s.within_one_accuracy:>10.0%}{b.accuracy:>13.0%}")

    print("\nPer-case detail (feature value: 1.0 = wide open, 0.0 = saturated)\n")
    print(f"{'case':16}{'outcome':18}{'actual':>7}{'$/cap':>9}{'feature':>9}")
    print("-" * 59)
    for r in base_rows:
        print(f"{r['case_id']:16}{r['outcome_label']:18}{r['actual']:>7}"
              f"{r['per_capita']:>9.2f}{r['competition']:>9.2f}")

    print("\nWhy it fails: per-capita imports conflate category *demand* with category")
    print("*saturation*. Japan and Korea are high-import markets and Strong Successes —")
    print("the feature penalizes them for the very thing that made them work. And Vietnam,")
    print("a top-two global coffee producer, imports almost nothing, so the case the")
    print("feature exists to fix reads as a wide-open market. See final report §4.5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
