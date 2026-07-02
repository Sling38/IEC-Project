"""Demo: the Checkpoint-2 validation test on the Starbucks ground truth.

Scores every curated market-entry case with the entry-viability model, compares the
predictions against the documented outcome labels, and prints:

  1. a per-case prediction table,
  2. aggregate score + bucket metrics (default weights, uncalibrated),
  3. the same after in-sample threshold calibration and an honest leave-one-out run,
  4. an error analysis pointing at which cases/features to iterate on.

Uses the bundled offline signal fixtures so it runs without network access; live
runs would swap in WorldBankClient / Comtrade / Google Trends pulls.

Usage::

    python scripts/demo_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from marketfit.groundtruth import GroundTruthLoader
from marketfit.validation import (
    ValidationHarness,
    analyze_errors,
    load_signal_fixtures,
)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def main() -> int:
    cases = GroundTruthLoader().load()
    signals = load_signal_fixtures()

    print("=== Per-case predictions vs. ground truth (default weights) ===")
    report = ValidationHarness().run(cases, signals)
    print(report.to_frame().to_string(index=False))
    print("\n" + report.summary())

    print("\n=== In-sample threshold calibration (optimistic) ===")
    calibrated = ValidationHarness().run(cases, signals, calibrate=True)
    print(calibrated.summary())

    print("\n=== Leave-one-out success/struggle (honest small-sample estimate) ===")
    loo_bucket, loo_frame = ValidationHarness().leave_one_out(cases, signals)
    print(loo_frame.to_string(index=False))
    print(
        f"LOO bucket: acc={loo_bucket.accuracy:.0%}  precision={loo_bucket.precision:.2f}  "
        f"recall={loo_bucket.recall:.2f}  F1={loo_bucket.f1:.2f}"
    )

    print("\n=== Error analysis ===")
    analysis = analyze_errors(report)
    print(analysis.summary())
    for err in analysis.errors:
        drivers = ", ".join(f"{f}({v:.2f})" for f, v in err.implicated)
        print(f"  - {err.case_id:16} {err.note}  [{drivers}]")

    print("\nNote: illustrative offline fixtures — the deliverable is the harness/metrics, not the accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
