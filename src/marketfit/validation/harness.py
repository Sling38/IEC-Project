"""Validation harness: score the ground-truth cases and measure the fit.

This is the Checkpoint-2 "first validation test on Starbucks for ground-truth
comparison". It wires the pieces the two partners built at Checkpoint 1-2 together:

    ground-truth cases  ->  feature vectors  ->  scorer  ->  metrics vs. labels
    (groundtruth)           (features)           (scoring)    (validation.metrics)

The harness is deliberately decoupled from the network clients: the caller supplies
a ``signals`` mapping (``iso3 -> {"macro": ..., "product_import_usd": ...,
"demand_interest": ...}``), which can come from the bundled offline fixtures or from
live ingestion pulls. That keeps validation deterministic and unit-testable.

It offers two evaluation modes:

* :meth:`ValidationHarness.run` — score every case with the scorer as-is (optionally
  calibrating the success threshold in-sample first, which is noted as optimistic).
* :meth:`ValidationHarness.leave_one_out` — honest small-sample evaluation: for each
  case, calibrate the threshold on the *other* cases and predict the held-out one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd

from marketfit.features import FeatureBuilder, FeatureVector
from marketfit.groundtruth.labels import label_for
from marketfit.scoring import MarketFitScorer
from marketfit.scoring.model import ScoreResult
from marketfit.validation.metrics import (
    BucketMetrics,
    ScoreMetrics,
    bucket_metrics,
    score_metrics,
)

# Bundled offline signal fixtures shared with the scoring demo.
DEFAULT_FIXTURES_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "fixtures" / "country_signals.json"
)


def load_signal_fixtures(path: Optional[Path] = None) -> Dict[str, dict]:
    """Load the offline ``iso3 -> signals`` fixtures, dropping ``_comment`` keys."""
    fixtures_path = Path(path) if path else DEFAULT_FIXTURES_PATH
    data = json.loads(fixtures_path.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


@dataclass(frozen=True)
class CaseResult:
    """Prediction vs. ground truth for a single validation case."""

    case_id: str
    country_iso3: str
    hs_code: str
    outcome_label: str
    predicted_score: int
    actual_score: int
    predicted_success: bool
    actual_success: bool
    composite: float
    score_result: ScoreResult  # retained so error analysis can read the drivers

    @property
    def score_error(self) -> int:
        """Signed error: ``predicted - actual`` (>0 = over-predicted)."""
        return self.predicted_score - self.actual_score

    @property
    def bucket_correct(self) -> bool:
        return self.predicted_success == self.actual_success


@dataclass(frozen=True)
class ValidationReport:
    """Per-case results plus aggregate score and bucket metrics."""

    cases: List[CaseResult]
    score: ScoreMetrics
    bucket: BucketMetrics
    skipped: Tuple[str, ...] = ()
    calibrated_threshold: Optional[float] = None

    def to_frame(self) -> pd.DataFrame:
        """Tidy per-case table (excludes the nested ``ScoreResult``)."""
        rows = [
            {
                "case_id": c.case_id,
                "country_iso3": c.country_iso3,
                "outcome_label": c.outcome_label,
                "actual_score": c.actual_score,
                "predicted_score": c.predicted_score,
                "score_error": c.score_error,
                "actual_success": c.actual_success,
                "predicted_success": c.predicted_success,
                "bucket_correct": c.bucket_correct,
                "composite": round(c.composite, 4),
            }
            for c in self.cases
        ]
        columns = [
            "case_id", "country_iso3", "outcome_label", "actual_score",
            "predicted_score", "score_error", "actual_success",
            "predicted_success", "bucket_correct", "composite",
        ]
        return pd.DataFrame(rows, columns=columns)

    def summary(self) -> str:
        """Human-readable metrics block for reports / the demo."""
        s, b = self.score, self.bucket
        lines = [
            f"Cases evaluated: {s.n}"
            + (f"  (skipped {len(self.skipped)}: {', '.join(self.skipped)})"
               if self.skipped else ""),
        ]
        if self.calibrated_threshold is not None:
            lines.append(f"Success threshold (in-sample calibrated): {self.calibrated_threshold:.3f}")
        lines += [
            "Score (1-5):  "
            f"MAE={s.mae}  RMSE={s.rmse}  exact={s.exact_accuracy:.0%}  "
            f"within-1={s.within_one_accuracy:.0%}  bias={s.bias:+.2f}  "
            f"rank_rho={s.rank_correlation:+.2f}",
            "Bucket:       "
            f"acc={b.accuracy:.0%}  precision={b.precision:.2f}  recall={b.recall:.2f}  "
            f"F1={b.f1:.2f}  (TP={b.tp} FP={b.fp} FN={b.fn} TN={b.tn})",
        ]
        return "\n".join(lines)


@dataclass
class ValidationHarness:
    """Runs the scorer over ground-truth cases and reports how well it fits."""

    scorer: MarketFitScorer = field(default_factory=MarketFitScorer)
    builder: FeatureBuilder = field(default_factory=FeatureBuilder)

    # -- feature assembly --------------------------------------------------

    def _build(
        self, cases: pd.DataFrame, signals: Mapping[str, dict]
    ) -> Tuple[List[Tuple[pd.Series, FeatureVector]], List[str]]:
        """Pair each case with its feature vector; collect cases lacking signals."""
        paired: List[Tuple[pd.Series, FeatureVector]] = []
        skipped: List[str] = []
        for _, row in cases.iterrows():
            sig = signals.get(row["country_iso3"])
            if not sig:
                skipped.append(row["case_id"])
                continue
            fv = self.builder.from_signals(
                country_iso3=row["country_iso3"],
                hs_code=row["hs_code"],
                macro=sig.get("macro", {}),
                product_import_usd=sig.get("product_import_usd"),
                demand_interest=sig.get("demand_interest"),
            )
            paired.append((row, fv))
        return paired, skipped

    @staticmethod
    def _case_result(row: pd.Series, result: ScoreResult) -> CaseResult:
        actual_success = label_for(row["outcome_label"]).is_success
        return CaseResult(
            case_id=row["case_id"],
            country_iso3=row["country_iso3"],
            hs_code=str(row["hs_code"]),
            outcome_label=row["outcome_label"],
            predicted_score=result.score,
            actual_score=int(row["outcome_score"]),
            predicted_success=result.is_success,
            actual_success=actual_success,
            composite=result.composite,
            score_result=result,
        )

    # -- evaluation --------------------------------------------------------

    def run(
        self,
        cases: pd.DataFrame,
        signals: Mapping[str, dict],
        calibrate: bool = False,
    ) -> ValidationReport:
        """Score every case and return a :class:`ValidationReport`.

        If ``calibrate`` is set, the success threshold is first tuned on these same
        cases. That is *in-sample* (optimistic) — use :meth:`leave_one_out` for an
        honest estimate. The calibrated threshold is recorded on the report.
        """
        paired, skipped = self._build(cases, signals)

        calibrated_threshold = None
        if calibrate and paired:
            labeled = [
                (fv, label_for(row["outcome_label"]).is_success)
                for row, fv in paired
            ]
            calibrated_threshold = self.scorer.calibrate_threshold(labeled)

        results = [self._case_result(row, self.scorer.score(fv)) for row, fv in paired]
        report = ValidationReport(
            cases=results,
            score=score_metrics(
                [c.predicted_score for c in results],
                [c.actual_score for c in results],
            ),
            bucket=bucket_metrics(
                [c.predicted_success for c in results],
                [c.actual_success for c in results],
            ),
            skipped=tuple(skipped),
            calibrated_threshold=calibrated_threshold,
        )
        return report

    def leave_one_out(
        self, cases: pd.DataFrame, signals: Mapping[str, dict]
    ) -> Tuple[BucketMetrics, pd.DataFrame]:
        """Leave-one-out success/struggle evaluation (honest for a tiny sample).

        For each case, a fresh scorer is calibrated on the *other* cases and used to
        predict the held-out one, so no case influences its own threshold. Returns
        the aggregate :class:`BucketMetrics` and a per-fold DataFrame.
        """
        paired, _ = self._build(cases, signals)
        preds: List[bool] = []
        actuals: List[bool] = []
        rows: List[dict] = []

        for i, (row, fv) in enumerate(paired):
            others = [
                (f, label_for(r["outcome_label"]).is_success)
                for j, (r, f) in enumerate(paired)
                if j != i
            ]
            fold_scorer = MarketFitScorer(weights=dict(self.scorer.weights))
            threshold = fold_scorer.calibrate_threshold(others)
            result = fold_scorer.score(fv)
            actual_success = label_for(row["outcome_label"]).is_success

            preds.append(result.is_success)
            actuals.append(actual_success)
            rows.append(
                {
                    "case_id": row["case_id"],
                    "fold_threshold": round(threshold, 4),
                    "composite": round(result.composite, 4),
                    "predicted_success": result.is_success,
                    "actual_success": actual_success,
                    "correct": result.is_success == actual_success,
                }
            )

        frame = pd.DataFrame(
            rows,
            columns=[
                "case_id", "fold_threshold", "composite",
                "predicted_success", "actual_success", "correct",
            ],
        )
        return bucket_metrics(preds, actuals), frame
