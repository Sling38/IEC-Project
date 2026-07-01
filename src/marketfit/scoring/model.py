"""The MarketFit entry-viability scoring model.

Takes a normalized :class:`~marketfit.features.FeatureVector` and produces a
**1-5 entry-viability score** — the same scale the ground-truth outcome labels use
(see :mod:`marketfit.groundtruth.labels`) — so predictions are directly comparable
against documented outcomes at validation time.

Design: a transparent **weighted linear model** over the eight features. With only
a handful of ground-truth cases a heavy ML model would overfit, so the default is
an interpretable weighted sum with documented priors. Every prediction exposes a
per-feature contribution breakdown, which the Checkpoint 2 rationale generator
(LLM) turns into prose. Weights can optionally be calibrated from ground truth via
:meth:`MarketFitScorer.calibrate_threshold`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from marketfit.features.engineering import FEATURE_NAMES, FeatureVector

# Default feature weights (sum to 1.0). These encode the two Checkpoint-1 agents:
# "macro-fit" (size/wealth/growth/stability/openness/connectivity) and
# "trade-history + demand" (existing_trade, consumer_demand).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "market_size": 0.18,
    "purchasing_power": 0.15,
    "growth": 0.12,
    "price_stability": 0.08,
    "openness": 0.10,
    "connectivity": 0.10,
    "existing_trade": 0.15,
    "consumer_demand": 0.12,
}


@dataclass(frozen=True)
class ScoreResult:
    """Output of the scorer for one ``(product, country)`` pair.

    Attributes
    ----------
    score:
        Integer 1-5 entry-viability score (comparable to ground-truth labels).
    composite:
        Raw weighted composite in [0, 1] before mapping to the 1-5 scale.
    is_success:
        Coarse success/struggle bucket (``composite >= threshold``).
    contributions:
        Per-feature contribution to ``composite`` (weight x feature value),
        already renormalized for any missing features. Sums to ``composite``.
    used_weights:
        The (renormalized) weights actually applied, for transparency.
    missing:
        Features that were unavailable and excluded from scoring.
    """

    country_iso3: str
    hs_code: str
    score: int
    composite: float
    is_success: bool
    contributions: Dict[str, float]
    used_weights: Dict[str, float]
    missing: tuple = ()

    def top_drivers(self, n: int = 3) -> List[tuple]:
        """Return the ``n`` features contributing most to the score."""
        ranked = sorted(self.contributions.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:n]

    def top_gaps(self, n: int = 3) -> List[tuple]:
        """Return the ``n`` features contributing least (biggest missed potential)."""
        # Gap = weight - contribution (how much of the available weight went unused).
        gaps = {
            f: self.used_weights[f] - self.contributions.get(f, 0.0)
            for f in self.used_weights
        }
        return sorted(gaps.items(), key=lambda kv: kv[1], reverse=True)[:n]


@dataclass
class MarketFitScorer:
    """Weighted linear entry-viability scorer.

    Parameters
    ----------
    weights:
        Feature weights. Defaults to :data:`DEFAULT_WEIGHTS`. Need not sum to 1 —
        they are renormalized over the features actually present.
    success_threshold:
        Composite cutoff (in [0, 1]) separating "success" from "struggle".
    """

    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    success_threshold: float = 0.5

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"Unknown feature(s) in weights: {sorted(unknown)}")

    # -- scoring -----------------------------------------------------------

    def score(self, features: FeatureVector) -> ScoreResult:
        """Score one feature vector into a :class:`ScoreResult`."""
        present = {f: v for f, v in features.values.items() if f in self.weights}
        total_w = sum(self.weights[f] for f in present)
        if total_w <= 0:
            raise ValueError(
                f"No usable features for {features.country_iso3}:{features.hs_code}; "
                f"cannot score (missing={features.missing})."
            )
        # Renormalize weights over the present features so a missing signal
        # neither silently drags the score down nor inflates it.
        used_weights = {f: self.weights[f] / total_w for f in present}
        contributions = {f: used_weights[f] * present[f] for f in present}
        composite = sum(contributions.values())

        return ScoreResult(
            country_iso3=features.country_iso3,
            hs_code=features.hs_code,
            score=self._to_score(composite),
            composite=composite,
            is_success=composite >= self.success_threshold,
            contributions=contributions,
            used_weights=used_weights,
            missing=features.missing,
        )

    @staticmethod
    def _to_score(composite: float) -> int:
        """Map a [0, 1] composite to an integer 1-5 entry-viability score."""
        # 1 + 4*composite spreads [0,1] across [1,5]; round to nearest integer.
        return int(round(1 + 4 * max(0.0, min(1.0, composite))))

    # -- optional calibration ---------------------------------------------

    def calibrate_threshold(
        self, labeled: Iterable[Tuple[FeatureVector, bool]]
    ) -> float:
        """Pick the success threshold that best separates labeled cases.

        ``labeled`` is an iterable of ``(feature_vector, is_success)`` pairs.
        Scans candidate thresholds and keeps the one with the highest accuracy,
        updating :attr:`success_threshold` in place. Returns the chosen threshold.
        (Intended for the Checkpoint-2 validation harness; kept dependency-free
        given the small sample.)
        """
        scored = [(self.score(fv).composite, truth) for fv, truth in labeled]
        if not scored:
            return self.success_threshold

        candidates = sorted({c for c, _ in scored})
        best_t, best_acc = self.success_threshold, -1.0
        for t in candidates:
            acc = sum((c >= t) == truth for c, truth in scored) / len(scored)
            if acc > best_acc:
                best_acc, best_t = acc, t
        self.success_threshold = best_t
        return best_t
