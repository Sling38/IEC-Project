"""Entry-viability scoring for MarketFit.

Maps an engineered feature vector to a 1-5 entry-viability score with a
per-feature contribution breakdown for rationale generation.
"""

from marketfit.scoring.model import (
    DEFAULT_WEIGHTS,
    MarketFitScorer,
    ScoreResult,
)

__all__ = ["DEFAULT_WEIGHTS", "MarketFitScorer", "ScoreResult"]
