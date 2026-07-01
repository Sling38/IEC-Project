"""Feature engineering for the MarketFit scoring model.

Turns the ingested signals (World Bank macro, Comtrade/BACI trade, Google Trends
demand) into a normalized, interpretable feature vector for the scorer.
"""

from marketfit.features.engineering import (
    FEATURE_NAMES,
    FeatureBuilder,
    FeatureVector,
)

__all__ = ["FEATURE_NAMES", "FeatureBuilder", "FeatureVector"]
