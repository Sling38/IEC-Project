"""Demo-UI support for MarketFit (Checkpoint 3).

Pure, offline-testable helpers the Streamlit app (``app/streamlit_app.py``) calls:
signal assembly (bundled fixtures or live ingestion pulls), ground-truth lookup,
and the feature->score convenience wrapper. Keeping the logic here leaves the
Streamlit file a thin view layer, consistent with how the validation harness is
decoupled from its demo script.
"""

from marketfit.demo.data import (
    MarketAssessment,
    assess_market,
    find_case,
    live_signals,
    signals_for,
)

__all__ = [
    "MarketAssessment",
    "assess_market",
    "find_case",
    "live_signals",
    "signals_for",
]
