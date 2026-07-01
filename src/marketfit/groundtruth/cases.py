"""Loader for the curated market-entry ground-truth cases.

The cases live in a small hand-curated CSV (``data/ground_truth/``) rather than
being pulled from an API — they encode documented historical outcomes with sources.
This loader reads that CSV, validates it against the outcome taxonomy in
:mod:`marketfit.groundtruth.labels`, and returns a tidy :class:`pandas.DataFrame`
the validation harness (Checkpoint 2) can join against model predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from marketfit.groundtruth.labels import OUTCOME_LABELS, score_for

# Repo-root/data/ground_truth/starbucks_market_entries.csv
DEFAULT_CASES_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "ground_truth"
    / "starbucks_market_entries.csv"
)

# Columns every case row must provide.
REQUIRED_COLUMNS = [
    "case_id",
    "company",
    "product",
    "hs_code",
    "country",
    "country_iso3",
    "comtrade_reporter_m49",
    "google_trends_geo",
    "trends_keyword",
    "entry_year",
    "outcome_label",
    "outcome_score",
    "source_url",
    "notes",
]


@dataclass
class GroundTruthLoader:
    """Loads and validates the ground-truth market-entry cases.

    Parameters
    ----------
    path:
        Path to the cases CSV. Defaults to :data:`DEFAULT_CASES_PATH`.
    """

    path: Path = DEFAULT_CASES_PATH

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def load(self, validate: bool = True) -> pd.DataFrame:
        """Return the cases as a tidy DataFrame.

        ``hs_code`` and ``google_trends_geo`` are read as strings so leading zeros
        (e.g. ``"0901"``) survive. ``entry_year``, ``comtrade_reporter_m49`` and
        ``outcome_score`` are integers. Set ``validate=False`` to skip the
        consistency checks in :meth:`validate`.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"Ground-truth cases not found at {self.path}. "
                "Expected the curated CSV committed under data/ground_truth/."
            )
        df = pd.read_csv(
            self.path,
            dtype={
                "hs_code": str,
                "google_trends_geo": str,
                "country_iso3": str,
                "trends_keyword": str,
            },
        )
        df.columns = [c.strip() for c in df.columns]
        for int_col in ("entry_year", "comtrade_reporter_m49", "outcome_score"):
            if int_col in df.columns:
                df[int_col] = df[int_col].astype("Int64")
        if validate:
            self._validate(df)
        return df.reset_index(drop=True)

    def validate(self) -> pd.DataFrame:
        """Load with validation and return the frame (raises on any problem)."""
        return self.load(validate=True)

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Ground-truth CSV is missing columns: {missing}")

        problems: List[str] = []

        dupes = df["case_id"][df["case_id"].duplicated()].tolist()
        if dupes:
            problems.append(f"duplicate case_id(s): {sorted(set(dupes))}")

        for _, row in df.iterrows():
            cid = row["case_id"]
            label = row["outcome_label"]
            if label not in OUTCOME_LABELS:
                problems.append(
                    f"{cid}: unknown outcome_label {label!r} "
                    f"(expected one of {list(OUTCOME_LABELS)})"
                )
                continue
            expected = score_for(label)
            if int(row["outcome_score"]) != expected:
                problems.append(
                    f"{cid}: outcome_score {row['outcome_score']} does not match "
                    f"label {label!r} (expected {expected})"
                )

        if problems:
            raise ValueError(
                "Ground-truth validation failed:\n  - " + "\n  - ".join(problems)
            )
