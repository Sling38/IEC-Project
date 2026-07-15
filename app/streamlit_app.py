"""MarketFit demo UI (Checkpoint 3).

Enter a (product, market) pair, see the predicted 1-5 entry-viability score with
its feature breakdown, and compare against the curated ground-truth outcome when
the pair is one of the documented cases.

Run with::

    streamlit run app/streamlit_app.py

All heavy lifting lives in :mod:`marketfit.demo` (tested offline); this file is
the view layer only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from marketfit.demo import assess_market, signals_for
from marketfit.features.engineering import FEATURE_NAMES
from marketfit.groundtruth import GroundTruthLoader, label_for
from marketfit.scoring import MarketFitScorer
from marketfit.validation import ValidationHarness, analyze_errors, load_signal_fixtures

# --- palette (validated with the dataviz six-checks script) -----------------
BLUE = "#2a78d6"      # predicted / primary series
AQUA = "#1baf7a"      # actual / ground truth (below 3:1 on white -> direct labels)
TRACK = "#e8e8e3"     # neutral "available weight" track
INK = "#3d3d3a"       # primary text
INK_2 = "#6e6e69"     # secondary text
GRID = "#ececea"      # recessive grid

FEATURE_LABELS = {
    "market_size": "Market size",
    "purchasing_power": "Purchasing power",
    "growth": "GDP growth",
    "price_stability": "Price stability",
    "openness": "Trade openness",
    "connectivity": "Connectivity",
    "existing_trade": "Existing trade",
    "consumer_demand": "Consumer demand",
}


# --- cached data -------------------------------------------------------------

@st.cache_data
def _cases() -> pd.DataFrame:
    return GroundTruthLoader().load()


@st.cache_data
def _fixture_countries() -> list:
    return sorted(load_signal_fixtures())


@st.cache_data(ttl=3600, show_spinner="Pulling signals…")
def _signals(country_iso3: str, hs_code: str, source: str, keyword: str,
             m49: int | None):
    return signals_for(
        country_iso3, hs_code, source=source, keyword=keyword or None,
        comtrade_reporter_m49=m49,
    )


# --- charts ------------------------------------------------------------------

def _bare_axes(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def contribution_chart(assessment) -> plt.Figure:
    """Bullet-style bars: contribution (blue) inside the available weight (track)."""
    result = assessment.result
    names = [f for f in FEATURE_NAMES if f in result.used_weights or f in assessment.features.missing]
    y = range(len(names))

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(names) + 0.9))
    for i, f in enumerate(names):
        weight = result.used_weights.get(f, 0.0)
        contrib = result.contributions.get(f, 0.0)
        ax.barh(i, weight, height=0.5, color=TRACK, zorder=2)
        ax.barh(i, contrib, height=0.5, color=BLUE, zorder=3)
        if f in assessment.features.missing:
            ax.text(0.004, i, "signal unavailable", va="center", ha="left",
                    fontsize=8, color=INK_2, style="italic", zorder=4)
        else:
            ax.text(contrib + 0.004, i, f"{contrib:.2f}", va="center", ha="left",
                    fontsize=8.5, color=INK, zorder=4)

    ax.set_yticks(list(y))
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in names], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(result.used_weights.values()) * 1.25)
    ax.set_xlabel("Contribution to composite (gray = available weight)",
                  fontsize=9, color=INK_2)
    ax.xaxis.grid(True, color=GRID, zorder=1)
    ax.set_axisbelow(True)
    _bare_axes(ax)
    fig.tight_layout()
    return fig


def predicted_vs_actual_chart(frame: pd.DataFrame) -> plt.Figure:
    """Dumbbell plot: predicted (blue) vs actual (aqua) 1-5 score per case."""
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(frame) + 1.0))
    ys = range(len(frame))
    for i, (_, row) in enumerate(frame.iterrows()):
        a, p = row["actual_score"], row["predicted_score"]
        ax.plot([a, p], [i, i], color=GRID, lw=2, zorder=2)
        ax.scatter([p], [i], s=90, color=BLUE, zorder=3)
        ax.scatter([a], [i], s=90, color=AQUA, zorder=3)
        # Direct labels (relief for the aqua series; skip when overlapping).
        ax.text(p, i - 0.32, str(int(p)), ha="center", fontsize=8.5, color=INK)
        if a != p:
            ax.text(a, i - 0.32, str(int(a)), ha="center", fontsize=8.5, color=INK)

    ax.set_yticks(list(ys))
    ax.set_yticklabels(frame["case_id"], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Entry-viability score (1–5)", fontsize=9, color=INK_2)
    ax.xaxis.grid(True, color=GRID, zorder=1)
    ax.set_axisbelow(True)
    _bare_axes(ax)
    ax.scatter([], [], s=90, color=BLUE, label="Predicted")
    ax.scatter([], [], s=90, color=AQUA, label="Actual (ground truth)")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    return fig


# --- page --------------------------------------------------------------------

st.set_page_config(page_title="MarketFit", page_icon="🌍", layout="centered")

st.title("MarketFit")
st.caption(
    "Market-entry viability for a (product, market) pair — trade history, "
    "macro fit, and consumer demand combined into one 1–5 score, validated "
    "against documented outcomes."
)

with st.sidebar:
    st.header("Query")
    source_label = st.radio(
        "Data source",
        ["Bundled fixtures (offline)", "Live APIs (cached)"],
        help="Fixtures cover the 7 curated markets and run with no network. "
        "Live mode pulls World Bank / Comtrade / Google Trends through the "
        "cached ingestion clients and degrades per signal if a feed is down.",
    )
    source = "live" if source_label.startswith("Live") else "fixtures"

    options = _fixture_countries() + ["Other…"]
    choice = st.selectbox("Market (ISO-3)", options, index=options.index("ITA"))
    country = st.text_input("ISO-3 code", value="USA").strip().upper() if choice == "Other…" else choice

    hs_code = st.text_input("Product HS code", value="0901",
                            help="e.g. 0901 = coffee").strip()
    keyword = st.text_input("Consumer-demand keyword", value="Starbucks",
                            help="Google Trends search term (live mode only)").strip()

cases = _cases()
tab_assess, tab_validate = st.tabs(["Market assessment", "Validation vs. ground truth"])


with tab_assess:
    case_hint = cases[cases["country_iso3"] == country]
    m49 = int(case_hint["comtrade_reporter_m49"].iloc[0]) if not case_hint.empty else None

    signals, notes = _signals(country, hs_code, source, keyword, m49)
    if signals is None:
        st.warning(" ".join(notes))
        st.stop()

    assessment = assess_market(country, hs_code, signals, cases=cases, notes=tuple(notes))
    result = assessment.result

    col1, col2, col3 = st.columns(3)
    col1.metric("Entry-viability score", f"{result.score} / 5")
    col2.metric("Assessment", "Success" if result.is_success else "Struggle")
    col3.metric("Composite", f"{result.composite:.2f}")

    if assessment.has_ground_truth:
        gt = assessment.ground_truth
        actual = label_for(gt["outcome_label"])
        match = (
            "bucket and score(±1) match"
            if result.is_success == actual.is_success and abs(result.score - actual.score) <= 1
            else "bucket matches" if result.is_success == actual.is_success
            else "prediction disagrees with the documented outcome"
        )
        st.success(
            f"**Ground truth ({gt['case_id']}, entered {gt['entry_year']}):** "
            f"{gt['outcome_label']} — actual score {actual.score}/5 vs predicted "
            f"{result.score}/5 → {match}.\n\n{gt['notes']}  \n[source]({gt['source_url']})"
        )
    else:
        st.info("No curated ground-truth case for this pair — showing the model's prediction only.")

    st.subheader("What drives the score")
    st.pyplot(contribution_chart(assessment), use_container_width=True)
    st.caption("Blue = each signal's contribution to the composite; gray = the weight "
               "available to it. Weights renormalize over the signals present.")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Top drivers**")
        for f, v in result.top_drivers(3):
            st.markdown(f"- {FEATURE_LABELS.get(f, f)} (+{v:.2f})")
    with d2:
        st.markdown("**Biggest gaps**")
        for f, v in result.top_gaps(3):
            st.markdown(f"- {FEATURE_LABELS.get(f, f)} (−{v:.2f} unused)")

    for note in assessment.notes:
        st.caption(f"⚠ {note}")

    with st.expander("Raw input signals"):
        macro = assessment.signals.get("macro", {})
        if macro:
            st.dataframe(
                pd.DataFrame(sorted(macro.items()), columns=["indicator", "value"]),
                hide_index=True, use_container_width=True,
            )
        st.write(
            {
                "product_import_usd": assessment.signals.get("product_import_usd"),
                "demand_interest": assessment.signals.get("demand_interest"),
                "missing_features": list(assessment.features.missing),
            }
        )


with tab_validate:
    st.caption(
        "Scores every curated case from the bundled offline fixtures and compares "
        "against documented outcomes — the Checkpoint-2 validation loop, live."
    )
    c1, c2 = st.columns(2)
    calibrate = c1.checkbox("Calibrate threshold (in-sample)", value=False)
    show_loo = c2.checkbox("Leave-one-out evaluation", value=False)

    harness = ValidationHarness(scorer=MarketFitScorer())
    report = harness.run(cases, load_signal_fixtures(), calibrate=calibrate)

    st.code(report.summary(), language=None)
    st.pyplot(predicted_vs_actual_chart(report.to_frame()), use_container_width=True)
    st.dataframe(report.to_frame(), hide_index=True, use_container_width=True)

    st.subheader("Error analysis")
    st.code(analyze_errors(report).summary(), language=None)

    if show_loo:
        st.subheader("Leave-one-out")
        loo_metrics, loo_frame = harness.leave_one_out(cases, load_signal_fixtures())
        st.code(
            f"acc={loo_metrics.accuracy:.0%}  precision={loo_metrics.precision:.2f}  "
            f"recall={loo_metrics.recall:.2f}  F1={loo_metrics.f1:.2f}  "
            f"(TP={loo_metrics.tp} FP={loo_metrics.fp} FN={loo_metrics.fn} TN={loo_metrics.tn})",
            language=None,
        )
        st.dataframe(loo_frame, hide_index=True, use_container_width=True)
