"""MarketFit demo UI (Checkpoint 3).

Enter a (product, market) pair, see the predicted 1-5 entry-viability score with
its feature breakdown, and compare against the curated ground-truth outcome when
the pair is one of the documented cases.

Run with::

    streamlit run app/streamlit_app.py

All heavy lifting lives in :mod:`marketfit.demo` (tested offline); this file is
the view layer only. Visual language follows the reference dataviz palette
(six-checks validated): thin marks, hairline grid, status colors reserved for
state, text in ink tokens — never in series colors.
"""

from __future__ import annotations

import html as _html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from marketfit.demo import assess_market, signals_for
from marketfit.features.engineering import FEATURE_NAMES
from marketfit.groundtruth import GroundTruthLoader, label_for
from marketfit.ingestion import iso3_to_geo
from marketfit.scoring import MarketFitScorer
from marketfit.validation import ValidationHarness, analyze_errors, load_signal_fixtures

# --- palette (reference dataviz palette; validated with the six-checks script) ---
SURFACE = "#fcfcfb"    # chart surface
BLUE = "#2a78d6"       # categorical slot 1 — predicted / primary series
BLUE_100 = "#cde2fb"   # lighter step of the same ramp (meter/bullet tracks)
AQUA = "#1baf7a"       # categorical slot 2 — actual / ground truth (sub-3:1 → relief
                       # via legend + table view, both always present)
INK = "#0b0b0b"        # primary text
INK_2 = "#52514e"      # secondary text
MUTED = "#898781"      # axis / muted labels
GRID = "#e1e0d9"       # hairline gridline
BASELINE = "#c3c2b7"   # axis baseline
GOOD_TEXT = "#006300"  # success text on light surface
CRIT = "#d03b3b"       # critical (4.68:1 on light)

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

COUNTRY_NAMES = {
    "AUS": "Australia", "CHN": "China", "IND": "India", "ITA": "Italy",
    "JPN": "Japan", "KOR": "South Korea", "VNM": "Vietnam", "USA": "United States",
    "GBR": "United Kingdom", "DEU": "Germany", "FRA": "France", "BRA": "Brazil",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
})


# --- chrome -------------------------------------------------------------------

CSS = """
<style>
:root {
  --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --border: rgba(11,11,11,0.10); --blue: #2a78d6; --blue-100: #cde2fb;
}
/* Hide the deploy button + menu but NOT the toolbar itself — it holds the
   sidebar expand control (stExpandSidebarButton). */
footer, .stAppDeployButton, [data-testid="stMainMenu"] { visibility: hidden; }
h1, h2, h3 { letter-spacing: -0.01em; }

.mf-brand { display: flex; align-items: baseline; gap: .55rem; margin-bottom: .1rem; }
.mf-brand .logo { width: 12px; height: 12px; border-radius: 3px; background: var(--blue);
  align-self: center; }
.mf-brand .name { font-size: 1.9rem; font-weight: 700; color: var(--ink); letter-spacing: -0.02em; }
.mf-sub { color: var(--ink2); font-size: .92rem; margin: 0 0 .4rem 0; }

.mf-hero { display: flex; align-items: center; gap: 2.2rem; padding: 1.2rem 1.4rem;
  border: 1px solid var(--border); border-radius: 14px; background: var(--surface); }
.mf-hero .score { font-size: 58px; font-weight: 650; line-height: 1; color: var(--ink); }
.mf-hero .of { font-size: 22px; font-weight: 500; color: var(--muted); }
.mf-hero .scale { font-size: .74rem; color: var(--muted); margin-top: .35rem; }
.mf-hero .right { flex: 1; min-width: 0; }

.mf-badge { display: inline-flex; align-items: center; gap: .35rem; padding: .28rem .7rem;
  border-radius: 999px; font-size: .84rem; font-weight: 600; margin-bottom: .55rem; }
.mf-badge.good { background: rgba(12,163,12,.10); color: #006300; }
.mf-badge.bad  { background: rgba(208,59,59,.08); color: #b02a2a; }
.mf-badge.mid  { background: rgba(250,178,25,.16); color: #7a5200; }

.mf-meter { position: relative; height: 8px; border-radius: 4px; background: var(--blue-100); }
.mf-meter .fill { position: absolute; inset: 0 auto 0 0; border-radius: 4px; background: var(--blue); }
.mf-meter .notch { position: absolute; top: -3px; bottom: -3px; width: 2px; background: #898781; }
.mf-meter-labels { display: flex; justify-content: space-between; font-size: .72rem;
  color: var(--muted); margin-top: .3rem; }

.mf-card { border: 1px solid var(--border); border-radius: 14px; background: var(--surface);
  padding: 1rem 1.2rem; margin: .35rem 0; }
.mf-card .head { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap;
  margin-bottom: .35rem; }
.mf-card .title { font-weight: 650; color: var(--ink); font-size: .95rem; }
.mf-card .meta { color: var(--muted); font-size: .8rem; }
.mf-card .body { color: var(--ink2); font-size: .86rem; line-height: 1.45; }
.mf-card .body a { color: var(--blue); }
.mf-vs { display: flex; gap: 1.6rem; margin: .45rem 0 .5rem 0; }
.mf-vs .cell .k { font-size: .72rem; text-transform: uppercase; letter-spacing: .05em;
  color: var(--muted); }
.mf-vs .cell .v { font-size: 1.05rem; font-weight: 650; color: var(--ink); }

.mf-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
  gap: .7rem; margin: .4rem 0 .7rem 0; }
.mf-tile { border: 1px solid var(--border); border-radius: 12px; background: var(--surface);
  padding: .75rem .9rem; }
.mf-tile .label { font-size: .72rem; text-transform: uppercase; letter-spacing: .05em;
  color: var(--ink2); }
.mf-tile .value { font-size: 1.65rem; font-weight: 650; color: var(--ink); line-height: 1.25; }
.mf-tile .sub { font-size: .74rem; color: var(--muted); }

.mf-chip { display: inline-flex; align-items: center; gap: .4rem; padding: .26rem .6rem;
  border: 1px solid var(--border); border-radius: 999px; font-size: .8rem; color: var(--ink2);
  margin: 0 .3rem .35rem 0; background: var(--surface); }
.mf-chip .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.mf-chip b { color: var(--ink); font-weight: 600; }

.mf-note { display: flex; gap: .5rem; align-items: baseline; padding: .45rem .7rem;
  border-radius: 10px; background: rgba(250,178,25,.14); color: #5c4a12; font-size: .8rem;
  margin: .25rem 0; }
</style>
"""


def _fmt_market(code: str) -> str:
    if code == "Other…":
        return code
    iso2 = iso3_to_geo(code)
    flag = "".join(chr(0x1F1E6 + ord(c) - 65) for c in iso2) if len(iso2) == 2 else "🌐"
    return f"{flag} {COUNTRY_NAMES.get(code, code)} ({code})"


def _tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="mf-tile"><div class="label">{label}</div><div class="value">{value}</div>{sub_html}</div>'


def _tiles(*tiles: str) -> None:
    st.markdown(f'<div class="mf-tiles">{"".join(tiles)}</div>', unsafe_allow_html=True)


def _chips(items, dot_color: str) -> str:
    return "".join(
        f'<span class="mf-chip"><span class="dot" style="background:{dot_color}"></span>'
        f"{FEATURE_LABELS.get(f, f)} <b>{v:+.2f}</b></span>"
        for f, v in items
    )


def _hero(result, threshold: float = 0.5) -> None:
    badge = (
        '<span class="mf-badge good">✓ Likely success</span>'
        if result.is_success
        else '<span class="mf-badge bad">⚠ Likely struggle</span>'
    )
    fill = max(0.0, min(1.0, result.composite)) * 100
    st.markdown(
        f"""
<div class="mf-hero">
  <div>
    <div class="score">{result.score}<span class="of"> / 5</span></div>
    <div class="scale">1 = withdrew · 5 = strong success</div>
  </div>
  <div class="right">
    {badge}
    <div class="mf-meter"><div class="fill" style="width:{fill:.1f}%"></div>
      <div class="notch" style="left:{threshold * 100:.0f}%"></div></div>
    <div class="mf-meter-labels"><span>composite {result.composite:.2f}</span>
      <span>success threshold {threshold:.2f}</span></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _ground_truth_card(gt: pd.Series, result) -> None:
    actual = label_for(gt["outcome_label"])
    delta = result.score - actual.score
    if result.is_success == actual.is_success and abs(delta) <= 1:
        badge = '<span class="mf-badge good">✓ Prediction matches the outcome</span>'
    elif result.is_success == actual.is_success:
        badge = f'<span class="mf-badge mid">◐ Bucket matches · score off by {delta:+d}</span>'
    else:
        badge = '<span class="mf-badge bad">✕ Prediction disagrees with the outcome</span>'
    notes = _html.escape(str(gt["notes"]))
    st.markdown(
        f"""
<div class="mf-card">
  <div class="head"><span class="title">Ground truth · {gt["case_id"]}</span>
    <span class="meta">entered {gt["entry_year"]}</span>{badge}</div>
  <div class="mf-vs">
    <div class="cell"><div class="k">Documented outcome</div>
      <div class="v">{gt["outcome_label"]}</div></div>
    <div class="cell"><div class="k">Actual score</div><div class="v">{actual.score} / 5</div></div>
    <div class="cell"><div class="k">Predicted</div><div class="v">{result.score} / 5</div></div>
  </div>
  <div class="body">{notes} <a href="{gt["source_url"]}" target="_blank">source ↗</a></div>
</div>
""",
        unsafe_allow_html=True,
    )


# --- charts ---------------------------------------------------------------------

def _bare_axes(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_facecolor("none")


def contribution_chart(assessment) -> plt.Figure:
    """Bullet bars: contribution (blue) inside the available weight (light-blue track)."""
    result = assessment.result
    names = [f for f in FEATURE_NAMES if f in result.used_weights or f in assessment.features.missing]

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(names) + 0.8))
    fig.patch.set_alpha(0)
    for i, f in enumerate(names):
        weight = result.used_weights.get(f, 0.0)
        contrib = result.contributions.get(f, 0.0)
        ax.barh(i, weight, height=0.45, color=BLUE_100, zorder=2)
        ax.barh(i, contrib, height=0.45, color=BLUE, zorder=3)
        if f in assessment.features.missing:
            ax.text(0.004, i, "signal unavailable", va="center", ha="left",
                    fontsize=8, color=MUTED, style="italic", zorder=4)
        else:
            ax.text(contrib + 0.004, i, f"{contrib:.2f}", va="center", ha="left",
                    fontsize=8.5, color=INK_2, zorder=4)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in names], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    xmax = max(result.used_weights.values()) * 1.25
    ax.set_xlim(0, xmax)
    ax.set_xticks([t / 100 for t in range(0, int(xmax * 100) + 1, 5)])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".") or "0")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    _bare_axes(ax)
    fig.tight_layout()
    return fig


def predicted_vs_actual_chart(frame: pd.DataFrame) -> plt.Figure:
    """Dumbbell plot: predicted (blue) vs actual (aqua) 1-5 score per case."""
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(frame) + 1.0))
    fig.patch.set_alpha(0)
    for i, (_, row) in enumerate(frame.iterrows()):
        a, p = row["actual_score"], row["predicted_score"]
        if a != p:
            ax.plot([a, p], [i, i], color="#d8d7d0", lw=2, solid_capstyle="round", zorder=2)
        # 2px surface ring keeps overlapping dots legible (marks spec).
        ax.scatter([p], [i], s=110, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.scatter([a], [i], s=110, color=AQUA, edgecolor=SURFACE, linewidth=2, zorder=3)

    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame["case_id"], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Entry-viability score (1–5)", fontsize=9, color=MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    _bare_axes(ax)
    ax.scatter([], [], s=90, color=BLUE, label="Predicted")
    ax.scatter([], [], s=90, color=AQUA, label="Actual (ground truth)")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    return fig


# --- cached data ------------------------------------------------------------------

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


# --- page ---------------------------------------------------------------------------

st.set_page_config(page_title="MarketFit", page_icon="🌍", layout="centered",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="mf-brand"><span class="logo"></span><span class="name">MarketFit</span></div>'
    '<p class="mf-sub">Market-entry viability for a (product, market) pair — trade history, '
    "macro fit, and consumer demand combined into one 1–5 score, validated against "
    "documented outcomes.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Query")
    options = _fixture_countries() + ["Other…"]
    choice = st.selectbox("Market", options, index=options.index("ITA"),
                          format_func=_fmt_market)
    country = st.text_input("ISO-3 code", value="USA").strip().upper() if choice == "Other…" else choice

    hs_code = st.text_input("Product (HS code)", value="0901",
                            help="Harmonized System code — e.g. 0901 = coffee").strip()
    keyword = st.text_input("Consumer-demand keyword", value="Starbucks",
                            help="Google Trends search term (live mode only)").strip()

    st.divider()
    source_label = st.radio(
        "Data source",
        ["Bundled fixtures", "Live APIs"],
        captions=["Offline; covers the 7 curated markets", "Cached pulls; degrades per signal"],
        help="Live mode pulls World Bank / Comtrade / Google Trends through the "
        "cached ingestion clients and degrades per signal if a feed is down.",
    )
    source = "live" if source_label.startswith("Live") else "fixtures"

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

    st.markdown(f"#### {_fmt_market(country)} · HS {hs_code}")
    _hero(result)

    if assessment.has_ground_truth:
        _ground_truth_card(assessment.ground_truth, result)
    else:
        st.caption("No curated ground-truth case for this pair — showing the model's prediction only.")

    st.subheader("What drives the score")
    st.pyplot(contribution_chart(assessment), width="stretch")
    st.caption("Blue = each signal's contribution to the composite; the light track = the "
               "weight available to it. Weights renormalize over the signals present.")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Top drivers**")
        st.markdown(_chips(result.top_drivers(3), BLUE), unsafe_allow_html=True)
    with d2:
        st.markdown("**Biggest gaps**")
        st.markdown(_chips([(f, -v) for f, v in result.top_gaps(3)], BLUE_100),
                    unsafe_allow_html=True)

    for note in assessment.notes:
        st.markdown(f'<div class="mf-note">⚠ <span>{_html.escape(note)}</span></div>',
                    unsafe_allow_html=True)

    with st.expander("Raw input signals (table view)"):
        macro = assessment.signals.get("macro", {})
        if macro:
            st.dataframe(
                pd.DataFrame(sorted(macro.items()), columns=["indicator", "value"]),
                hide_index=True, width="stretch",
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
    calibrate = c1.toggle("Calibrate threshold (in-sample)", value=False)
    show_loo = c2.toggle("Leave-one-out evaluation", value=False)

    harness = ValidationHarness(scorer=MarketFitScorer())
    report = harness.run(cases, load_signal_fixtures(), calibrate=calibrate)
    s, b = report.score, report.bucket

    n_pos = sum(c.actual_success for c in report.cases)
    baseline = max(n_pos, s.n - n_pos) / s.n if s.n else 0.0
    _tiles(
        _tile("Bucket accuracy", f"{b.accuracy:.0%}", f"always-success baseline {baseline:.0%}"),
        _tile("Within ±1", f"{s.within_one_accuracy:.0%}", f"exact {s.exact_accuracy:.0%}"),
        _tile("MAE", f"{s.mae:.2f}", f"RMSE {s.rmse:.2f} · bias {s.bias:+.2f}"),
        _tile("Rank ρ", f"{s.rank_correlation:+.2f}", f"n = {s.n} cases"),
    )
    if report.calibrated_threshold is not None:
        st.caption(f"Success threshold calibrated in-sample to "
                   f"**{report.calibrated_threshold:.3f}** (optimistic — see leave-one-out).")
    if report.skipped:
        st.caption(f"Skipped (no signals): {', '.join(report.skipped)}")

    st.subheader("Predicted vs. actual")
    st.pyplot(predicted_vs_actual_chart(report.to_frame()), width="stretch")

    st.subheader("Per-case results")
    st.dataframe(
        report.to_frame(),
        hide_index=True,
        width="stretch",
        column_config={
            "case_id": st.column_config.TextColumn("case"),
            "country_iso3": st.column_config.TextColumn("market"),
            "outcome_label": st.column_config.TextColumn("documented outcome"),
            "actual_score": st.column_config.NumberColumn("actual"),
            "predicted_score": st.column_config.NumberColumn("predicted"),
            "score_error": st.column_config.NumberColumn("Δ"),
            "actual_success": st.column_config.CheckboxColumn("succeeded"),
            "predicted_success": st.column_config.CheckboxColumn("pred. success"),
            "bucket_correct": st.column_config.CheckboxColumn("bucket ✓"),
            "composite": st.column_config.ProgressColumn(
                "composite", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )

    st.subheader("Error analysis")
    analysis = analyze_errors(report)
    if not analysis.errors:
        st.success("No prediction errors: every case's bucket and score (±1) matched.")
    else:
        st.markdown(
            f"**{len(analysis.errors)}** case(s) with errors — "
            f"{analysis.n_bucket_errors} bucket, {analysis.n_magnitude_errors} magnitude ≥ 2; "
            f"{analysis.over_predictions} over-, {analysis.under_predictions} under-predicted. "
            f"Most implicated: "
            + ", ".join(f"{FEATURE_LABELS.get(f, f)} ({n}×)"
                        for f, n in analysis.implicated_features[:3])
        )
        for err in analysis.errors:
            arrow = "↑ over-predicted" if err.direction == "over" else "↓ under-predicted"
            chips = _chips(err.implicated, BLUE if err.direction == "over" else BLUE_100)
            badge_cls = "bad" if err.is_bucket_error else "mid"
            st.markdown(
                f"""
<div class="mf-card">
  <div class="head"><span class="title">{err.case_id}</span>
    <span class="meta">{err.outcome_label} — predicted {err.predicted_score} vs actual {err.actual_score}</span>
    <span class="mf-badge {badge_cls}">{arrow} ({err.score_error:+d})</span></div>
  <div class="body">{_html.escape(err.note)}</div>
  <div style="margin-top:.45rem">{chips}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    if show_loo:
        st.subheader("Leave-one-out")
        st.caption("For each case the threshold is calibrated on the other cases only — "
                   "an honest small-sample estimate.")
        loo_metrics, loo_frame = harness.leave_one_out(cases, load_signal_fixtures())
        _tiles(
            _tile("LOO accuracy", f"{loo_metrics.accuracy:.0%}",
                  f"TP {loo_metrics.tp} · FP {loo_metrics.fp} · FN {loo_metrics.fn} · TN {loo_metrics.tn}"),
            _tile("Precision", f"{loo_metrics.precision:.2f}"),
            _tile("Recall", f"{loo_metrics.recall:.2f}"),
            _tile("F1", f"{loo_metrics.f1:.2f}"),
        )
        st.dataframe(
            loo_frame,
            hide_index=True,
            width="stretch",
            column_config={
                "composite": st.column_config.ProgressColumn(
                    "composite", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "correct": st.column_config.CheckboxColumn("correct"),
            },
        )
