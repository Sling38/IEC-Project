# IEC-Project — MarketFit

**MarketFit** is a multi-agent system that assesses international market-entry
viability for a `(product, country)` pair by combining three data layers:
historical **trade flows**, **macroeconomic fit**, and real-time **consumer demand**.

This repo currently implements the **Checkpoint 1 data-ingestion pipeline**.

## Checkpoint 1 — Data ingestion (Alexander: Comtrade/BACI + World Bank)

Pulls and normalizes the macro/trade input signals for any `(product, market)` pair
and caches every response locally so free-tier rate limits don't block downstream work.

| Module | Source | Signal | Key needed? |
|--------|--------|--------|-------------|
| [`world_bank.py`](src/marketfit/ingestion/world_bank.py) | World Bank WDI API | Macro indicators (GDP, population, inflation, trade %, internet use…) | No |
| [`comtrade.py`](src/marketfit/ingestion/comtrade.py) | UN Comtrade v1 API | Live bilateral trade flows by HS code | Optional* |
| [`baci.py`](src/marketfit/ingestion/baci.py) | CEPII BACI bulk CSVs | Cleaned historical bilateral trade (offline, no limits) | No |
| [`cache.py`](src/marketfit/ingestion/cache.py) | — | On-disk JSON cache shared by the API clients | — |

\* Comtrade uses the free rate-limited **preview** endpoint with no key; set
`COMTRADE_API_KEY` to use the full subscription endpoint.

## Checkpoint 1 — Demand signal + ground truth (Samuel: Google Trends + label curation)

Adds the third input signal (real-time **consumer demand**) and the curated
**ground-truth** dataset that later checkpoints validate predictions against.

| Module | Source | Signal | Key needed? |
|--------|--------|--------|-------------|
| [`trends.py`](src/marketfit/ingestion/trends.py) | Google Trends (pytrends) | Consumer search interest for a term in a market | No |
| [`groundtruth/labels.py`](src/marketfit/groundtruth/labels.py) | — | Outcome-label taxonomy + 1–5 success metric | — |
| [`groundtruth/cases.py`](src/marketfit/groundtruth/cases.py) | [curated CSV](data/ground_truth/starbucks_market_entries.csv) | Documented historical market-entry outcomes | — |

Google Trends is an unofficial, rate-limited endpoint, so — like Comtrade/World
Bank — every response is cached under `data/cache/trends/`; `pytrends` is imported
lazily (only on a cache miss).

### Success metric (defined before modeling)

Every validation case gets one of four outcome labels, each mapped to a **1–5
entry-viability score**. The Checkpoint 2 agent emits the same 1–5 score, so
"the prediction matched reality" is concrete and comparable:

| Label | Meaning | Score |
|---|---|---|
| Strong Success | Entered and became a large, durable market | 5 |
| Moderate Success | Entered and sustained a viable presence | 4 |
| Struggled | Entered but under-performed / grew painfully | 2 |
| Withdrew | Entered then materially retreated or exited | 1 |

### Ground-truth cases

Seven documented Starbucks (`coffee`, HS `0901`) market entries spanning all four
labels, each row wired to the other signals via `comtrade_reporter_m49`,
`google_trends_geo`, and `trends_keyword`, and carrying a `source_url`:

| case_id | country | entry_year | outcome_label |
|---|---|---|---|
| SBUX-JPN-1996 | Japan | 1996 | Strong Success |
| SBUX-CHN-1999 | China | 1999 | Strong Success |
| SBUX-KOR-1999 | South Korea | 1999 | Strong Success |
| SBUX-ITA-2018 | Italy | 2018 | Moderate Success |
| SBUX-IND-2012 | India | 2012 | Moderate Success |
| SBUX-VNM-2013 | Vietnam | 2013 | Struggled |
| SBUX-AUS-2000 | Australia | 2000 | Withdrew |

```bash
# Ground-truth cases + taxonomy (offline) then a live Trends demand snapshot
python scripts/demo_trends.py --country ITA --keyword Starbucks
python scripts/demo_trends.py --skip-trends          # offline only
```

```python
from marketfit.ingestion import GoogleTrendsClient, iso3_to_geo
from marketfit.groundtruth import GroundTruthLoader, score_for

# Consumer-demand snapshot: search interest for "Starbucks" in Japan.
trends = GoogleTrendsClient()
demand = trends.demand_snapshot("Starbucks", geo=iso3_to_geo("JPN"))

# Curated, validated ground-truth cases (raises if a label/score is inconsistent).
cases = GroundTruthLoader().load()
score_for("Withdrew")  # -> 1
```

### Install & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Hello-World demo: macro snapshot + coffee (HS 0901) imports for Italy (M49 380)
python scripts/demo_ingestion.py --country ITA --reporter 380 --hs 0901 --year 2022
```

### Usage

```python
from marketfit.ingestion import WorldBankClient, ComtradeClient, BaciLoader

# Macro snapshot (one row per indicator, most recent year) — no API key.
wb = WorldBankClient()
snapshot = wb.latest_snapshot("ITA")

# Live trade flow: Italy's coffee imports from the world, 2022.
ct = ComtradeClient()                       # set COMTRADE_API_KEY for full tier
flows = ct.get_trade(reporter=380, partner=0, hs_code="0901", period=2022, flow="M")

# Offline historical trade from BACI files in data/baci/ (HS-code trace via prefix).
baci = BaciLoader()
coffee_2022 = baci.load_year(2022, hs_prefix="0901")
```

### Data samples (metadata + sample rows)

**World Bank `latest_snapshot("ITA")`** — normalized macro snapshot:

| indicator_code | indicator | year | value |
|---|---|---|---|
| NY.GDP.PCAP.CD | GDP per capita (current US$) | 2022 | 34,776 |
| SP.POP.TOTL | Population, total | 2022 | 58,940,425 |
| IT.NET.USER.ZS | Individuals using the Internet (%) | 2022 | 84.9 |

**Comtrade / BACI trade frame** — tidy columns:

`year, exporter_iso3, exporter, importer_iso3, importer, hs_code, commodity, trade_value_usd, quantity_ton`

| year | exporter | importer | hs_code | commodity | trade_value_usd |
|---|---|---|---|---|---|
| 2022 | Brazil | Italy | 090111 | Coffee; not roasted | 1,500,000,000 |

*(BACI values are converted from thousands-USD to USD; Comtrade returns `primaryValue` in USD.)*

### Getting BACI data

BACI isn't checked in (large). Download an HS release from
[CEPII](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37),
unzip into `data/baci/`, then call `BaciLoader().load_year(<year>)`.

### Tests

Offline tests exercise every module (cache-seeded WB/Trends payloads, synthetic
BACI files, sample Comtrade rows, ground-truth consistency) with no network
dependency:

```bash
python -m pytest tests/               # or run each file directly:
python tests/test_ingestion.py        # Comtrade / BACI / World Bank
python tests/test_trends.py           # Google Trends normalization + snapshot
python tests/test_groundtruth.py      # label taxonomy + curated cases
python tests/test_features.py         # feature engineering
python tests/test_scoring.py          # entry-viability scoring model
python tests/test_validation.py       # validation harness, metrics, error analysis
python tests/test_demo.py             # demo-UI helpers + app compile check
```

## Checkpoint 2 — Scoring model + feature engineering (Alexander)

Turns the three ingested signals into an interpretable feature vector and scores a
`(product, country)` pair on the **same 1–5 entry-viability scale** the ground-truth
labels use, so predictions are directly comparable to documented outcomes.

| Module | Role |
|--------|------|
| [`features/engineering.py`](src/marketfit/features/engineering.py) | Normalizes signals into 8 `[0,1]` features (size, wealth, growth, price stability, openness, connectivity, existing trade, consumer demand) |
| [`scoring/model.py`](src/marketfit/scoring/model.py) | Weighted linear scorer → 1–5 score + success/struggle bucket, with a per-feature contribution breakdown for rationale generation |

Design notes: features use **fixed, documented reference ranges** (not scalers fit
on the tiny sample) so scores stay stable and explainable. The scorer is a
transparent **weighted sum** — with only a handful of ground-truth cases a heavy ML
model would overfit — and it **renormalizes weights** over whichever signals are
present, so a missing feed neither drags nor inflates the score. Every prediction
exposes `top_drivers()` / `top_gaps()` to feed the Checkpoint-2 LLM rationale.

```python
from marketfit.features import FeatureBuilder
from marketfit.scoring import MarketFitScorer

fv = FeatureBuilder().from_signals(
    "ITA", "0901",
    macro=WorldBankClient().latest_snapshot("ITA").pipe(FeatureBuilder.snapshot_to_macro),
    product_import_usd=1_600_000_000, demand_interest=40,
)
result = MarketFitScorer().score(fv)
print(result.score, result.is_success, result.top_drivers(3))
```

```bash
# Score all ground-truth cases from bundled offline fixtures (preview of validation)
python scripts/demo_scoring.py            # add --calibrate to tune the threshold
```

Sample run over the 7 curated Starbucks cases (illustrative offline fixtures):
score within ±1 of the actual label on **5/7**, success/struggle bucket correct on
**5/7**. Live runs swap the fixtures for real ingestion pulls; Samuel's Checkpoint-2
validation harness turns this into proper metrics + error analysis.

## Checkpoint 2 — Validation harness + metrics + error analysis (Samuel)

Runs the scorer over the curated ground-truth cases and measures how well predictions
match documented outcomes — the "first validation test on Starbucks for ground-truth
comparison" — then diagnoses where the model is wrong to drive feature iteration.

| Module | Role |
|--------|------|
| [`validation/metrics.py`](src/marketfit/validation/metrics.py) | Score metrics (MAE, RMSE, exact/within-1, bias, Spearman ρ) + bucket metrics (accuracy, precision, recall, F1, confusion) — no scikit-learn/scipy |
| [`validation/harness.py`](src/marketfit/validation/harness.py) | `ValidationHarness` → `ValidationReport` (per-case table + metrics); in-sample **and** leave-one-out evaluation |
| [`validation/error_analysis.py`](src/marketfit/validation/error_analysis.py) | Flags bucket + magnitude errors, over/under-prediction bias, and the features most implicated in each miss |

Design notes: metrics are computed for **both** heads the scorer emits — the ordinal
1–5 score and the binary success/struggle bucket. Because the ground-truth set is
tiny, the harness reports honest **leave-one-out** accuracy (each case's threshold is
calibrated on the *other* cases) alongside the optimistic in-sample number, so we
don't fool ourselves. Error analysis reads each miss's `top_drivers`/`top_gaps` to
name the signals to revisit.

```bash
python scripts/demo_validation.py     # per-case table, metrics, LOO, error analysis
```

```python
from marketfit.groundtruth import GroundTruthLoader
from marketfit.validation import ValidationHarness, analyze_errors, load_signal_fixtures

report = ValidationHarness().run(GroundTruthLoader().load(), load_signal_fixtures())
print(report.summary())               # MAE/RMSE/within-1/bias/ρ + acc/precision/recall/F1
print(analyze_errors(report).summary())
```

Sample run over the 7 cases (offline fixtures, default weights): bucket **acc 71%**
(precision 0.71, recall 1.00) in-sample, **43% under leave-one-out** — the harness
surfacing the small-sample optimism. Error analysis flags the two over-predicted
failures (Vietnam, Australia) and implicates `market_size`/`purchasing_power`, i.e.
the model rewards big/rich markets and misses entrenched local competition — a
concrete next-iteration signal. *(The point is the harness/metrics, not the accuracy.)*

## Checkpoint 3 — Demo UI (Alexander)

Streamlit interface: enter a `(product, market)` pair and see the predicted 1–5
entry-viability score, what drives it, and the documented ground-truth outcome when
the pair is one of the curated cases.

```bash
streamlit run app/streamlit_app.py
```

| Piece | Role |
|--------|------|
| [`app/streamlit_app.py`](app/streamlit_app.py) | The UI (view layer only): assessment tab + validation tab |
| [`demo/data.py`](src/marketfit/demo/data.py) | Tested helpers behind the UI: signal assembly (fixtures **or** live pulls), ground-truth lookup, feature→score wrapper |

**Market assessment tab** — score / bucket / composite metrics, the ground-truth
comparison panel (with source link) when available, a per-feature contribution
chart (blue = contribution, gray = available weight), top drivers & gaps, and the
raw input signals.

**Validation tab** — runs the Checkpoint-2 harness live: metrics summary, a
predicted-vs-actual dumbbell chart across all 7 cases, the per-case table, error
analysis, plus optional in-sample calibration and leave-one-out evaluation.

Data sources: **bundled fixtures** (offline, the 7 curated markets) or **live APIs**
(any market — pulls World Bank/Comtrade/Trends through the cached ingestion clients
and *degrades per signal* if a feed is down, reporting what was skipped; the scorer
renormalizes over the signals present).

## Checkpoint 3 — Final validation + documentation (Samuel)

Finalized validation results — including a **live-data run** through the real APIs
(2026-07-15, 20/21 case-signals landed) that confirmed the fixture-based findings:
bucket accuracy 71% in-sample (equal to the always-success baseline, stated
explicitly), leave-one-out 57% live vs 43% on fixtures, and the same two
over-predicted failures (Vietnam, Australia) implicating the missing
competition/category-maturity signal.

| Document | Contents |
|---|---|
| [`docs/final_report.md`](docs/final_report.md) | Final documentation: methods, live + fixture validation results, baseline confrontation, error analysis, limitations, reproducibility |
| [`docs/checkpoint3_report.md`](docs/checkpoint3_report.md) | Checkpoint 3 class-report draft (milestones, match, evidence, skill learning) |

## Roadmap

- **Checkpoint 1** ✅ trade + macro ingestion (Alexander) · ✅ Google Trends + ground-truth labels (Samuel)
- **Checkpoint 2** ✅ scoring model + feature engineering (Alexander) · ✅ validation harness + metrics + error analysis (Samuel) · LLM rationale → documented as future work ([final report §5–6](docs/final_report.md))
- **Checkpoint 3** ✅ Streamlit demo UI (Alexander) · ✅ final validation + documentation (Samuel)

GitHub: https://github.com/Sling38/IEC-Project
