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

## Roadmap

- **Checkpoint 1** ✅ trade + macro ingestion (Alexander) · ✅ Google Trends + ground-truth labels (Samuel)
- **Checkpoint 2** ✅ scoring model + feature engineering (Alexander) · validation harness + metrics (Samuel) · LLM rationale
- **Checkpoint 3** Streamlit demo UI, final validation

GitHub: https://github.com/Sling38/IEC-Project
