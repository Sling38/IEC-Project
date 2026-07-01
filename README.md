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

Offline tests exercise all three modules (cache-seeded WB payload, synthetic BACI
files, sample Comtrade rows) with no network dependency:

```bash
python tests/test_ingestion.py        # or: python -m pytest tests/
```

## Roadmap

- **Checkpoint 1** ✅ trade + macro ingestion (this) · Google Trends + ground-truth labels (Samuel)
- **Checkpoint 2** scoring model + rationale, validation harness
- **Checkpoint 3** Streamlit demo UI, final validation

GitHub: https://github.com/Sling38/IEC-Project
