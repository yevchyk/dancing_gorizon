# Candle Data Inventory

_Auto-generated 2026-06-07 17:13 UTC — regenerate with `python -m src.run_data_inventory --write`._

> Single source of truth for what candle data exists on this machine. Stores are declared in `src/markets.py`. Tokenized equities live INSIDE the `crypto_feature` store (`data/candles`) mixed with crypto — the `EQUITY_TICKERS` set in `markets.py` is the canonical list.

## Stores

| store | market | on_disk | files | min | max | med_span_d | dir |
|---|---|---|---|---|---|---|---|
| crypto_feature | crypto | yes | 331 | 2022-05-25 16:00 | 2026-06-07 17:12 | 410.0 | `data\candles` |
| crypto_target_1m | crypto | **ABSENT** | 0 | - | - | 0.0 | `data\fast_v1\candles_1m` |
| tradfi_1m | tradfi | **ABSENT** | 0 | - | - | 0.0 | `data\nasdaq\okx_candles_1m` |
| tradfi_1m_legacy | tradfi | **ABSENT** | 0 | - | - | 0.0 | `data\nasdaq\candles_1m` |
| liquid_mixed | mixed | **ABSENT** | 0 | - | - | 0.0 | `data\okx_liquid\candles_mixed` |
| liquid_1m | mixed | **ABSENT** | 0 | - | - | 0.0 | `data\okx_liquid\candles_1m` |
| okx_stable_200 | mixed | yes | 200 | 2022-05-25 16:00 | 2026-06-05 12:27 | 895.9 | `data\okx_stable\candles_mixed` |
| bluechip | crypto | **ABSENT** | 0 | - | - | 0.0 | `data\bluechip\candles_1m` |

## `crypto_feature` (`data/candles`) — LIVE production store

- total symbols on disk: **331**  (crypto: 298, tokenized equities: 33)
- freshness (max candle): newest `2026-06-07 17:12`, oldest-tail `2026-06-01 12:07` UTC
- ⚠️ quarantined/corrupt (in `_corrupt/`, need re-fetch): ['DOGE_USDT_SWAP.parquet']

### Tokenized equities / ETFs (in `data/candles`)

| ticker | rows | start | end | span_d |
|---|---|---|---|---|
| AAPL | 50871 | 2026-03-03 16:00 | 2026-06-07 17:03 | 96.0 |
| ADBE | 6216 | 2026-06-02 16:00 | 2026-06-07 17:03 | 5.0 |
| AMAT | 11041 | 2026-05-27 16:00 | 2026-06-07 17:03 | 11.0 |
| AMD | 46720 | 2026-03-10 16:00 | 2026-06-07 17:03 | 89.0 |
| AMZN | 52604 | 2026-02-25 16:00 | 2026-06-07 17:03 | 102.0 |
| ANTHROPIC | 32429 | 2026-05-06 16:00 | 2026-06-07 17:12 | 32.0 |
| ARM | 33056 | 2026-04-28 16:00 | 2026-06-07 17:03 | 40.0 |
| ASML | 6201 | 2026-06-02 16:00 | 2026-06-07 17:03 | 5.0 |
| AVGO | 34763 | 2026-04-28 16:00 | 2026-06-07 17:03 | 40.0 |
| COIN | 51898 | 2026-02-25 16:00 | 2026-06-07 17:03 | 102.0 |
| COST | 35246 | 2026-04-26 16:00 | 2026-06-07 17:12 | 42.0 |
| CRCL | 50139 | 2026-02-25 16:00 | 2026-06-07 17:12 | 102.0 |
| CRWD | 11612 | 2026-05-25 16:00 | 2026-06-07 17:03 | 13.0 |
| CSCO | 27810 | 2026-05-18 16:00 | 2026-06-07 17:12 | 20.0 |
| GOOGL | 50165 | 2026-03-03 16:00 | 2026-06-07 17:03 | 96.0 |
| HOOD | 52906 | 2026-02-24 16:00 | 2026-06-07 17:04 | 103.0 |
| INTC | 48868 | 2026-02-25 16:00 | 2026-06-07 17:04 | 102.0 |
| IWM | 40162 | 2026-04-09 16:00 | 2026-06-07 17:04 | 59.0 |
| META | 50861 | 2026-03-03 16:00 | 2026-06-07 17:04 | 96.0 |
| MRVL | 30974 | 2026-05-11 16:00 | 2026-06-07 17:04 | 27.0 |
| MSFT | 50158 | 2026-03-03 16:00 | 2026-06-07 17:04 | 96.0 |
| MSTR | 51754 | 2026-02-24 16:00 | 2026-06-07 17:04 | 103.0 |
| MU | 50891 | 2026-03-03 16:00 | 2026-06-07 17:04 | 96.0 |
| NFLX | 48829 | 2026-03-10 16:00 | 2026-06-07 17:04 | 89.0 |
| NOW | 7537 | 2026-06-01 16:00 | 2026-06-07 17:04 | 6.0 |
| NVDA | 47508 | 2026-03-03 16:00 | 2026-06-07 17:04 | 96.0 |
| ORCL | 48834 | 2026-03-10 16:00 | 2026-06-07 17:04 | 89.0 |
| PLTR | 52609 | 2026-02-25 16:00 | 2026-06-07 17:04 | 102.0 |
| QCOM | 27107 | 2026-05-18 16:00 | 2026-06-07 17:04 | 20.0 |
| QQQ | 47486 | 2026-03-03 16:00 | 2026-06-07 17:04 | 96.0 |
| SPY | 50859 | 2026-03-03 16:00 | 2026-06-07 17:04 | 96.0 |
| TSLA | 51762 | 2026-02-24 16:00 | 2026-06-07 17:04 | 103.0 |
| TSM | 48832 | 2026-03-10 16:00 | 2026-06-07 17:05 | 89.0 |

