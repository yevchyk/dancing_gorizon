# Horizon-Conditioned Model — Build Brief (for the implementing agent)

> Read this top to bottom before writing any code. It is a spec, not a suggestion.
> Owner is blunt and wants honest stats + clean tables. No fake results.
> If a number here is wrong vs the data, STOP and report — do not "fix" it silently.

---

## 0. What we are building (one paragraph)

Two **separate** binary CatBoost classifiers — an **UP model** and a **DOWN model** —
that see the market on 4 timeframes (5m / 15m / 1h / 4h) and receive the forecast
**horizon in minutes** as an input feature. Each model outputs a probability. A
thin code layer combines them:

```
up>=0.70 & down<=0.30  -> LONG
down>=0.70 & up<=0.30  -> SHORT
both high               -> volatility, unclear -> skip
both low                -> NO_TRADE (dead zone appears by itself)
```

The dead zone is **not a third class** — it is just the `0` label of both models.

---

## 1. Data — exactly what exists (already verified)

- Store: **`crypto_feature`** in `src/markets.py` (dir `data/candles`). Use
  `from src.markets import get; get("crypto_feature")`.
- Columns per parquet: `open, high, low, close, volume`, index `timestamp` (UTC).
- **Multi-resolution per symbol.** For BTC: `1d` 2022→2024-05, `1h` 2024-05→2025-09,
  `5m` 2025-09-27→recent, `1m` last ~2 weeks.
- **Genuine 5m-or-finer data starts ~2025-09-27.** That is the hard ceiling.
- **227 symbols** have ≥200 days of 5m-or-finer history. After dropping the
  blacklist (`src/config.py::BLACKLIST_SYMBOLS`, which already contains SPX + the
  8 toxics + others) → **218 symbols** = our training universe.

### Usable window
**2025-09-27 → now ≈ 249 days (~8 months) of real 5m.** You CANNOT go deeper at 5m.
Do not fabricate 5m from the older 1h/1d tiers.

---

## 2. THREE TRAPS that will silently break this (read twice)

1. **In the 5m era there are NO native 1h / 4h candles.** The store switched 1h→5m
   on 2025-09-27. You MUST build 15m/1h/4h by **resampling UP from the 5m grid**
   (15m = 3×5m, 1h = 12×5m, 4h = 48×5m). If you read "1h" straight from the file you
   get a hole in the middle of history. Base grid = a uniform 5m series (resample the
   recent 1m tier DOWN to 5m so the whole window is one clean 5m grid).

2. **Leakage / embargo.** The target looks up to +180 min ahead. At every split
   boundary purge a gap of **≥180 min**: a train sample at time `t` with horizon `h`
   must satisfy `t + h < holdout_start - margin`. No train sample's target window may
   touch the holdout. Same for the early-stopping validation slice.

3. **No scaling, time-based split only.** Features are ratios (already stationary),
   trees need no normalization. Do NOT z-score/min-max with global stats (that leaks
   the future). Do NOT random-split — strictly chronological.

---

## 3. Feature spec (exact)

Base = uniform 5m grid per symbol (resample 1m→5m for the recent tail). Build the
other TFs by resampling UP. For each timeframe use the last **N_POINTS = 30**
**completed** bars at/just before the sample time `t` (NEVER the currently forming bar).

Per point `i` (i = 0 newest .. 29 oldest):

| Timeframe | Features per point | BTC included? |
|---|---|---|
| 5m  | `rel_coin`, `vol_ratio` | **no** |
| 15m | `rel_coin`, `vol_ratio` | **no** |
| 1h  | `rel_coin`, `rel_btc`, `vol_ratio` | **yes** |
| 4h  | `rel_coin`, `rel_btc`, `vol_ratio` | **yes** |

> Owner rule: **BTC reference only on 1h and 4h.** On 5m/15m it is noise — do not add it.

Definitions (on each TF series, `k` = index of last completed bar ≤ `t`):
- `rel_coin[i]  = close[k-i]      / close[k-i-1]`        (1.01 = +1%)
- `rel_btc[i]   = btc_close[k-i]  / btc_close[k-i-1]`    (BTC_USDT_SWAP resampled to same TF, aligned by timestamp)
- `vol_ratio[i] = volume[k-i]     / volume[k-i-1]`       (guard div-by-zero → 1.0)

Plus 2 horizon columns:
- `horizon_minutes` (raw int)
- `horizon_log = ln(1 + horizon_minutes)`

**Column count:** (2+2+3+3)·30 = **300** + 2 horizon = **302**. (`N_POINTS` is tunable;
it controls the count and the lookback: 5m→2.5h, 15m→7.5h, 1h→30h, 4h→5d.)

Suggested column naming: `c5m_rel_{i}`, `c5m_vol_{i}`, `c15m_rel_{i}`, `c15m_vol_{i}`,
`c1h_rel_{i}`, `c1h_btc_{i}`, `c1h_vol_{i}`, `c4h_rel_{i}`, `c4h_btc_{i}`, `c4h_vol_{i}`,
`horizon_minutes`, `horizon_log`.

---

## 4. Horizon conditioning (the core mechanic — do not get this wrong)

For ONE base snapshot `(symbol, t)` you emit **multiple rows**, one per horizon, each
with its own `horizon_minutes`, its own label, and its own weight.

- `HORIZON_ANCHORS = [5, 15, 30, 60, 120, 180]` (minutes)
- `+ 2 random` horizons per snapshot: random ints in `[5, 180]` (fills the gaps so the
  model interpolates smoothly instead of making step-functions at the anchors).
- So ~8 rows per snapshot.

---

## 5. Target + weight (close-at-horizon, owner-approved)

On the 5m grid, for horizon `h` minutes (`t+h` must exist):
```
ret = close_5m[t + h] / close_5m[t] - 1          # entry/exit on 5m close
up_label   = 1 if ret*100 >=  thr(h) else 0
down_label = 1 if ret*100 <= -thr(h) else 0      # both 0 = dead zone
weight     = 1 + min(abs(ret*100) / 3.0, 1.0) * 4   # range 1..5, CAPPED at 3% move
```

Dead-zone thresholds `thr(h)` in **percent** (owner's starter grid — put these in the
truth file, they WILL be tuned):

| horizon (min) | 5 | 15 | 30 | 60 | 120 | 180 |
|---|---|---|---|---|---|---|
| thr % | 0.4 | 0.6 | 0.8 | 1.1 | 1.5 | 1.8 |

For random horizons between anchors, **linearly interpolate** `thr(h)`.

> The weight cap at 3% is deliberate: moves above 3% are rare pumps/dumps/liquidations.
> We refuse to let the model overfit to manipulation.

---

## 6. Split — walk-forward (owner-approved)

Sort all samples by base timestamp `t`. Run **3 folds**, each trains ONLY on its past:

| Fold | Test window (7 days) | Purpose |
|---|---|---|
| 1 (primary) | last 7 days | live-like check |
| 2 | an earlier **down/red** week | regime stress |
| 3 | an earlier **sideways/bull** week | regime stress |

Pick folds 2 & 3 from BTC behavior over the 8 months (you choose; report which weeks
and why). For each fold:
- train = samples with `t + h < test_start - EMBARGO` (`EMBARGO = 180 min`),
- early-stopping validation = the last ~10% of train **by time** (also embargoed),
- test = samples whose `t` is inside the 7-day window.

> Rationale: the owner's previous model died on a regime change because it was only
> tested on one (rising) week. Folds 2–3 exist to catch exactly that. If the model is
> only good on fold 1, say so plainly.

---

## 7. Models

Two `CatBoostClassifier`, identical features, different label+weight:
```python
params = dict(
    loss_function="Logloss", eval_metric="AUC",
    task_type="GPU", devices="0",          # RTX 4070
    iterations=4000, learning_rate=0.05, depth=6,
    l2_leaf_reg=3.0, random_seed=42,
    od_type="Iter", od_wait=200,           # early stopping
)
```
- UP model: `fit(X, up_label, sample_weight=w, eval_set=val_up)`
- DOWN model: `fit(X, down_label, sample_weight=w, eval_set=val_down)`
- Save `up.cbm`, `down.cbm`, the feature-name list, and a JSON snapshot of all config
  constants used (universe, N_POINTS, horizons, thresholds, window, embargo).

---

## 8. Sizing / performance (machine: 34 GB RAM, 16 CPU, RTX 4070 12 GB)

- `SAMPLE_STRIDE_MIN = 120` (one base snapshot every 2 h). Default chosen so the first
  full run fits comfortably: 218 syms × 249 d × 12 snaps/day × 8 horizons ≈ **5.2 M rows**
  → ~6 GB as float32. Tunable down to 60 (≈10 M rows) once it works.
- Build the dataset as **per-symbol parquet shards** under `data/hc/dataset/`, then
  load+concat in chunks. Downcast all feature columns to `float32`.
- Train on **GPU** (quantized Pool compresses to ~1 byte/feature in VRAM → fits 12 GB).
- If RAM gets tight: raise the stride or lower `N_POINTS` before anything else.

---

## 9. Eval — clean tables only (owner loves these)

On each fold's test set, predict `up_prob`, `down_prob`, then emit:

- **Table A — by horizon** (5/15/30/60/120/180): n, base_rate, AUC, and precision
  (win-rate) of the positive at `prob ≥ 0.70`.
- **Table B — calibration**: prob buckets 0.5–0.6 … 0.9–1.0 → n, realized win-rate.
  (Does 0.70 actually mean ~70%?)
- **Table C — by symbol**: top/bottom 15 by edge at `prob ≥ 0.70` (min sample guard).
- **Table D — decision level**: apply the LONG/SHORT rule from §0; report count,
  win-rate, and avg return **net of 0.15% round-trip fee**.
- Print per fold AND state plainly whether the edge survives folds 2–3.

Write results to `docs/HC_MODEL_RESULTS.md` + a `.csv`. No cherry-picking; if a fold
is bad, show it.

---

## 10. Deliverables

1. `configs/hc_universe.json` — the 218 symbols (generated by the filter below; list them).
2. `src/run_hc_dataset.py` — builds the per-symbol parquet shards.
3. `src/run_hc_train.py` — trains `up.cbm` + `down.cbm`, saves config snapshot.
4. `src/run_hc_eval.py` — produces the tables in §9.
5. `TRADING_MODEL_TRUTH.md` (repo root) — the editable truth file (structure below),
   filled with the REAL numbers from this run.
6. `docs/HC_MODEL_RESULTS.md` — the result tables.

### Universe filter (use this, don't invent your own)
```python
from src.markets import get
from src import config as C
import pandas as pd
s = get("crypto_feature"); bl = set(C.BLACKLIST_SYMBOLS); keep = []
for p in s.files():
    ts = pd.to_datetime(pd.read_parquet(p, columns=["timestamp"])["timestamp"], utc=True).sort_values()
    if len(ts) < 10: continue
    fine = ts[ts.diff().dt.total_seconds() <= 300]
    if len(fine) and (fine.max()-fine.min()).total_seconds()/86400 >= 200 and p.stem not in bl:
        keep.append(p.stem)
# expect len(keep) == 218
```

---

## 11. Order of operations — SMOKE TEST FIRST (do not skip)

0. **Smoke run**: 5 symbols × last 20 days × anchors only. Verify: no NaN, shapes
   match §3, no row has `t+h` in the future, models train, eval prints. Only then scale.
1. Generate `configs/hc_universe.json` (assert count == 218).
2. Build full dataset → report final row count, disk size, peak RAM.
3. Train both models (fold 1 first), confirm AUC sane (> 0.5, not 0.99 → 0.99 = leak).
4. Run all 3 folds + eval tables.
5. Write `TRADING_MODEL_TRUTH.md` + `docs/HC_MODEL_RESULTS.md`.

> An AUC near 0.99 or a holdout win-rate that looks too good = LEAK. Stop and audit
> §2 before celebrating.

---

## 12. DO-NOT checklist

- [ ] DO NOT use the currently-forming bar — only completed bars ≤ `t`.
- [ ] DO NOT read 1h/4h from the store in the 5m era — resample up from 5m.
- [ ] DO NOT add `rel_btc` to 5m or 15m (BTC only on 1h & 4h).
- [ ] DO NOT z-score / min-max with global stats — ratios only, no scaling.
- [ ] DO NOT random-split — chronological only, 180-min embargo at every boundary.
- [ ] DO NOT let a train sample's `t..t+h` overlap a test window.
- [ ] DO NOT include blacklist symbols — use the universe json.
- [ ] DO NOT report a number you didn't measure.

---

## 13. `TRADING_MODEL_TRUTH.md` structure to create (editable by owner)

Short, practical, fill the TODOs with real numbers from the run:

```
# Trading Model Truth

## Idea
Horizon-conditioned multi-timeframe CatBoost. Two separate models: UP, DOWN.

## Universe
218 OKX crypto perps (crypto_feature, >=200d of 5m, blacklist removed).
File: configs/hc_universe.json

## Data window
5m era: 2025-09-27 -> <run date>  (~249 days). Hard ceiling (older = 1h/1d only).

## Timeframes & features
5m : rel_coin, vol_ratio
15m: rel_coin, vol_ratio
1h : rel_coin, rel_btc, vol_ratio
4h : rel_coin, rel_btc, vol_ratio
N_POINTS=30 per TF -> 302 columns (incl. horizon_minutes, horizon_log)

## Horizons
anchors 5/15/30/60/120/180 + 2 random per snapshot (5..180)

## Target (close-at-horizon)
up=1 if ret>=thr(h); down=1 if ret<=-thr(h); else dead zone.
thr% : 5=0.4 15=0.6 30=0.8 60=1.1 120=1.5 180=1.8   <-- TUNE ME
weight = 1 + min(|ret%|/3,1)*4  (cap at 3%)

## Split
walk-forward 3 folds (last 7d + 2 regime weeks), embargo 180 min.

## Output
up_prob, down_prob -> long/short/skip/no-trade.

## Results (TODO from run)
fold1 / fold2 / fold3:  AUC up=__ down=__ ; win@0.70=__ ; net avg ret=__
best horizon: __   worst: __   best symbols: __   failed: __
```
