# Dancing Taras (TT) — design of a new training paradigm

> Status: **SPEC / plan** (2026-06-15). Code is partly written; tests are NOT to be
> run yet (user's call). This is the authoritative design of the new direction; once it
> is confirmed on data, the key conclusions get folded into `CLAUDE.md` §13.

## 0. In one paragraph — what the break is
All previous models (`hc_final` d7/d8, OLD/NEW, v4, band-spec) did **one** thing: binary
classification of `P(profit)` with the horizon fed in as an **input feature**, querying
the model at several horizons separately. TT changes the **objective** itself: the model
**regresses the entire future price curve** (the return trajectory) in a single output,
and the horizon becomes the **OUTPUT axis**, not an input. From the curve we then *derive*
the signal, the best horizon, and the confidence. On top sits a separate **ranker with
abstention** that, out of a pile of candidates, picks what to bet on and what to "not even
look at."

Why this isn't a random idea but a cure for a known bug: §9 of CLAUDE.md proved that
dense-querying corrupts (max-prob across horizons = an extreme order statistic that picks
the most over-confident horizon). If the horizon is the output axis rather than an input,
that pathology vanishes at the root: the whole curve is produced consistently in one pass,
with no re-query.

---

## 1. Closed decisions (user's answers + elegant defaults)

### A. Target — "the graph"
- **A1. Cumulative log-return.** Target at node h = `log(close[t+h]/close[t_entry])`.
  Cumulative (rather than step return) was chosen deliberately: the curve is smooth by
  construction (neighbouring nodes are strongly correlated) → **free "smoothing"** without
  a custom penalty (answer to B9).
- **A2 (elegant default). Normalize by the symbol's realized volatility.**
  Target = cumulative-log-return / σ_realized(symbol, t). Reason: the features are
  symbol-blind (§6) → the target must also be scale-free, otherwise the MultiRMSE loss
  would be dominated by high-volatility alts. At decision-time we multiply back by the
  current σ → we get the expected % move, which we then compare against cost.
- **A3/A4. Grid = minimal (1-min), horizons 1..240 min = 240 nodes.**
  Binance has a year of 1-min candles (§12) → the old limit "1-min only for 22 days" (that
  was OKX, the `min1_2to120` memory) does NOT apply. A continuous query (e.g. h=2.5 min) =
  **interpolation on the predicted curve**, not a separate model run. So "give 2.5 → get
  2m 3s" comes for free, because the curve *is* a continuous object; the nodes are just the
  points where we supervise it.
  - Knob `--horizon-step-min` (default 1) lets you coarsen the grid (e.g. 1-min up to 120,
    then 2-min up to 240 ≈ 180 nodes) if 240 outputs become too expensive in training.

### B. Loss — "a formula that equates two graphs"
- **B5 (elegant default). Not one trajectory but a DISTRIBUTION — in two layers:**
  the median curve (Phase 1, MultiRMSE) **+** a quantile "fan" layer (Phase 2). This is a
  direct answer to the random-walk trap: a bare MSE regression converges to the conditional
  mean (≈0 for returns) and the edge disappears; what saves it is precisely the confidence
  layer — it tells "flat with confidence" from "flat because unknown".
- **B6 (elegant default). Closeness metric = L2 on the cumulative curve + per-node
  standardization.** The cumulative form encodes both level and shape (it's a sequence of
  accumulated values) → a separate cosine/DTW term isn't needed in Phase 1. DTW is kept in
  reserve (it needs a custom objective / a neural net, and we're on CatBoost — E17).
- **B7. We do NOT bake cost into the loss** (user's call: "less external model noise"). The
  curve head learns to predict the **clean** return; cost (Binance RT ~0.126%,
  `configs/binance_costs.json`) is applied at the decision stage. This is cleaner: the model
  is not rewarded/punished for tiny moves around the threshold.
- **B8 (elegant default). Horizon weighting = per-node standardization (equal
  information).** Cumulative return grows ~√h → late nodes are larger in magnitude and
  would dominate the loss unweighted. We standardize each node (z by horizon) → all horizons
  weigh equally.
- **B9. Smoothness is "free" via A1 (cumulative) + shared MultiRMSE trees** (all 240 heads
  share tree structure → neighbouring nodes don't diverge). An explicit curvature penalty
  `λ·Σ(p₋−2p+p₊)²` is kept as **Phase 1.5**, only if the curves come out jagged (in CatBoost
  this is a custom multi-target objective — fiddly, hence not the default).

### C. From curve → to trade
- **C10 (elegant default).** The signal is **derived**, not classified separately. For a
  candidate (symbol, scan, h): read the curve at h (interpolation) → expected move; denorm
  by the current σ → expected %; edge = expected% − cost; side = sign of the curve;
  confidence = width of the quantile fan at h (Phase 2). Narrow fan + median past cost =
  high-conviction; wide fan = "don't look" → into abstention.
- **C11. Pipeline, not joint-multi-task** (the "what's best" answer + a CatBoost limit:
  one model can't train heterogeneous heads). Order: curve+confidence → their outputs become
  features → ranker (stacking). This realizes "we pass the time + a pile of positions; the
  model says which position and which points on the graph are most likely."

### D. Ranking + abstention (mandatory base)
- **D12. Two groups (as requested):**
  - primary `group_id = scan` (all symbol×horizon candidates of that minute) — rank "which
    position is best RIGHT NOW";
  - secondary "supergroup" = a sliding window of the **last 60 scans** — rank "which moment
    in the last hour is even worth entering at all". We train TWO rankers (intra-scan and
    cross-time); at inference we combine the two scores = "a prediction from two points".
    For now (user's call) it's **training only** — both group columns are built in the
    dataset, the inference combination is enabled later.
- **D13 (elegant default). Graded relevance = realized clean return** (not a binary label).
  Winners are ranked by the strength of their net, not just "won/not" → the ranker learns
  more finely.
- **D14 (elegant default). A no-trade null in EVERY group, relevance = 0.** Any candidate
  must beat "do nothing". If the best net ≤ 0 → null on top → abstention. This is the learned
  "better not to look" (consistent with §2 "detect the bad day and sit in cash", but learned
  rather than heuristic).
- **D15 (elegant default). `QuerySoftMax`** — built for "pick the best in the group", maps
  directly onto a top-k engine. Reserve: `YetiRankPairwise`.
- **D16 (elegant default). Multi-leg per (symbol,scan) is kept, but as ONE risk unit**
  (correlated, §10) — the stake is split, not counted as independent bets.

### E. Architecture / framework
- **E17. CatBoost** (user's call). Implementation: `MultiRMSE` for the curve + quantile
  models for the fan + ranking mode (`QuerySoftMax`) for the selector.
- **E18 (elegant default). 3-seed ensemble** per head (like v4/band) for tail stability.

### F. Features / inputs — the **MAXIMAL schema** (user's call: "I want maximal")
- **F19. The superset of all curves + the full BTC curve + v5 regime.** Implemented in
  `src/tt/schema_tt.py` (`FEATURE_COLUMNS_TT`, **561 features**, the hc pipeline is
  untouched):
  - curves: c1m + c5m + c15m + c1h(+`c1h_btc`) + c4h(+`c4h_btc`) — 12 cols/point;
  - **1-min is BACK as input** (on OKX it was dead under the 0.75% wall — but Binance 0.126%
    + curve regression is a different game);
  - **BTC both as a curve (`c1h_btc`/`c4h_btc`) and as scalars** (regime block);
  - +18 v5 regime scalars (BTC ret/vol, breadth/panic, own volatility, funding).
- **F19b. Wider input window: `N_POINTS` 30 → 45** (×1.5, user's call "even half again as
  much"). It lengthens the history of EVERY timeframe: c1m 30→45 min, c5m →225 min, c15m
  →11.25 h, c1h →45 h, c4h →7.5 d.
- **F20 (confirmed). The horizon is NO LONGER an input feature** — it's the output axis;
  `horizon_minutes`/`horizon_log` are REMOVED from features (the tail = only
  hour_sin/cos+weekday). This also kills off-anchor miscalibration (§9).
- **F21. We do NOT add identity features** (symbol name etc., §6) — the schema stays
  symbol-blind; "maximal" = depth of curves + BTC + regime, not identity.

### G. Data
- **G22. Binance only** (`data/binance/candles`, a year × 1-min × ~200 syms).
- **G23. The last 4 days are NOT TOUCHED** (the user's test). Our train-cutoff =
  `data-edge − 4 days`. Our internal val/holdout is a tail slice WITHIN the available range
  (e.g. the second-to-last ~48h before the −4d edge), train = everything before it. The
  4-day zone = the user's final unseen test; we don't peek into it (consistent with §12's
  holdout rule and §4).
- **G24 (elegant default). Unrealized long horizons at the edge** (where t+h falls past the
  available boundary) → mask the target node (NaN / weight 0 on that (row, node));
  MultiRMSE with per-target weights handles it.

### H. How we judge (documented, NOT executed yet)
- Curve head: per-horizon correlation/MAE of the predicted vs realized curve + directional
  hit-rate past cost.
- Derived trading signal + ranker: **top-k precision** (= the real objective) and the
  correctness of abstention (how often the null is correctly on top in bad windows).
- Since ranking breaks the gradators (§0): either recalibrate score→probability (isotonic)
  to keep the familiar gradator, or judge top-k precision directly.
- Comparison on the same holdout against `hc_final` d7/d8 and the portfolio.
- **A→B disjoint-confirmation** of any "good zones" (§11) — mandatory.

### I. Scope / housekeeping
- **I27. New namespace, the current hc pipeline untouched:** code in `src/tt/`, datasets
  `data/tt_*`, models `models/tt_*`.
- **I28.** This file is the main design doc; `CLAUDE.md` §13 is the short pointer.
- **I29. Research-only for now.** The live engine (by analogy with `hc_v4_live_engine`)
  comes later, after the curve+ranker are confirmed OOS.

---

## 2. Architecture (3 layers, pipeline)

```
              features (561: max-curves + BTC + v5 regime, N_POINTS=45)   one row = (symbol, scan)
                          │
        ┌─────────────────┼───────────────────────────┐
        ▼                 ▼                             ▼
 [Layer 1: CURVE]   [Layer 2: CONFIDENCE]        (outputs 1+2 as
 MultiRMSE          quantiles p10/p50/p90         extra features)
 240-vector         on a coarser grid                    │
 cum.log-return     → fan width = conviction             ▼
 (vol-norm)                                       [Layer 3: RANKER + ABSTENTION]
        │                 │                        QuerySoftMax
        └──────┬──────────┘                        group=scan (+ supergroup of 60 scans)
               ▼                                    no-trade null (rel=0)
   decision: edge = denorm(curve[h]) − cost;        graded rel = realized net
   side = sign; conviction = fan[h]                 → "what to bet / what to skip"
```

- **Layer 1 (the heart of TT).** `MultiRMSE`, one row per (symbol, scan), target = a
  240-vector of vol-norm cumulative-log-return. Replaces the old "row per (snapshot,
  horizon)" scheme → fewer rows, a consistent curve per pass.
- **Layer 2.** Quantile CatBoost models (p10/p50/p90) on a coarser horizon grid (confidence
  changes more slowly than the mean) → the fan. Gives "which points on the graph are most
  likely."
- **Layer 3.** Ranking (`QuerySoftMax`), consumes features + the outputs of layers 1–2
  (stacking), two groups (scan + 60-scan), a no-trade null. Gives "which position is most
  likely and which one is better not to look at."

---

## 3. Implementation plan (phases; we write code, tests — NOT now)

- **Phase 0 — scaffolding. ✅ BUILT (2026-06-15):**
  - `src/tt/schema_tt.py` — the maximal schema (561 features, `N_POINTS=45`) + the target
    spec (vol-norm cumulative curve, 1-min 1..240).
  - `src/tt/data_tt.py` — `build_symbol_curve_tt`: one row per (symbol, scan), maximal
    features + a 240-node vol-norm cumulative target on the 1-min close series (entry =
    base+5min), HOLDOUT-guard (no node reads past the cutoff). Groups for the ranker are
    derived from `base_time`+`symbol` at the Phase 3 stage (not stored here).
  - `src/run_tt_dataset.py` — CLI: attaches the binance store, computes cutoff = edge −
    `--holdout-days` (4), a market-frame pre-pass, workers, summary. Flags
    `--no-regime`/`--limit-symbols`/`--days` for a fast smoke.
- **Phase 1 — the curve. ✅ BUILT (2026-06-15):**
  - `src/tt/train_tt.py` + `src/run_tt_train.py` — `MultiRMSE` multi-output, per-node
    standardization (B8; sd≈√h, verified), 3-seed, GPU by default (MultiRMSE on GPU
    confirmed; 240-dim leaves are tiny). Produces `models/tt_curve` + `standardizer.json`
    (mu/sd per node) + `feature/target_names.json`.
  - **Continued training**: `--continue-from <model_dir>` adds another `--iterations` trees
    on top (init_model), reusing the standardizer. **CatBoost does NOT support
    GPU-continuation → continued training auto-switches to CPU.** Alternative — a full
    retrain on GPU with a higher cap.
  - TODO: a curve-read utility with interpolation (continuous-h query) — for Phase 2/3.
- **Phase 1.5 (conditional).** If the curves are jagged — a custom multi-target objective
  with a curvature term (B9).
- **Phase 2 — confidence.** Quantile models (p10/p50/p90) → `models/tt_quant`.
- **Phase 3 — ranker + abstention.** `QuerySoftMax`, two groups, a no-trade null, stacking
  of layers 1–2 → `models/tt_ranker_scan`, `models/tt_ranker_w60`.
- **Phase 4 (later). Live** — a separate engine `src/trading/tt_live_engine.py` + runner,
  self-prefetch (like v4/portfolio). Only after OOS confirmation.

---

## 4. Risks / hypotheses TT tests
- **Are SHORT horizons alive on Binance?** On OKX 2–15 min were cost-dead (the 0.75% wall,
  §2; confirmed by v4/band). Binance RT ~0.126% — a wall ~6× lower → 2–5 min might come
  alive. TT with a 1-min grid is the cleanest way to test this honestly.
- **The random-walk trap** (B5) — the main curve risk; saved by the confidence layer +
  abstention. If the fan is wide everywhere → the model honestly says "no edge".
- **240 outputs** — training is harder; there's a knob to coarsen the grid.
- **In-sample zone-picking** (§11) — any "good horizon zones" are A→B confirmed, otherwise
  the winrate is optimistic (the Zhnyvar/Snaiper rake).
- **Ranking breaks the gradators** (§0) — judge top-k precision or recalibrate.

---

## 5. Phase 1 — FIRST RESULT (`models/tt_curve`, 2026-06-15)
Build: 1.19M rows, 178 syms, window 2025-06-23..2026-06-11 (~353 days), 561 features, 240
nodes. Training: depth 7, cap 6000, 3-seed, GPU. cutoff 2026-06-11 11:25 UTC (the 11–15 Jun
holdout was NOT touched).
- **Early-stop stops early** (best_iter 42/123/157) — NOT the §5 trap (a clear val minimum,
  overfit beyond it), but the expected random-walk weakness of the MEDIAN curve (B5). Don't
  force 6000 — it overfits.
- **The signal is REAL and grows with the horizon** (val tail ~6 weeks, OOS-within-dataset,
  NOT the holdout): per-node corr(pred,true) 0.12 (h5) → 0.41 (h240); dir-hit 50%→66%.
- **Conviction (|pred|) — a working gate.** Top-2% of magnitude: dir-hit 72% (h30) →
  82% (h60) → 86% (h240).
- **Live zone = h≥30, core h60–240.** Short 5–15 min are DEAD even on Binance: dir-hit 62%
  at top-2%, but the implied move 0.03–0.09% < cost 0.126% → the headline hypothesis "short
  comes alive" is rather NEGATIVE (the move is too small, not just the direction).
- **CAVEATS:** ONE ~6-week val window, NOT cross-regime A→B (§11) → optimistic; dir-hit+move
  = a proxy, the real net needs an engine (Phase 3); top-2% = low volume.
- **Next:** Phase 2 (the quantile fan — though magnitude already works as conviction) +
  Phase 3 (a ranker h≥30 with a no-trade null) + A→B confirmation of the zones.

## 5b. ⚠️ OOS CORRECTION (2026-06-15) — the earlier conviction numbers = IN-SAMPLE
The holdout-split in the explorer exposed it: the "val tail" (Apr29–Jun11), on which §5
showed 72–86%, was TRAINED ON by the final model (final fit on ALL rows) → it is NOT unseen.
On the STERILE holdout (11–15 Jun, training never saw it) the edge DROPS:
- conv≥0.9: in-sample **72%/+1.06%** vs **HOLDOUT 53%/+0.53%**; conv≥0.8: 69% vs 46%.
- By horizon OOS (conv≥0.9): h30 **40%** / h60 48% / h120 54% / **h180 64%/+1.11%** /
  **h240 63%/+1.12%**. So the conviction gate does NOT generalize to short/mid; the real
  OOS-edge remains ONLY at **h≥180**.
- **New curve-native slices of the holdout (tab filters, h≥180 conv≥0.9, base n=1523 →
  63.4%/+1.12%):** ASSET — equities **80%/+1.34% (n=145)** vs crypto 61.7%/+1.09% → equities
  = the cleanest OOS slice (confirms the transfer-edge on equities, §6 memory). SIDE — longs
  carry it (63.5%, n=1456), shorts are dead (61% but net~0, n=67). SNR/|move|/persist on
  this NARROW zone are MARGINAL (snr≥2: 63.6 vs 63.4; persist degenerate over 2 horizons;
  |move|≥0.3 slightly worse) — they'll matter on a WIDER zone. Lead: **h≥180 × long ×
  (especially) equities**.
- **Conclusion:** as trained, the TT curve is OVERFIT. The priority shifts: (1) judge ONLY
  on the sterile holdout, not on the in-sample val; (2) fight overfit (regularization /
  fewer features / drop h<180); (3) the "short alive on Binance" hypothesis — confirmed
  DEAD. A→B (§11) is all the more mandatory.

## 6. Explorer (browser) — a dedicated "🌀 TT" tab (2026-06-15)
TT is a curve, not P(up), so it does NOT fit the prob explorer (it gave bugs / nonsensical
maxconc) → a SEPARATE, self-contained tab.
- `src/run_tt_export_html.py` scores the curve on a window **up to now** from
  `data/tt_now/dataset` (the training `tt_curve` stays sterile) → writes `window.TT` into
  `data.js` (other binance sims untouched). Each leg (10 fields, 2026-06-15)
  `[sym_idx,tmin,h,side,conv,net,holdout,move,snr,persist]` + an `eq` array (1=equity) per
  symbol: **conv** = the RELATIVE spread (percentile of |pred| by horizon 0..1); **move** =
  expected signed % move (EV vs cost); **snr** = the POINTWISE spread = |pred|/spread-between-
  seeds (clarity / seed agreement — needs per-seed predict, so the export stacks seeds);
  **persist** = the fraction of horizons ≥h where the curve holds direction; **holdout=1**
  if base_time ≥ train-cutoff (11 Jun). net = side×realized% − binance cost. ⚠️ tmin via
  `to_ns` (base_time in parquet is microseconds; `.astype(int64)` gave 1970 — fixed).
- The tab (index.html `page-tt` + self-contained JS, no prob engine): a conviction slider,
  horizon selection, period, dedup, + **curve-native filters (2026-06-15): SNR (pointwise
  spread), |move| (EV vs cost ≈0.13%), persist, side, asset (crypto/equities)** — the first
  three are IMPOSSIBLE in a scalar-prob model (they come from the curve shape). Deliberately
  NOT ported: p_opp, the engine maxconc/cooldown (junk), and hour-of-day (DST + the
  in-sample zone-pick risk §11). THE KEY thing — **two cards IN-SAMPLE vs unseen HOLDOUT** +
  a per-horizon table. It's exactly what exposed §5b.
- **Per-day chart (2026-06-15, `ttDayChart`/`initTtDayTip`):** a canvas with a METRIC toggle
  (win-rate / avg net / total net / avg conv / count), grouped by Kyiv days (UTC+3), full
  scale over the current filters (period ignored — the holdout is marked with a yellow zone
  + a divider, more transparent bars = IS), green/red by sign (net) or by 50% (win), a
  tooltip with all the day's metrics. Reuses `cv()`+`#hourTip`. Verified via canvas pixels
  (a screenshot times out — the page is 39 MB).
- Panel: `POST /api/ttexport` chains a `tt_now` build (reuse the market frame) → export.
  Button "🌀 TT" (the tab) + "🌀 TT curve → explorer" (the header). server.py.
- Verified in the browser (preview hc-panel, 2026-06-15): the 10-field export was
  regenerated on tt_curve (537k legs / 178 syms / 114.6k holdout legs), 0 console errors,
  the new filters change the selection correctly, the cards/table match a direct computation
  (HO h180+240 conv≥0.9 = 63%/±2.4/n1523/+1.115%). The screenshot times out (39 MB page) —
  verification is textual (eval/snapshot), not visual.
