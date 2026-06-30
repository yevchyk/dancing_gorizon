# BINANCE_V5_PLAN.md — market-regime features + calibration (next training)

> ⚠️ **SUPERSEDED (2026-06-15): the sealed-exam discipline is RETIRED.** Ignore
> "frozen cutoffs / selection on PROBE / FINAL 5d sealed / separate exam" below.
> Replacement = **CLAUDE.md §12**: ONE live data pool + HOLDOUT RULE (test period
> is a plain runtime cutoff = data-edge − N days, never a sealed fixture). The v5
> ideas (regime features, isotonic calibration on a holdout slice) still stand;
> just fit/judge on a normal holdout, not a sealed window.

Status: DRAFT 2026-06-11 late evening — co-designed with the user; UPDATE
tomorrow with the it20k overnight verdict (budget + whether d12 stays in play).
Discipline: BINANCE_PLAN.md pre-registration still rules — same frozen cutoffs,
all selection on PROBE, FINAL 5d stays sealed, judge = NET calibration ladder.

## 0. Hypotheses (what v5 is for)
- H1 (user): the model is blind to MARKET STATE (BTC block was dropped in v3);
  re-injecting a compact regime block helps timing — especially LONGS, which
  need "the market stopped falling / bounces" context the per-symbol curves
  can't see.
- H2 (user, supported by the 2026-06-11 iteration-curve probe): 12000 iters
  undertrain; budget goes to ~20000 (exact number = tomorrow's it20k verdict).
- H3: a post-hoc calibration layer can straighten the deep models' probability
  ladders cheaper than retraining (and is legal if fitted on unseen-by-model data).

## 1. New feature block (v5 schema = v3/v4 + ~17 cols, ALL symbol-blind, %-space)
All computed from the Binance 1m store only (no external APIs), at base_time,
causal (data <= base only), NaN-free after a 24h warm-up. Live parity is
mandatory: same functions for dataset and live engine, gate extended in
`run_binance_parity_check`.

A. BTC reference (the "вприски" — compact scalars, not the old 60-col curve):
   - btc_ret_15m / 1h / 4h / 24h: log-return of BTC close over the lookback (4)
   - btc_vol_1h / 24h: std of 5m log-returns over the window, annualis-free raw (2)
   - btc_range_pos_24h: (close − min24h) / (max24h − min24h), 0..1 (1)

B. Symbol vs market (relative strength):
   - rs_1h / rs_24h: symbol log-ret minus btc log-ret over the window (2)

C. Breadth / own panic index (computed over the FROZEN trade universe file —
   NOT the live watchlist — or parity breaks when bans change):
   - breadth_above_4h: share of universe with close > own 4h SMA (1)
   - breadth_red_1h: share with negative 1h return (1)
   - panic_cascade: share with 1h return < −2% (1)
   - univ_vol_1h: cross-sectional MEDIAN of per-symbol 1h realized vol (1)
   (fear&greed external index: rejected for v1 — external dependency, daily
   granularity, kills parity/self-sufficiency. Our breadth/panic IS the index.)

D. Symbol's own volatility context (user's "сумарна вола"):
   - sym_vol_1h / sym_vol_24h: std of 5m log-returns (2)
   - sym_vol_ratio: sym_vol_1h / max(sym_vol_24h, eps) — vol expanding? (1)
   - sym_range_pos_24h: position in own 24h range, 0..1 (1)

Implementation notes:
- dataset: one market pre-pass builds a time-indexed MARKET frame (A+C) once,
  symbols join it by base_time; B/D computed per symbol. Builder: `hc/data_v5.py`
  (+ `schema_v5.py`), new `src/run_binance_dataset_v5.py` re-using the v4 grid
  (stride 60m + jitter, per-symbol cost+funding thresholds — unchanged) BUT with
  the user's horizon decision: **30..320 by 5** (59 values; "далі воно бачить
  погано"), anchors (30, 120, 320) + 3 random. Funding term in the threshold
  stays med|rate|×h/480 — it is time-proportional, not tied to the max horizon.
- +1 col per user: `funding_level` = current 8h funding rate of the symbol
  normalized by its own yearly median |rate| (crowding signal).
- live: market frame computed in the engine from the same frozen universe file;
  bit-parity vs dataset is the launch gate (extend parity check to v5 cols).

## 2. Training arms (GPU budget ≈ a day; final numbers after it20k verdict)
1. d8_v5 @ <budget> × 3 seeds — the main arm (favourite depth + regime block).
2. d8_v4 @ <budget> × 3 seeds — ABLATION CONTROL on the same budget, so "regime
   block helps" is separable from "more iterations help". (If tonight's
   binance_y1_d8_it20k turns out clean, it IS this control — no extra run.)
3. (optional, after 1-2 land) d12_v5 @ <budget> × 1 seed — only if tonight's
   d12_it20k probe shows the deep model coming alive with budget.
Trainer unchanged (`run_hc_prod_train --random-val`, cutoff = frozen probe_from,
od_wait 300 stays — let early stop bite if 20000 saturates).

## 3. Calibration layer (H3 — answers "на чому калібрувати")
- What: per-head isotonic regression p_raw -> p_cal (fallback: Platt if probe
  shows isotonic overfits thin tails); fitted SEPARATELY per depth family.
- Where fitted (two legal options, pick by sample size):
  a) PROBE window [T−10d, T−5d): it is the selection window by pre-registration —
     fitting a calibrator there is selection, the exam stays clean. Must pass
     its own A→B check (fit on probe-half-A, ladder must hold on half-B).
  b) Inner calibration slice: extra cutoff T−25d → model sees nothing of
     [T−25d, T−10d), calibrator fits there, probe stays a pure judge. Costlier
     (one more refit) but cleaner. Use (b) only if (a) fails its A→B.
- NEVER: fit on train period itself (model memorized it → too-mild correction)
  or on the FINAL window (burns the exam).
- Success metric: monotone p_cal ladder on the untouched half with avg-NET
  rising by bucket; deep models (d10/d12) are the patients most likely to gain.

## 4. Judge & decision (unchanged discipline)
- `run_binance_export` probe sims for every arm (+ explorer live-book mode) and
  the formal judge: NET ladder, A→B probe halves, long/short split, per-day,
  per-symbol concentration. Picks written into BINANCE_PLAN.md before any exam.
- Exam stays the single shot on [T−5d, T]; v5 does not get a separate exam
  window — it competes on probe, and only the chosen ONE config sits the exam.

## 5. DECIDED with the user (2026-06-11 night)
1. Horizons: **30–320 / 5** ("далеко воно бачить погано") — grid + anchors
   updated in §1; labels formula unchanged.
2. Breadth universe = frozen 153 trade file. ✓
3. Ablation arm: KEPT (explained: when two things change at once — budget AND
   features — a win can't be attributed; the twin run with old features at the
   same budget isolates the feature effect). Tonight's `binance_y1_d8_it20k`
   doubles as this control for free if it lands clean.
4. Calibration: explained to the user — it is NOT retraining and NOT depth
   choice; it's a small correction table ON TOP of a finished model ("model
   says 0.90, history says that 0.90 wins 44% → translate 0.90→0.44").
   User believes in d12 → calibrate BOTH d10 and d12 (it's cheap); a deep
   model whose lies are SYSTEMATIC is exactly what a calibrator can revive.
5. `funding_level` feature: IN. ✓

## 6. Risks / do-not (v5-specific)
- Parity is THE trap: breadth/market cols must use the frozen universe list and
  identical warm-up windows in dataset and live; extend the parity gate FIRST,
  build second.
- Regime features can become a regime MEMORIZER on one crash year — watch the
  A→B halves and the long/short split on probe before celebrating.
- Do not silently change labels/grid/universe in the same arm — one variable
  at a time (regime block is the variable; budget is controlled by the ablation).
- CLAUDE.md + BINANCE_PLAN.md rules apply in full.

## 7. FILLED 2026-06-12 (it20k verdicts are in)
- [x] d8 @20000 vs 12000: 20k WORSE (2-seed ens p≥0.85: 49.2%/−0.53 vs 57%/+0.07)
      → **budget = 12000** (the cap acts as regularization; val_auc keeps lying).
- [x] d12 @20000 ×3 seeds: −1.02% at p≥0.85 (vs −1.25 @12k) — deep model does
      NOT come alive → **arm 3 (d12_v5) cancelled**. d12 only gets the cheap
      §3 calibration attempt.
- [x] 28000 refit idea: dead with the budget verdict.
- [x] Ablation control (§2 arm 2): NOT needed as a run — budget stayed 12000,
      so the existing `binance_y1_d8` IS the same-budget old-features control.
- BUILD DONE 2026-06-12 ~21:00: schema_v5 (323 cols) + hc/data_v5 (shared
  feature fns = parity by construction) → run_binance_dataset_v5 (7.32M rows,
  market frame pre-pass + funding series) → smoke+check PASS → d8_v5 @12000
  ×3 seeds, val_auc 0.826. Exported `binance d8 v5 (probe)`.

## 8. V5 VERDICT (2026-06-12 ~22:00) — NEGATIVE. Regime block did NOT help.
Pocket judge v5 vs controls (probe, A/B halves):
- FLAGSHIP long 95-240 p≥0.80: d12 176 legs 62.5%/+1.11 (A+0.79/B+1.48) vs
  v5 819 legs 60.3%/+0.55 (A+0.71/**B+0.16**). v5 fires 4.6× MORE signals at
  LOWER avg and a DECAYING second half.
- NIGHT long 65-120: d12 62/79%/+1.55 stable vs v5 203/77%/+0.99, B decays.
- NIGHT short 125-240: d12 215/60%/+0.99 (A+0.52/B+2.21) vs v5 434/54.6%/
  **−0.44 with A−2.08 / B+1.98 SIGN FLIP** — v5 BROKE the shorts.
- Global p≥0.85 tail: v5 −0.599 (n=939) vs control d8 +0.07 (n=114).
Diagnosis: the regime block became a REGIME MEMORIZER on the one crash year
(exactly the §6 risk). It inflates CONFIDENCE (more legs clear thresholds) not
CORRECTNESS; the universal A>B decay is the tell. **The capacity axis (depth,
iters) AND the regime-feature axis are both spent on this data.** The real edge
stays the cross-model POCKETS on the OLD models (d8/d10/d12), unchanged.
Next levers (not capacity, not these features): (a) §3 calibration layer is the
only cheap thing left to try; (b) the edge we HAVE (pockets) → exam → live; the
honest move may be to STOP squeezing model variants and ship the pocket
portfolio. Decision pending with user.

## 8. VERDICT 2026-06-12 23:20 — regime block did NOT beat control on probe
- d8_v5 (323 feats, 3-seed ens, `models/binance_y1_v5_d8`) probe-tail ladder:
  p≥0.70 n=15620 44.3%/−0.41 · p≥0.80 n=3206 43.8%/−0.73 ·
  **p≥0.85 n=939 47.6%/−0.60** · p≥0.90 n=124 49.2%/+0.18.
- Control d8-12k (305 feats): p≥0.85 n=114 **57%/+0.07**. v5 fires ~8× more at
  the same nominal confidence and LOSES — the regime block fed overconfidence,
  same failure mode as more depth / more iterations. The pre-registered
  question ("does it beat 57%/+0.07 at the tail?") → **NO**.
- p≥0.90 barely green on n=124 with 49% win = tail-driven, not a signal.
- Per pre-commitment: regime features rejected for the exam candidate;
  d8-12k (305) stays THE candidate. Remaining cheap ideas: isotonic calibration
  on probe (d10/d12/v5 alike), short-side timing analysis. Exam still untouched.
- Explorer: exported as "binance d8 v5 (probe)" (data.js) and added to the
  secret exporter MODELS (own dataset path data/binance_y1_v5/dataset).
