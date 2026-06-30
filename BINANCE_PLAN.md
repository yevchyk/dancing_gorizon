# BINANCE_PLAN.md — the year-rebuild (next agent: read this FIRST, then CLAUDE.md)

> ⚠️ **SUPERSEDED (2026-06-15, user): the SEALED-EXAM pre-registration is RETIRED.**
> The "тихий тест"/sealed exam, the probe/final/now split, `binance_cutoffs.json`
> + the freeze script, `data_secret.js`/`data_now.js` + the secret export scripts,
> and the "AGENT STAYS BLIND" discipline below are ALL GONE — those files were
> deleted and that machinery removed from the explorer/server. **Ignore every
> exam / freeze / cutoffs / secret / agent-blind / probe-vs-final instruction in
> this document.** What replaces it (authoritative): **CLAUDE.md §12** — ONE live
> data pool (`data.js`, last ~12d to now), and the **HOLDOUT RULE** = the test
> period is a plain runtime holdout (train cutoff = data-edge − N days), never a
> code fixture. The research findings further down (cost facts, build steps,
> pocket picks, iteration-curve verdicts) still hold — read them as history, not
> as a sealed protocol.

Status date: 2026-06-10. Owner decision: full migration to Binance USDT-M —
data now, execution later. This file is the pre-registration: decisions here are
LOCKED with the user; execute, don't relitigate.

## 0. TLDR for a cold agent
Rebuild the HC model family on 365d of Binance 1-min data with HONEST per-symbol
costs (measured ~0.12–0.21% round-trip) instead of OKX's flat-0.75% lie.
Horizons 30–480 min. Final exam = LAST 5 DAYS, untouched by everything, reported
ONCE. Judge = NET calibration (avg net per p-bucket, net of per-symbol cost) —
not win% alone, not engine PnL. All prep tooling exists and is verified.

## 1. Why (one paragraph of history)
OKX live lost real money on thin names (H, MRVL): real round-trip cost there is
0.75–2.5% while every sim/label assumed flat 0.75% → models считали wins what
physically can't clear cost, and the "edge" of the best model (min1 flat d12) on
the full universe was a junk-name mirage (liquid-only it was NEGATIVE). Binance
liquid perps measured 4–6× cheaper (median 0.126% RT). At threshold≈cost the game
changes from "find rare big moves" to "direction with a small buffer": base win
rate ≈ 50%, the money lives in the high-p_dir tail and in AVG NET, not win%.

## 2. What exists on disk (all verified working 2026-06-10)
- `data/binance/candles/*.parquet` — 175 liquid syms × 365d × 1m (~525k rows
  each, gap 0.00%), plus 25 extra being fetched. Schema identical to the OKX
  store: timestamp = bar OPEN time UTC (VERIFIED vs OKX: lag-0 corr 0.9966,
  ±1 lags ~0 → no off-by-one), open/high/low/close/volume, RangeIndex.
- `src/binance_fetcher.py` — parallel throttled fetcher (~200 calls/min under
  the 2400 weight/min limit), RESUME-able: re-running tops existing files up to
  now and skips complete ones. 175 syms × 365d took ~3.5h.
- `configs/binance_universe_liquid.json` — 175 trading universe (≥$10M/day).
- `configs/binance_universe_train_extra.json` — +25 survivorship-fix names
  (old listings ≥13mo, NOW $1–10M/day = the year's faded losers: COMP, GALA,
  SAND, AXS, APE, LDO, STRK, PYTH, KAS...). TRAIN-ONLY, never trade them.
- `src/binance_costs.py` → `configs/binance_costs.json` — honest per-symbol RT
  cost = Binance taker fee 0.10% RT + Corwin-Schultz spread (measured from 1m
  H/L, 30d window, median) + 0.01% impact floor. Result: p25=0.120, median=0.126,
  p95=0.173, max(good)=0.205. 153 trusted / 21 flagged `rows<50000` (listings
  younger ~35d, incl tokenized stocks ARM/SAMSUNG/MRVL/ORCL/SOXL) → flagged are
  EXCLUDED from train v1.
- `src/binance_okx_align.py` — timestamp-convention check. Re-run after any big
  refetch; must say "ALIGNED (lag 0 wins)".
- `src/binance_pick_extra.py` — regenerates the extra-25 selection if needed.
- `src/binance_funding.py` — year of 8h funding rates per symbol →
  `configs/binance_funding.json` (med|rate| feeds the label threshold, §3.8).
- `src/binance_freeze_cutoffs.py` — freezes train universe + cutoffs (§4.5).
- `src/run_binance_dataset.py` — the year dataset builder (§5).

## 3. LOCKED decisions (agreed with user, do not change silently)
1. **Labels vs honest cost**: win = directional move from entry (t+5m close on
   the 1-min series, same EXEC_ENTRY_DELAY_MIN=5 convention) to exit (t+5m+h)
   beats THAT symbol's full cost: `rt_cost_pct` (configs/binance_costs.json)
   **+ med|funding| × h/480** (configs/binance_funding.json). No flat threshold
   anywhere. (User 2026-06-10: funding is ALWAYS in — "щоб нейронка ігнорувала
   ризик безтолковий, тільки супер впевненість високого росту".)
2. **Horizons 30..480 min**, 5-min aligned, ~6–8 sampled per snapshot (v4-style
   row bounding). Short 2–15m intentionally absent in v1 (they died on OKX cost;
   at Binance cost they may revive — that's experiment #2 AFTER v1 ships).
3. **Train window = full year minus last 10d.**
4. **PROBE = [T−10d, T−5d)** — ALL hyperparam/zone/floor selection happens here,
   and zone-picks must be A→B confirmed across the probe's two halves.
5. **FINAL = [T−5d, T] — UNTOUCHED.** No training, no tuning, no zone-picking,
   no peeking. Evaluated ONCE at the very end, result reported as-is. (User
   explicitly chose this over rolling walk-forward: any surface I iterate on
   becomes an optimization target.)
   CONTROLLED RELAXATION (user request 2026-06-12): the USER may peek privately
   via the explorer's "🤫 тихий тест" source (`data_secret.js`, generated by
   `src.run_binance_export_secret` / POST /api/secretgen). THE AGENT STAYS
   BLIND: Claude never reads that file, never adds statistics to that script's
   stdout, never asks about it — and the user NEVER reveals anything from it
   (numbers, "good/bad", model hints). If revealed, the exam is burned and must
   be treated as contaminated in all reports.
6. **Judge = NET calibration**: per p-bucket realized win% AND avg net (net of
   per-symbol cost), monotonicity of the ladder, stability across the two halves
   of the window, long/short balance, per-day consistency. avg net outranks win%.
7. **Model**: v4-style schema (c1m 1-min curve + 5m/15m/1h/4h curves + hour/
   weekday time feats, NO BTC ref, NO symbol identity — symbol-blind transfers).
   Depth sweep {8, 10, 12} × 3 seeds, `--random-val`, ~9000 iters (no time-tail
   early-stop trivialization — CLAUDE.md §5).
8. **Funding** (NOT in candles — separate 8h cash flow at 00/08/16 UTC):
   DECIDED 2026-06-10 — always folded into the label threshold as expected
   adverse funding `med|rate| × h/480`, fetched per symbol by
   `src.binance_funding` → `configs/binance_funding.json`. No materiality
   debate needed; the threshold is the full honest cost of the hold.
9. **Survivorship honesty**: the universe = liquid TODAY, so in-train-year
   aggregate stats are inflated by selection. The extra-25 losers partially
   de-bias; true corpses (delisted) aren't fetchable — accepted limitation.
   Only the FINAL-window numbers are honest claims.

## 4. Pre-flight (in order, before any dataset build) — ALL TOOLING EXISTS
1. Top-up candles to NOW (both resume): `python -m src.binance_fetcher --days 365`
   and `python -m src.binance_fetcher --universe configs/binance_universe_train_extra.json --days 365`.
2. `python -m src.binance_okx_align` → must print ALIGNED.
3. `python -m src.binance_costs --window-days 30` → refresh the cost map so the
   extra-25 get measured costs too (labels need them).
4. `python -m src.binance_funding` → configs/binance_funding.json (year of 8h
   funding per symbol; med|rate| feeds the label threshold per §3.8).
5. `python -m src.binance_freeze_cutoffs` → FREEZES configs/binance_train_universe.json
   (liquid minus flagged plus extra25, trusted-cost only) + configs/binance_cutoffs.json
   ({t_end, final_from=T−5d, probe_from=T−10d}). Refuses to overwrite without --force.
   Builder, trainer and judge READ these files, never recompute "now".

## 5. Dataset build — `python -m src.run_binance_dataset` (exists)
- v4 pipeline re-pointed at the Binance store via markets.REGISTRY["binance_feature"]
  + HC.STORE_KEY patch; HC.HC_ERA_START patched to 2025-06-01 (the OKX value
  2025-09-27 would silently cut 3.5 months of the year).
- Train universe: frozen file (≈179 syms). Labels per §3.1 via threshold_fn
  (per-symbol cost + funding×h/480) — new optional params threaded through
  `hc/data_v4.build_symbol_frame_v4` (threshold_fn, grid_offset_min) and
  `hc/data_v2._anchor_grid_v2` (offset_min); defaults keep OKX behavior intact.
- Grid: stride 60m with per-symbol 0..55m jitter (stable_seed(sym,7)); horizons
  30..480 step 5 (91 values), every snapshot = anchors (30,120,480) + 3 random
  → 6/snap, ≈ 8760×179×6 ≈ **9.4M rows** (≈11.5GB f32, fits the 32GB box).
- Output `data/binance_y1/dataset` as PER-SYMBOL shards, resume-able (kill and
  re-run freely; `--fresh` to rebuild). Writes feature_names.json +
  dataset_summary.json. Smoke first: `--limit-symbols 2 --days 3 --out-dir
  data/binance_smoke/dataset`.
- The dataset contains the FULL year incl. probe+final windows — the cutoff
  discipline lives in the TRAINER (`--cutoff-local` = probe_from for sweeps,
  final_from for the exam fit) and the judge. Never train past probe_from
  during selection.

## 6. Train → select → exam
1. `run_hc_prod_train` on the dataset, cutoff from the cutoffs file,
   `--random-val`, depths {8,10,12} × 3 seeds. GPU available (RTX 4070 12GB) —
   CatBoost `task_type=GPU` makes depth-12 feasible; verify identical-ish quality
   on one small CPU/GPU A/B before trusting it.
2. On PROBE: pick depth, horizon zones (A→B across probe halves), p_dir floor.
   Write the picks INTO THIS FILE under "## Probe picks" before touching FINAL.
3. Run FINAL once. Produce: net calibration ladder (all buckets), two-half
   split, long/short split, per-day table, n per bucket. Report verbatim — no
   retuning afterwards, whatever it shows.
4. Then: forward shadow on Binance prices (the only real test), and the Binance
   executor as a separate workstream (port LiveTrader's order path; OKX live
   stays running small meanwhile).

## 7. Compute (measured on this box, 2026-06-10)
RAM 32GB · GPU RTX 4070 12GB · D: 73GB free · binance candles ≈1.1GB on disk.
Dataset ≈9–10M rows × ~305 f32 ≈ 12GB → fits, but don't go bigger; CatBoost
pool build peaks ~2× data (GPU quantization helps). 3 seeds × 3 depths = budget
a day end-to-end; feature build is the slow part, training on GPU is hours.

## 8. Move-to-D — CANCELLED (2026-06-10)
User decided to stay on C: if space suffices — it does: C: has ~57GB free, the
year dataset is ~5-7GB parquet (+~1.5GB candles already on disk). If a future
move happens anyway: stop engines, move the whole folder (`.env` travels — never
print/commit it), then verify `.venv\Scripts\python.exe -V` (recreate venv if
broken); all repo paths are relative.

## 9. Do NOT (hard rules, learned the expensive way)
- Do NOT touch FINAL before the single exam; do NOT pick zones on the window you
  report (in-sample zone-picking inflated d7 to fake 75–81% before).
- Do NOT judge by win% alone — at threshold≈cost, win≈"direction right"; the
  money metric is avg net per bucket, net of per-symbol cost.
- Do NOT train on the 21 flagged listings; do NOT trade the extra-25.
- Do NOT query horizons outside the trained grid (off-grid = overconfident junk,
  CLAUDE.md §9).
- Do NOT trust in-train-year aggregate stats (survivorship, §3.9).
- CLAUDE.md still applies in full (gradators-first, volume↔winrate wall,
  bad-day rule, no symbol identity features).

## Executor status (2026-06-11) — BUILT, shadow running, live GATED
- `src/trading/binance_executor.py` `BinanceExecutor` — USDT-M fapi order path
  (market enter / reduce-only close / positions / balance / leverage+margin /
  exchangeInfo lot+minNotional sizing / testnet base). Same Executor interface
  as OKX → LiveTrader unchanged.
- `src/run_binance_live.py` — portfolio runner on the BINANCE store (HC store
  patched like run_binance_dataset → live features == dataset, verified
  bit-exact by `src/run_binance_parity_check.py`: feat diff 0.0, prob 2.6e-08).
  Builds applied via BinancePortfolioEngine (no OKX blacklist; build bans only).
  Config: `configs/builds/binance_shadow_portfolio.json` (d10 long + d10 short,
  $5×3, maxconc 12, cooldown 30m, min_p_dir 0.55 = explorer floor),
  universe `configs/binance_universe_trade.json` (153 = liquid ∩ trusted-cost).
- Modes: `--shadow` (default, running detached via binance_shadow.ps1 →
  binance_shadow.log) | `--testnet` (needs BINANCE_TESTNET_API_KEY/SECRET) |
  `--live` (real $; needs BINANCE_API_KEY/SECRET; self-guards: creds check,
  duplicate-runner check). LIVE STAYS OFF until the §6 exam is reported.

## Depth sweep result (2026-06-11 evening — facts; depth pick = formal judge)
- All 3 families done (3 seeds × 2 heads each, cutoff = frozen probe_from,
  12000-iter cap hit by EVERY head → final refit budget 14000–16000).
- val_auc (random-val): d8 0.807 / d10 0.865 / d12 0.905 — rises with capacity.
- Probe quick-look (mixed legs, floor 0.55, p_dir≥0.85 bucket, NET of full cost):
  d8 n=114 win 57% avg +0.07% · d10 n=354 win 48% −0.75% · d12(2 seeds) n=999
  win 43% −1.24%. Deeper ⇒ 3× more "confident" calls each step, monotonically
  worse honest net — textbook overconfidence; val_auc must NOT pick the depth.
- Status: executor built + parity-gated (see §6.4 note), explorer probe sims
  exported for all 3 depths, forward-shadow of the user's two d10 builds running
  since 16:48 Kyiv. Next: formal probe judge (NET ladder + A→B halves +
  long/short + per-day) → depth+zone picks written HERE → single FINAL exam.

## Iteration-curve probe (2026-06-11 ~22:00) — user's "undertrained" hypothesis
- Truncated the EXISTING models with ntree_end on the probe tail (p>=0.85,
  mixed legs, floor 0.55): d8: 6k −2.09% → 9k −0.30% → 12k **+0.07%** (still
  rising at the cap). d12: 3k −4.53 → 6k −2.24 → 9k −2.04 → 12k −1.25 (rising
  too, but deep underwater; naive extrapolation needs ~18-24k trees to reach 0).
- Verdict: at fixed depth the honest tail IMPROVES with budget — the 12000 cap
  undertrains. Depth still hurts (same-budget comparison stands), but budget
  and capacity interact: deeper needs disproportionately more trees.
- QUEUED overnight (22:08): d8@20000 ×3 seeds → models/binance_y1_d8_it20k,
  d12@20000 ×1 seed → models/binance_y1_d12_it20k (separate dirs, 12k families
  untouched; cutoff = frozen probe_from, exam still sealed). Compare on probe.
- VERDICT (2026-06-12 ~02:00, user accepted): d8@20k probe tail p≥0.85 with a
  2-seed ensemble: n=327 49.2% −0.525% (1-seed: 513/51.9%/−0.149; p≥0.90 bucket
  −3.75%) vs d8-12k 3-seed n=114 57% +0.07%. val_auc rose 0.807→0.832. So a
  model TRAINED at 20k ≠ a 12k-trained model truncation-extrapolated: real 20k
  budget = the same overconfidence trap as depth. **12k cap LOCKED** (it acts
  as regularization); it20k discarded, seed43 killed mid-train. USER DECISIONS
  (2026-06-12 ~02:30): d8 budget LOCKED ≤15k ("може навіть 14000") for the final
  refit; d12@20k REVIVED ×3 seeds and trained FIRST (its truncation curve was
  still rising deep underwater — needs the real test) → launched 02:40 Kyiv,
  models/binance_y1_d12_it20k, log binance_d12_20k.log, ETA ~10:00. Judge on
  probe vs d12-12k 3v3 when done.

## Overnight notes (2026-06-11, mid-run — facts for the judge, no decisions)
- d8 seed41 HIT the iteration cap (best_iter 11999/12000, BOTH heads, random-val
  still improving) → when the chosen depth gets its FINAL refit at
  cutoff=final_from, raise the cap to 14000–16000 (keep od_wait 300). Do NOT
  restart the running sweep for this — the depth ranking needs equal budgets.
- Probe-window market context: BTC −18.5%, median symbol −17.6%, 86% of
  universe down, every day red. Judge must read sides against this beta:
  mid-confidence SHORT win% is crash-inflated; on seed41 the LONG tail
  (p≥0.85) was net-positive AGAINST the falling market while the SHORT tail
  was deeply negative (fires en masse AFTER cascades, eats the bounce) — but
  A/B halves are unstable on 1 seed → no side/zone picks until the judge.
- Explorer export: `python -m src.run_binance_export` (probe-only hard guard;
  auto-picks every COMPLETED seed — re-run as seeds land, refresh the page).

## 10. After v1 ships (parked experiments, in priority order)
0. Market-regime features + calibration layer — SPEC'D, see `BINANCE_V5_PLAN.md`
   (drafted 2026-06-11 with the user; budget numbers pend the it20k verdict —
   now in: 12k locked, so v5 budget = depth 8 × 12000 × 3 seeds).
0a. NEXT-TRAINING grid (user, 2026-06-12 night): extend horizons to ~540m —
   "найкращі ставки шортів десь там" (far end of the grid). Caveat for the
   judge: the probe was a crash window (BTC −18.5%), long-h short win% is
   beta-inflated — confirm on a non-crash window before trusting.
   UPDATE same night, pocket scan: long-h SHORTS bled in ALL three depths ON A
   CRASH WINDOW (d8 −2.25 / d10 −1.86 / d12 −1.21 at h245-480) — they fire
   AFTER the cascade and eat the bounce. User's revised thesis: model shorts
   via TIMING, not probe zone-picks — "їхня якість з самого низу" → candidate
   features: time-since-local-extreme / drawdown-position-in-move / hour gates
   (fold into BINANCE_V5_PLAN). NO short builds from this probe.
0a'. PROBE POCKET PICKS (2026-06-12, written per §6.2; exam still sealed):
   - LONG h95-240 @ p_dir≥0.80 — cross-confirmed by ALL THREE depths with both
     A/B halves green (d8 57/+0.98 · d10 82 legs 62.2%/+1.29 · d12 176 legs
     62.5%/+1.11), longs against a −18% tape. Pocket #1, the flagship.
   - LONG h245-480 @ p_dir≥0.85 (d10/d12) — A/B green but win ~39% (lottery
     profile, net from fat winners) → small weight, drop if shadow disagrees.
   - Shadow portfolio SWITCHED to these (3 builds: d12+d10 long 95-240,
     d12 marathon) — configs/builds/binance_shadow_portfolio.json; the old
     d10 long+short pair backed up as binance_shadow_portfolio_v1_d10pair.json
     (its 6h-window summaries were window-fit fantasy; short leg bled live).
0a''. NIGHT POCKETS (deep miner 2026-06-12 ~16:30, gates: n>=40, both halves
   green, >=2 models confirm; saved builds in folder "нічні probe 12.06"):
   - SHORT h125-180 p>=0.70 KYIV NIGHT 0-7 — THE FIRST LIVE SHORT, confirmed by
     THREE models (d10 73.5%/+1.01 · d12 67%/+1.17 · d12-20k 67%/+0.47);
     neighbour 185-240 confirmed by two → continuous night short zone 125-240.
   - LONG h65-120 p>=0.80 night — d12 79%/+1.55, d12-20k 76.7%/+1.28; the
     p>=0.70 variant confirmed by ALL FOUR models. Night = model reads the
     (Asian-session) tape better BOTH ways.
   - p_opp<=0.10 cleanup KILLS shorts everywhere (keeps the trend-shorts fired
     off cascade bottoms) — do NOT use it on shorts.
   - Day/evening shorts: dead in every cell. User's "shorts need timing" thesis
     confirmed literally — the first working short came from an HOUR filter.
   - Caveat: probe = crash week; night shorts may be its artifact → forward
     shadow first, NOT added to the running pockets-shadow without user OK.
0b. NEXT-TRAINING grid shape (user, 2026-06-12 night): replace the uniform
   30..480/5 grid (91 pts) with ~100 LOG-SPACED points 5→540 (e.g. 5,7,9,11,…,
   500,520,540): dense where P(win|h) bends fast (short h), sparse where it is
   flat (long h) — equal information per grid step (move ~ √t physics), and it
   re-admits short horizons at Binance costs (absorbs item 1 below). Snapshot
   still samples ~6 pts + log-spread anchors (e.g. {15,60,240,540}). Inference
   must still query GRID POINTS ONLY (off-grid lesson, CLAUDE.md §9).
1. Short horizons 10–25m at Binance costs (the "short = dead" verdict was an
   OKX-cost artifact; physics untested at 0.12%). — absorbed by 0b.
2. Maker entries: fee 0.02%/side + no spread crossing ⇒ ~0.05–0.08% RT, but
   adverse selection must be MEASURED (passive fills skew toward moves against
   you), never assumed.
3. Tokenized-stock perps (the flagged list) once they accumulate history.
4. Binance executor → live, starting at $5×3 scale like the OKX book.
