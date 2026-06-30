# Dancing Horizon — working instructions (read first, don't re-derive)

Hard-won rules. Follow these so we stop repeating mistakes.

## 12. BINANCE workstream — single live data pool (read this first)
- **HOLDOUT RULE — the "тихий тест"/sealed exam is RETIRED (2026-06-15, user).**
  The test period is a PLAIN runtime holdout chosen at TRAIN time, never a
  permanent code fixture. Training cutoff = data-edge − N days, computed at
  runtime (`run_binance_overnight.py`, `HOLDOUT_DAYS`); judge on that unseen tail
  (§4). There is NO sealed-exam pipeline, NO `binance_cutoffs.json`, NO freeze
  script, NO `data_secret.js`/`data_now.js`, NO secret/agent-blind files, NO
  probe/exam/now split. **Do NOT reintroduce any of it — keep the test window out
  of the explorer/server code.** (`BINANCE_PLAN.md` is the OLD pre-registration:
  superseded; ignore its exam/freeze/secret mechanics, the research notes still hold.)
- Data: `data/binance/candles` (365d × 1m × ~200 syms, schema == OKX store,
  timestamp alignment verified lag-0). Costs: `configs/binance_costs.json`
  (median RT 0.126% — the flat 0.75% figure below is OKX-only history).
- **ONE data source in the explorer** = `reports/sim_explorer/data.js` (no «Дані»
  dropdown anymore). It covers the last ~12 days up to NOW. Refresh with the panel
  button **«докачати + ребілд до зараз (Binance)»** (`POST /api/binancenow` →
  `_binance_now`): fetch candles to now → rebuild `data/binance_now/dataset` →
  `run_binance_export --all --fresh` (writes data.js over the full window).
  `run_binance_export` auto-reads the freshest dataset (binance_now if present,
  else binance_y1; v5 → v5 dataset); model sims are named without a window suffix
  (legacy "(probe)"/"(now)" names still resolve via `canonSim` + the registry).
- Trading: executor IS built — `src/run_binance_live.py` (--shadow default /
  --testnet / --live) + `src/trading/binance_executor.py`; feature parity
  live==dataset via `src/run_binance_parity_check.py`. Forward-shadow runs via
  `binance_shadow.ps1` → `binance_shadow.log`. REAL --live is gated behind
  forward-shadow validation (go SMALL via the runner), NOT behind any exam; the
  panel exposes save/shadow/testnet only. OKX live stays small meanwhile.
- **Exit-test / engine-test** (`run_binance_exittest`, `run_binance_engine_exittest`;
  panel «🎬 тест виходу» / «рахувати з виходами») run on the single live window
  (last --days to now), with optional `from_min/to_min` sub-window. Engine-test
  reports FREQUENCY (`signals` pre-book vs `taken` after-book, per-day), PER-BUILD
  +/- after cross-dedup (n, /day, win, HOLD$/EXIT$, early), and PER-DAY (Kyiv)
  consistency. Settings echoed (floor/book/notional/exit).

## 0. Judge by GRADATORS, not by the engine  ← most important
- A "gradator" / calibrator = the table of **realized winrate + avg net per
  probability/spread bucket** (RAW90/RAW85/SPREAD80/SPREAD70/…, the scorecard
  "MASK SUMMARY"). It answers *where the edge actually is*.
- **Always evaluate a model by its gradators, NOT by a specific engine's PnL.**
  The engine (selection rule + caps + sizing + cooldown) is configured **after**
  we draw conclusions from the gradators. Engine PnL conflates model quality with
  config choices. Look at the calibration first; tune the engine last.

## 1. How the sim works (mechanics — confirmed in code)
- **5-minute cadence.** Per scan: features at `t`, **entry at t+5m**, exit at
  `t+5m+horizon`. Leak-free. (`EXEC_ENTRY_DELAY_MIN=5`)
- **Horizon-conditioned, continuous.** `horizon_minutes`/`horizon_log` are input
  features; the win/loss threshold is **linearly interpolated** across horizons
  (`threshold_pct` = `np.interp`). So the model can output P(profit) for **any**
  horizon → a continuous curve over time.
- **We do NOT sweep every minute.** Training uses anchors {5,15,30,60,120,180}
  +2 random; evals used a sparse set (~30/45/60/90). Denser horizons = untapped
  lever (see §3).
- **Look at raw probabilities** (`up_prob`/`down_prob`) — that's the gradator input.
- Cost basis: fee 0.15% + slip 0.60% = **0.75%** all-coins (0.45% liquid-only).
  Per-instrument slippage is more honest for the $-sim (indices tight, thin names wide).

## 2. The volume↔winrate wall (proven ~5×, stop re-testing)
- **Loosening the gate for volume ALWAYS kills winrate** → into the losing zone.
  Frequency multiplies exposure, it does NOT dilute losses.
- The edge is the **high-conviction tail** (RAW85/90, SPREAD80, or `bdw`
  p_dir≥0.80 & p_opp≤0.05). Below ~0.80 it loses.
- On **bad/calm crypto days there is NO extractable edge by any method** (raw,
  blend, overlay, calibrator, from-scratch model, EV-regressor — all failed OOS,
  several inverted). Don't try to squeeze a bad day; detect it and sit out.

## 3. To get "same winrate, much MORE volume" — multiply the space, don't loosen
`signals = symbols × horizons × scans × P(clear high bar)`. Keep the bar; grow
the rest:
1. **Dense horizon grid** — model is continuous-horizon. BUT it is NOT free:
   querying horizons the model was NOT trained on (off-anchor) surfaces
   **miscalibrated, overconfident** signals → dedup-by-max-prob cherry-picks them
   → MORE candidates but LOWER winrate (tested: RAW90 73%→60%, bdw 74%→65%).
   **Fix = TRAIN on a dense horizon grid first** (see §9), then it pays off.
2. **More symbols** (more liquid crypto + equities, esp. semis/tech cluster).
3. **Multi-leg per (symbol,scan)** when several horizons/sides clear the bar.
4. Ensemble seeds to stabilise the tail.
Realistic "imba" = not a higher winrate, but ×5–10 volume at the same ~62–65%.

## 4. Validation protocol (our standard)
- **Hold out the LAST N hours (24/48) as the unknown test.** Train on everything
  before. Test must be truly unseen.
- **No middle/center holdouts. No symbol-holdouts.** (Decision: memorise all the
  rest.) The internal probe-val is a TAIL slice used only to pick iterations.
- A single short holdout is a **narrow exam** (esp. weekends) — 1–2 tail trades
  = noise. Prefer 48h, or repeat the holdout at a few cutoffs for sample.

## 5. Training (don't ship a 3-second model)
- Proven-good models used **~2500–6000 iterations, depth 6–7, val_auc 0.82–0.89**.
- If the recent val window is a bad regime it **early-stops at ~3 trees → flat,
  useless model**. Use `--no-early-stop` and train full iterations there.
- More capacity (depth 7, more iters) is fine; watch overfit on the holdout.
- `--no-early-stop` makes val_auc in-sample (≈0.99) — meaningless; judge on the holdout.

## 6. Symbol identity / transfer
- The 302 features are **symbol-blind** (relative price/vol curves + BTC ref).
  This is WHY models **transfer to never-seen symbols** (equities Group B worked).
- **Do NOT add the raw symbol name as a feature** — tested, it didn't help and
  hurt at volume; it also memorises and breaks transfer + held-out-symbol checks.
  If instrument-awareness is wanted, use *generalising descriptors*
  (asset_class, realised vol, liquidity, beta), not identity.
- Equities are tokenized `*_USDT_SWAP` on OKX, mixed INTO `data/candles`. The
  high-conviction tail transfers to them (semis/tech: MU, ORCL, QCOM, PLTR, META,
  TSM, MRVL). Indices (SPY/QQQ) rarely fire — fixed threshold favours volatile names.

## 7. Data hygiene (we kept "losing" data)
- Canonical inventory: `reports/DATA_INVENTORY.md` — regenerate with
  `python -m src.run_data_inventory --write`.
- Equity tickers are the source of truth in `src/markets.py:EQUITY_TICKERS`.
- The normal fetcher only covers the HC universe → off-universe symbols (most
  equities) go STALE unless fetched explicitly (`CandleFetcher.fetch_symbol(update=True)`).
- Watch for corrupt parquets (footer) — quarantine to `data/candles/_corrupt/` and refetch.
- Combined train universe (crypto+equities): `configs/hc_universe_plus_equities.json`.

## 9. Horizons & the FINAL dense build
- `HORIZON_ANCHORS=(5,15,30,60,120,180)` were a sparse default; the target is
  close on a 5-min grid so horizons must be 5-min-aligned. Anchors are NOT special
  — they were just cheap. The model is horizon-conditioned, so train it DENSE.
- **Dense training grid = full 5–180 every 5 min (36 horizons)**: build with
  `--random-count 30 --random-step-min 5` → every snapshot gets all 36. This lets
  the model learn the real continuous P(profit|horizon) curve instead of
  interpolating noise between anchors.
- **FINAL build config (max everything):**
  - universe: `configs/hc_universe_full.json` (all on-disk minus toxic blacklist +
    all equities ≈ 319 symbols), built by the `_tmp_uni2` recipe.
  - dataset: `data/hc_final/dataset`, dense 5–180, stride 120, ~14 days.
  - train: `run_hc_prod_train --no-early-stop --depth 7 --iterations 6000`,
    cutoff = edge − 48h (last 48h = unseen test).
  - judge: `run_hc_dense_eval` gradator frontier on the 48h holdout (dense horizons,
    now calibrated) — look at RAW85/90, SPREAD80, bdw.
- **FINAL RESULT (`models/hc_final`, 48h date-OOS holdout, eval 251 syms):**
  the fatter model (dense-trained 5–180, 315 syms, depth 7, 6000 iters) is clearly
  BETTER at the tail than the prior depth-7 model:
  | gate (SPARSE 30/60/90) | prior blind | **hc_final** |
  |---|---|---|
  | bdw .80&opp.05 | 65% / +0.54 | **79% / +1.22 (66 trades)** |
  | SPREAD80 | 62% / +0.09 | **78% / +1.09 (63)** |
  | RAW90 | 67% / +0.55 | **78% / +0.97 (37)** |
  - **Dense-horizon QUERYING still hurts even after dense training** (bdw dense
    60%/+0.09 vs sparse 79%/+1.22): max-prob across many horizons is an extreme
    order statistic that cherry-picks the model's most-overconfident horizon, and
    short horizons (10–20m) rarely beat the 0.75% cost. → **At inference use the
    sparse {30,60,90} + high-conviction gates.** Dense helped TRAINING, not querying.
  - More volume at this winrate ⇒ more SYMBOLS (+ multi-leg), NOT dense horizons.

## 10. How to ANALYZE a model (do this BEFORE wiring any engine)
A model is not "good/bad" — it has a *shape*. Profile it with the funnel
(`src/run_hc_funnel.py`) on a holdout, then pick controllers per model. Never
just shove a new model into the previous engine config.
Stages to read, in order:
1. **p_dir threshold curve** (dedup'd) — the master quantity↔quality knob. Find
   where net turns positive and how volume falls as you tighten.
2. **Per-horizon edge** at a fixed p_dir — which time-slices actually carry edge.
   Different models differ a lot (d7: only ~30–60m; d8: ~all 30–160m). Query only
   the horizons that pay; the rest are pure dilution.
3. **Multi-leg vs dedup-1/scan** — dedup-by-max-p_dir is an ANTI-PATTERN (it
   cherry-picks the model's most-overconfident horizon → lower winrate than the
   average leg). If per-horizon edge is broad (d8), MULTI-LEG recovers ~4× volume
   at the SAME winrate. If per-horizon edge is narrow/noisy (d7), multi-leg adds noise.
4. **opp cap** — for over-confident models (d7) `opp≤0.05` is essential; for a
   well-calibrated model (d8) it's redundant (high p_dir already implies low opp).
5. **spread curve** — alternative gate; cross-check against p_dir.
Risk note: multi-leg = several legs on the SAME (symbol,scan) → correlated. Size
them as ONE risk unit (split the stake), don't count them as independent bets.
Only after this profile do you choose: floor, horizon set, multi-leg cap, opp cap.

## 11. Parallel per-model engines + OOS zone selection
- **Don't force one config on all models.** Profile each model (funnel) and give
  it its OWN controllers, then run them as PARALLEL strategies combined into ONE
  portfolio. Current pair:
  - **Zhnyvar** = d7 (`hc_final`): narrow edge → tight horizons {30,40,50,60} +
    `opp≤0.05` + `p_dir≥0.85`. (d7 over-fires → needs the opp clean-up.)
  - **Snaiper** = d8 (`hc_final_d8`): clean across the curve → wide horizons
    {20–120(+160)} + `p_dir≥0.85`, opp cap NOT needed (high p_dir already clean).
- **Portfolio rule:** combine both engines but **cross-dedup** — never hold two
  positions on the same (symbol,scan); keep the higher p_dir. Shared
  max-concurrent + per-symbol cooldown across BOTH engines (one risk book).
- **OOS horizon-zone selection (critical):** pick "good zones" on window A,
  CONFIRM on a disjoint window B. NEVER select the horizon set on the same window
  you report winrate on — in-sample zone-picking inflates winrate (we saw d7 jump
  to 75–81% that way). The Zhnyvar/Snaiper horizon sets are still pending this
  A→B confirmation; treat current winrates as optimistic until then.

## 8. Key paths
- Train: `src/run_hc_dataset.py` (build) → `src/run_hc_prod_train.py` (cutoff, --no-early-stop).
- Gradator eval on a holdout: `src/run_hc_tagged_eval.py` (works for any model dir).
- Scorecard gradators on a window: `src/run_hc_scorecard_analysis.py` (--symbols, MASK SUMMARY).
- Models: `models/hc_exec_stride120_nonoverlap` (OLD), `hc_exec_to20260604_prod` (NEW).
- **Manual explorer + control panel**: `python -m dh.webapp.server` → http://127.0.0.1:8765
  (localhost-only). Buttons: fetch candles, regen data, jobs+logs, model info, shadow
  engine launch, save builds (with description → `configs/builds/`). Static-only fallback:
  open `reports/sim_explorer/index.html`. Data regen: `python -m src.run_hc_export_html`.
  Server code: `dh/webapp/server.py`. Single-engine `/api/live` stays shadow-only.
- **PORTFOLIO LIVE machine (build→live translator, the real $5×3)**:
  - Engine: `src/trading/hc_portfolio_engine.py` (`HCPortfolioEngine`) — runs N saved
    explorer builds as ONE risk book. Reuses tested `HCLiveEngine.snapshot` (scores each
    DISTINCT model once), applies each build's level filters (same `_cond/_apply` as the
    explorer + `run_hc_build`: side/p_dir/p_opp/spread/horizon-set/hour-of-day-Kyiv/asset/lean),
    then CROSS-DEDUPS across builds (1 pos/symbol, higher p_dir wins). LiveTrader gives the
    shared book: shared max-concurrent + per-symbol cooldown + battle-tested OKX order path.
  - Runner: `src/run_hc_portfolio_live.py --portfolio configs/builds/portfolio_5x3.json`
    `--shadow` (default, no orders) | `--demo` (OKX sandbox) | `--live` (REAL $, self-guards
    on OKX creds in `.env`). Stake = `--stake-margin $ × --leverage` = notional/pos.
  - Config: `configs/builds/portfolio_5x3.json` = d8_long + old_best + new_badday, $5×3,
    maxconc 12, cooldown 30m, universe `hc_universe_full.json`.
  - Panel: Білди page → «🏦 Портфель — один риск-банк» → mode shadow/demo/LIVE + stake/lev →
    `POST /api/portfolio`. LIVE button has a confirm() guard.
  - CAVEAT (live): the 3 builds' horizon/hour zones were picked IN-SAMPLE on ONE short
    window (6h) → winrates (71/78/82%) are optimistic; expect lower live. Go SMALL.
- Dense-trained: `models/hc_final` (d7) + `models/hc_final_d8` (d8). Funnel:
  `src/run_hc_funnel.py`; $-extraction: `src/run_hc_extract.py`; engine sim+hourly:
  `src/run_hc_engine_sim.py`.
- **Zhnyvar engine** (best operating point, live candidate): spec in
  `reports/ZHNYVAR_ENGINE.md`. d7 · h{30,40,50,60} · p_dir≥0.85 · opp≤0.05 ·
  risk-unit sizing. 24h OOS: 21 units, 71% win (tail-driven $ — go live SMALL).

## 13. Танцюючий Тарас (ТТ) — new training paradigm (SPEC, 2026-06-15)
- **NOT another P(profit) classifier — a paradigm change.** Full spec + build plan:
  `DANCING_TARAS.md` (authoritative for ТТ; read it before touching ТТ code).
- Core: regress the **forward price CURVE** (cumulative log-return, vol-normalized,
  1-min grid 1..240) as a `MultiRMSE` multi-output — **horizon is the OUTPUT axis,
  not an input feature** (kills §9 off-anchor miscalibration; continuous-h query =
  interpolate the predicted curve, no re-query). Then derive signal/best-horizon
  from the curve; a quantile "fan" layer gives per-point confidence; a `QuerySoftMax`
  ranker (group=scan + 60-scan supergroup, no-trade null = learned abstention) picks
  which candidates to trade vs skip. Pipeline (not joint), CatBoost, 3-seed, BTC ref kept.
- Code namespace: `src/tt/`, datasets `data/tt_*`, models `models/tt_*` (does NOT
  touch the hc pipeline). Binance only. Research-only first; live engine later.
- **MAXIMAL input schema** (`src/tt/schema_tt.py`, 561 feats, `N_POINTS=45`): all
  curve blocks c1m+c5m+c15m+c1h(+`c1h_btc`)+c4h(+`c4h_btc`) + 18 v5 regime scalars +
  time tail; horizon DROPPED from features (it's the output axis). 1-min curve is
  back as input. Phase 0 builder DONE: `src/tt/data_tt.py` + `src/run_tt_dataset.py`
  (cutoff = edge − holdout-days; `--no-regime` for fast smoke).
- **HOLDOUT: last 4 days are RESERVED for the user's own test — do NOT touch/look.**
  Train cutoff = data-edge − 4 days. Tests are NOT to be run yet (user's call).
- Key hypothesis ТТ tests: short horizons (2–15m), cost-dead on OKX's 0.75% wall,
  may be ALIVE on Binance's ~0.126% RT cost.
