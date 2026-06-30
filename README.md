# 🪩 Dancing Horizon

**Horizon-conditioned machine-learning trading signals for crypto (and tokenized equities).**
A research engine that learns, for any symbol and *any* future horizon, the calibrated
probability that a trade clears its cost — then trades only the high-conviction tail.

> Built with [CatBoost](https://catboost.ai/) · gradient-boosted decision trees ·
> probability **calibration** · leak-free walk-forward backtesting ·
> OKX & Binance live execution.

---

## What is this?

Dancing Horizon is a **quantitative trading / signal-research system** for short-horizon
directional bets on perpetual swaps. Most ML-for-trading projects predict "up or down"
at one fixed horizon and then drown in transaction costs. This one does two things
differently, and both turned out to matter:

1. **Horizon is a model input, not a fixed choice.** A single pair of gradient-boosted
   models (one for *up*, one for *down*) is **conditioned on the horizon** you ask about.
   Feed it `horizon_minutes = 30` or `= 120` and it returns `P(profit | that horizon)`.
   Because horizon is continuous, you get a whole **probability-vs-time curve** per
   signal instead of one number.

2. **We judge the model by its calibration, not by a pretty equity curve.** The core
   artifact is a *gradator* — a table of **realized win-rate and average net P&L per
   probability bucket**. It answers the only question that matters: *where is the edge
   actually real?* Engines, position sizing, and caps are tuned **afterwards**, on top
   of a model we already trust.

The whole thing is **symbol-blind**: the ~302 features describe *relative* price/volume
shape plus a BTC reference, never the ticker name. That's why models trained on crypto
**transfer to never-before-seen instruments** — including tokenized equities (semis/tech
like MU, ORCL, QCOM, PLTR, META, TSM, MRVL).

**Keywords:** algorithmic trading, machine learning trading signals, CatBoost,
gradient boosting, crypto quant, probability calibration, walk-forward backtest,
horizon-conditioned model, OKX, Binance, perpetual futures, market microstructure.

---

## 🤪 A brutally honest note on the code (read this part)

Let me set expectations.

**This is vibecode.** It was hammered out at 3 a.m. across dozens of sessions, with
files named `_tmp_ntree_probe.py`, three half-abandoned data schemas, a `src/` folder
nobody is allowed to edit "casually," and a `dh/` folder that was supposed to be the
clean rewrite and is now its own kind of swamp. There are functions that take a
`--no-early-stop` flag because the *alternative* was a model that silently collapsed to
three trees. There is a comment in the codebase that is, essentially, *"do NOT reintroduce
the secret-exam pipeline, we ripped it out for a reason."* The git history this repo is
*replacing* contained a folder literally called `data_secret.js`.

It is, by any reasonable software-engineering standard, **a mess.**

**And yet — IT SEES SOMETHING.** 👀

On clean out-of-sample windows, the high-conviction tail (`p_dir ≥ 0.90`) lands
**73–87% directional win-rate** on **millions** of held-out samples — not on the days it
was fitted, on genuinely unseen days. The calibration is *monotone*: higher predicted
probability really does mean higher realized win-rate, bucket after bucket, across
disjoint time slices. The dense-trained models hit **78–79% / +1.0–1.2% net** on a 48-hour
date-OOS holdout across 250+ symbols. That is not nothing. That is the spaghetti
**quietly working.**

So: the engineering is embarrassing, the result is not. This README documents *both*
honestly, because the lessons below were paid for in real losing days, and they're the
actual value here — not the code style.

---

## The one idea that pays: the high-conviction tail

Five separate experiments, three model families, multiple regimes — same wall every time:

- **Loosening the gate to get more volume *always* kills win-rate** and pushes you below
  50%. Frequency multiplies exposure; it does **not** dilute losses.
- **The edge lives in the tail.** `up_prob`/`down_prob ≥ ~0.80–0.90` (or a clean spread,
  or `p_dir ≥ 0.80 & p_opp ≤ 0.05`). Below ~0.80 it *loses*.
- **On calm / bad days there is no extractable edge by any method** we tried (raw, blend,
  overlay, calibrator, from-scratch model, EV-regressor — all failed OOS, several
  inverted). You don't squeeze a dead day. You **detect it and sit out.** The model
  firing ~0 signals on a flat day is the system working *correctly.*

To get *"same win-rate, much more volume,"* you don't loosen the gate — you **multiply the
space**: `signals = symbols × horizons × scans × P(clear the high bar)`. Keep the bar,
grow the rest (more symbols, multi-leg per scan, ensemble seeds). Realistic "imba" is not
a higher win-rate — it's **×5–10 volume at the same ~62–65%.**

---

## How it works (the flow: model → engine → your desk)

1. **Data → features.** Every 5 minutes, per symbol, build a row of ~302 features at
   `base_time = now − 5m`: four timeframes (5m / 15m / 1h / 4h, 30 points each) of
   relative price + volume ratios, a BTC reference on the higher timeframes, plus the
   `horizon_minutes` you're asking about. Symbol-blind by design.

2. **Training.** Two CatBoost models per run (UP, DOWN), **horizon-conditioned**, trained
   up to a **cutoff date** — everything after the cutoff is clean out-of-sample. Leak-free
   target: features at `t`, **entry at `t+5m`**, exit at `t+5m+horizon`. (The original
   mirror-leak gave a fake 90%+; fixing it is what made the rest trustworthy.)

3. **Calibration ("gradators").** Bin signals by score, measure the **real** realized
   win-rate and net per bin. This is how we learned *where* the edge is — and it's the
   thing you should look at first, last, and always.

4. **Sim / backtest.** On a mature OOS window: executable outcomes (entry+5m → exit at
   horizon), a slot scheduler (one position/symbol or multi-leg), conviction sizing,
   slippage, capital caps. Always reported OOS — never the days a threshold was fit on.

5. **Engines.** A selection rule on top of probabilities: threshold + ranking + portfolio
   caps + cooldown. Different models get **different controllers** (a narrow-edge model
   needs a tight horizon set and an opposite-probability cap; a clean model can run wide).
   Multiple engines run in **parallel as one shared risk book** with cross-dedup.

6. **Live.** OKX / Binance executors (real fills, reduce-only partial closes), a position
   manager with per-leg deadlines, and a 5-minute scan loop. Live is gated behind
   forward-shadow validation. Default mode everywhere is **`--shadow`** (no orders).

7. **Report.** One command ties any date range together: `python -m dh.report.stats`.

---

## What works · what doesn't (hard-won, keep)

**Works**
- Leak-free timing (features at `t`, enter at `t+5m`).
- The high-conviction tail (`p_dir ≥ ~0.90`).
- **Calibration tables over cascade filters** — use score generators as an *OR pool*, then
  rank, instead of stacking hard AND-filters.
- Conviction sizing (size ∝ `p_dir − p_opp`).
- **No stops / no take-profit** — winners must run to the horizon; caps and stops hurt the
  edge.
- Letting the system sit out bad days.
- **Dense-horizon *training*** (5–180 min on a 5-minute grid) — it makes the model learn
  the real continuous `P(profit | horizon)` curve.

**Does not work (stop trying)**
- More frequency / lower threshold → the losing zone.
- Adding the **raw symbol name** as a feature → didn't help, hurt at volume, and *broke*
  the symbol transfer that makes this thing interesting.
- Opposite-prob filters / blends / overlays beyond the raw tail → ~nothing.
- Trading every day → the edge is regime-dependent.
- **Dense-horizon *querying* at inference** → max-prob across many horizons is an extreme
  order statistic that cherry-picks the model's most over-confident horizon. Train dense;
  *query* sparse ({30, 60, 90}) with high-conviction gates.

---

## ⚠️ Data & models are NOT in this repo

The full research store is **~46 GB** (29.6 GB of 1-minute candles across ~200 symbols ×
365 days, plus 16 GB of trained model artifacts). That does not belong on GitHub, so it
is **git-ignored**. What ships instead:

- ✅ all the **code** (`src/`, `dh/`, `configs/`)
- ✅ a **tiny runnable sample** — `data/binance_smoke/` (a handful of symbols) and one
  small trained model, `models/hc_exec_to20260604_prod/` — so the quickstart actually runs
- ✅ the **decision records** in `reports/*.md` (the real story of what we learned)
- ❌ the full candle store, the big models, logs, `.venv`, and `.env`

The candle store is **rebuilt locally from the exchange public APIs** — the fetchers are
included, no keys required to download market data. It's a research engine, not a dataset
release.

---

## Quickstart (runs on the bundled sample, no keys)

```powershell
# 1. environment
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# 2. shadow a live engine on the sample model — NO real orders, no keys needed.
#    This loads the bundled model, scores the sample candles, prints any signals.
.\.venv\Scripts\python -m src.run_hc_live --shadow --once --selection-mode quality `
  --model-dir models\hc_exec_to20260604_prod `
  --stake-margin 5 --leverage 1 --top-per-scan 3 --max-concurrent 6
```

`--shadow` is the default everywhere: it computes and logs signals but **places no
orders**. A quiet market legitimately produces zero signals — the model only fires on
high-conviction setups.

---

## The full pipeline (fetch → build → train → evaluate → run)

The bundled sample is just enough to see the machinery move. To do real research you
rebuild the data and train your own models. Everything runs from the project root via
`python -m ...`.

### 1. Fetch candles (public data, no keys)

```powershell
# OKX — one full sweep of the candle store universe, then exit
.\.venv\Scripts\python -m src.run_fetcher --once --universe store --workers 10

# OKX — rebuild / backfill the separate long-history 200-symbol store
.\.venv\Scripts\python -m src.run_okx_stable200_build
.\.venv\Scripts\python -m src.run_okx_stable200_backfill --workers 4 --update

# Binance — the overnight driver chains: candle top-up -> funding history ->
# honest cost model -> alignment check -> full 365d x 1m dataset -> depth sweep.
.\.venv\Scripts\python -m src.run_binance_overnight
```

Candles land under `data/...` (OKX: `data/candles`, Binance: `data/binance*`). Check
coverage any time with:

```powershell
.\.venv\Scripts\python -m src.run_data_inventory
```

### 2. Build a training dataset

Turn raw candles into leak-free feature/label rows (features at `t`, entry at `t+5m`,
exit at `t+5m+horizon`). The horizon grid is dense by design (train dense, query sparse):

```powershell
# OKX HC dataset
.\.venv\Scripts\python -m src.run_hc_dataset

# Binance v5 dataset (dense 30..320 every 5 min). --fresh rebuilds from scratch.
.\.venv\Scripts\python -m src.run_binance_dataset_v5 --workers 8 --fresh
```

### 3. Train

Two horizon-conditioned CatBoost models (UP, DOWN), trained up to a **cutoff date** so
everything after the cutoff is clean out-of-sample. Use `--no-early-stop` on calm windows
so the model doesn't collapse to three trees:

```powershell
.\.venv\Scripts\python -m src.run_hc_prod_train --no-early-stop --depth 7 --iterations 6000
```

The cutoff is `data-edge − HOLDOUT_DAYS`, computed at runtime — the last N days are simply
held out as the unseen test. There is no frozen/sealed test fixture anywhere in the code.

### 4. Evaluate by calibration (this is the important step)

Look at the **gradators** — realized win-rate and net per probability bucket — on the
held-out tail. This, not an equity curve, tells you whether the model is real.

```powershell
# scorecard / calibration on a holdout window (MASK SUMMARY = the gradator table)
.\.venv\Scripts\python -m src.run_hc_scorecard_analysis --date 2026-06-01 --days 4 --model old

# per-horizon edge / multi-leg / opp-cap profile before wiring any engine
.\.venv\Scripts\python -m src.run_hc_funnel
```

### 5. Explore & run

```powershell
# full stats for a date range (the main report tool)
.\.venv\Scripts\python -m dh.report.stats --date 2026-06-04 --days 1 --models new,old --slip 0.6

# local web explorer + control panel (localhost only)
.\.venv\Scripts\python -m dh.webapp.server          # -> http://127.0.0.1:8765

# shadow a single engine (safe) or a portfolio of saved builds as one risk book
.\.venv\Scripts\python -m src.run_hc_live --shadow --selection-mode bad_day_worker `
  --model-dir models\hc_exec_to20260604_prod --bdw-raw 0.80 --bdw-opp 0.05
.\.venv\Scripts\python -m src.run_hc_portfolio_live --portfolio configs\builds\portfolio_5x3.json --shadow
```

### 6. Going live (optional, at your own risk)

Copy `.env.example` to `.env` and add **your own** exchange keys. Keep `OKX_DEMO=1` while
testing. Live runners (`--live`) self-guard on missing credentials and should only be used
after forward-shadow validation. Start small.

```powershell
.\.venv\Scripts\python -m src.run_hc_live --live --selection-mode bad_day_worker `
  --model-dir models\hc_exec_to20260604_prod --bdw-raw 0.80 --bdw-opp 0.05 `
  --stake-margin 5 --leverage 2 --top-per-scan 3 --max-concurrent 6
```

---

## Repo layout

```
dancing_horizon/
├── dh/              clean reusable layer — config, data, models, sim, calibration, report
├── src/             the vendored proven engine (training, live executors, runners)
├── configs/         universes, thresholds, cost models, saved builds
├── reports/         decision records (*.md) — the actual hard-won knowledge
├── models/          [git-ignored except one sample model]
├── data/            [git-ignored except data/binance_smoke]
├── CLAUDE.md        the no-bullshit working notes (rules we stopped re-deriving)
└── README.md        you are here
```

`CLAUDE.md` and `DANCING_TARAS.md` are the unvarnished engineering logs — if you actually
want to understand the project's reasoning, read those.

---

## ⚖️ Disclaimer

This is **research and educational** software. It is **not financial advice**, not a
recommendation, and not a promise of profit. Markets change; an edge measured on past
out-of-sample windows can vanish forward. Trading leveraged perpetual futures can lose you
more than your stake. **If you point this at a real account, you do so entirely at your own
risk.** Default to `--shadow`. Start small. The authors are not responsible for your P&L.

## License

MIT — see [`LICENSE`](LICENSE). Do what you want; just don't blame us.
