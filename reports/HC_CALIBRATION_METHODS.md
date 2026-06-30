# HC Calibration & Signal-Combination Methods

Reference for all the calibration / signal-combination tests we run, so future
calibration work is consistent. "Calibration" = bin signals by some score, measure
the REAL realized winrate + avg net in each bin (does the score predict outcome?).

## Conventions (use the same everywhere)
- Entry = `base_time + 5m`, exit = `entry + horizon` (executable, leak-free).
- `gross_move` = (exit/entry − 1)·100 for a long (short = −gross_move).
- Cost = fee + slippage. Two standard levels:
  - **all coins**: fee 0.15% + slip 0.60% = **0.75%**
  - **liquid only**: fee 0.15% + slip 0.30% = **0.45%**
- `net = side·gross_move − cost`; **win = net > 0**.
- "Edge zone" = the score range where avg_net > 0.

## Data sources (what is clean / no-peek)
- **OLD model** (`hc_exec_stride120_nonoverlap`, trained to 2026-05-26): clean OOS =
  **June 1–4** (rich, ~1.8M signals) — best for big-sample calibration.
- **NEW model** (`hc_exec_to20260604_prod`, trained to 2026-06-04 20:00 UTC): clean OOS =
  **June 5+ only** (thin, calm day → few tail signals).
- **Both-clean** (needed for any old+new combo): **June 5 only**.

## Methods (labels)
| ID | Name | Definition | Status / finding |
|----|------|-----------|------------------|
| **C1** | RAW | bin by raw `up_prob` (long) / `down_prob` (short), one model, one horizon | ✅ OLD Jun1-4: monotonic; edge only ≥0.85-0.90 (0.90-0.95→73% win) |
| **C2** | OPP_FILTER | C1 but require opposite prob ≤ cap (0.20) | ✅ ~no effect vs C1 (opp adds nothing) |
| **C3** | SPREAD | bin by `p_dir − p_opp` | ✅ monotonic; spread≥0.9 → ~86% win (≈ same as C1 tail) |
| **C4** | MODEL_BLEND_AVG | `p = (p_old + p_new)/2` then C1 ("скласти й /2") | ⏳ June5 only (thin) |
| **C5** | MODEL_OVERLAY_MIN | `p = min(p_old, p_new)` (both agree) then C1 | ✅ June5: thins sample, no edge (tail empty) |
| **C6** | HORIZON_MEAN | per (symbol,scan): `p̄ = mean(p_dir over horizons 30/45/60/90)`; trade symbol by p̄; outcome = mean over horizons | ✅ OLD Jun1-4: +4-7pp winrate vs C1 in tail (0.90-95→78.9% vs 74.8%) but fewer trades; denoise helps a little, edge still only ≥0.85 |
| **C7** | SPREAD_MEAN | per (symbol,scan): `s̄ = mean(up−down over horizons)`; trade by s̄ | ✅ OLD Jun1-4: s̄≥0.90 → 94.8% win but only ~15/day; cleanest filter, very rare |
| **C8** | REGIME / NO-TRADE | causal market-state and scorecard activity gate | diagnostic built: `src/run_hc_regime_gate_analysis.py`; not live-final |
| **C9** | TEMPORAL_IMPULSE | same side probability rising over T/T-10/T-20/T-30, especially `temporal_prob_slope_30m` | ✅ OLD Jun1-4: useful generator; stability-hard filters were worse than impulse |

## Scorecard operating model

The current best structure is NOT a cascade of hard filters.  Use an OR generator
pool, then rank inside it:

`RAW90 OR SPREAD80 OR HMEAN85 OR SMEAN70 OR TPROB_SLOPE50`

OLD clean OOS (`2026-06-01..2026-06-04`) pool results:

| slice | n | win | avg_net |
|---|---:|---:|---:|
| RAW90 | 1,532 | 80.1% | +1.646% |
| SPREAD80 | 4,762 | 71.9% | +1.087% |
| HMEAN85 | 3,604 | 70.0% | +1.200% |
| SMEAN70 | 11,976 | 60.9% | +0.583% |
| TPROB_SLOPE50 | 18,288 | 62.9% | +0.563% |
| POOL_ANY | 23,974 | 59.8% | +0.476% |

Current operating points from `src/run_hc_scorecard_frontier.py`:

| mode | config | result on OLD Jun1-4 |
|---|---|---|
| quality | `score>=~1.95`, `top3`, `cap6`, `cd30` | 121 trades, 87.6% win, +2.286% avg net |
| balanced | `score>=~1.20`, `top6`, `cap6`, `cd30` | 443 trades, 80.1% win, +1.430% avg net |
| max squeeze | `score>=~0.89`, `top20`, `cap20`, `cd30` | 1,271 trades, 71.4% win, +0.767% avg net |

Fresh `2026-06-05` warning:

- OLD lower-score modes lose hard; quality mode produces zero trades.
- NEW produces no viable frontier on Jun5, but is safer because its high tail is
  almost empty (`RAW90=1`, `RAW85=14`).
- NEW dry-tail pockets exist but are tiny: `p_dir>=0.85` has `n=14`, avg
  `+0.168%`; `spread>=0.75` has `n=30`, avg `+0.331%`; the best small portfolio
  profile tested had only `9` trades.  Shadow only until forward validation.
- Therefore only the quality operating point is live-safe right now.  Balanced
  and wider modes stay shadow/research until C8 is validated forward.

## Key findings so far (the wall we keep hitting)
- Raw probability **is** calibrated **in active regimes** (OLD, June 1-4): real edge in the high tail (≥0.85-0.90), losing below ~0.80.
- **June 5 (calm/adverse) = no edge by ANY method** (raw, blend, overlay, spread, threshold sweep). You cannot extract signal that is not in the data.
- NEW model is better-behaved than OLD on fresh data (conservative, fewer false-confident signals) but its profitable tail is unconfirmed (June 5 too thin).
- The main remaining lever = forward validation of **C8 regime/no-trade** and NEW-specific score thresholds.

## Runner scripts
- `outputs/analysis/_calib.py` — C1/C2/C3 on OLD June 1-4.
- `outputs/analysis/_calib_new.py` — C1 + C5 overlay, NEW vs OLD on June 5.
- `outputs/analysis/_calib_combo.py` — C6/C7 (horizon-mean) on OLD June 1-4.
- `outputs/analysis/_cmp_blend.py` — C4 blend sweep on June 5.
- `src/run_hc_scorecard_analysis.py` — frozen legs + lift/incremental/generator diagnostics.
- `src/run_hc_scorecard_frontier.py` — scorecard frontier + portfolio profiles.
- `src/run_hc_regime_gate_analysis.py` — causal market-state gate diagnostics.
- `src/run_hc_live.py --selection-mode quality|squeezer|bad_day_worker` — explicit
  live/shadow modes; defaults are unchanged. Packaged set: `bad_day_worker` (NEW,
  `p_dir>=0.80 AND p_opp<=0.05`) is the calm-day worker running live at
  `$5 x 2x = $10` notional; `quality` (NEW) is the litmus-paper regime canary at
  `$5 x 1x = $5` notional —
  run it as a standing shadow canary and watch whether it triggers. The `squeezer`
  good-day engine is deferred and OLD-only for now.

## Current report links

- Decision record: `reports/HC_SCORECARD_AND_REGIME_20260605.md`
- OLD frontier: `outputs/analysis/hc_scorecard_frontier/old_2026-06-01_4d_h30-90_p50_slip0p6/HC_SCORECARD_FRONTIER.md`
- OLD Jun5 frontier: `outputs/analysis/hc_scorecard_frontier/old_2026-06-05_0p5d_h30-90_p50_slip0p6/HC_SCORECARD_FRONTIER.md`
- NEW Jun5 frontier: `outputs/analysis/hc_scorecard_frontier/new_2026-06-05_0p5d_h30-90_p50_slip0p6/HC_SCORECARD_FRONTIER.md`
- Regime comparison: `outputs/analysis/hc_regime_gate/old_good_old_new_bad/HC_REGIME_GATE_ANALYSIS.md`
