# HC Scorecard + Regime Summary (2026-06-05)

This is the current decision record for the HC scoring work.

## What changed

- We stopped treating every signal component as a hard filter.
- Candidate generation is now an OR pool:
  `RAW90 OR SPREAD80 OR HMEAN85 OR SMEAN70 OR TPROB_SLOPE50`.
- Inside that pool, a transparent scorecard ranks legs.
- Portfolio simulation then applies one active position per symbol, `top_per_scan`,
  `max_open`, and cooldown.
- The live timing bugs were fixed separately: fresh anchor after fetch, confirmed
  candles only, fill-time based opens, and deadline exits independent of scan time.

## Clean OLD OOS: 2026-06-01..2026-06-04

OLD model cutoff: `2026-05-26`, so this window is clean OOS.

Generator pool:

| slice | n | win | avg net |
|---|---:|---:|---:|
| RAW90 | 1,532 | 80.1% | +1.646% |
| SPREAD80 | 4,762 | 71.9% | +1.087% |
| HMEAN85 | 3,604 | 70.0% | +1.200% |
| SMEAN70 | 11,976 | 60.9% | +0.583% |
| TPROB_SLOPE50 | 18,288 | 62.9% | +0.563% |
| POOL_ANY | 23,974 | 59.8% | +0.476% |

Portfolio profiles, with `$8 margin * 5x = $40 notional` per trade:

| profile | config | n | win | avg net | avg USD/day | best day |
|---|---|---:|---:|---:|---:|---:|
| quality | `thr~1.95 top3 cap6 cd30` | 121 | 87.6% | +2.286% | +$27.7/day | +$49.1 |
| balanced | `thr~1.20 top6 cap6 cd30` | 443 | 80.1% | +1.430% | +$63.4/day | +$90.0 |
| aggressive | `thr~0.90 top10 cap10 cd30` | 814 | 72.9% | +0.927% | +$75.5/day | +$106.3 |
| max squeeze | `thr~0.89 top20 cap20 cd30` | 1,271 | 71.4% | +0.767% | +$97.6/day | +$130.7 |

The maximum daily squeeze observed in this clean OLD OOS window was about
`+$130.7` on 2026-06-04, but that is the wide/high-risk mode.

## NEW on 2026-06-01..2026-06-04 (in-sample diagnostic only)

NEW is trained to `2026-06-04 20:00 UTC`, so this window is NOT clean OOS for
NEW.  Treat these numbers only as "how NEW behaves in the familiar active
regime", not as proof of forward edge.

Diagnostic generator pool:

| slice | n | win | avg net |
|---|---:|---:|---:|
| SPREAD80 | 13,982 | 84.9% | +1.591% |
| RAW90 | 11,870 | 81.1% | +1.488% |
| HMEAN85 | 19,300 | 74.1% | +1.083% |
| SMEAN70 | 28,824 | 74.3% | +1.072% |
| POOL_ANY | 39,987 | 68.0% | +0.783% |

Diagnostic frontier best wide mode was about `+$496` over the 4-day window with
`$8 margin * 5x`, but again this is in-sample and must not be used as a live
expectation.

## Fresh adverse window: 2026-06-05

This is the important warning.

OLD on 2026-06-05, partial `~10.75h` mature window:

| profile | trades | actual USD | extrapolated / 24h |
|---|---:|---:|---:|
| quality | 0 | $0.0 | $0.0/day |
| balanced | 23 | -$12.8 | -$28.6/day |
| aggressive | 61 | -$30.7 | -$68.5/day |
| max squeeze | 83 | -$42.8 | -$95.5/day |

Conclusion: OLD lower-score modes must not trade this regime.  Quality mode avoids
the damage by producing no trades.

NEW on 2026-06-05:

| slice/profile | n/trades | result |
|---|---:|---:|
| RAW90 | 1 | statistically useless |
| RAW85 | 14 | too thin |
| scorecard POOL_ANY | 181 | avg net -0.641% |
| quality profile | 0 trades | sits out |
| balanced profile | 3 trades | -$0.7 actual |
| aggressive/max | 12 trades | -$2.6 actual |

Conclusion: NEW is useful right now mostly because it is conservative and does
not create a false large tail in the bad fresh window.  There is not enough clean
NEW OOS to claim its profitable tail yet.

Additional NEW dry-tail check on the same `2026-06-05` window found small
positive pockets:

| NEW rule | n | win | avg net |
|---|---:|---:|---:|
| `p_dir >= 0.85` | 14 | 57.1% | +0.168% |
| `spread >= 0.75` | 30 | 46.7% | +0.331% |
| `p_dir >= 0.80 and p_opp <= 0.05` | 17 | 47.1% | +1.332% |
| `p_dir >= 0.82 and spread >= 0.70` | 22 | 59.1% | +0.230% |

Portfolio constraints reduce these to tiny numbers.  Best small profile tested:
`p_dir >= 0.82 and spread >= 0.70`, `top3`, `cap6`: `9` trades, avg `+0.673%`,
about `+$2.4` actual over the partial window.  This is promising for shadow, not
enough for live sizing.

## Simple OLD tail sanity check on 2026-06-05

We explicitly checked whether the bad day was overcomplicated, and whether simply
raising raw probability or relative opposite-probability evidence would recover
an edge.

OLD `2026-06-05`, partial mature window:

| rule | n | win | avg net |
|---|---:|---:|---:|
| `p_dir >= 0.85` | 194 | 44.3% | -0.685% |
| `p_dir >= 0.88` | 50 | 56.0% | -0.196% |
| `p_dir >= 0.90` | 13 | 53.8% | -0.255% |
| `p_dir >= 0.92` | 1 | 100.0% | +1.737% |
| `spread >= 0.80` | 129 | 30.2% | -1.012% |
| `spread >= 0.85` | 28 | 35.7% | -1.059% |
| `spread >= 0.90` | 0 | n/a | n/a |

Combinations such as `p_dir >= 0.88/0.90` plus `spread` floors or `p_opp` caps
also had no positive bucket with `n >= 5`.  The least-bad non-tiny rule was
`p_dir >= 0.88 and spread >= 0.60`: `n=50`, win `56.0%`, avg net `-0.196%`.

Conclusion: simply raising raw probability or spread does not rescue OLD on
2026-06-05.  The single positive `p_dir >= 0.92` trade is not a tradable rule.

## Current recommendation

- Live-safe mode: use the quality operating point only:
  `score >= ~1.95`, `top_per_scan=3`, `max_open=6`, `cooldown=30`.
- Balanced/aggressive modes are research/shadow until C8 regime/no-trade logic is
  validated on more clean forward days.
- Do not judge by winrate alone.  Use total net, avg net, drawdown, N, and regime
  robustness.

## Key artifacts

- Scorecard discovery script:
  `src/run_hc_scorecard_analysis.py`
- Scorecard frontier script:
  `src/run_hc_scorecard_frontier.py`
- Regime gate analysis script:
  `src/run_hc_regime_gate_analysis.py`
- OLD scorecard report:
  `outputs/analysis/hc_scorecard_frontier/old_2026-06-01_4d_h30-90_p50_slip0p6/HC_SCORECARD_FRONTIER.md`
- OLD bad-day report:
  `outputs/analysis/hc_scorecard_frontier/old_2026-06-05_0p5d_h30-90_p50_slip0p6/HC_SCORECARD_FRONTIER.md`
- NEW bad-day report:
  `outputs/analysis/hc_scorecard_frontier/new_2026-06-05_0p5d_h30-90_p50_slip0p6/HC_SCORECARD_FRONTIER.md`
- NEW in-sample active diagnostic:
  `outputs/analysis/hc_scorecard_frontier/new_2026-06-01_4d_h30-90_p50_slip0p6_INSAMPLE_DIAGNOSTIC/HC_SCORECARD_FRONTIER.md`
- Regime comparison report:
  `outputs/analysis/hc_regime_gate/old_good_old_new_bad/HC_REGIME_GATE_ANALYSIS.md`

## Can NEW be adapted?

Yes, but not by pretending 2026-06-05 contains a hidden tradable edge.

Reasonable path:

1. Keep NEW as the live/shadow production model because it sits out bad regimes.
2. Apply the same scorecard framework to NEW, but do not fit NEW thresholds on
   2026-06-05 alone.
3. Accumulate several clean forward days after the NEW cutoff
   (`2026-06-04 20:00 UTC`), then fit/validate NEW-specific score thresholds.
4. If retraining is needed, train the model or a second-stage ranker toward
   expectancy/net-PnL labels, not only raw direction probability.

Current evidence: NEW gives safety/conservatism now; it does not yet give a
confirmed profitable high-volume tail.

## Live adapter modes

`src/trading/hc_live_engine.py` now supports explicit selection modes.  Defaults
are unchanged.

- `plain`: old behavior, `p_dir >= high AND p_opp <= opp_cap`.
- `squeezer`: good-day extractor, `p_dir >= high OR spread >= spread_floor`.
  Example: `--selection-mode squeezer --high 0.90 --spread-floor 0.80`.
- `quality`: ultra-strict guard, effectively `p_dir >= max(high, 0.94)` OR
  `spread >= 0.92`.  This is the live-safe/no-trade-first mode.
- `bad_day_worker`: calm/bad-regime dry-pocket extractor, **AND** gate
  `p_dir >= bdw_raw (0.80) AND p_opp <= bdw_opp (0.05)`.  Distinct from squeezer:
  this is an intersection, not a union.  Flags: `--bdw-raw`, `--bdw-opp`.

Use `squeezer` in shadow until C8 regime permission is forward-validated.

## Packaged operating set (2026-06-05)

Per the live sizing decision, the packaged modes trade small (down from the
`$8 * 5x = $40` used in the simulations above).  Each mode is sized for its role:

- **bad_day_worker** (the calm/bad-day worker): runs on **NEW**
  (`models/hc_exec_to20260604_prod`), gate `p_dir>=0.80 AND p_opp<=0.05`.  This is
  the NEW dry pocket from `2026-06-05` (n=17, avg net +1.332%).  Sized
  **`$5 * 2x = $10` notional**, runs **live and standing** — this is the one that
  actually trades.  It is thin and forward-unproven on a single partial bad day, so
  keep `top_per_scan` and `max_concurrent` small and sanity-check in shadow first.
- **quality = litmus paper** (the regime canary): runs on **NEW**, ultra-strict
  `p_dir>=0.94 OR spread>=0.92`.  Its job is not to earn — it is the **litmus test
  for regime**: when quality starts producing signals, the calm/штиль regime is
  lifting and the active modes become candidates.  Zero quality signals = stay in
  bad_day_worker / sit-out posture.  Sized **`$5 * 1x = $5` notional** on purpose:
  it is a standing shadow canary the user glances at in the logs, not a sizer.
  (Distinct from the scorecard `quality` profile above that earned `+$27.7/day`;
  the live mode is a behavioural proxy with the same sit-out-on-bad-days property,
  not the same score>=1.95 engine.)

Pending (deferred, do later): the **squeezer (вижимач)** good-day engine.  For now
it is **OLD-only** (`models/hc_exec_stride120_nonoverlap`), because NEW has no
clean forward active day to extract from yet.  See "Open / next" in README.

Ready commands are in README under "Packaged live modes ($5 x 3x)".
