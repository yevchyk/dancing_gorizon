# Zhnyvar (Жнивар) — engine spec & operating manual

High-conviction, multi-horizon, risk-unit harvester on the dense-trained model.
Chosen from the funnel analysis (`src/run_hc_funnel.py`) as the best
quantity/quality balance. Candidate for SMALL live forward-testing.

## Models (both kept)
- **`models/hc_final`** = d7: depth 7, 6000 iters, dense horizons 5–180,
  319-symbol universe (crypto+equities), cutoff = edge−48h. The workhorse.
- **`models/hc_final_d8`** = d8: depth 8, 9000 iters (border 16). Conservative
  "sniper" — fewer, very clean signals. Kept as a complement.
- Both are **symbol-blind** (no tag — tested, the tag didn't help, hurt at volume).

## Zhnyvar config (operating point)
| controller | value | why (from funnel) |
|---|---|---|
| model | `hc_final` (d7) | most independent risk-units at high winrate |
| horizons | **{30,40,50,60}** | d7's edge lives only ~30–60m; rest is noise |
| p_dir gate | **≥0.85** | best $/day balance (0.80=more vol, 0.90=cleaner) |
| opp cap | **≤0.05** | essential for d7 (it over-fires; opp cleans it) |
| sizing | **risk-unit**: 1 stake per (symbol,scan); multi-leg splits it | legs on one symbol are correlated — NOT independent bets |
| max concurrent | 15 | |
| cooldown | 30m / symbol | |
| universe | `configs/hc_universe_full.json` (319) | max symbols = the real volume lever |

## Validation (48h funnel, risk-unit, $15/unit)
- p_dir≥0.85: ~28 units/day, **78% unit-win, +0.75% net/unit, ~$3.2/day** @ $15.
- p_dir≥0.80: ~40/day, 73%, +0.48, ~$2.9/day.
- p_dir≥0.90: ~18/day, 82%, +0.78, ~$2.2/day.

## 24h OOS sim (Jun6 15:45→Jun7 15:45 UTC, $15/unit)
- 21 units, **71% win**, +0.56% net/unit, **+$1.76 total**, maxDD −$0.28.
- ⚠️ Tail-driven: HOME +14.8% carried it; one −15.1% nearly cancelled it. Quiet
  weekend window → few units, mostly small crypto (equities barely fired).

## Honest status / risk
- Winrate (71–78%) is real and holds across windows; **$ is outlier-dependent on
  any single window** and the sample is small (21–58 units).
- No stops (project rule: winners run to horizon) → expect occasional −10–15% legs;
  conviction + winrate carry it on average.
- **Go live SMALL** (forward test, e.g. $5–15/unit) — do NOT size up on one 24h.
- Validate on ≥2–3 more windows before scaling.

## Snaiper (d8) + DUAL portfolio (run both in parallel)
Two engines, one risk book. Each profiled with its own controllers:
- **Zhnyvar** = d7 `hc_final`: h{30,40,50,60} · p_dir≥0.85 · opp≤0.05.
- **Snaiper** = d8 `hc_final_d8`: h{20,30,40,50,60,70,80,90,120,160} · p_dir≥0.85
  · opp cap NOT needed (d8 is clean across the curve).
Portfolio rule: **cross-dedup** (one position per symbol/scan, keep higher p_dir)
+ shared max-concurrent (15) + per-symbol cooldown (30m). Risk-unit sizing.

24h OOS dual sim (Jun6 15:45→Jun7 15:45 UTC, $15/unit):
- **28 units, 68% win, +0.51% net/unit, +$2.14/day.**
  Zhnyvar 19 (68%, +1.58), Snaiper 9 (67%, +0.56).
- Steady 04:00–11:00 core ≈ +$1.6 on ~17 units (NOT tail-driven); the +2.22
  (HOME) and −2.26 outliers ~cancel. Dual > single (28 vs 21 units, +$2.14 vs +$1.76).
- Still ONE window; horizon zones not yet A→B OOS-confirmed → go live SMALL.

## Run
```
# DUAL portfolio sim + hourly (Zhnyvar d7 + Snaiper d8 in parallel) — the main one
python -m src.run_hc_dual_sim --hours 24
# single-engine sim + hourly report (any 24h)
python -m src.run_hc_engine_sim --hours 24
# funnel (profile a model before trusting it)
python -m src.run_hc_funnel --model-dir models/hc_final --cutoff-local "<edge-48h Kyiv>"
# extraction $/curve across the p_dir knob
python -m src.run_hc_extract --model-dir models/hc_final --cutoff-local "<...>" --horizons 30,40,50,60 --opp-cap 0.05
```
Live executor is NOT yet wired for this exact config (risk-unit multi-leg) — that
is the next build before real orders.

## $ scaling
Linear in notional: $15/unit → ~$2–3/day; $150/unit → ~$20–30/day at ~$2k deployed
(maxconc×notional). Real money lever = MORE SYMBOLS, not more horizons/legs.
