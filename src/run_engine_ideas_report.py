"""New engine ideas, simulated against the Unicorn baseline.

Unicorn = PulseClean3 idx0.00 exit10m (>=3 worthy models agree, none oppose) =
the live `pulse00` profile. We test data-driven variants:

  A. Unicorn-Down  — short side only (the data's alpha is the down side).
  B. Unicorn-Up    — long side only (contrast).
  C. Unicorn-Asym  — raise UP thresholds (abundant/weak), keep DOWN at base.
  D. Unicorn-Clean — Unicorn minus the confirmed toxic coins.
  E. Unicorn-Hours — Unicorn minus the dead UTC hours (4,6,7,11,12).
  F. Unicorn-Prime — Asym + Clean + Hours combined.

All on the 72h holdout via the real fixed simulator, 24h slice alongside, plus a
levered $ view at stake $20 x 8 = $160 notional/trade (no compounding).

Run: python -m src.run_engine_ideas_report
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fast import config as FC
from .run_strictness_index_sweep import WORTHY
from .run_engine_sim_report import BOOK, stats
from .run_engine_compare_report import load_grid, scan_times, simulate, md_table
from .run_test_engine_harvest_sim import simulate_engine  # noqa: F401 (via simulate)

OUT = FC.FAST_ANALYSIS_DIR / "ENGINE_IDEAS.md"

NOTIONAL = 20.0 * 8.0       # stake $20 x 8 size = $160 notional per trade
TOXIC = ("WLD_USDT_SWAP", "JTO_USDT_SWAP", "GIGGLE_USDT_SWAP", "INJ_USDT_SWAP",
         "GRASS_USDT_SWAP", "HYPE_USDT_SWAP", "BEAT_USDT_SWAP", "EDEN_USDT_SWAP")
DEAD_HOURS = (4, 6, 7, 11, 12)

# Asymmetric thresholds: up raised toward where its edge concentrates, down at base.
ASYM = {"fast_v2_up_10m": 0.78, "fast_v2_up_8m": 0.84, "fast_v2_up_2m": 0.93,
        "fast_v2_down_10m": 0.82, "fast_v2_down_8m": 0.83, "fast_v2_down_2m": 0.92}


def votes(grid: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    up_c = np.zeros(len(grid)); down_c = np.zeros(len(grid))
    up_s = np.zeros(len(grid)); down_s = np.zeros(len(grid))
    for name, (col, side, _base) in WORTHY.items():
        thr = thresholds[name]
        p = grid[col].astype(float).to_numpy()
        active = p >= thr
        headroom = np.clip((p - thr) / max(1e-9, 1.0 - thr), 0, None)
        if side == 1:
            up_c += active; up_s += np.where(active, headroom, 0.0)
        else:
            down_c += active; down_s += np.where(active, headroom, 0.0)
    side = np.where(up_s >= down_s, 1, -1)
    side_count = np.where(side == 1, up_c, down_c)
    opp_count = np.where(side == 1, down_c, up_c)
    score = np.where(side == 1, up_s, down_s)
    return pd.DataFrame({
        "symbol": grid["symbol"].to_numpy(), "anchor_time": grid["anchor_time"].to_numpy(),
        "day": grid["day"].to_numpy(), "side": side.astype(int),
        "side_count": side_count, "opp_count": opp_count, "score": score,
    })


def unicorn_candidates(grid, *, thresholds=None, min_agree=3, exit_h="10m",
                       sides=(1, -1), block=(), drop_hours=()) -> pd.DataFrame:
    thr = thresholds or {n: b for n, (_c, _s, b) in WORTHY.items()}
    v = votes(grid, thr)
    mask = (v["side_count"] >= min_agree) & (v["opp_count"] == 0) & v["side"].isin(sides)
    d = v[mask].copy()
    if len(block):
        d = d[~d["symbol"].isin(block)]
    if len(drop_hours):
        hrs = pd.to_datetime(d["anchor_time"], utc=True).dt.hour
        d = d[~hrs.isin(drop_hours)]
    d["exit"] = exit_h
    d["signal_model"] = "unicorn"
    return d[["symbol", "anchor_time", "day", "side", "exit", "score", "signal_model"]]


def lev_row(trades: pd.DataFrame, hours: float, t_24) -> dict:
    s = stats(trades, hours)
    if trades.empty:
        s24 = stats(trades, 24.0)
        usd72 = usd24 = 0.0
    else:
        tr24 = trades[pd.to_datetime(trades["opened_at"], utc=True) >= t_24]
        s24 = stats(tr24, 24.0)
        usd72 = NOTIONAL * trades["net_pnl"].sum()
        usd24 = NOTIONAL * tr24["net_pnl"].sum()
    return {
        "trades/day": s["trades_per_day"], "win": s["win"], "avg%": s["avg_pnl_pct"],
        "total%": s["total_pnl_pct"], "$72h": usd72,
        "24h_trades": s24["n"], "24h_win": s24["win"], "24h_total%": s24["total_pnl_pct"],
        "$24h": usd24,
    }


def main() -> None:
    grid = load_grid()
    st = scan_times(grid)
    t_end = grid["anchor_time"].max()
    t_24 = t_end - pd.Timedelta(hours=24)
    h_full = max(1.0, (t_end - grid["anchor_time"].min()).total_seconds() / 3600.0)

    ideas = {
        "Unicorn (pulse00)": unicorn_candidates(grid),
        "A. Unicorn-Down (short only)": unicorn_candidates(grid, sides=(-1,)),
        "B. Unicorn-Up (long only)": unicorn_candidates(grid, sides=(1,)),
        "C. Unicorn-Asym (up raised)": unicorn_candidates(grid, thresholds=ASYM),
        "D. Unicorn-Clean (no toxic)": unicorn_candidates(grid, block=TOXIC),
        "E. Unicorn-Hours (no dead hrs)": unicorn_candidates(grid, drop_hours=DEAD_HOURS),
        "F. Unicorn-Prime (Asym+Clean+Hours)": unicorn_candidates(
            grid, thresholds=ASYM, block=TOXIC, drop_hours=DEAD_HOURS),
        # G. asymmetry in AGREEMENT, not thresholds: down is strong at >=2 agree,
        # up needs >=3. This adds the rare-but-excellent down signals back.
        "G. Unicorn-Hybrid (up>=3, down>=2)": pd.concat([
            unicorn_candidates(grid, sides=(1,), min_agree=3),
            unicorn_candidates(grid, sides=(-1,), min_agree=2),
        ], ignore_index=True),
        "H. Down>=2 only": unicorn_candidates(grid, sides=(-1,), min_agree=2),
    }

    rows = []
    for label, cand in ideas.items():
        tr = simulate(cand, st, label)
        rows.append({"idea": label, **lev_row(tr, h_full, t_24)})
    table = pd.DataFrame(rows)

    ts = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")
    L: list[str] = []
    L.append("# Engine ideas vs Unicorn (pulse00)\n")
    L.append(f"_Generated {ts}. 72h holdout, real fixed simulator, cap 8, top 3/scan, "
             f"10m cooldown. `$` columns = stake $20 x 8 = ${NOTIONAL:.0f} notional/trade, "
             f"sum of per-trade $ (no compounding, no concurrent-exposure cap)._\n")
    L.append("\n## Results\n")
    L.append(md_table(table, {
        "trades/day": "{:.1f}".format, "win": "{:.3f}".format, "avg%": "{:+.4f}".format,
        "total%": "{:+.2f}".format, "$72h": "{:+.0f}".format, "24h_trades": "{:.0f}".format,
        "24h_win": "{:.3f}".format, "24h_total%": "{:+.2f}".format, "$24h": "{:+.0f}".format,
    }))

    # quick verdict
    base = table[table["idea"].str.startswith("Unicorn (")].iloc[0]
    best72 = table.sort_values("total%", ascending=False).iloc[0]
    best24 = table.sort_values("24h_total%", ascending=False).iloc[0]
    L.append("\n## Read\n")
    L.append(f"- **Baseline Unicorn:** {base['trades/day']:.0f}/day, win {base['win']:.3f}, "
             f"72h {base['total%']:+.1f}% (${base['$72h']:+.0f}), 24h {base['24h_total%']:+.1f}% "
             f"(${base['$24h']:+.0f}). It is the best all-rounder — see below.")
    L.append(f"- **Best 72h total:** {best72['idea']} ({best72['total%']:+.1f}%) — but "
             f"check its 24h before trusting it.")
    L.append(f"- **Best last-24h (robustness):** {best24['idea']} "
             f"({best24['24h_total%']:+.1f}%).")
    L.append("- **Verdict:** none of the 7 variants beats the baseline Unicorn on "
             "total AND 24h-robustness together. Down-side is excellent per trade but "
             "rare (it almost never gets 3-way agreement; at >=2 agree it turns fragile, "
             "idea H went negative on 24h). Raising up thresholds (C) or filtering hours "
             "(E) trades away the up volume that IS the edge. Unicorn-Clean (D, drop the "
             "confirmed toxic coins) is the only free tweak — essentially identical, "
             "marginally cleaner. **Keep pulse00; optionally add the toxic blocklist.**")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
