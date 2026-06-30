"""Date -> full stats.

    python -m dh.report.stats --date 2026-06-04 [--days 1] [--models new,old] [--slip 0.6]

Builds features for the period, scores the chosen model(s), computes executable
outcomes (entry+5m, exit at horizon) and prints:
  * headline: signal counts in the working tail (>=0.90), regime activity
  * C1 raw calibration (winrate by probability), C6 horizon-mean, C7 spread-mean
A markdown copy is saved under reports/by_date/.

Honest reminders baked into the output:
  * edge lives only in the high-conviction tail (>=~0.90); below ~0.80 loses;
  * a calm/low-activity period legitimately yields ~no signals (don't force trades).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dh import config as cfg
from dh import data, models, sim
from dh import calibration as cal

KEYS = ["symbol", "base_time", "horizon_minutes"]


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df[cols].itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body])


def run(date: str, days: float, which_models: list[str], slip: float) -> str:
    syms = data.universe()
    entries = sim.date_grid(date, days)
    out = [f"# Dancing Horizon - stats for {date} (+{days}d, slip {slip}%)",
           f"_universe={len(syms)} tradable, {len(entries)} scans (5-min), horizons={cfg.HORIZONS_DEFAULT}_", ""]
    if len(entries) == 0:
        out.append("No scans in window (future date or no data)."); return "\n".join(out)

    feats = data.build_features(syms, entries, cfg.HORIZONS_DEFAULT, cfg.ENTRY_DELAY_MIN)
    if feats.empty:
        out.append("No features built (no candle data for this window)."); return "\n".join(out)
    moves = sim.outcomes(feats)
    cost = cfg.cost(slip)

    # regime activity: average # of high-conviction (>=0.85) rows per scan, any model-agnostic proxy
    for which in which_models:
        sc = models.score(feats, which)
        d = sc[KEYS].copy()
        d["up_prob"] = sc["up_prob"].values
        d["down_prob"] = sc["down_prob"].values
        d = d.merge(moves, on=KEYS, how="inner")
        out += [f"\n## Model: {which.upper()}  (trained to {cfg.MODEL_CUTOFF[which]})  rows_with_outcome={len(d)}"]
        if d.empty:
            out.append("_no matured rows_"); continue

        tail_long = int((d["up_prob"] >= cfg.EDGE_PROB).sum())
        tail_short = int((d["down_prob"] >= cfg.EDGE_PROB).sum())
        n_scans = d["base_time"].nunique()
        active = (((d["up_prob"] >= 0.85) | (d["down_prob"] >= 0.85)).sum()) / max(1, n_scans)
        regime = "ACTIVE" if active >= 5 else ("MODERATE" if active >= 1 else "CALM (sit out)")
        out += [f"- working-tail signals (>= {cfg.EDGE_PROB}): long={tail_long}, short={tail_short}",
                f"- regime: **{regime}** (~{active:.1f} signals>=0.85 per scan)"]

        g = cal.horizon_mean(d)
        out += ["", "**C1 RAW LONG** (winrate by up_prob)", _md(cal.c1_raw(d, "long", cost)),
                "", "**C1 RAW SHORT** (by down_prob)", _md(cal.c1_raw(d, "short", cost)),
                "", f"**C6 HORIZON-MEAN LONG** (denoised, {len(g)} symbol-scans)", _md(cal.c6_horizon_mean(g, "long", cost)),
                "", "**C7 SPREAD-MEAN LONG**", _md(cal.c7_spread_mean(g, cost))]

    text = "\n".join(out)
    dest = cfg.REPORTS_DIR / "by_date" / f"{date}_{days}d_slip{slip}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    out.append(f"\n_saved -> {dest}_")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument("--models", default="new,old")
    ap.add_argument("--slip", type=float, default=cfg.SLIP_ALL)
    a = ap.parse_args()
    which = [m.strip() for m in a.models.split(",") if m.strip()]
    print(run(a.date, a.days, which, a.slip))


if __name__ == "__main__":
    main()
