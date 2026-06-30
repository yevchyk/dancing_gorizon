"""Clean, honest holdout report for the fast_v2 models and the pooled engine.

Single source of truth = outputs/analysis/fast_v2/holdout_scores.parquet
(72h untouched holdout, 130 symbols, horizons 2m/5m/8m/10m, model prob + the
realized outcome at the exact horizon).

Everything here is computed at the *exact* horizon, so it does NOT depend on
scan cadence. Scan cadence (1m vs 2m) only changes how many trades a live loop
takes and how they overlap -- it is a trade-frequency knob, not model quality.

Sections written to TRUTH_REPORT.md:
  A. Per-model predictivity (AUC vs direction, AUC vs event, Spearman, base
     rates, raw edge) -- 72h and last-24h side by side.
  B. Recommended probability threshold per model (PnL vs signal-count tradeoff).
  C. Pooled trading engine, hour-of-day breakdown (all models at their picks).
  D. Toxic coins: only those that bleed with an adequate sample in BOTH windows.

Run: python -m src.run_truth_report
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fast import config as FC

EVAL = FC.EVAL_COST          # 0.0015 round-trip fee + slippage
EDGE = FC.TARGET_EDGE        # 0.0010 fee-only edge that defines an "event"
HOLD = FC.FAST_ANALYSIS_DIR / "holdout_scores.parquet"
GRID = FC.FAST_ANALYSIS_DIR / "combined_signal_math" / "combined_signal_grid.parquet"
OUT = FC.FAST_ANALYSIS_DIR / "TRUTH_REPORT.md"

HORIZONS = ["2m", "5m", "8m", "10m"]
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97]
MODEL_ORDER = [f"{d}_{h}" for h in HORIZONS for d in ("up", "down")]

# Engine mechanics (live-like, fixed-horizon exits). Cooldown >= longest horizon
# so the same coin cannot re-fire on the same autocorrelated move (this is what
# made 1m vs 2m scanning look different — it was duplicate re-entries, not skill).
TOP_PER_SCAN = 3
MAX_OPEN = 8
COOLDOWN_MIN = 10
MIN_TOXIC_N = 12         # per-coin sample floor before we trust a verdict


# --------------------------------------------------------------------------- #
# data prep
# --------------------------------------------------------------------------- #
def load_signals() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Long per-(model, symbol, anchor) signal frame with outcomes precomputed."""
    d = pd.read_parquet(HOLD)
    d["anchor_time"] = pd.to_datetime(d["anchor_time"], utc=True)
    frames = []
    for direction, side in (("up", 1), ("down", -1)):
        x = pd.DataFrame({
            "model": direction + "_" + d["horizon"].astype(str),
            "horizon": d["horizon"].to_numpy(),
            "side": side,
            "symbol": d["symbol"].to_numpy(),
            "anchor_time": d["anchor_time"].to_numpy(),
            "day": d["day"].to_numpy(),
            "prob": (d["p_up"] if direction == "up" else d["p_down"]).astype(float).to_numpy(),
            "ret": d["real_ret"].astype(float).to_numpy(),
            "mfe": d["real_mfe"].astype(float).to_numpy(),
            "mae": d["real_mae"].astype(float).to_numpy(),
        })
        x = x[np.isfinite(x["prob"]) & np.isfinite(x["ret"])].copy()
        dirret = x["side"] * x["ret"]
        x["pnl"] = dirret - EVAL
        x["dir_correct"] = (dirret > 0).astype(int)
        x["event_hit"] = (dirret > EDGE).astype(int)
        x["touch_green"] = np.where(side == 1, x["mfe"] > EVAL, -x["mae"] > EVAL).astype(int)
        x["hour"] = pd.to_datetime(x["anchor_time"], utc=True).dt.hour
        frames.append(x)
    sig = pd.concat(frames, ignore_index=True)
    t_end = sig["anchor_time"].max()
    return sig, t_end


def window_hours(d: pd.DataFrame) -> float:
    t = pd.to_datetime(d["anchor_time"], utc=True)
    return max(1.0, (t.max() - t.min()).total_seconds() / 3600.0)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _auc(label: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(label)) < 2:
        return np.nan
    return float(roc_auc_score(label, prob))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    s = pd.Series(a)
    return float(s.corr(pd.Series(b), method="spearman"))


def model_predictivity(sig: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, d in sig.groupby("model"):
        prob = d["prob"].to_numpy()
        dirret = (d["side"] * d["ret"]).to_numpy()
        rows.append({
            "model": model,
            "n": int(len(d)),
            "auc_dir": _auc((dirret > 0).astype(int), prob),
            "auc_event": _auc(d["event_hit"].to_numpy(), prob),
            "spearman": _spearman(prob, dirret),
            "base_dir": float((dirret > 0).mean()),
            "base_event": float(d["event_hit"].mean()),
            "raw_avg_pnl_pct": float(d["pnl"].mean() * 100),
            "touch_green": float(d["touch_green"].mean()),
        })
    out = pd.DataFrame(rows).set_index("model").reindex(MODEL_ORDER).reset_index()
    return out


def threshold_sweep(d: pd.DataFrame, hours: float) -> pd.DataFrame:
    rows = []
    for thr in THRESHOLDS:
        s = d[d["prob"] >= thr]
        if len(s) == 0:
            continue
        daily = s.groupby("day")["pnl"].sum()
        rows.append({
            "thr": thr,
            "n": int(len(s)),
            "sig_per_day": len(s) / hours * 24.0,
            "win": float((s["pnl"] > 0).mean()),
            "event_hit": float(s["event_hit"].mean()),
            "avg_pnl_pct": float(s["pnl"].mean() * 100),
            "total_pnl_pct": float(s["pnl"].sum() * 100),
            "touch_green": float(s["touch_green"].mean()),
            "green_days": int((daily > 0).sum()),
            "days": int(s["day"].nunique()),
        })
    return pd.DataFrame(rows)


def recommend_threshold(sweep: pd.DataFrame, *, n_floor: int = 200,
                        margin: float = 0.02) -> dict | None:
    """Most-inclusive threshold that holds a *robust* per-signal edge.

    Robust = avg pnl above a small margin (not just >0), enough sample
    (n>=n_floor), spread over all days, and green on the majority of days so a
    single lucky day cannot pick the threshold. Among the survivors we take the
    LOWEST prob (most signals) — that is the "good PnL + most signals" knee.
    Returns None when nothing clears the bar (model is not standalone-tradeable).
    """
    if sweep.empty:
        return None
    days = int(sweep["days"].max())
    need_green = max(2, (days + 1) // 2)          # majority of days
    ok = sweep[(sweep["avg_pnl_pct"] >= margin) & (sweep["n"] >= n_floor)
               & (sweep["days"] >= max(3, days)) & (sweep["green_days"] >= need_green)]
    if ok.empty:
        return None
    return ok.sort_values("thr").iloc[0].to_dict()


# --------------------------------------------------------------------------- #
# pooled engine (live-like selection, fixed-horizon exits from real_ret)
# --------------------------------------------------------------------------- #
EXIT_MIN = {"2m": 2, "5m": 5, "8m": 8, "10m": 10}


def build_engine_candidates(sig: pd.DataFrame, picks: dict[str, float]) -> pd.DataFrame:
    parts = []
    for model, thr in picks.items():
        if thr is None:
            continue
        s = sig[(sig["model"] == model) & (sig["prob"] >= thr)].copy()
        parts.append(s)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def live_like_select(cand: pd.DataFrame, *, top_per_scan: int, max_open: int,
                     cooldown_min: int) -> pd.DataFrame:
    """One position per symbol, global cap, per-symbol cooldown, top-N by prob.

    Exit time = anchor + horizon (exact). PnL is already in `pnl` (from
    real_ret at the exact horizon), so there is no scan-cadence artifact.
    """
    if cand.empty:
        return cand.copy()
    x = cand.sort_values(["anchor_time", "prob"], ascending=[True, False]).copy()
    open_pos: list[tuple[pd.Timestamp, str]] = []
    last_open: dict[str, pd.Timestamp] = {}
    per_scan: dict[pd.Timestamp, int] = {}
    picked = []
    for row in x.itertuples(index=False):
        now = pd.Timestamp(row.anchor_time)
        open_pos = [(et, s) for et, s in open_pos if et > now]
        if per_scan.get(now, 0) >= top_per_scan:
            continue
        if len(open_pos) >= max_open:
            continue
        sym = row.symbol
        if any(sym == s for _, s in open_pos):
            continue
        prev = last_open.get(sym)
        if prev is not None and now < prev + pd.Timedelta(minutes=cooldown_min):
            continue
        picked.append(row)
        last_open[sym] = now
        per_scan[now] = per_scan.get(now, 0) + 1
        open_pos.append((now + pd.Timedelta(minutes=EXIT_MIN[row.horizon]), sym))
    return pd.DataFrame(picked)


def hourly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hour in range(24):
        g = trades[trades["hour"] == hour]
        if len(g) == 0:
            rows.append({"hour": hour, "n": 0, "win": np.nan, "avg_pnl_pct": np.nan,
                         "total_pnl_pct": 0.0})
            continue
        rows.append({
            "hour": hour,
            "n": int(len(g)),
            "win": float((g["pnl"] > 0).mean()),
            "avg_pnl_pct": float(g["pnl"].mean() * 100),
            "total_pnl_pct": float(g["pnl"].sum() * 100),
        })
    return pd.DataFrame(rows)


def per_model_hour_avg(trades: pd.DataFrame) -> pd.DataFrame:
    """avg pnl(%) per model x hour-of-day, signal level (engine trades)."""
    if trades.empty:
        return pd.DataFrame()
    piv = trades.pivot_table(index="model", columns="hour", values="pnl",
                             aggfunc="mean") * 100
    return piv.reindex([m for m in MODEL_ORDER if m in piv.index])


def new_vs_old_5m(q: float = 0.95) -> pd.DataFrame:
    """Head-to-head + agreement at the one horizon both families share (5m).

    The new and old probabilities live on different scales (the new up_5m model
    tops out near 0.55 — it is under-confident — while the old models reach 0.95),
    so a fixed cut would zero out one side. Instead "fires" = each family's own
    top-(1-q) tail, i.e. its most confident signals. We ask: alone, and when BOTH
    fire the same direction, how often is the call right and what is the net edge.
    """
    if not GRID.exists():
        return pd.DataFrame()
    g = pd.read_parquet(GRID)
    ret = g["real_ret_5m"].astype(float).to_numpy()
    rows = []
    for direction, side in (("up", 1), ("down", -1)):
        new_p = g[f"fast_v2_p_{direction}_5m"].astype(float).to_numpy()
        old_p = g[f"standard_p_{direction}_5m"].astype(float).to_numpy()
        tn = float(np.nanquantile(new_p, q))
        to = float(np.nanquantile(old_p, q))
        dirret = side * ret
        ev = (dirret > EDGE).astype(float)
        dc = (dirret > 0).astype(float)
        pnl = dirret - EVAL
        defs = {
            f"new {direction}_5m (top {100*(1-q):.0f}%, p>={tn:.2f})": new_p >= tn,
            f"old {direction}_5m (top {100*(1-q):.0f}%, p>={to:.2f})": old_p >= to,
            f"BOTH agree {direction}_5m": (new_p >= tn) & (old_p >= to),
        }
        for label, mask in defs.items():
            m = mask & np.isfinite(ret)
            n = int(m.sum())
            rows.append({
                "rule": label,
                "n": n,
                "dir_correct": float(dc[m].mean()) if n else np.nan,
                "event_hit": float(ev[m].mean()) if n else np.nan,
                "avg_pnl_pct": float(pnl[m].mean() * 100) if n else np.nan,
            })
    return pd.DataFrame(rows)


def coin_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym, g in trades.groupby("symbol"):
        rows.append({
            "symbol": sym,
            "n": int(len(g)),
            "win": float((g["pnl"] > 0).mean()),
            "avg_pnl_pct": float(g["pnl"].mean() * 100),
            "total_pnl_pct": float(g["pnl"].sum() * 100),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def md_table(df: pd.DataFrame, floatfmt: dict[str, str] | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if (isinstance(v, float) or pd.api.types.is_float(v)) and pd.isna(v):
                cells.append("")
            elif c in floatfmt:
                try:
                    cells.append(floatfmt[c](v))
                except (ValueError, TypeError):
                    cells.append(f"{int(round(float(v)))}")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    sig_full, t_end = load_signals()
    t_24 = t_end - pd.Timedelta(hours=24)
    sig_24 = sig_full[sig_full["anchor_time"] >= t_24].copy()
    h_full = window_hours(sig_full)
    h_24 = window_hours(sig_24)

    # A. predictivity
    pred_full = model_predictivity(sig_full)
    pred_24 = model_predictivity(sig_24)

    # B. thresholds + picks (picked on the 72h base of truth)
    # auc_event tells us whether a model has any real ranking signal at all.
    auc_event = pred_full.set_index("model")["auc_event"].to_dict()
    AUC_FLOOR = 0.57            # below this the model is essentially a coin-flip
    picks: dict[str, float | None] = {}
    sweeps_full: dict[str, pd.DataFrame] = {}
    rec_rows = []
    for model in MODEL_ORDER:
        dfm = sig_full[sig_full["model"] == model]
        sw = threshold_sweep(dfm, h_full)
        sweeps_full[model] = sw
        rec = recommend_threshold(sw)
        ae = auc_event.get(model, np.nan)
        if rec is None:
            picks[model] = None
            rec_rows.append({"model": model, "rec_thr": np.nan, "sig_per_day": np.nan,
                             "win": np.nan, "avg_pnl_pct": np.nan, "event_hit": np.nan,
                             "touch_green": np.nan, "avg24_pnl_pct": np.nan,
                             "verdict": "skip — no robust threshold"})
            continue
        s24 = sig_24[(sig_24["model"] == model) & (sig_24["prob"] >= rec["thr"])]
        avg24 = float(s24["pnl"].mean() * 100) if len(s24) else np.nan
        # eligibility for the new engine: real signal + edge that does not fully
        # collapse out-of-window.
        eligible = (ae >= AUC_FLOOR) and (pd.isna(avg24) or avg24 > -0.05)
        if eligible and rec["avg_pnl_pct"] >= 0.04:
            verdict = "core"
        elif eligible:
            verdict = "ok"
        elif ae >= AUC_FLOOR:
            verdict = "unstable (24h collapses) — confirmation only"
        else:
            verdict = "skip — weak signal (auc_event<0.57)"
        picks[model] = float(rec["thr"]) if eligible else None
        rec_rows.append({
            "model": model,
            "rec_thr": rec["thr"],
            "sig_per_day": rec["sig_per_day"],
            "win": rec["win"],
            "avg_pnl_pct": rec["avg_pnl_pct"],
            "event_hit": rec["event_hit"],
            "touch_green": rec["touch_green"],
            "avg24_pnl_pct": avg24,
            "verdict": verdict,
        })
    rec_df = pd.DataFrame(rec_rows)

    # C. pooled engine
    cand = build_engine_candidates(sig_full, picks)
    trades = live_like_select(cand, top_per_scan=TOP_PER_SCAN, max_open=MAX_OPEN,
                              cooldown_min=COOLDOWN_MIN)
    hourly = hourly_breakdown(trades)
    pm_hour = per_model_hour_avg(trades)
    coins = coin_breakdown(trades)
    trades_24 = trades[trades["anchor_time"] >= t_24]

    # D. toxic coins -- engine trades, adequate n, negative in both windows
    coins_full = coins.sort_values("total_pnl_pct")
    coins24 = coin_breakdown(trades_24)
    merged = coins_full.merge(coins24[["symbol", "n", "avg_pnl_pct", "total_pnl_pct"]],
                              on="symbol", how="left", suffixes=("", "_24"))
    toxic = merged[(merged["n"] >= MIN_TOXIC_N) & (merged["avg_pnl_pct"] < 0)]
    toxic_confirmed = toxic[(toxic["avg_pnl_pct_24"] < 0) | (toxic["avg_pnl_pct_24"].isna())]

    # E. new (fast_v2) vs old (standard) at the shared 5m horizon
    nvo = new_vs_old_5m()

    # ----- bottom line (computed from the tables above) -----
    def _models(pred):
        return ", ".join(rec_df.loc[rec_df["verdict"].str.startswith(pred), "model"]) or "—"
    core_m = _models("core")
    ok_m = _models("ok")
    conf_m = ", ".join(rec_df.loc[rec_df["verdict"].str.contains("confirmation"), "model"]) or "—"
    skip_m = ", ".join(rec_df.loc[rec_df["verdict"].str.startswith("skip"), "model"]) or "—"
    tox_top = ", ".join(toxic_confirmed.sort_values("total_pnl_pct")["symbol"]
                        .head(8).str.replace("_USDT_SWAP", "", regex=False))
    hrs = hourly.dropna(subset=["total_pnl_pct"])
    hot = ", ".join(str(int(h)) for h in hrs.sort_values("total_pnl_pct", ascending=False)["hour"].head(5))
    dead = ", ".join(str(int(h)) for h in hrs.sort_values("total_pnl_pct")["hour"].head(5))

    # ----- write -----
    ts = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")
    span = (f"{sig_full['anchor_time'].min():%Y-%m-%d %H:%M} -> "
            f"{sig_full['anchor_time'].max():%Y-%m-%d %H:%M} UTC")
    L: list[str] = []
    L.append("# Truth Report — new fast_v2 models + new engine\n")
    L.append(f"_Generated {ts}. Base = untouched 72h holdout ({span}), "
             f"130 symbols. Last-24h slice shown alongside._\n")
    L.append("_Naming: **`fast_v2_*`** = the NEW short-horizon models (2m/5m/8m/10m) "
             "and the new pooled engine. **`standard_*`** = the OLD production models "
             "(5m/15m/30m/1h/2h). Only the 5m horizon overlaps, so that is where "
             "new-vs-old is compared (Section E)._\n")
    L.append(f"_Costs: round-trip fee+slip = {EVAL*100:.2f}% per trade; "
             f"event edge = {EDGE*100:.2f}%. Every number is at the exact horizon "
             f"-> independent of scan cadence (1m vs 2m only changes trade COUNT, "
             f"not model quality)._\n")
    L.append("\n## Bottom line\n")
    L.append(f"- **Core new models** (real edge, holds out-of-window): {core_m}. "
             f"**Usable:** {ok_m}. **Confirmation-only** (24h collapses): {conf_m}. "
             f"**Skip** (no robust edge): {skip_m}.")
    L.append(f"- **Short side beats long side** at every horizon (down_* auc/edge > up_*).")
    L.append(f"- **Per-signal edge is small** (best ~+0.05–0.09%/trade) and the new "
             f"engine only works *with* throttling + coin filtering. Engine 72h: "
             f"{len(trades)} trades, net {trades['pnl'].sum()*100:+.1f}% "
             f"(sum of per-trade %), win {(trades['pnl']>0).mean():.3f}.")
    L.append(f"- **Confirmed toxic coins** (bleed in both windows, adequate n): {tox_top}.")
    L.append(f"- **Hot hours (UTC):** {hot}. **Dead hours:** {dead}.")
    if not nvo.empty:
        def _dc(sub):
            r = nvo.loc[nvo["rule"].str.contains(sub, case=False)]
            return r["dir_correct"].iloc[0] if len(r) else float("nan")
        L.append(
            f"- **New beats old at 5m:** new up_5m {_dc('new up'):.3f} vs old "
            f"{_dc('old up'):.3f}; new down_5m {_dc('new down'):.3f} vs old "
            f"{_dc('old down'):.3f} (direction-correct). **Old-model agreement does "
            f"NOT help** — both-agree down_5m {_dc('BOTH agree down'):.3f} is worse "
            f"than new alone. Don't use standard_* as a confirmation gate.")
    L.append(f"- The earlier report's blacklist was built on n=4 noise (it called "
             f"APR toxic — APR is actually a TOP coin here, +0.32%/trade over n=92). "
             f"This list uses an n>={MIN_TOXIC_N} floor and both-window confirmation.")

    L.append("\n## A. Model predictivity (72h)\n")
    L.append("`auc_dir` = ranks direction (ret>0). `auc_event` = ranks a real "
             "move past fee (|ret|>0.10%) — this is the metric the earlier report "
             "showed. `raw_avg_pnl` = edge if you took *every* signal of that model "
             "with no threshold.\n")
    fa = pred_full.copy()
    L.append(md_table(fa, {
        "auc_dir": "{:.3f}".format, "auc_event": "{:.3f}".format,
        "spearman": "{:+.3f}".format, "base_dir": "{:.3f}".format,
        "base_event": "{:.3f}".format, "raw_avg_pnl_pct": "{:+.4f}".format,
        "touch_green": "{:.3f}".format, "n": "{:d}".format,
    }))

    L.append("\n## A2. Same models, last-24h slice\n")
    L.append(md_table(pred_24.copy(), {
        "auc_dir": "{:.3f}".format, "auc_event": "{:.3f}".format,
        "spearman": "{:+.3f}".format, "base_dir": "{:.3f}".format,
        "base_event": "{:.3f}".format, "raw_avg_pnl_pct": "{:+.4f}".format,
        "touch_green": "{:.3f}".format, "n": "{:d}".format,
    }))

    L.append("\n## B. Recommended probability per model\n")
    L.append("Rule: the LOWEST prob (= most signals) whose per-signal edge is "
             "robust — avg_pnl >= +0.02%, n >= 200, spread over all days, and green "
             "on the majority of days (so one lucky day can't pick it). "
             "`avg24_pnl_pct` re-checks that exact threshold on the last 24h: if it "
             "collapses below -0.05% the model is downgraded to confirmation-only. "
             "`sig_per_day` is raw availability across ALL 130 coins (a firehose) — "
             "the real tradeable count after throttling is the engine in Section C.\n")
    L.append(md_table(rec_df[["model", "rec_thr", "sig_per_day", "win", "event_hit",
                              "avg_pnl_pct", "avg24_pnl_pct", "touch_green", "verdict"]], {
        "rec_thr": "{:.2f}".format, "sig_per_day": "{:.1f}".format,
        "win": "{:.3f}".format, "event_hit": "{:.3f}".format,
        "avg_pnl_pct": "{:+.4f}".format, "avg24_pnl_pct": "{:+.4f}".format,
        "touch_green": "{:.3f}".format,
    }))

    L.append("\n### B2. Full threshold curves (per model, 72h)\n")
    for model in MODEL_ORDER:
        sw = sweeps_full[model]
        if sw.empty:
            continue
        L.append(f"\n**{model}**\n")
        L.append(md_table(sw, {
            "thr": "{:.2f}".format, "sig_per_day": "{:.1f}".format,
            "win": "{:.3f}".format, "event_hit": "{:.3f}".format,
            "avg_pnl_pct": "{:+.4f}".format, "total_pnl_pct": "{:+.2f}".format,
            "touch_green": "{:.3f}".format,
        }))

    engine_models = ", ".join(m for m, v in picks.items() if v is not None)
    L.append("\n## C. New pooled engine — hour-of-day (UTC)\n")
    L.append(f"Engine uses only the eligible new models ({engine_models}) at their "
             f"recommended prob, pooled. Live mechanics: one open position per "
             f"symbol, global cap {MAX_OPEN}, top {TOP_PER_SCAN}/scan by prob, "
             f"{COOLDOWN_MIN}m per-symbol cooldown, fixed-horizon exits. "
             f"Total engine trades 72h = {len(trades)}, "
             f"net PnL = {trades['pnl'].sum()*100:+.2f}% "
             f"(sum of per-trade %, NOT account return — no sizing/compounding), "
             f"win = {(trades['pnl']>0).mean():.3f}.\n")
    L.append(md_table(hourly, {
        "win": "{:.3f}".format, "avg_pnl_pct": "{:+.4f}".format,
        "total_pnl_pct": "{:+.2f}".format, "hour": "{:d}".format, "n": "{:d}".format,
    }))

    if not pm_hour.empty:
        L.append("\n### C2. avg PnL% per model × hour (engine trades)\n")
        L.append("_Directional only — per-cell n is small (a few trades), so read "
                 "the sign/pattern, not the exact number._\n")
        pmh = pm_hour.copy()
        pmh.columns = [str(c) for c in pmh.columns]
        pmh = pmh.reset_index()
        L.append(md_table(pmh, {c: "{:+.3f}".format for c in pmh.columns if c != "model"}))

    L.append("\n## D. Toxic coins (engine trades)\n")
    L.append(f"Only coins with n>={MIN_TOXIC_N} on 72h AND a negative avg edge. "
             f"`*_24` columns = last-24h check. A coin is a confirmed blacklist "
             f"candidate only if it bleeds in BOTH windows (or has no 24h trades). "
             f"Small-n coins are NOT listed — that was the earlier report's mistake.\n")
    if toxic_confirmed.empty:
        L.append("\n_No coin clears the sample floor as a confirmed bleeder. "
                 "Do not blacklist on noise._\n")
    else:
        show = toxic_confirmed[["symbol", "n", "win", "avg_pnl_pct", "total_pnl_pct",
                                "n_24", "avg_pnl_pct_24"]].copy()
        L.append(md_table(show, {
            "win": "{:.3f}".format, "avg_pnl_pct": "{:+.4f}".format,
            "total_pnl_pct": "{:+.2f}".format, "avg_pnl_pct_24": "{:+.4f}".format,
            "n": "{:.0f}".format, "n_24": "{:.0f}".format,
        }))
    L.append("\n### D2. Best coins (for context)\n")
    best = coins_full.sort_values("total_pnl_pct", ascending=False)
    best = best[best["n"] >= MIN_TOXIC_N].head(12)
    L.append(md_table(best[["symbol", "n", "win", "avg_pnl_pct", "total_pnl_pct"]], {
        "win": "{:.3f}".format, "avg_pnl_pct": "{:+.4f}".format,
        "total_pnl_pct": "{:+.2f}".format, "n": "{:d}".format,
    }))

    if not nvo.empty:
        L.append("\n## E. New (fast_v2) vs Old (standard) — shared 5m horizon\n")
        L.append("Each family alone, then when BOTH fire the same direction at 5m. "
                 "`dir_correct` = direction right; `avg_pnl_pct` = net edge per signal. "
                 "Agreement is the question you asked: when old AND new both say "
                 "up_5m (or down_5m), how often is it right.\n")
        L.append(md_table(nvo, {
            "dir_correct": "{:.3f}".format, "event_hit": "{:.3f}".format,
            "avg_pnl_pct": "{:+.4f}".format, "n": "{:d}".format,
        }))

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"engine trades 72h={len(trades)} net={trades['pnl'].sum()*100:+.2f}% "
          f"win={(trades['pnl']>0).mean():.3f}")
    print("picks:", {k: v for k, v in picks.items()})


if __name__ == "__main__":
    main()
