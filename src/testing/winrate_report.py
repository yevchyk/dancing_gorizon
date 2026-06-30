"""Persists a full test run into one timestamped folder under test_results/."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


class WinRateReport:
    def __init__(self, out_root: Path):
        self.out_root = Path(out_root)

    def new_run_dir(self) -> Path:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = self.out_root / f"run_{ts}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(self, run_dir: Path, *, scored: pd.DataFrame, threshold_summary: pd.DataFrame,
              percentile_summary: pd.DataFrame, top_coins: pd.DataFrame,
              deceptive_coins: pd.DataFrame) -> None:
        scored.to_parquet(run_dir / "scored.parquet", index=False)
        threshold_summary.to_csv(run_dir / "threshold_summary.csv", index=False)
        percentile_summary.to_csv(run_dir / "percentile_summary.csv", index=False)
        top_coins.to_csv(run_dir / "top_coins.csv", index=False)
        deceptive_coins.to_csv(run_dir / "deceptive_coins.csv", index=False)
        self._write_text(run_dir, threshold_summary, percentile_summary)

    def _write_text(self, run_dir: Path, ts_summary: pd.DataFrame,
                    pct_summary: pd.DataFrame) -> None:
        lines = ["=== THRESHOLD SUMMARY (win rate per model) ===", ""]
        for name, g in ts_summary.groupby("model"):
            lines.append(name)
            for _, r in g.iterrows():
                lines.append(
                    f"  >{r['threshold']:.2f}  n={int(r['n_signals']):>5}  "
                    f"win={r['win_rate']:.3f}  base={r['base_rate']:.3f}  lift={r['lift']}"
                )
            lines.append("")

        best = (ts_summary[ts_summary["n_signals"] >= 20]
                .sort_values("win_rate", ascending=False).head(20))
        lines.append("=== LEADERBOARD (win_rate, n>=20) ===")
        for _, r in best.iterrows():
            lines.append(
                f"  {r['model']:<12} >{r['threshold']:.2f}  "
                f"win={r['win_rate']:.3f}  n={int(r['n_signals'])}  lift={r['lift']}"
            )

        lines += ["", "=== PERCENTILE SUMMARY (adaptive per-model thresholds) ===", ""]
        for name, g in pct_summary.groupby("model"):
            lines.append(name)
            for _, r in g.iterrows():
                lines.append(
                    f"  top{r['top_pct']:>4.1f}%  abs>={r['abs_threshold']:.3f}  "
                    f"n={int(r['n_signals']):>5}  win={r['win_rate']:.3f}  "
                    f"base={r['base_rate']:.3f}  lift={r['lift']}"
                )
            lines.append("")
        (run_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
