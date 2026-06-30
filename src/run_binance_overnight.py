"""Overnight driver for the Binance year-rebuild (BINANCE_PLAN.md §4-§6 step 1).

Chains, with hard gates (any failure aborts the rest and is logged):
  0. wait for the candle top-up to finish (binance_topup.log ends with DONE)
  1. funding history     -> configs/binance_funding.json
  2. honest costs        -> configs/binance_costs.json (now incl extra-25)
  3. Binance/OKX alignment check (must print ALIGNED)
  4. freeze train universe (skipped if already frozen)
  5. smoke dataset (2 syms) + bit-exact label/threshold verification
  6. FULL year dataset, 8 parallel workers  -> data/binance_y1/dataset
  7. dataset verification on a 12-shard sample
  8. depth sweep d8 -> d10 -> d12, 3 seeds each, --random-val, GPU.
     HOLDOUT RULE: the test period is a plain runtime holdout — train cutoff =
     (now - HOLDOUT_DAYS), so the last HOLDOUT_DAYS stay unseen. It is NOT a
     frozen/sealed config; nothing about the test window leaks into the code.
Writes progress to stdout (redirect to binance_overnight.log) and a final
OVERNIGHT DONE / OVERNIGHT FAILED marker line.

  python -m src.run_binance_overnight
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PY = sys.executable
# children must never die on printing exotic symbol names under cp1251 consoles
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
ROOT = Path(__file__).resolve().parents[1]
TOPUP_LOG = ROOT / "binance_topup.log"
DATASET = "data/binance_y1/dataset"
DEPTHS = (8, 10, 12)
SEEDS = "41,42,43"
ITERATIONS = 12000
HOLDOUT_DAYS = 5  # last N days kept unseen as the holdout (runtime cutoff, never frozen)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(name: str, args: list[str], must_contain: str | None = None) -> None:
    log(f"START {name}: {' '.join(args)}")
    t0 = time.time()
    p = subprocess.run([PY, "-m", *args], cwd=ROOT, capture_output=True, text=True,
                       env=ENV, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    for line in out.strip().splitlines()[-15:]:
        print(f"    {line}", flush=True)
    dt = (time.time() - t0) / 60
    if p.returncode != 0:
        raise RuntimeError(f"{name} exited {p.returncode} after {dt:.1f}m")
    if must_contain and must_contain not in out:
        raise RuntimeError(f"{name}: expected '{must_contain}' in output")
    log(f"OK {name} ({dt:.1f}m)")


def main() -> None:
    try:
        # 0. top-up must be finished (it writes 'DONE ...' as its last line)
        t0 = time.time()
        while True:
            if TOPUP_LOG.exists() and "DONE" in TOPUP_LOG.read_text(errors="ignore")[-2000:]:
                break
            if time.time() - t0 > 45 * 60:
                raise RuntimeError("top-up did not finish within 45m")
            log("waiting for candle top-up...")
            time.sleep(60)
        log("top-up finished")

        run("funding", ["src.binance_funding"])
        run("costs", ["src.binance_costs", "--window-days", "30"])
        run("align", ["src.binance_okx_align"], must_contain="ALIGNED (lag 0 wins)")

        run("smoke-build", ["src.run_binance_dataset", "--limit-symbols", "2",
                            "--days", "5", "--out-dir", "data/binance_smoke/dataset", "--fresh"])
        run("smoke-check", ["src.binance_dataset_check",
                            "--dataset", "data/binance_smoke/dataset"],
            must_contain="ALL CHECKS PASS")

        run("dataset", ["src.run_binance_dataset", "--workers", "8", "--out-dir", DATASET])
        run("dataset-check", ["src.binance_dataset_check", "--dataset", DATASET,
                              "--sample", "12"], must_contain="ALL CHECKS PASS")

        cutoff = (pd.Timestamp.utcnow().floor("min") - pd.Timedelta(days=HOLDOUT_DAYS)).isoformat()
        log("TRAIN PHASE STARTING — close GPU apps now "
            "(dataset load gives a few minutes before the GPU engages)")
        log(f"training cutoff (runtime holdout, now-{HOLDOUT_DAYS}d) = {cutoff}")
        for d in DEPTHS:
            run(f"train-d{d}", ["src.run_hc_prod_train",
                                "--dataset-dir", DATASET,
                                "--model-dir", f"models/binance_y1_d{d}",
                                "--cutoff-local", cutoff,
                                "--seeds", SEEDS, "--depth", str(d),
                                "--iterations", str(ITERATIONS),
                                "--random-val", "--task-type", "GPU"])
        log("OVERNIGHT DONE")
    except Exception as e:
        log(f"OVERNIGHT FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
