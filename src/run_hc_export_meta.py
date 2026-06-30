"""Inject window.LIQUID + window.COST into the explorer data.js (no re-scoring).

LIQUID = configs/hc_universe_liquid.json (the tradeable set; powers the
"liquid only" checkbox). COST = per-symbol honest round-trip % from hc.costs
(bar-range based; powers the threshold-vs-cost traffic light). Appended at the
END so SIM_META stays line 1 (the server reads that).
"""

from __future__ import annotations

import json

from . import config as C
from .hc.costs import cost_fn_from_store

OUT = C.ROOT / "reports" / "sim_explorer" / "data.js"
LIQ = C.CONFIGS_DIR / "hc_universe_liquid.json"


def main() -> None:
    txt = OUT.read_text(encoding="utf-8")
    syms = None
    keep = []
    for line in txt.splitlines():
        if line.startswith("window.LIQUID=") or line.startswith("window.COST="):
            continue  # drop any previous injection
        keep.append(line)
        if line.startswith("window.SYMS="):
            syms = json.loads(line[len("window.SYMS="):].rstrip().rstrip(";"))
    if syms is None:
        raise SystemExit("no window.SYMS in data.js")

    liq_all = json.loads(LIQ.read_text(encoding="utf-8"))
    liq_all = liq_all.get("symbols", liq_all) if isinstance(liq_all, dict) else liq_all
    sym_set = set(syms)
    liquid = [s for s in liq_all if s in sym_set]

    cost_fn = cost_fn_from_store()
    cost = {s: round(float(cost_fn(s)), 3) for s in syms}

    body = "\n".join(keep).rstrip("\n")
    extra = ("\nwindow.LIQUID=" + json.dumps(liquid) +
             ";\nwindow.COST=" + json.dumps(cost, separators=(",", ":")) + ";\n")
    OUT.write_text(body + extra, encoding="utf-8")
    print(f"injected LIQUID={len(liquid)} (of {len(syms)} syms) + COST={len(cost)} into {OUT}")
    lo = min(cost.values()); hi = max(cost.values())
    print(f"cost range: {lo:.2f}% .. {hi:.2f}%  (liquid floor 0.45%)")


if __name__ == "__main__":
    main()
