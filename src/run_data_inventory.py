"""Show the candle-data landscape so we never confuse / lose stores again.

  python -m src.run_data_inventory               # one row per store
  python -m src.run_data_inventory crypto_feature  # per-symbol coverage of one store
  python -m src.run_data_inventory --write         # (re)write reports/DATA_INVENTORY.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from . import config as C
from .markets import REGISTRY, get, EQUITY_TICKERS, is_equity

DOC_PATH = C.ROOT / "reports" / "DATA_INVENTORY.md"


def _fmt(ts) -> str:
    return "-" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def landscape() -> None:
    print(f"{'key':<18}{'market':<8}{'role':<10}{'res':<26}{'files':>6}{'decl':>6}"
          f"{'miss':>6}  {'min':<11}{'max':<11}{'medspan':>8}")
    print("-" * 116)
    for store in REGISTRY.values():
        s = store.summary()
        print(f"{s['key']:<18}{s['market']:<8}{s['role']:<10}{s['res']:<26}"
              f"{s['files']:>6}{s['declared']:>6}{s['missing']:>6}  "
              f"{_fmt(s['min']):<11}{_fmt(s['max']):<11}{s['med_span_d']:>7.1f}d")
    print()
    for store in REGISTRY.values():
        print(f"  {store.key:<18} {store.description}")


def detail(key: str) -> None:
    store = get(key)
    cov = store.coverage().sort_values("span_d", ascending=False)
    print(f"{store.key}  ({store.market}/{store.role}/{store.resolution})  dir={store.store_dir}")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        show = cov.copy()
        show["min"] = show["min"].map(_fmt)
        show["max"] = show["max"].map(_fmt)
        show["span_d"] = show["span_d"].round(1)
        print(show.to_string(index=False))


def _md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return out


def write_doc(path: Path = DOC_PATH) -> None:
    now = pd.Timestamp.now(tz="UTC")
    L: list[str] = []
    L.append("# Candle Data Inventory")
    L.append("")
    L.append(f"_Auto-generated {now:%Y-%m-%d %H:%M} UTC — regenerate with "
             f"`python -m src.run_data_inventory --write`._")
    L.append("")
    L.append("> Single source of truth for what candle data exists on this machine. "
             "Stores are declared in `src/markets.py`. Tokenized equities live INSIDE "
             "the `crypto_feature` store (`data/candles`) mixed with crypto — the "
             "`EQUITY_TICKERS` set in `markets.py` is the canonical list.")
    L.append("")

    # ---- 1) store landscape ----
    L.append("## Stores")
    L.append("")
    rows = []
    for st in REGISTRY.values():
        exists = st.store_dir.exists()
        s = st.summary() if exists else None
        rows.append({
            "store": st.key, "market": st.market, "on_disk": "yes" if exists else "**ABSENT**",
            "files": s["files"] if s else 0,
            "min": _fmt(s["min"]) if s else "-", "max": _fmt(s["max"]) if s else "-",
            "med_span_d": s["med_span_d"] if s else 0.0,
            "dir": f"`{st.store_dir.relative_to(C.ROOT)}`",
        })
    L += _md_table(pd.DataFrame(rows),
                   ["store", "market", "on_disk", "files", "min", "max", "med_span_d", "dir"])
    L.append("")

    # ---- 2) crypto_feature breakdown (crypto vs tokenized equities) ----
    cf = REGISTRY["crypto_feature"]
    cov = cf.coverage()
    cov["is_eq"] = cov["symbol"].map(is_equity)
    eq = cov[cov["is_eq"]].copy()
    cr = cov[~cov["is_eq"]].copy()
    L.append("## `crypto_feature` (`data/candles`) — LIVE production store")
    L.append("")
    L.append(f"- total symbols on disk: **{len(cov)}**  "
             f"(crypto: {len(cr)}, tokenized equities: {len(eq)})")
    if len(cov):
        L.append(f"- freshness (max candle): newest `{_fmt(cov['max'].max())}`, "
                 f"oldest-tail `{_fmt(cov['max'].min())}` UTC")
    quar = (cf.store_dir / "_corrupt")
    if quar.exists():
        bad = sorted(p.name for p in quar.glob("*.parquet"))
        L.append(f"- ⚠️ quarantined/corrupt (in `_corrupt/`, need re-fetch): {bad or 'none'}")
    L.append("")

    # ---- 3) tokenized equities table (the precious, easily-lost set) ----
    L.append("### Tokenized equities / ETFs (in `data/candles`)")
    L.append("")
    if len(eq):
        eq["ticker"] = eq["symbol"].str.replace("_USDT_SWAP", "", regex=False)
        eq["start"] = eq["min"].map(_fmt)
        eq["end"] = eq["max"].map(_fmt)
        eq["span_d"] = eq["span_d"].round(1)
        eq = eq.sort_values("ticker")
        L += _md_table(eq, ["ticker", "rows", "start", "end", "span_d"])
    else:
        L.append("_none found_")
    L.append("")
    declared_missing = sorted(t for t in EQUITY_TICKERS
                              if not (cf.store_dir / f"{t}_USDT_SWAP.parquet").exists())
    if declared_missing:
        L.append(f"> Declared in `EQUITY_TICKERS` but NOT on disk: {declared_missing}")
        L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {path}  ({len(cov)} symbols, {len(eq)} equities)")


def main() -> None:
    args = sys.argv[1:]
    if "--write" in args:
        write_doc()
    elif args:
        detail(args[0])
    else:
        landscape()


if __name__ == "__main__":
    main()
