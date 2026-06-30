"""Single source of truth for candle data stores.

The repo accumulated several overlapping candle stores + symbol lists, which is
easy to confuse (e.g. "nasdaq" is really 22 OKX stock/index perps, NOT the real
Nasdaq-100; the "100 positions" live in okx_liquid, a mixed tradfi+crypto list).

This module names every store ONCE with its market / role / resolution / symbol
source, and gives a uniform API to inspect coverage. Inspect, don't guess.

CLI:  python -m src.run_data_inventory            # landscape table
      python -m src.run_data_inventory liquid_mixed  # per-symbol coverage
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config as C


@dataclass(frozen=True)
class Store:
    key: str            # stable handle used in code
    market: str         # crypto | tradfi | mixed
    role: str           # feature | target | candidate
    resolution: str     # 1m | mixed | 5m/1h/1d
    store_dir: Path
    symbols_file: Path | None
    description: str

    # --- listing -----------------------------------------------------------
    def files(self) -> list[Path]:
        return sorted(self.store_dir.glob("*.parquet")) if self.store_dir.exists() else []

    def symbols_on_disk(self) -> list[str]:
        return [p.stem for p in self.files()]

    def declared_symbols(self) -> list[str]:
        if not self.symbols_file or not self.symbols_file.exists():
            return []
        d = json.loads(self.symbols_file.read_text(encoding="utf-8"))
        return d.get("symbols", d) if isinstance(d, dict) else d

    # --- data --------------------------------------------------------------
    def load(self, symbol: str) -> pd.DataFrame | None:
        p = self.store_dir / f"{symbol}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()

    # --- inspection --------------------------------------------------------
    def coverage(self) -> pd.DataFrame:
        rows = []
        for p in self.files():
            try:
                ts = pd.to_datetime(pd.read_parquet(p, columns=["timestamp"])["timestamp"], utc=True)
                rows.append({"symbol": p.stem, "rows": int(len(ts)),
                             "min": ts.min(), "max": ts.max(),
                             "span_d": (ts.max() - ts.min()).total_seconds() / 86400})
            except Exception as exc:
                rows.append({"symbol": p.stem, "rows": 0, "min": pd.NaT, "max": pd.NaT,
                             "span_d": 0.0, "error": str(exc)})
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        cov = self.coverage()
        declared = self.declared_symbols()
        on_disk = set(self.symbols_on_disk())
        missing = [s for s in declared if s not in on_disk] if declared else []
        return {
            "key": self.key, "market": self.market, "role": self.role, "res": self.resolution,
            "files": len(cov), "declared": len(declared), "missing": len(missing),
            "min": cov["min"].min() if len(cov) else pd.NaT,
            "max": cov["max"].max() if len(cov) else pd.NaT,
            "med_span_d": round(float(cov["span_d"].median()), 1) if len(cov) else 0.0,
        }


CONFIGS = C.ROOT / "configs"

REGISTRY: dict[str, Store] = {
    "crypto_feature": Store(
        "crypto_feature", "crypto", "feature", "5m/1h/1d",
        C.CANDLES_DIR, None,
        "Production multi-res feature store. Feeds LIVE trading + the fast curve features.",
    ),
    "crypto_target_1m": Store(
        "crypto_target_1m", "crypto", "target", "1m",
        C.DATA_DIR / "fast_v1" / "candles_1m", None,
        "Rolling 1m target store for fast_* training. ROLLS FORWARD (live overwrites it) "
        "-> always `run_fast_v3 --stage refetch` before recomputing targets.",
    ),
    "tradfi_1m": Store(
        "tradfi_1m", "tradfi", "target", "1m",
        C.DATA_DIR / "nasdaq" / "okx_candles_1m", CONFIGS / "nasdaq_symbols.json",
        "OKX stock/index perps (AAPL/NVDA/QQQ/SPX/...). NOT the real Nasdaq-100. "
        "69 syms after the 2026-06-03 backfill. Source for the fast_nasdaq experiment.",
    ),
    "tradfi_1m_legacy": Store(
        "tradfi_1m_legacy", "tradfi", "target", "1m",
        C.DATA_DIR / "nasdaq" / "candles_1m", None,
        "Older tradfi 1m (bare ticker filenames, e.g. AAPL.parquet). Superseded by tradfi_1m.",
    ),
    "liquid_mixed": Store(
        "liquid_mixed", "mixed", "candidate", "mixed(1m7d/5m240d/1H730d/1D1460d)",
        C.DATA_DIR / "okx_liquid" / "candles_mixed", CONFIGS / "okx_liquid_symbols_100.json",
        "The '100 positions': 100 liquid OKX instruments (tradfi + crypto mixed). "
        "Built by run_okx_liquid_backfill. Deep history, mixed resolution.",
    ),
    "liquid_1m": Store(
        "liquid_1m", "mixed", "candidate", "1m",
        C.DATA_DIR / "okx_liquid" / "candles_1m", CONFIGS / "okx_liquid_symbols_100.json",
        "okx_liquid 1m slice (short, clean 1m). Companion to liquid_mixed.",
    ),
    "okx_stable_200": Store(
        "okx_stable_200", "mixed", "candidate", "mixed(1m7d/5m240d/1H730d/1D1460d)",
        C.DATA_DIR / "okx_stable" / "candles_mixed", CONFIGS / "okx_stable_200.json",
        "Separate long-history OKX universe: only live USDT swaps with enough OKX-native "
        "history, ranked to 200. Built by run_okx_stable200_build "
        "and backfilled by run_okx_stable200_backfill.",
    ),
    "bluechip": Store(
        "bluechip", "crypto", "target", "1m",
        C.DATA_DIR / "bluechip" / "candles_1m", CONFIGS / "bluechip_symbols.json",
        "Top-120 liquid OKX crypto perps by 24h volume (age>=150d, toxics+equity excluded). "
        "Deep 1m (~200d backfill 2026-06-04) for a 2-month-train crypto model. NOT only blue-chips.",
    ),
}


def get(key: str) -> Store:
    if key not in REGISTRY:
        raise KeyError(f"unknown store '{key}'. Known: {', '.join(REGISTRY)}")
    return REGISTRY[key]


# Tokenized US equities / ETFs that live INSIDE the crypto_feature store
# (data/candles) mixed with the crypto perps. Listed here as the source of truth
# so we never "lose" them again. All stored as OKX `<TICKER>_USDT_SWAP` perps.
EQUITY_TICKERS: frozenset[str] = frozenset({
    "AAPL", "ADBE", "AMAT", "AMD", "AMZN", "ANTHROPIC", "ARM", "ASML", "AVGO",
    "COIN", "COST", "CRCL", "CRWD", "CSCO", "GOOGL", "HOOD", "INTC", "IWM",
    "META", "MRVL", "MSFT", "MSTR", "MU", "NFLX", "NOW", "NVDA", "ORCL",
    "PLTR", "QCOM", "QQQ", "SPY", "TSLA", "TSM",
})


def is_equity(symbol: str) -> bool:
    """True if a store symbol like 'NVDA_USDT_SWAP' is a tokenized equity/ETF."""
    return symbol.split("_")[0] in EQUITY_TICKERS
