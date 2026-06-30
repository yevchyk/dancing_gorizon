"""Live adapter for the fast_bluechip models (top-120 liquid OKX crypto perps).

Mirrors the Krykun adapter but points at the bluechip experiment. Two profiles:
  bluechip_unicorn : AGREEMENT long-only noVeto, >=N up-models agree, exit 100m. $60
  bluechip         : FLAT POOL of the high-win zones (own-horizon exits). $30

WARNING: bluechip is bull-beta and bleeds in crashes (no regime gate yet). The
short stop-loss does NOT help (crypto volatility whipsaws it). Use a regime gate
before real money. Engine here is for sim/paper analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .. import config as C
from ..database import CandleStore
from ..fast.curve import FastCurve
from .timeutil import index_to_ns

TAG = "bluechip"
CFG_PATH = C.DATA_DIR / "fast_bluechip" / "datasets" / f"config_{TAG}.json"
MODELS_DIR = C.MODELS_DIR / "fast_bluechip" / TAG
STORE_DIR = C.DATA_DIR / "bluechip" / "candles_1m"
SYMBOLS_FILE = C.ROOT / "configs" / "bluechip_symbols.json"
SAFETY_OCO_PCT = 0.05

PROFILES = {
    "bluechip_unicorn": {
        "kind": "agreement", "notional_usd": 60.0,
        "agreement_threshold": 0.85, "min_agree": 3, "exit_horizon": "100m",
    },
    "bluechip": {
        "kind": "flat_pool", "notional_usd": 30.0,
        "up_thresholds": {"18m": 0.90, "24m": 0.92, "32m": 0.90, "100m": 0.75},
    },
}
STACKS = {"bluechip_pair": ["bluechip_unicorn", "bluechip"]}


@dataclass
class BlueSignal:
    symbol: str; model: str; side: str; horizon: str; move_pct: float
    prob: float; score: float; agree: int; size_usd: float; source: str; engine: str = ""


class BluechipEngine:
    def __init__(self, profile: str = "bluechip_unicorn") -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown bluechip profile: {profile}")
        self.profile = profile
        self.cfg = PROFILES[profile]
        meta = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        offsets = tuple(float(x) for x in meta["curve_offsets_min"])
        self.curve = FastCurve(len(offsets), 2.0, max(offsets), offsets_min=offsets)
        self.columns = self.curve.columns()
        self.labels = [h["label"] for h in meta["horizons"]]
        self.h_minutes = {h["label"]: int(h["minutes"]) for h in meta["horizons"]}
        self._models = {}
        for lab in self.labels:
            for side in ("up", "down"):
                name = f"{side}_{lab}"
                self._models[name] = (joblib.load(MODELS_DIR / f"{name}.joblib"),
                                      joblib.load(MODELS_DIR / f"{name}_columns.joblib"))

    def describe(self) -> str:
        if self.cfg["kind"] == "flat_pool":
            th = ", ".join(f"up_{k}>={v}" for k, v in self.cfg["up_thresholds"].items())
            return f"bluechip[{self.profile}] flat long pool; {th}; ${self.cfg['notional_usd']:.0f}"
        return (f"bluechip[{self.profile}] agreement>={self.cfg['min_agree']}@"
                f"{self.cfg['agreement_threshold']}; exit={self.cfg['exit_horizon']}; "
                f"${self.cfg['notional_usd']:.0f}")

    def build_watchlist(self, store: CandleStore, logger=None) -> list[str]:
        data = json.loads(SYMBOLS_FILE.read_text(encoding="utf-8"))
        syms = data.get("symbols", data) if isinstance(data, dict) else data
        watch = [s for s in syms if store.load(s) is not None and not store.load(s).empty]
        if logger:
            logger.event(f"bluechip watchlist: {len(watch)} crypto symbols")
        return watch

    def snapshot(self, store: CandleStore, symbols: list[str], now: pd.Timestamp) -> pd.DataFrame:
        anchor = pd.Timestamp(now); anchor_ns = np.array([int(anchor.value)], dtype="int64")
        rows = []
        for sym in symbols:
            c = store.load(sym)
            if c is None or c.empty:
                continue
            c = c.sort_index(); ts_ns = index_to_ns(c.index); close = c["close"].to_numpy("float64")
            feats, valid = self.curve.build_matrix(ts_ns, close, anchor_ns)
            if not bool(valid[0]):
                continue
            ei = int(np.searchsorted(ts_ns, anchor_ns[0], side="right")) - 1
            if ei < 0 or c.index[ei] < anchor - pd.Timedelta(minutes=2):
                continue
            entry = float(close[ei])
            if not np.isfinite(entry) or entry <= 0:
                continue
            row = {"symbol": sym, "anchor_time": anchor, "entry_price": entry}
            row.update({col: float(feats[0, i]) for i, col in enumerate(self.columns)})
            rows.append(row)
        return pd.DataFrame(rows)

    def _score(self, feat: pd.DataFrame) -> pd.DataFrame:
        out = feat[["symbol", "anchor_time", "entry_price"]].copy()
        for name, (model, cols) in self._models.items():
            out[f"p_{name}"] = model.predict_proba(feat[cols])[:, 1]
        return out

    def decide(self, feat: pd.DataFrame, top_n: int = 3) -> list[BlueSignal]:
        if feat.empty:
            return []
        scored = self._score(feat)
        sigs = []
        notional = float(self.cfg["notional_usd"])
        for row in scored.itertuples(index=False):
            if self.cfg["kind"] == "flat_pool":
                for lab, thr in self.cfg["up_thresholds"].items():
                    p = float(getattr(row, f"p_up_{lab}"))
                    if p >= thr:
                        sigs.append(BlueSignal(row.symbol, f"bluechip_up_{lab}_p{thr}", "long",
                                               lab, SAFETY_OCO_PCT, p, 20 + (p - thr), 1,
                                               notional, "bluechip", self.profile))
            else:
                thr = float(self.cfg["agreement_threshold"])
                up = sum(1 for lab in self.labels if float(getattr(row, f"p_up_{lab}")) >= thr)
                if up >= int(self.cfg["min_agree"]):
                    ex = str(self.cfg["exit_horizon"])
                    pmax = max(float(getattr(row, f"p_up_{lab}")) for lab in self.labels)
                    sigs.append(BlueSignal(row.symbol, f"bluechip_long_agree{up}_exit{ex}",
                                           "long", ex, SAFETY_OCO_PCT, pmax, 100 + up, up,
                                           notional, "bluechip_unicorn", self.profile))
        sigs.sort(key=lambda x: x.score, reverse=True)
        picked, used = [], set()
        for x in sigs:
            if x.symbol in used:
                continue
            picked.append(x); used.add(x.symbol)
            if len(picked) >= top_n:
                break
        return picked


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for prof in ("bluechip_unicorn", "bluechip"):
        e = BluechipEngine(prof)
        print(f"OK {prof}: {len(e._models)} models, horizons {e.labels}")
        print("    ", e.describe())
