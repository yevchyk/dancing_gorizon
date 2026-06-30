"""Parity gate: live feature build (run_binance_live path) must equal the
binance_y1 dataset features bit-for-bit, and the live engine's probabilities
must equal scoring the dataset rows directly. 4 symbols x 2 anchors each.
Run for every model family BEFORE any testnet/live flip:

  python -m src.run_binance_parity_check --model-dir models/binance_y1_d10
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# same store patch as run_binance_live / run_binance_dataset
from src.markets import REGISTRY, Store
from src import config as C
from src.hc import config as HC

REGISTRY["binance_feature"] = Store(
    "binance_feature", "crypto", "feature", "1m",
    C.ROOT / "data/binance/candles", C.ROOT / "configs/binance_train_universe.json",
    "parity check")
HC.STORE_KEY = "binance_feature"
HC.HC_ERA_START = pd.Timestamp("2025-06-01T00:00:00Z")

from src.database import CandleStore  # noqa: E402
from src.hc.data import prepare_btc_frames, prepare_timeframes  # noqa: E402
from src.hc.data_v3 import _build_feature_matrix_v3, prepare_1m  # noqa: E402
from src.hc import schema_v3 as S3  # noqa: E402
from src.trading.hc_v4_live_engine import HCV4LiveEngine  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--model-dir", type=Path, default=Path("models/binance_y1_d10"))
_ap.add_argument("--dataset", type=Path, default=Path("data/binance_y1/dataset"))
_args = _ap.parse_args()

DATASET = _args.dataset
MODEL = _args.model_dir
SYMS = ["BTC_USDT_SWAP", "ETH_USDT_SWAP", "SOL_USDT_SWAP", "DOGE_USDT_SWAP"]
FEATS = json.loads((MODEL / "feature_names.json").read_text(encoding="utf-8"))

store = CandleStore(Path("data/binance/candles"))
btc = prepare_btc_frames()
engine = HCV4LiveEngine(model_dir=MODEL, high=0.0, horizons=(30, 120, 480),
                        universe_path=Path("configs/binance_universe_trade.json"))

worst_feat = 0.0
worst_prob = 0.0
checked = 0
for sym in SYMS:
    shard = pd.read_parquet(DATASET / f"{sym}.parquet")
    shard["base_time"] = pd.to_datetime(shard["base_time"], utc=True)
    bts = sorted(shard["base_time"].unique())
    for bt in (bts[-2], bts[len(bts) // 2]):
        rows = shard[shard["base_time"] == bt]
        # (a) feature parity: rebuild the curve at this anchor the LIVE way
        candles = store.load(sym)
        anchors = pd.DatetimeIndex([bt])
        prepared = prepare_timeframes(candles, btc)
        p1m = prepare_1m(candles)
        mat, valid = _build_feature_matrix_v3(anchors, prepared, p1m, HC.N_POINTS)
        assert bool(valid[0]), f"{sym} @ {bt}: live matrix invalid"
        live_curve = mat[0].astype("float32")
        ds_curve = rows.iloc[0][S3.CURVE_COLUMNS_V3].to_numpy(dtype="float32")
        df_feat = float(np.max(np.abs(live_curve - ds_curve)))
        worst_feat = max(worst_feat, df_feat)

        # (b) end-to-end: engine snapshot probs vs direct scoring of dataset rows
        snap = engine.snapshot(store, [sym], now=pd.Timestamp(bt) + pd.Timedelta(minutes=5))
        assert not snap.empty, f"{sym} @ {bt}: empty live snapshot"
        for h in (30, 120, 480):
            ds_row = rows[rows["horizon_minutes"] == h]
            if ds_row.empty:
                continue
            X = ds_row[FEATS]
            up = np.mean([u.predict_proba(X)[:, 1] for u, _ in engine._folds], axis=0)[0]
            dn = np.mean([d.predict_proba(X)[:, 1] for _, d in engine._folds], axis=0)[0]
            live = snap[snap["horizon_minutes"] == h].iloc[0]
            dp = max(abs(float(live["up_prob"]) - up), abs(float(live["down_prob"]) - dn))
            worst_prob = max(worst_prob, float(dp))
            checked += 1
    print(f"{sym}: feat_maxdiff={worst_feat:.3e} prob_maxdiff={worst_prob:.3e}")

print(f"\nchecked {checked} (symbol,anchor,horizon) points")
print(f"WORST feature diff: {worst_feat:.3e}")
print(f"WORST prob diff:    {worst_prob:.3e}")
if worst_feat == 0.0 and worst_prob < 1e-6:
    print("PARITY OK — live build == dataset, bit-for-bit")
else:
    print("PARITY FAILED — investigate before any run")
