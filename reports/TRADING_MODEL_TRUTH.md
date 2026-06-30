# Trading Model Truth

## Idea
Horizon-conditioned multi-timeframe CatBoost. Two separate models: UP, DOWN.

## Universe
218 OKX crypto perps (`crypto_feature`, >=200d of 5m, blacklist removed).
File: `configs/hc_universe.json`

## Data window
5m era source: 2025-09-27 -> 2026-06-03 13:25 UTC in the current local store.
Hard ceiling: older tiers are 1h/1d only and are not used as fake 5m.

Current caveat: strict BTC/4h alignment rejects the recent BTC gaps, so the smoke
HC dataset's valid snapshots stop at 2026-05-26 20:00 UTC even though source candles
continue later. Do not call a full fold "latest live-like" until BTC gaps are fixed
or the owner approves a different missing-data rule.

## Timeframes & features
5m : rel_coin, vol_ratio
15m: rel_coin, vol_ratio
1h : rel_coin, rel_btc, vol_ratio
4h : rel_coin, rel_btc, vol_ratio
N_POINTS=30 per TF -> 302 columns including `horizon_minutes`, `horizon_log`.

## Horizons
Anchors: 5/15/30/60/120/180.
Full run adds 2 random 5m-aligned horizons per snapshot in [5, 180].

## Target (close-at-horizon)
up=1 if ret>=thr(h); down=1 if ret<=-thr(h); else dead zone.
thr%: 5=0.4 15=0.6 30=0.8 60=1.1 120=1.5 180=1.8
weight = 1 + min(|ret%|/3,1)*4 (cap at 3%).

## Split
Walk-forward 3 folds: last 7d + 2 BTC-regime stress weeks. Embargo: 180 min.

## Output
`up_prob`, `down_prob` -> long/short/skip/no-trade.

## Results
Full run is pending. Do not fill this section with smoke or guessed numbers.

fold1 / fold2 / fold3: AUC up=__ down=__ ; win@0.70=__ ; net avg ret=__
best horizon: __   worst: __   best symbols: __   failed: __
