# HC Toxic Block 2026-06-05

System: `Tantsiuiuchyi_Horyzont` / HC live.

## Change

Added an HC-only blacklist in `src/config.py`:

- `HC_BLACKLIST_SYMBOLS`
- `hc_blacklist_symbols() = BLACKLIST_SYMBOLS union HC_BLACKLIST_SYMBOLS`

The HC blacklist is used by:

- `src/trading/hc_live_engine.py`
- `src/run_fetcher.py --universe hc`
- `src/run_hc_live_hourly_report.py`
- `src/run_hc_live_threshold_sweep.py`
- `src/run_hc_threshold_live_sim.py`
- `src/run_hc_offgrid_sim.py`

This is intentionally not merged into the base `BLACKLIST_SYMBOLS`, so older
non-HC engines are not silently changed.

## Blocked HC Symbols

`ZEC_USDT_SWAP` is explicitly blocked after the live loss on 2026-06-05.

The rest were selected from 72h HC trade simulations where the symbol showed
negative aggregate edge and/or repeated poor winrate across modes:

`ACT_USDT_SWAP`, `ALGO_USDT_SWAP`, `APE_USDT_SWAP`, `APR_USDT_SWAP`,
`APT_USDT_SWAP`, `ASTER_USDT_SWAP`, `BEAT_USDT_SWAP`, `COAI_USDT_SWAP`,
`CORE_USDT_SWAP`, `CRV_USDT_SWAP`, `DASH_USDT_SWAP`, `DOT_USDT_SWAP`,
`DYDX_USDT_SWAP`, `EGLD_USDT_SWAP`, `ETHFI_USDT_SWAP`, `ETHW_USDT_SWAP`,
`FARTCOIN_USDT_SWAP`, `GMT_USDT_SWAP`, `ICP_USDT_SWAP`, `IP_USDT_SWAP`,
`LAYER_USDT_SWAP`, `LDO_USDT_SWAP`, `LINK_USDT_SWAP`, `MERL_USDT_SWAP`,
`MET_USDT_SWAP`, `NEAR_USDT_SWAP`, `ONDO_USDT_SWAP`, `OP_USDT_SWAP`,
`PENDLE_USDT_SWAP`, `RENDER_USDT_SWAP`, `STRK_USDT_SWAP`, `SUI_USDT_SWAP`,
`TAO_USDT_SWAP`, `TIA_USDT_SWAP`, `TON_USDT_SWAP`, `TRUMP_USDT_SWAP`,
`UMA_USDT_SWAP`, `VIRTUAL_USDT_SWAP`, `WIF_USDT_SWAP`, `WLFI_USDT_SWAP`,
`XLM_USDT_SWAP`, `ZEN_USDT_SWAP`.

Universe effect after the patch:

- raw HC universe: 218 symbols
- HC-blocked in universe: 43 symbols
- tradable HC universe: 175 symbols

## Checks

Compilation passed for the touched modules.

Post-block 24h live-like sim, ending `2026-06-05 12:00 Europe/Kiev`, with
`threshold_shift=-0.02`, `opp_cap=0.20`, `top_per_scan=50`, `max_open=6`,
`stake_margin=8`, `leverage=5`, `conviction=on`:

- trades: 23
- winrate: 100%
- avg net per trade: +3.23%
- PnL: +$44.34
- max drawdown: $0.00
- output: `outputs/analysis/hc_live_hourly/to_20260605_1200_kyiv_toxicblocked_h24`

Important caveat: those 23 trades were all on 2026-06-04 12:00-18:00 Kyiv.
The last 12h window after that did not have enough strict high-quality signals.

Threshold sweep after the toxic block over the last 12h found no configuration
with `win >= 72%` and at least 4 trades. Lowering thresholds increased count,
but winrate collapsed to roughly 30-50%, so frequency alone is not a valid fix
for that window.

Output:

`outputs/analysis/hc_live_threshold_sweep/to_20260605_1200_kyiv_toxicblocked`
