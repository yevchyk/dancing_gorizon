"""Central configuration: paths, curve shape, horizon and model specs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CANDLES_DIR = DATA_DIR / "candles"
DATASETS_DIR = DATA_DIR / "datasets"
CHUNKS_DIR = DATA_DIR / "chunks"
MODELS_DIR = ROOT / "models"
DIRECTION_MODELS_DIR = MODELS_DIR / "direction"
STABILITY_MODELS_DIR = MODELS_DIR / "stability"
OUTPUTS_DIR = ROOT / "outputs"
TEST_RESULTS_DIR = OUTPUTS_DIR / "test_results"
TRADING_LOGS_DIR = OUTPUTS_DIR / "trading_logs"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
CONFIGS_DIR = ROOT / "configs"

# --- Curve shape (feature engineering) ---
CURVE_POINTS = 300            # time points along the log curve
CURVE_METRICS = ("price_ratio",)   # relative price only; trees infer trend from lags
CURVE_COLUMNS = CURVE_POINTS * len(CURVE_METRICS)   # 300
CURVE_MIN_STEP_MIN = 5.0      # finest spacing near t=0 (OKX keeps 1m only ~17d)
CURVE_MAX_DEPTH_MIN = 60 * 24 * 60   # ~2 months lookback

# --- Dataset sampling ---
ANCHORS_PER_SYMBOL = 200
TRAIN_START_OFFSET_DAYS = 120   # -4 months
TRAIN_END_OFFSET_DAYS = 10      # exclude last 10 days (holdout)
HOLDOUT_DAYS = 10

# --- TRAIN/TEST DATE SPEC (never test a model on data it trained on) ---
# Two distinct regimes, do not confuse them:
#
# (A) PRODUCTION models in models/reg/ and models/dir_prob/ train on everything
#     BEFORE the cutoff below. Their ONLY valid (unseen) test window is the last
#     HOLDOUT_DAYS. Reporting them on anything earlier is in-sample / invalid.
DATA_SNAPSHOT = "2026-05-30"          # latest candle/anchor at training time
PROD_TRAIN_CUTOFF = "2026-05-20"      # = snapshot - HOLDOUT_DAYS; holdout starts here
PROD_HOLDOUT = ("2026-05-20", "2026-05-30")   # the only window prod models never saw
#
# (B) WALK-FORWARD (run_*_walkforward) does NOT use the production models. Each
#     fold trains a FRESH model strictly before its test slice, so its multi-week
#     span IS valid OOS -- it simulates "retrain every WF_TEST_DAYS, trade next
#     WF_TEST_DAYS". Use this for multi-window stats; use (A) for the single
#     freshest unseen check.
WF_TRAIN_DAYS = 90
WF_TEST_DAYS = 14
WF_FOLDS = 4

# --- Training ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Testing thresholds ---
PROB_THRESHOLDS = (0.70, 0.75, 0.80, 0.82, 0.85)
TEST_WINDOW_DAYS = 14   # anchor ~2 weeks back

# --- Trading engine ---
SCAN_INTERVAL_MIN = 5          # how often the live loop scans
COOLDOWN_MIN = 90              # min minutes between trades on the same symbol
MAX_CONCURRENT = 10           # max simultaneously open paper/live positions
TRADE_SIZE_USD = 5.0          # notional per directional entry
LIVE_WATCHLIST_SIZE = 80      # trade only the top-N most liquid coins (live loop)
LIVE_UPDATE_LOOKBACK_MIN = 180  # how much recent 1m history to refresh each scan
MAX_RISK_PER_TRADE = 0.01     # fraction of equity risked per trade (sizing)
DAILY_STOP_PCT = 0.10         # halt new entries after this daily drawdown
# Per-model probability cutoffs come from block-5 percentile analysis.
# Fallback if no tuned threshold is supplied for a model:
DEFAULT_SIGNAL_THRESHOLD = 0.80
# Don't trade directional signal if the stability model for that horizon
# fires above this (market expected to go flat -> touch unlikely).
STABILITY_VETO_THRESHOLD = 0.90
# Exit simulation: stop distance as a multiple of the model's move_pct.
# 1.0 = symmetric (target +move%, stop -move%). If neither hit within the
# horizon, exit at the horizon's closing price.
STOP_PCT_RATIO = 1.0
OKX_FEE_PER_SIDE = 0.05        # % taker fee per side (entry + exit)

# --- Strategy layer (on top of raw model signals) ---
# Regime filter: only take longs in an up regime, shorts in a down regime.
REGIME_LOOKBACK_MIN = 240      # trend reference window (4h)
REGIME_BAND = 0.0              # |trend| must exceed this to count as up/down
# Horizon agreement: require at least N same-direction models to fire together.
AGREEMENT_MIN = 2
# Models cleared for trading (from the clean-window analysis). None = all.
TRADEABLE_MODELS = ("up_15m", "up_30m", "up_1h", "down_5m")

# --- v4 high-confidence engine (the winning mechanic) ---
# Trade only very confident, CLEAN directional signals: the model's direction
# prob must clear SIGNAL_FLOOR while the opposite side stays below CLEAN_OPP_MAX
# (a quiet opposite = no whipsaw). Rank by the spread (p_dir - p_opp). Exit at
# the horizon; OCO left wide as a crash safety net only.
SIGNAL_FLOOR = 0.82
CLEAN_OPP_MAX = 0.30
CONF_TOP_PER_SCAN = 3
# Harvest works best on the fast 5m/15m models (catch quick green wiggles); the
# slower horizons touch green but only tiny. Keep only up/down 5m+15m.
CONF_EXCLUDE = ("up_30m", "down_30m", "up_1h", "down_1h", "up_2h", "down_2h")
# #4 require >=N horizons agreeing on a symbol/side (raises win-rate, cuts whipsaw).
CONF_MIN_AGREE = 2
# #3 size the position by spread (0.5x..2x of base) -- bet more on conviction.
CONF_SIZE_BY_SPREAD = True
# --- v5 green-harvest: every scan, close any open position that's in profit ---
GREEN_HARVEST = True        # close positions the moment they're green (net of cost)
HARVEST_COST_PCT = 0.15     # a position counts as green when pnl > this (fee+slip)
# Coins that fired confident signals but consistently lost (deception analysis,
# high-signal / robust cases). Skipped by the live trader for now.
BLACKLIST_SYMBOLS = ("BSB_USDT_SWAP", "UB_USDT_SWAP", "LAB_USDT_SWAP",
                     "TRUTH_USDT_SWAP",
                     # tokenized stocks (out-of-distribution for crypto models)
                     "NVDA_USDT_SWAP", "AMD_USDT_SWAP", "TSLA_USDT_SWAP",
                     "AAPL_USDT_SWAP", "MSFT_USDT_SWAP", "GOOGL_USDT_SWAP",
                     "META_USDT_SWAP", "AMZN_USDT_SWAP", "COIN_USDT_SWAP",
                     "MSTR_USDT_SWAP", "SPX_USDT_SWAP", "QQQ_USDT_SWAP",
                     "AMD_USDT_SWAP", "PLTR_USDT_SWAP", "HOOD_USDT_SWAP",
                     # toxic-everywhere on the fast_v2 worthy universe: negative
                     # per-signal edge in BOTH the 72h holdout and the last-24h
                     # slice with adequate sample (NOT BEAT/APR — those are good).
                     "WLD_USDT_SWAP", "GIGGLE_USDT_SWAP", "INJ_USDT_SWAP",
                     "JTO_USDT_SWAP", "EDEN_USDT_SWAP", "HYPE_USDT_SWAP",
                     "GRASS_USDT_SWAP")

# Extra HC-only blocklist for "Tantsiuiuchyi Horyzont".
# Built from the 2026-06-02..2026-06-05 72h HC trade simulations plus the live
# ZEC incident. Kept separate from BLACKLIST_SYMBOLS so older engines are not
# silently changed by HC-specific toxicity research.
HC_BLACKLIST_SYMBOLS = (
    "ACT_USDT_SWAP",
    "ALGO_USDT_SWAP",
    "APE_USDT_SWAP",
    "APR_USDT_SWAP",
    "APT_USDT_SWAP",
    "ASTER_USDT_SWAP",
    "BEAT_USDT_SWAP",
    "COAI_USDT_SWAP",
    "CORE_USDT_SWAP",
    "CRV_USDT_SWAP",
    "DASH_USDT_SWAP",
    "DOT_USDT_SWAP",
    "DYDX_USDT_SWAP",
    "EGLD_USDT_SWAP",
    "ETHFI_USDT_SWAP",
    "ETHW_USDT_SWAP",
    "FARTCOIN_USDT_SWAP",
    "GMT_USDT_SWAP",
    "ICP_USDT_SWAP",
    "IP_USDT_SWAP",
    "LAYER_USDT_SWAP",
    "LDO_USDT_SWAP",
    "LINK_USDT_SWAP",
    "MERL_USDT_SWAP",
    "MET_USDT_SWAP",
    "NEAR_USDT_SWAP",
    "ONDO_USDT_SWAP",
    "OP_USDT_SWAP",
    "PENDLE_USDT_SWAP",
    "RENDER_USDT_SWAP",
    "STRK_USDT_SWAP",
    "SUI_USDT_SWAP",
    "TAO_USDT_SWAP",
    "TIA_USDT_SWAP",
    "TON_USDT_SWAP",
    "TRUMP_USDT_SWAP",
    "UMA_USDT_SWAP",
    "VIRTUAL_USDT_SWAP",
    "WIF_USDT_SWAP",
    "WLFI_USDT_SWAP",
    "XLM_USDT_SWAP",
    "ZEC_USDT_SWAP",
    "ZEN_USDT_SWAP",
)


def hc_blacklist_symbols() -> set[str]:
    """Symbols excluded from HC live/research trading universes."""
    return set(BLACKLIST_SYMBOLS) | set(HC_BLACKLIST_SYMBOLS)


@dataclass(frozen=True)
class HorizonSpec:
    """One forecast horizon."""
    minutes: int
    label: str
    move_pct: float          # directional target threshold (e.g. 0.02 = +2%)
    stability_range: float   # max path range to count as "stable"
    lookback_min: float      # how far back the model's curve input reaches


HORIZONS = (
    HorizonSpec(5,   "5m",  0.020, 0.04, 1 * 24 * 60),    # 1 day of curve
    HorizonSpec(15,  "15m", 0.025, 0.05, 3 * 24 * 60),    # 3 days
    HorizonSpec(30,  "30m", 0.030, 0.06, 7 * 24 * 60),    # 7 days
    HorizonSpec(60,  "1h",  0.040, 0.08, 30 * 24 * 60),   # 30 days
    HorizonSpec(120, "2h",  0.050, 0.08, 60 * 24 * 60),   # full curve (~2 months)
)

DIRECTIONS = ("up", "down")
