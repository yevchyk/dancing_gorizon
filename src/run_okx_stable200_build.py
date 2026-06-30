"""Build a separate OKX stable-200 universe from live exchange metadata.

The goal is not "top 200 crypto". It is a calmer production universe with
enough exchange-native history for training: filter short-history contracts
first, then rank the remaining OKX USDT swaps. The output is a config consumed by
run_okx_stable200_backfill and can be rebuilt when OKX listings change.

Example:
  python -m src.run_okx_stable200_build
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config as C
from .database import OKXClient


DEFAULT_OUT = C.CONFIGS_DIR / "okx_stable_200.json"
DEFAULT_CSV = C.CONFIGS_DIR / "okx_stable_200.csv"
DEFAULT_STORE_DIR = C.DATA_DIR / "okx_stable" / "candles_mixed"

ASSET_CLASS_BY_INST_CATEGORY = {
    "1": "crypto",
    "3": "tradfi",
    "4": "commodity",
}

ETF_INDEX_BASES = {
    "EWT",
    "EWJ",
    "EWY",
    "INFQ",
    "IWM",
    "QQQ",
    "SOXL",
    "SPX",
    "SPY",
    "SPCX",
    "URNM",
    "USAR",
    "USO",
    "XLE",
}

CRYPTO_BLUECHIP_BASES = {
    "AAVE",
    "ADA",
    "ATOM",
    "AVAX",
    "BCH",
    "BNB",
    "BTC",
    "ETC",
    "ETH",
    "FIL",
    "HBAR",
    "LTC",
    "OKB",
    "SOL",
    "TRX",
    "UNI",
    "XRP",
}

# Event/meme/stablecoin-like symbols that usually add noise instead of calmer
# cross-market structure. This list is deliberately conservative and only
# applies to instCategory=1 crypto; OKX stock/commodity swaps are never removed
# by it.
DEFAULT_EXCLUDED_CRYPTO_BASES = {
    "BABY",
    "BOME",
    "BONK",
    "BRETT",
    "DOGE",
    "FARTCOIN",
    "FLOKI",
    "GIGGLE",
    "HMSTR",
    "JELLYJELLY",
    "MEME",
    "MEW",
    "MOODENG",
    "MUBARAK",
    "NEIRO",
    "PEPE",
    "PIPPIN",
    "PNUT",
    "POPCAT",
    "SATS",
    "SHIB",
    "STABLE",
    "TRUMP",
    "TURBO",
    "USDC",
    "USELESS",
    "WIF",
}


@dataclass(frozen=True)
class Candidate:
    rank: int
    symbol: str
    inst_id: str
    base: str
    asset_class: str
    inst_category: str
    age_days: float
    turnover_24h: float
    range_24h_pct: float
    abs_return_24h_pct: float
    score: float
    selected_reason: str


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _age_days(list_time_ms: str, now: datetime) -> float:
    try:
        listed = datetime.fromtimestamp(int(list_time_ms) / 1000.0, tz=UTC)
    except (TypeError, ValueError, OSError):
        return 0.0
    return max(0.0, (now - listed).total_seconds() / 86400.0)


def _inst_symbol(inst_id: str) -> str:
    return inst_id.upper().replace("-", "_")


def _base(inst_id: str) -> str:
    return inst_id.upper().replace("-USDT-SWAP", "")


def _asset_class(base: str, inst_category: str) -> str:
    if inst_category == "4":
        return "commodity"
    if inst_category == "3":
        return "etf_index" if base in ETF_INDEX_BASES else "tradfi"
    return ASSET_CLASS_BY_INST_CATEGORY.get(inst_category, "other")


def _ticker_metrics(ticker: dict[str, Any]) -> tuple[float, float, float]:
    last = _num(ticker.get("last"))
    open_24h = _num(ticker.get("open24h"))
    high_24h = _num(ticker.get("high24h"))
    low_24h = _num(ticker.get("low24h"))
    vol_ccy = _num(ticker.get("volCcy24h"))
    turnover = last * vol_ccy if last > 0 and vol_ccy > 0 else 0.0
    range_pct = ((high_24h - low_24h) / last * 100.0) if last > 0 and high_24h >= low_24h else 0.0
    abs_ret_pct = abs(last / open_24h - 1.0) * 100.0 if last > 0 and open_24h > 0 else 0.0
    return turnover, range_pct, abs_ret_pct


def _score(inst_category: str, base: str, age_days: float, turnover: float,
           range_pct: float, abs_ret_pct: float) -> float:
    liquidity = math.log10(max(turnover, 1.0))
    age = math.log10(max(age_days, 1.0))
    calm = max(0.0, 1.0 - min(range_pct, 30.0) / 30.0)
    drift = max(0.0, 1.0 - min(abs_ret_pct, 20.0) / 20.0)

    if inst_category == "3":
        class_bonus = 4.0
    elif inst_category == "4":
        class_bonus = 3.5
    elif base in CRYPTO_BLUECHIP_BASES:
        class_bonus = 1.0
    else:
        class_bonus = 0.0

    return round(class_bonus + liquidity * 1.8 + age * 1.2 + calm * 1.4 + drift * 0.7, 6)


def _load_live_candidates(client: OKXClient, now: datetime) -> list[dict[str, Any]]:
    instruments = client.get_json("/api/v5/public/instruments", {"instType": "SWAP"}).get("data", [])
    tickers = {
        str(t.get("instId")): t
        for t in client.get_json("/api/v5/market/tickers", {"instType": "SWAP"}).get("data", [])
    }

    rows: list[dict[str, Any]] = []
    for inst in instruments:
        inst_id = str(inst.get("instId", "")).upper()
        if not inst_id.endswith("-USDT-SWAP"):
            continue
        if inst.get("state") != "live" or inst.get("settleCcy") != "USDT":
            continue

        inst_category = str(inst.get("instCategory", ""))
        if inst_category not in ASSET_CLASS_BY_INST_CATEGORY:
            continue

        base = _base(inst_id)
        turnover, range_pct, abs_ret_pct = _ticker_metrics(tickers.get(inst_id, {}))
        age_days = _age_days(str(inst.get("listTime", "")), now)
        rows.append({
            "inst_id": inst_id,
            "symbol": _inst_symbol(inst_id),
            "base": base,
            "inst_category": inst_category,
            "asset_class": _asset_class(base, inst_category),
            "age_days": age_days,
            "turnover_24h": turnover,
            "range_24h_pct": range_pct,
            "abs_return_24h_pct": abs_ret_pct,
            "score": _score(inst_category, base, age_days, turnover, range_pct, abs_ret_pct),
        })
    return rows


def _crypto_blocklist(*, include_hc_blacklist: bool, allow_meme_crypto: bool) -> set[str]:
    blocked: set[str] = set()
    if not allow_meme_crypto:
        blocked.update(DEFAULT_EXCLUDED_CRYPTO_BASES)
    if include_hc_blacklist:
        blocked.update(
            sym.replace("_USDT_SWAP", "")
            for sym in C.hc_blacklist_symbols()
            if sym.endswith("_USDT_SWAP")
        )
    return blocked


def build_universe(args: argparse.Namespace) -> dict[str, Any]:
    now = datetime.now(tz=UTC)
    all_candidates = _load_live_candidates(OKXClient(timeout=args.timeout), now)
    short_history = [
        c for c in all_candidates
        if c["age_days"] < args.min_history_age_days
    ]
    candidates = [
        c for c in all_candidates
        if c["age_days"] >= args.min_history_age_days
    ]
    blocked_crypto = _crypto_blocklist(
        include_hc_blacklist=args.apply_hc_blacklist,
        allow_meme_crypto=args.allow_meme_crypto,
    )

    noncrypto = [
        c for c in candidates
        if c["inst_category"] in {"3", "4"}
        and c["turnover_24h"] >= args.min_noncrypto_turnover
    ]
    noncrypto.sort(key=lambda c: (
        0 if c["asset_class"] in {"tradfi", "etf_index"} else 1,
        -float(c["turnover_24h"]),
        c["symbol"],
    ))

    crypto_rejected: list[dict[str, Any]] = []
    crypto: list[dict[str, Any]] = []
    crypto_reserve: list[dict[str, Any]] = []
    for c in candidates:
        if c["inst_category"] != "1":
            continue
        reason = ""
        if c["base"] in blocked_crypto:
            reason = "blocked_crypto"
        elif c["turnover_24h"] < args.min_crypto_turnover:
            reason = "turnover_lt_min"
        if reason:
            item = dict(c)
            item["reject_reason"] = reason
            crypto_rejected.append(item)
            # If OKX does not have enough long-history, non-meme symbols to
            # reach 200, fall back to long-history blocked crypto. History
            # depth is the hard requirement for this training universe.
            if reason == "blocked_crypto" and c["turnover_24h"] >= args.min_crypto_turnover:
                reserve = dict(c)
                reserve["reserve_reason"] = reason
                crypto_reserve.append(reserve)
            continue
        crypto.append(c)

    crypto.sort(key=lambda c: (-float(c["score"]), -float(c["turnover_24h"]), c["symbol"]))
    crypto_reserve.sort(key=lambda c: (-float(c["score"]), -float(c["turnover_24h"]), c["symbol"]))

    needed_crypto = max(0, int(args.target_count) - len(noncrypto))
    selected_raw = noncrypto + crypto[:needed_crypto]
    if len(selected_raw) < args.target_count:
        selected_raw.extend(crypto_reserve[: int(args.target_count) - len(selected_raw)])
    if len(selected_raw) < args.target_count:
        raise SystemExit(
            f"Only {len(selected_raw)} symbols selected for target={args.target_count}. "
            "Lower --min-history-age-days/--min-crypto-turnover or use --allow-meme-crypto."
        )

    selected: list[Candidate] = []
    for rank, c in enumerate(selected_raw[:args.target_count], 1):
        if c["inst_category"] in {"3", "4"}:
            reason = "long_history_okx_tradfi_or_commodity"
        elif "reserve_reason" in c:
            reason = f"long_history_crypto_reserve_{c['reserve_reason']}"
        else:
            reason = "long_history_liquid_crypto_fill"
        selected.append(Candidate(
            rank=rank,
            symbol=str(c["symbol"]),
            inst_id=str(c["inst_id"]),
            base=str(c["base"]),
            asset_class=str(c["asset_class"]),
            inst_category=str(c["inst_category"]),
            age_days=round(float(c["age_days"]), 3),
            turnover_24h=round(float(c["turnover_24h"]), 6),
            range_24h_pct=round(float(c["range_24h_pct"]), 6),
            abs_return_24h_pct=round(float(c["abs_return_24h_pct"]), 6),
            score=round(float(c["score"]), 6),
            selected_reason=reason,
        ))

    return {
        "name": "okx_stable_200",
        "generated_at_utc": now.isoformat(),
        "source": {
            "exchange": "OKX",
            "inst_type": "SWAP",
            "settle_ccy": "USDT",
            "instruments_endpoint": "/api/v5/public/instruments?instType=SWAP",
            "tickers_endpoint": "/api/v5/market/tickers?instType=SWAP",
        },
        "target_count": int(args.target_count),
        "actual_count": len(selected),
        "store": "okx_stable_200",
        "store_dir": str(DEFAULT_STORE_DIR),
        "selection_policy": {
            "history_policy": "only select contracts whose OKX listTime age is at least min_history_age_days",
            "noncrypto_policy": "include live OKX instCategory 3/4 above min_noncrypto_turnover and min history",
            "crypto_policy": "fill remaining slots with long-history liquid instCategory 1 contracts",
            "min_history_age_days": int(args.min_history_age_days),
            "min_crypto_turnover": float(args.min_crypto_turnover),
            "min_noncrypto_turnover": float(args.min_noncrypto_turnover),
            "hc_blacklist_applied_to_crypto": bool(args.apply_hc_blacklist),
            "meme_crypto_blocklist_applied": not args.allow_meme_crypto,
            "note": "OKX currently has fewer than 200 long-history stock/ETF/commodity swaps, so long-history crypto fills the remainder.",
        },
        "counts": {
            "live_usdt_swaps_seen": len(all_candidates),
            "short_history_rejected": len(short_history),
            "long_history_candidates": len(candidates),
            "selected_tradfi": sum(1 for c in selected if c.asset_class in {"tradfi", "etf_index"}),
            "selected_commodity": sum(1 for c in selected if c.asset_class == "commodity"),
            "selected_crypto": sum(1 for c in selected if c.asset_class == "crypto"),
            "eligible_crypto_after_filters": len(crypto),
            "reserve_crypto_used": sum(
                1 for c in selected
                if c.selected_reason.startswith("long_history_crypto_reserve")
            ),
            "rejected_crypto": len(crypto_rejected),
        },
        "symbols": [c.symbol for c in selected],
        "ranked": [asdict(c) for c in selected],
        "rejected_crypto_sample": sorted(
            crypto_rejected,
            key=lambda c: (str(c.get("reject_reason")), -float(c.get("turnover_24h", 0.0))),
        )[:80],
        "short_history_rejected_sample": sorted(
            short_history,
            key=lambda c: (float(c.get("age_days", 0.0)), c.get("symbol", "")),
        )[:80],
    }


def write_outputs(payload: dict[str, Any], out_json: Path, out_csv: Path | None) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if out_csv is None:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(payload["ranked"][0].keys()) if payload.get("ranked") else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload.get("ranked", []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-count", type=int, default=200)
    ap.add_argument("--min-history-age-days", type=int, default=240)
    ap.add_argument("--min-crypto-turnover", type=float, default=100_000.0)
    ap.add_argument("--min-noncrypto-turnover", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--allow-meme-crypto", action="store_true",
                    help="do not treat old meme/event crypto as reserve-only")
    ap.add_argument("--apply-hc-blacklist", dest="apply_hc_blacklist", action="store_true",
                    help="apply the existing HC crypto blacklist to primary crypto fill symbols")
    ap.add_argument("--no-hc-blacklist", dest="apply_hc_blacklist", action="store_false",
                    help="do not push existing HC-blacklisted crypto into reserve")
    ap.set_defaults(apply_hc_blacklist=True)
    args = ap.parse_args()

    payload = build_universe(args)
    write_outputs(payload, args.out, None if args.no_csv else args.csv)

    counts = payload["counts"]
    print(
        f"wrote {payload['actual_count']} symbols -> {args.out} "
        f"(tradfi={counts['selected_tradfi']} commodity={counts['selected_commodity']} "
        f"crypto={counts['selected_crypto']})"
    )
    if not args.no_csv:
        print(f"wrote ranked csv -> {args.csv}")


if __name__ == "__main__":
    main()
