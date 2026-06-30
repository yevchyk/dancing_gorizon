"""Real Binance USDT-M futures execution backend (mirror of okx_executor.py).

Same Executor interface, so LiveTrader + HCPortfolioEngine run unchanged:
enter() places a MARKET order, force_close() cancels open orders and flattens
reduce-only, open_positions()/equity() feed the book sync. HC engines are
horizon-exit-only, so no TP/SL bracket is attached unless move_pct is passed
(then a TAKE_PROFIT_MARKET + STOP_MARKET pair is placed reduce-only).

Credentials come ONLY from the environment:
  live    -> BINANCE_API_KEY / BINANCE_SECRET_KEY
  testnet -> BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_SECRET_KEY (demo=True)
Safety: enter() refuses to place an order unless credentials are present and
live=True, so importing/dry-running can never accidentally trade.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import time
from datetime import datetime, timezone

import requests

from .executor import Executor, Fill

LIVE_BASE = "https://fapi.binance.com"
TESTNET_BASE = "https://testnet.binancefuture.com"


def to_binance_sym(symbol: str) -> str:
    """Store id 'BTC_USDT_SWAP' -> Binance 'BTCUSDT'."""
    return symbol.replace("_USDT_SWAP", "") + "USDT"


def to_store_sym(binance_sym: str) -> str:
    base = binance_sym[:-4] if binance_sym.endswith("USDT") else binance_sym
    return f"{base}_USDT_SWAP"


class BinanceExecutor(Executor):
    mode = "binance"

    def __init__(self, *, live: bool = False, demo: bool | None = None,
                 leverage: int = 1, recv_window_ms: int = 5000):
        self.live = live
        self.demo = (os.environ.get("BINANCE_TESTNET", "0") == "1") if demo is None else demo
        self.base = TESTNET_BASE if self.demo else LIVE_BASE
        self.leverage = int(leverage)
        self.recv_window_ms = int(recv_window_ms)
        if self.demo:
            self.api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
            self.secret = os.environ.get("BINANCE_TESTNET_SECRET_KEY", "")
        else:
            self.api_key = os.environ.get("BINANCE_API_KEY", "")
            self.secret = os.environ.get("BINANCE_SECRET_KEY", "")
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": "dancing-horizon/1.0",
                                "Accept": "application/json",
                                "X-MBX-APIKEY": self.api_key})
        self._time_offset_ms = 0
        self._time_synced = False
        self._filters: dict[str, dict] = {}
        self._hedge: bool | None = None
        self._lev_set: set[str] = set()
        self._restricted: set[str] = set()
        self._last_open_positions_ok = False

    # ---- auth / plumbing ----
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.secret)

    def _sync_time(self) -> None:
        try:
            r = self._s.get(self.base + "/fapi/v1/time", timeout=10).json()
            self._time_offset_ms = int(r["serverTime"]) - int(time.time() * 1000)
            self._time_synced = True
        except Exception:
            self._time_offset_ms = 0

    def _ts_ms(self) -> int:
        if not self._time_synced:
            self._sync_time()
        return int(time.time() * 1000) + self._time_offset_ms

    def _signed(self, method: str, path: str, params: dict | None = None,
                retry_skew: bool = True) -> dict:
        p = dict(params or {})
        p["timestamp"] = self._ts_ms()
        p["recvWindow"] = self.recv_window_ms
        qs = requests.compat.urlencode(p)
        sig = hmac.new(self.secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        url = f"{self.base}{path}?{qs}&signature={sig}"
        r = self._s.request(method, url, timeout=10)
        try:
            data = r.json()
        except ValueError:
            r.raise_for_status()
            raise RuntimeError(f"non-json response from {path}")
        # clock skew (-1021): resync once and retry
        if isinstance(data, dict) and data.get("code") == -1021 and retry_skew:
            self._sync_time()
            return self._signed(method, path, params, retry_skew=False)
        return data

    def _public(self, path: str, params: dict | None = None) -> dict | list:
        qs = ("?" + requests.compat.urlencode(params)) if params else ""
        r = self._s.get(self.base + path + qs, timeout=10)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _is_err(data) -> bool:
        return isinstance(data, dict) and int(data.get("code", 0) or 0) < 0

    # ---- exchange filters ----
    def _load_filters(self) -> None:
        if self._filters:
            return
        info = self._public("/fapi/v1/exchangeInfo")
        for s in info.get("symbols", []):
            f = {x["filterType"]: x for x in s.get("filters", [])}
            lot = f.get("LOT_SIZE") or f.get("MARKET_LOT_SIZE") or {}
            self._filters[s["symbol"]] = {
                "stepSize": float(lot.get("stepSize", 0) or 0),
                "minQty": float(lot.get("minQty", 0) or 0),
                "minNotional": float((f.get("MIN_NOTIONAL") or {}).get("notional", 0) or 0),
                "status": s.get("status", ""),
            }

    def instrument_info(self, inst: str) -> dict:
        self._load_filters()
        return self._filters.get(inst, {})

    @staticmethod
    def _fmt_qty(qty: float, step: float) -> str:
        s = f"{step:.12f}".rstrip("0")
        decimals = len(s.split(".")[1]) if "." in s and s.split(".")[1] else 0
        return f"{qty:.{decimals}f}"

    def _size(self, inst: str, price: float, size_usd: float, info: dict) -> str | None:
        step = info.get("stepSize") or 0
        if price <= 0 or step <= 0:
            return None
        raw = size_usd / price
        steps = max(1, round(raw / step))
        qty = steps * step
        min_qty = info.get("minQty") or 0
        if qty < min_qty:
            qty = math.ceil(min_qty / step - 1e-12) * step
        min_notional = info.get("minNotional") or 0
        if min_notional > 0 and qty * price < min_notional:
            qty = math.ceil(min_notional / price / step - 1e-12) * step
        return self._fmt_qty(qty, step)

    # ---- account state ----
    def _ensure_position_mode(self) -> None:
        if self._hedge is not None:
            return
        try:
            r = self._signed("GET", "/fapi/v1/positionSide/dual")
            self._hedge = bool(r.get("dualSidePosition", False)) if isinstance(r, dict) else False
        except Exception:
            self._hedge = False

    def set_leverage(self, inst: str) -> None:
        if inst in self._lev_set:
            return
        try:
            self._signed("POST", "/fapi/v1/marginType",
                         {"symbol": inst, "marginType": "CROSSED"})
        except Exception:
            pass  # -4046 "No need to change margin type" lands here too — fine
        try:
            r = self._signed("POST", "/fapi/v1/leverage",
                             {"symbol": inst, "leverage": self.leverage})
            if not self._is_err(r):
                self._lev_set.add(inst)
        except Exception:
            pass

    def equity(self) -> float:
        data = self._signed("GET", "/fapi/v2/balance")
        if self._is_err(data):
            raise RuntimeError(f"balance: {data.get('msg')}")
        for d in data:
            if d.get("asset") == "USDT":
                return float(d.get("availableBalance") or 0)
        return 0.0

    def open_positions(self) -> list[dict]:
        if not (self.live and self.has_credentials()):
            self._last_open_positions_ok = False
            return []
        try:
            data = self._signed("GET", "/fapi/v2/positionRisk")
        except Exception:
            self._last_open_positions_ok = False
            return []
        if self._is_err(data):
            self._last_open_positions_ok = False
            return []
        self._last_open_positions_ok = True
        out = []
        for p in data:
            amt = float(p.get("positionAmt", 0) or 0)
            if amt == 0:
                continue
            ps = p.get("positionSide", "BOTH")
            side = ("long" if ps == "LONG" else "short" if ps == "SHORT"
                    else ("long" if amt > 0 else "short"))
            out.append({"symbol": to_store_sym(p["symbol"]), "side": side})
        return out

    def _ticker_last(self, inst: str) -> float | None:
        try:
            r = self._public("/fapi/v1/ticker/price", {"symbol": inst})
            px = float(r.get("price") or 0)
            return px if px > 0 and math.isfinite(px) else None
        except Exception:
            return None

    # ---- trading ----
    def enter(self, symbol: str, side: str, price: float, size_usd: float,
              move_pct: float | None = None) -> Fill:
        inst = to_binance_sym(symbol)
        if not (self.live and self.has_credentials()):
            return Fill(symbol, side, price, size_usd, ok=False,
                        info="not live / no credentials")
        if inst in self._restricted:
            return Fill(symbol, side, price, size_usd, ok=False, info="restricted")
        self._ensure_position_mode()
        info = self.instrument_info(inst)
        if not info or info.get("status") != "TRADING":
            self._restricted.add(inst)
            return Fill(symbol, side, price, size_usd, ok=False,
                        info=f"no instrument / not TRADING: {inst}")
        qty = self._size(inst, price, size_usd, info)
        if qty is None or float(qty) <= 0:
            return Fill(symbol, side, price, size_usd, ok=False, info="size=0")
        actual_usd = float(qty) * price
        if actual_usd > size_usd * 3 or actual_usd < 1.0:
            return Fill(symbol, side, price, size_usd, ok=False,
                        info=f"notional ${actual_usd:.2f} out of range for ${size_usd:.0f}")

        self.set_leverage(inst)
        order_side = "BUY" if side == "long" else "SELL"
        payload = {"symbol": inst, "side": order_side, "type": "MARKET",
                   "quantity": qty, "newOrderRespType": "RESULT"}
        if self._hedge:
            payload["positionSide"] = "LONG" if side == "long" else "SHORT"
        resp = self._signed("POST", "/fapi/v1/order", payload)
        if self._is_err(resp):
            code = resp.get("code")
            if code in (-1121, -4140):       # invalid/delisted symbol
                self._restricted.add(inst)
            return Fill(symbol, side, price, size_usd, ok=False,
                        info=f"order rejected code={code} {resp.get('msg', '')}")

        fill_price = float(resp.get("avgPrice") or 0) or None
        if not fill_price:
            fill_price = self._order_avg_px(inst, resp.get("orderId")) or self._ticker_last(inst) or price
        tpsl_info = ""
        if move_pct:
            ok, detail = self._attach_tpsl(inst, side, qty, fill_price, move_pct)
            tpsl_info = f" tpsl={'ok' if ok else 'FAILED'}:{detail}"
        return Fill(symbol, side, fill_price, size_usd, ok=True, sz=qty,
                    info=f"binance qty={qty} fillPx={fill_price:.8g}{tpsl_info}",
                    filled_at=datetime.now(timezone.utc).isoformat())

    def _order_avg_px(self, inst: str, order_id) -> float | None:
        if not order_id:
            return None
        try:
            r = self._signed("GET", "/fapi/v1/order", {"symbol": inst, "orderId": order_id})
            if self._is_err(r):
                return None
            px = float(r.get("avgPrice") or 0)
            return px if px > 0 and math.isfinite(px) else None
        except Exception:
            return None

    def _attach_tpsl(self, inst: str, side: str, qty: str, entry: float,
                     move_pct: float) -> tuple[bool, str]:
        stop_ratio = 1.0
        if side == "long":
            tp, sl, close_side = entry * (1 + move_pct), entry * (1 - move_pct * stop_ratio), "SELL"
        else:
            tp, sl, close_side = entry * (1 - move_pct), entry * (1 + move_pct * stop_ratio), "BUY"
        ids = []
        for typ, trigger in (("TAKE_PROFIT_MARKET", tp), ("STOP_MARKET", sl)):
            payload = {"symbol": inst, "side": close_side, "type": typ,
                       "stopPrice": f"{trigger:.6g}", "quantity": qty,
                       "workingType": "MARK_PRICE"}
            if self._hedge:
                payload["positionSide"] = "LONG" if side == "long" else "SHORT"
            else:
                payload["reduceOnly"] = "true"
            r = self._signed("POST", "/fapi/v1/order", payload)
            if self._is_err(r):
                return False, f"{typ} code={r.get('code')} {r.get('msg', '')}"
            ids.append(str(r.get("orderId", "")))
        return True, ",".join(ids)

    def cancel_algos(self, inst: str) -> None:
        """Cancel all pending orders for an instrument (TP/SL leftovers)."""
        try:
            self._signed("DELETE", "/fapi/v1/allOpenOrders", {"symbol": inst})
        except Exception:
            pass

    def force_close(self, symbol: str) -> bool:
        inst = to_binance_sym(symbol)
        if not (self.live and self.has_credentials()):
            return False
        self._ensure_position_mode()
        self.cancel_algos(inst)
        try:
            data = self._signed("GET", "/fapi/v2/positionRisk", {"symbol": inst})
        except Exception:
            return False
        if self._is_err(data):
            return False
        closed = False
        for p in data:
            amt = float(p.get("positionAmt", 0) or 0)
            if amt == 0:
                continue
            info = self.instrument_info(inst)
            qty = self._fmt_qty(abs(amt), info.get("stepSize") or 0.001)
            close_side = "SELL" if amt > 0 else "BUY"
            payload = {"symbol": inst, "side": close_side, "type": "MARKET",
                       "quantity": qty}
            ps = p.get("positionSide", "BOTH")
            if self._hedge and ps in ("LONG", "SHORT"):
                payload["positionSide"] = ps
            else:
                payload["reduceOnly"] = "true"
            r = self._signed("POST", "/fapi/v1/order", payload)
            closed = closed or not self._is_err(r)
        return closed

    def partial_close(self, symbol: str, side: str, sz: str) -> bool:
        inst = to_binance_sym(symbol)
        if not (self.live and self.has_credentials()):
            return False
        try:
            if not sz or float(sz) <= 0:
                return False
        except (TypeError, ValueError):
            return False
        self._ensure_position_mode()
        close_side = "SELL" if side == "long" else "BUY"
        payload = {"symbol": inst, "side": close_side, "type": "MARKET", "quantity": str(sz)}
        if self._hedge:
            payload["positionSide"] = "LONG" if side == "long" else "SHORT"
        else:
            payload["reduceOnly"] = "true"
        try:
            r = self._signed("POST", "/fapi/v1/order", payload)
            return not self._is_err(r)
        except Exception:
            return False
