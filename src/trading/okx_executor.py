"""Real OKX execution backend (ported/cleaned from the old live_trader).

Directional, not straddle: a long enters market BUY and attaches an OCO
(take-profit at +move%, stop-loss at -move%*stop_ratio); a short is the mirror.
Credentials come ONLY from the environment (OKX_API_KEY/OKX_SECRET_KEY/
OKX_PASSPHRASE) — never hard-coded. Set demo=True (or OKX_DEMO=1) for OKX's
simulated-trading sandbox.

Safety: enter() refuses to place an order unless credentials are present and
live=True, so importing/dry-running can never accidentally trade.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
from datetime import datetime, timezone

import requests

from .. import config as C
from .executor import Executor, Fill

OKX_BASE = "https://www.okx.com"


def to_okx_inst(symbol: str) -> str:
    """Store id 'BTC_USDT_SWAP' -> OKX instId 'BTC-USDT-SWAP'."""
    return symbol.replace("_", "-")


class OKXExecutor(Executor):
    mode = "okx"

    def __init__(self, *, live: bool = False, demo: bool | None = None,
                 leverage: int = 1, td_mode: str = "cross", hedge: bool = False,
                 fee_per_side: float = C.OKX_FEE_PER_SIDE,
                 stop_ratio: float = C.STOP_PCT_RATIO):
        self.live = live
        self.demo = (os.environ.get("OKX_DEMO", "0") == "1") if demo is None else demo
        self.leverage = leverage
        self.td_mode = td_mode
        self.hedge = hedge
        self.fee_per_side = fee_per_side
        self.stop_ratio = stop_ratio
        self.api_key = os.environ.get("OKX_API_KEY", "")
        self.secret = os.environ.get("OKX_SECRET_KEY", "")
        self.passphrase = os.environ.get("OKX_PASSPHRASE", "")
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": "mlpredictor/2.0", "Accept": "application/json"})
        self._restricted: set[str] = set()
        self._mode_checked = False
        self._last_open_positions_ok = False

    def _ensure_position_mode(self) -> None:
        """Read the account's position mode once and set self.hedge accordingly.
        Hedge (long_short_mode) requires posSide on every order; net mode forbids
        it -> sending the wrong one yields sCode 51000 'Parameter posSide error'."""
        if self._mode_checked:
            return
        self._mode_checked = True
        try:
            cfg = self._get("/api/v5/account/config")
            if cfg.get("code") == "0" and cfg.get("data"):
                self.hedge = cfg["data"][0].get("posMode") == "long_short_mode"
        except Exception:
            pass

    # ---- auth ----
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.secret and self.passphrase)

    @staticmethod
    def _ts() -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    def _sign(self, ts: str, method: str, path: str, body: str = "") -> str:
        msg = f"{ts}{method.upper()}{path}{body}"
        return base64.b64encode(
            hmac.new(self.secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()

    def _headers(self, method: str, path: str, body: str = "") -> dict:
        ts = self._ts()
        h = {"OK-ACCESS-KEY": self.api_key,
             "OK-ACCESS-SIGN": self._sign(ts, method, path, body),
             "OK-ACCESS-TIMESTAMP": ts,
             "OK-ACCESS-PASSPHRASE": self.passphrase,
             "Content-Type": "application/json"}
        if self.demo:
            h["x-simulated-trading"] = "1"
        return h

    def _get(self, path: str, params: dict | None = None) -> dict:
        qs = ("?" + requests.compat.urlencode(params)) if params else ""
        full = path + qs
        r = self._s.get(OKX_BASE + full, headers=self._headers("GET", full), timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data) -> dict:
        body = json.dumps(data)
        r = self._s.post(OKX_BASE + path, data=body,
                         headers=self._headers("POST", path, body), timeout=10)
        r.raise_for_status()
        return r.json()

    def _public(self, path: str, params: dict | None = None) -> dict:
        qs = ("?" + requests.compat.urlencode(params)) if params else ""
        r = self._s.get(OKX_BASE + path + qs, timeout=10)
        r.raise_for_status()
        return r.json()

    def _ticker_last(self, inst: str) -> float | None:
        try:
            resp = self._public("/api/v5/market/ticker", {"instId": inst})
            if resp.get("code") != "0" or not resp.get("data"):
                return None
            px = float(resp["data"][0].get("last") or 0)
            return px if px > 0 and math.isfinite(px) else None
        except Exception:
            return None

    def _order_avg_px(self, inst: str, ord_id: str) -> float | None:
        if not ord_id:
            return None
        try:
            resp = self._get("/api/v5/trade/order", {"instId": inst, "ordId": ord_id})
            if resp.get("code") != "0" or not resp.get("data"):
                return None
            px = float(resp["data"][0].get("avgPx") or 0)
            return px if px > 0 and math.isfinite(px) else None
        except Exception:
            return None

    def _position_avg_px(self, inst: str, pos_side: str) -> float | None:
        try:
            resp = self._get("/api/v5/account/positions", {"instId": inst, "instType": "SWAP"})
            if resp.get("code") != "0":
                return None
            for pos in resp.get("data", []):
                psz = float(pos.get("pos", 0) or 0)
                if psz == 0:
                    continue
                if self.hedge and pos.get("posSide") != pos_side:
                    continue
                px = float(pos.get("avgPx") or 0)
                if px > 0 and math.isfinite(px):
                    return px
            return None
        except Exception:
            return None

    # ---- account ----
    def equity(self) -> float:
        resp = self._get("/api/v5/account/balance", {"ccy": "USDT"})
        if resp.get("code") != "0":
            raise RuntimeError(f"balance: {resp.get('msg')}")
        for d in resp["data"][0]["details"]:
            if d["ccy"] == "USDT":
                return float(d.get("availEq") or d.get("availBal") or 0)
        return 0.0

    def instrument_info(self, inst: str) -> dict:
        resp = self._public("/api/v5/public/instruments",
                            {"instType": "SWAP", "instId": inst})
        if resp.get("code") != "0" or not resp.get("data"):
            return {}
        d = resp["data"][0]
        return {"minSz": float(d["minSz"]), "ctVal": float(d["ctVal"]),
                "lotSz": float(d["lotSz"])}

    def set_leverage(self, inst: str) -> None:
        for ps in (("long", "short") if self.hedge else ("net",)):
            try:
                self._post("/api/v5/account/set-leverage",
                           {"instId": inst, "lever": str(self.leverage),
                            "mgnMode": self.td_mode, "posSide": ps})
            except Exception:
                pass

    def _size(self, inst: str, price: float, size_usd: float, info: dict) -> str | None:
        contract_usd = info["ctVal"] * price
        lot = info["lotSz"]
        if contract_usd <= 0:
            return None
        raw = size_usd / contract_usd
        if lot <= 0:
            return None
        steps = max(1, round(raw / lot))
        min_steps = max(1, math.ceil(info["minSz"] / lot - 1e-12))
        sz = max(steps, min_steps) * lot
        if lot >= 1:
            return str(int(round(sz)))
        return f"{sz:.12g}"

    # ---- trading ----
    def enter(self, symbol: str, side: str, price: float, size_usd: float,
              move_pct: float | None = None) -> Fill:
        inst = to_okx_inst(symbol)
        if not (self.live and self.has_credentials()):
            return Fill(symbol, side, price, size_usd, ok=False,
                        info="not live / no credentials")
        self._ensure_position_mode()
        if inst in self._restricted:
            return Fill(symbol, side, price, size_usd, ok=False, info="restricted")
        info = self.instrument_info(inst)
        if not info:
            return Fill(symbol, side, price, size_usd, ok=False, info="no instrument info")
        sz = self._size(inst, price, size_usd, info)
        if sz is None or float(sz) <= 0:
            return Fill(symbol, side, price, size_usd, ok=False, info="size=0")
        # safety: reject if the minimum lot forces a notional far above our size
        actual_usd = float(sz) * info["ctVal"] * price
        if actual_usd > size_usd * 3 or actual_usd < 1.0:
            return Fill(symbol, side, price, size_usd, ok=False,
                        info=f"notional ${actual_usd:.2f} out of range for ${size_usd:.0f}")

        self.set_leverage(inst)
        okx_side = "buy" if side == "long" else "sell"
        pos_side = side if self.hedge else "net"
        payload = {"instId": inst, "tdMode": self.td_mode, "side": okx_side,
                   "ordType": "market", "sz": sz}
        if self.hedge:
            payload["posSide"] = pos_side
        resp = self._post("/api/v5/trade/order", payload)
        d = (resp.get("data") or [{}])[0]
        if resp.get("code") != "0" or d.get("sCode", "0") != "0":
            scode = d.get("sCode", "")
            if scode in ("51155", "51001"):
                self._restricted.add(inst)
            return Fill(symbol, side, price, size_usd, ok=False,
                        info=f"order rejected sCode={scode} {d.get('sMsg','')}")

        ord_id = str(d.get("ordId") or "")
        fill_price = (
            self._order_avg_px(inst, ord_id)
            or self._position_avg_px(inst, pos_side)
            or self._ticker_last(inst)
            or price
        )
        oco_info = ""
        if move_pct:
            ok, detail = self._attach_oco(inst, side, pos_side, sz, fill_price, move_pct)
            oco_info = f" oco={'ok' if ok else 'FAILED'}:{detail}"
        price_info = "" if abs(fill_price - price) <= max(price, 1e-12) * 0.0005 else f" reqPx={price:.8g}"
        return Fill(symbol, side, fill_price, size_usd, ok=True, sz=str(sz),
                    info=f"okx sz={sz} fillPx={fill_price:.8g}{price_info}{oco_info}",
                    filled_at=datetime.now(timezone.utc).isoformat())

    def _attach_oco(self, inst: str, side: str, pos_side: str, sz: str,
                    entry: float, move_pct: float) -> tuple[bool, str]:
        if side == "long":
            tp = entry * (1 + move_pct)
            sl = entry * (1 - move_pct * self.stop_ratio)
            close_side = "sell"
        else:
            tp = entry * (1 - move_pct)
            sl = entry * (1 + move_pct * self.stop_ratio)
            close_side = "buy"
        algo = {"instId": inst, "tdMode": self.td_mode, "side": close_side,
                "ordType": "oco", "sz": sz,
                "tpTriggerPx": f"{tp:.6g}", "tpOrdPx": "-1",
                "slTriggerPx": f"{sl:.6g}", "slOrdPx": "-1"}
        if self.hedge:
            algo["posSide"] = pos_side
        try:
            resp = self._post("/api/v5/trade/order-algo", algo)
            d = (resp.get("data") or [{}])[0]
            if resp.get("code") == "0" and d.get("sCode", "0") == "0":
                return True, str(d.get("algoId", ""))
            return False, f"sCode={d.get('sCode','')} {d.get('sMsg','')}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def open_positions(self) -> list[dict]:
        if not (self.live and self.has_credentials()):
            self._last_open_positions_ok = False
            return []
        try:
            resp = self._get("/api/v5/account/positions", {"instType": "SWAP"})
        except Exception:
            self._last_open_positions_ok = False
            return []
        if resp.get("code") != "0":
            self._last_open_positions_ok = False
            return []
        self._last_open_positions_ok = True
        out = []
        for p in resp.get("data", []):
            psz = float(p.get("pos", 0) or 0)
            if psz == 0:
                continue
            store_sym = p["instId"].replace("-", "_")
            side = "long" if p.get("posSide") == "long" else (
                "short" if p.get("posSide") == "short" else
                ("long" if psz > 0 else "short"))
            out.append({"symbol": store_sym, "side": side})
        return out

    def cancel_algos(self, inst: str) -> None:
        """Cancel any pending OCO/TP-SL algos for an instrument (avoid orphans
        after a deadline close)."""
        try:
            resp = self._get("/api/v5/trade/orders-algo-pending",
                            {"instId": inst, "ordType": "oco", "instType": "SWAP"})
            orders = resp.get("data", []) if resp.get("code") == "0" else []
            if orders:
                self._post("/api/v5/trade/cancel-algos",
                           [{"algoId": o["algoId"], "instId": inst} for o in orders])
        except Exception:
            pass

    def force_close(self, symbol: str) -> bool:
        inst = to_okx_inst(symbol)
        if not (self.live and self.has_credentials()):
            return False
        self.cancel_algos(inst)
        resp = self._get("/api/v5/account/positions", {"instId": inst, "instType": "SWAP"})
        if resp.get("code") != "0":
            return False
        closed = False
        for pos in resp.get("data", []):
            psz = float(pos.get("pos", 0))
            if psz == 0:
                continue
            pos_side = pos.get("posSide", "net")
            if pos_side == "long" or (pos_side == "net" and psz > 0):
                close_side = "sell"
            else:
                close_side = "buy"
            sz = str(int(abs(psz))) if abs(psz) == int(abs(psz)) else str(abs(psz))
            payload = {"instId": inst, "tdMode": self.td_mode, "side": close_side,
                       "ordType": "market", "sz": sz, "reduceOnly": "true"}
            if self.hedge:
                payload["posSide"] = pos_side
            r = self._post("/api/v5/trade/order", payload)
            d = (r.get("data") or [{}])[0]
            closed = closed or (r.get("code") == "0" and d.get("sCode", "0") == "0")
        return closed

    def partial_close(self, symbol: str, side: str, sz: str) -> bool:
        """Reduce-only close of exactly `sz` contracts for ONE leg (multi-leg mode).

        Unlike force_close it does not cancel algos or flatten the whole symbol —
        it shrinks the netted position by this leg's contract size only.
        """
        inst = to_okx_inst(symbol)
        if not (self.live and self.has_credentials()):
            return False
        try:
            if not sz or float(sz) <= 0:
                return False
        except (TypeError, ValueError):
            return False
        self._ensure_position_mode()
        close_side = "sell" if side == "long" else "buy"
        payload = {"instId": inst, "tdMode": self.td_mode, "side": close_side,
                   "ordType": "market", "sz": str(sz), "reduceOnly": "true"}
        if self.hedge:
            payload["posSide"] = side
        try:
            r = self._post("/api/v5/trade/order", payload)
            d = (r.get("data") or [{}])[0]
            return r.get("code") == "0" and d.get("sCode", "0") == "0"
        except Exception:
            return False
