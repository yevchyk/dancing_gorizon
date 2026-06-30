"""Live trading loop: fresh candles -> snapshot@now -> strategy -> order.

Each scan:
  1. (optional) refresh candles for the watched symbols
  2. build the curve feature row at 'now' per symbol and score all 15 models
  3. detect regime, run the Strategy (threshold + agreement + regime + veto)
  4. for each entry: risk-check (PositionManager), place via the Executor
     (paper / shadow / OKX), attaching TP/SL = +-move_pct
  5. close positions whose horizon deadline passed (OKX OCO usually exits first;
     this is the safety net)

Backend-agnostic: pass PaperExecutor for dry simulation, ShadowExecutor to log
intended trades on live data, or OKXExecutor(live=True) for real orders.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from pathlib import Path

import pandas as pd

from .. import config as C
from ..database import CandleStore, CandleFetcher, OKXClient
from ..features import CurveBuilder
from ..training import ModelRegistry
from .strategy import Strategy
from .regime import RegimeDetector
from .position_manager import PositionManager, Position
from .trade_logger import TradeLogger
from .executor import Executor
from .timeutil import index_to_ns, ts_to_ns


class LiveTrader:
    def __init__(self, executor: Executor, *, symbols: list[str] | None = None,
                 thresholds: dict[str, float] | None = None,
                 trust_engine=None, top_per_scan: int = 3,
                 fetch: bool = True, scan_interval_min: int = C.SCAN_INTERVAL_MIN,
                 trade_size_usd: float = C.TRADE_SIZE_USD,
                 max_concurrent: int = C.MAX_CONCURRENT,
                 cooldown_min: int = C.COOLDOWN_MIN,
                 max_legs: int = 1,
                 green_harvest: bool | None = None,
                 scan_confirmation_sec: float = 65.0,
                 deadline_check_sec: float = 5.0,
                 max_anchor_lag_sec: float = 180.0,
                 store_root: str | Path | None = None,
                 fetcher=None):
        self.store = CandleStore(Path(store_root) if store_root is not None else C.CANDLES_DIR)
        self.curve = CurveBuilder(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
        self.trust_engine = trust_engine          # new engine (None => legacy strategy)
        self.top_per_scan = top_per_scan
        self.green_harvest = C.GREEN_HARVEST if green_harvest is None else green_harvest
        if trust_engine is None:
            self.registry = ModelRegistry.load_default()
            self.strategy = Strategy(self.registry, thresholds=thresholds)
        self.regime = RegimeDetector()
        # Live trading is one active position per symbol. The old multi-leg path
        # stacked horizons on one coin and disabled OCO, which is too easy to
        # misconfigure for real-money tests.
        self.max_legs = 1
        self.positions = PositionManager(max_concurrent=max_concurrent, cooldown_min=cooldown_min,
                                         max_legs=self.max_legs)
        self.executor = executor
        # injectable fetcher (e.g. Binance top-up adapter); default = OKX
        if fetcher is not None:
            self.fetcher = fetcher if fetch else None
        else:
            self.fetcher = CandleFetcher(OKXClient(), self.store) if fetch else None
        self.scan_interval_min = scan_interval_min
        self.scan_confirmation_sec = scan_confirmation_sec
        self.deadline_check_sec = max(1.0, float(deadline_check_sec))
        self.max_anchor_lag_sec = max(1.0, float(max_anchor_lag_sec))
        self.trade_size_usd = trade_size_usd
        self.logger = TradeLogger(C.TRADING_LOGS_DIR /
                                  f"live_{dt.datetime.now():%Y%m%d_%H%M%S}")
        self._position_lock = threading.RLock()
        self._deadline_stop = threading.Event()
        self._deadline_thread: threading.Thread | None = None
        self._horizon_min = {h.label: h.minutes for h in C.HORIZONS}
        if trust_engine is not None and hasattr(trust_engine, "horizon_minutes"):
            self._horizon_min.update(getattr(trust_engine, "horizon_minutes"))
        if symbols is not None:
            self.symbols = symbols
        elif trust_engine is not None and hasattr(trust_engine, "build_watchlist"):
            self.symbols = trust_engine.build_watchlist(
                self.store, C.LIVE_WATCHLIST_SIZE, self.logger)
        else:
            self.symbols = self._build_watchlist(C.LIVE_WATCHLIST_SIZE)

    def _engine_label(self) -> str:
        if self.trust_engine is None:
            return "legacy"
        profile = getattr(self.trust_engine, "profile", "")
        name = self.trust_engine.__class__.__name__
        return f"{name}:{profile}" if profile else name

    def _next_scan_slot(self) -> tuple[pd.Timestamp, float]:
        """Next wall-clock-aligned anchor and seconds to wait before scanning.

        For a 2m strategy this scans even UTC-minute anchors after the following
        1m candle has had time to confirm, e.g. anchor 16:50 at about 16:51:05.
        """
        interval_sec = max(60, int(round(float(self.scan_interval_min) * 60.0)))
        confirm_sec = max(0.0, float(self.scan_confirmation_sec))
        now = dt.datetime.now(dt.timezone.utc)
        now_sec = now.timestamp()
        anchor_sec = (int(now_sec) // interval_sec) * interval_sec
        scan_sec = anchor_sec + confirm_sec
        if now_sec >= scan_sec:
            anchor_sec += interval_sec
            scan_sec = anchor_sec + confirm_sec
        anchor = pd.Timestamp.fromtimestamp(anchor_sec, tz="UTC")
        return anchor, max(0.0, scan_sec - now_sec)

    @staticmethod
    def _utc(ts) -> pd.Timestamp:
        out = pd.Timestamp(ts)
        return out.tz_convert("UTC") if out.tzinfo else out.tz_localize("UTC")

    def _fresh_store_anchor(self) -> pd.Timestamp | None:
        """Newest confirmed candle timestamp currently available in the store."""
        wall_now = pd.Timestamp.now(tz="UTC")
        latest: pd.Timestamp | None = None
        for sym in self.symbols:
            candles = self.store.load(sym)
            if candles is None or candles.empty:
                continue
            ts = self._utc(candles.index.max())
            if ts > wall_now + pd.Timedelta(seconds=30):
                continue
            if latest is None or ts > latest:
                latest = ts
        if latest is not None:
            lag_sec = (wall_now - latest).total_seconds()
            if lag_sec > self.max_anchor_lag_sec:
                self.logger.event(
                    f"fresh_anchor_stale latest={latest.isoformat()} lag={lag_sec:.0f}s "
                    f"max={self.max_anchor_lag_sec:.0f}s"
                )
                return None
        return latest

    def _select_scan_anchor(self, requested: pd.Timestamp) -> pd.Timestamp:
        fresh = self._fresh_store_anchor()
        if fresh is None:
            return requested
        requested = self._utc(requested)
        if fresh != requested:
            self.logger.event(f"anchor_refresh requested={requested.isoformat()} fresh={fresh.isoformat()}")
        return fresh

    def _fill_timestamp(self, fill, fallback: pd.Timestamp) -> pd.Timestamp:
        raw = getattr(fill, "filled_at", "") or ""
        if raw:
            try:
                return self._utc(raw)
            except Exception:
                pass
        wall_now = pd.Timestamp.now(tz="UTC")
        fallback = self._utc(fallback)
        if abs((wall_now - fallback).total_seconds()) <= 3600:
            return wall_now
        return fallback

    def _store_price_at(self, symbol: str, now: pd.Timestamp) -> float | None:
        candles = self.store.load(symbol)
        if candles is None or candles.empty:
            return None
        now = self._utc(now)
        past = candles[candles.index <= now]
        if past.empty:
            return None
        px = float(past["close"].iloc[-1])
        return px if px > 0 else None

    def _build_watchlist(self, top_n: int) -> list[str]:
        """Top-N most liquid coins by recent 24h quote volume, minus blacklist.
        Liquidity filter doubles as a guard against the thin/deceptive coins."""
        blacklist = set(C.BLACKLIST_SYMBOLS)
        vols: list[tuple[str, float]] = []
        for sym in self.store.symbols():
            if sym in blacklist:
                continue
            c = self.store.load(sym)
            if c is None or c.empty:
                continue
            tail = c.iloc[-1440:]   # ~last day of finest bars
            vol = float((tail["close"] * tail["volume"]).sum())
            vols.append((sym, vol))
        vols.sort(key=lambda x: x[1], reverse=True)
        limit = int(top_n or 0)
        watch = [s for s, _ in (vols[:limit] if limit > 0 else vols)]
        scope = f"top {limit}" if limit > 0 else "all"
        self.logger.event(f"watchlist: {scope} liquid coins -> {len(watch)} symbols "
                          f"(of {len(vols)}), {len(blacklist)} blacklisted")
        return watch

    # ---- snapshot ----
    def _snapshot(self, now: pd.Timestamp) -> pd.DataFrame:
        if self.trust_engine is not None and hasattr(self.trust_engine, "snapshot"):
            return self.trust_engine.snapshot(self.store, self.symbols, now)
        rows = []
        for sym in self.symbols:
            candles = self.store.load(sym)
            if candles is None or candles.empty:
                continue
            past = candles[candles.index <= now]
            if past.empty:
                continue
            entry_time = past.index[-1]
            if entry_time < now - pd.Timedelta(minutes=2):
                continue
            curve = self.curve.build(candles, now)
            if curve is None:
                continue
            entry = float(past["close"].iloc[-1])
            rows.append({"symbol": sym, "anchor_time": now, "entry_price": entry, **curve})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df if self.trust_engine is not None else self.registry.score(df)

    def _open(self, sym, side, model, horizon, move_pct, prob, entry, now, now_dt, reason,
              size_usd=None, engine="", threshold: float = 0.0):
        size_usd = size_usd or self.trade_size_usd
        with self._position_lock:
            ok, why = self.positions.can_open(sym, now_dt)
        if not ok:
            return None
        # HC is horizon-exit-only. Other engines may still request a safety OCO.
        horizon_only = bool(getattr(self.trust_engine, "horizon_exit_only", False))
        oco_move = None if (horizon_only or self.max_legs > 1) else move_pct
        fill = self.executor.enter(sym, side, entry, size_usd, move_pct=oco_move)
        if not fill.ok:
            self.logger.log_decision(symbol=sym, model=model, side=side, prob=prob,
                                     threshold=threshold, action="skip", reason=fill.info,
                                     engine=engine)
            return None
        fill_time = self._fill_timestamp(fill, now)
        fill_dt = fill_time.to_pydatetime()
        actual_entry = float(fill.entry_price) if float(fill.entry_price or 0) > 0 else float(entry)
        leg_id = f"{sym}#{horizon}" if self.max_legs > 1 else sym
        pos = Position(symbol=sym, side=side, model=model, entry_price=entry,
                       size_usd=size_usd, move_pct=move_pct,
                       horizon=horizon, opened_at=fill_time.isoformat(), engine=engine,
                       sz=str(getattr(fill, "sz", "") or ""), leg_id=leg_id)
        pos.entry_price = actual_entry
        with self._position_lock:
            self.positions.open(pos, fill_dt)
        self.logger.log_decision(symbol=sym, model=model, side=side, prob=prob,
                                 threshold=threshold, action="open",
                                 reason=f"{self.executor.mode}:{reason} fill_at={fill_time.isoformat()} {fill.info}".strip(),
                                 engine=engine)
        self.logger.log_trade({"event": "open", "engine": engine, "symbol": sym,
                               "model": model, "side": side, "horizon": horizon,
                               "entry_price": actual_entry, "size_usd": size_usd,
                               "opened_at": fill_time.isoformat()})
        return pos

    def harvest_exits(self, snap: pd.DataFrame, now_dt) -> int:
        """Green harvest: close any open position currently in profit (net cost)."""
        price = dict(zip(snap["symbol"], snap["entry_price"]))
        closed = 0
        with self._position_lock:
            for key, pos in list(self.positions.open_positions.items()):
                cur = price.get(pos.symbol)
                if cur is None or pos.entry_price <= 0:
                    continue
                sgn = 1 if pos.side == "long" else -1
                pnl_pct = sgn * (cur / pos.entry_price - 1.0) * 100
                if pnl_pct > C.HARVEST_COST_PCT:
                    try:
                        if self.max_legs > 1 and hasattr(self.executor, "partial_close"):
                            self.executor.partial_close(pos.symbol, pos.side, pos.sz)
                        elif hasattr(self.executor, "force_close"):
                            self.executor.force_close(pos.symbol)
                    except Exception as e:
                        self.logger.event(f"harvest {pos.symbol} failed: {e}")
                    self.positions.close(key, pnl_pct, now_dt)
                    self.logger.log_trade({"event": "harvest_close", "engine": pos.engine,
                                           "symbol": pos.symbol, "model": pos.model, "side": pos.side,
                                           "horizon": pos.horizon, "entry_price": pos.entry_price,
                                           "exit_price": cur, "pnl_pct": round(pnl_pct, 3)})
                    closed += 1
        return closed

    def _scan_trust(self, snap: pd.DataFrame, now, now_dt) -> list[Position]:
        sigs = self.trust_engine.decide(snap, top_n=self.top_per_scan)
        entry_by_sym = dict(zip(snap["symbol"], snap["entry_price"]))
        opened = []
        for sig in sigs:
            mult = getattr(sig, "size_mult", 1.0)
            source = getattr(sig, "source", "")
            engine = getattr(sig, "engine", "") or self._engine_label()
            # Profile-supplied absolute size wins; else global stake * size_mult.
            size_usd = getattr(sig, "size_usd", None) or self.trade_size_usd * mult
            reason = (f"spread={sig.spread:.2f} agree={getattr(sig,'agree','?')} "
                      f"x{mult:.1f} {source}".strip()
                      if hasattr(sig, "spread") else
                      f"rr={getattr(sig,'rr',0):.1f} score={getattr(sig,'score',0):.4f}")
            pos = self._open(sig.symbol, sig.side, sig.model, sig.horizon, sig.move_pct,
                             sig.prob, float(entry_by_sym[sig.symbol]), now, now_dt, reason,
                             size_usd=size_usd, engine=engine,
                             threshold=float(getattr(sig, "threshold", 0.0) or 0.0))
            if pos is not None:
                opened.append(pos)
        return opened

    # ---- one cycle ----
    def sync_exchange_positions(self, now_dt) -> int:
        """Clear local positions that OKX already closed via OCO/manual action.

        The OKX executor exposes `_last_open_positions_ok` so an API error is not
        mistaken for "no open positions".
        """
        if getattr(self.executor, "mode", "") not in ("okx", "binance"):
            return 0
        backend = self.executor.open_positions()
        if not bool(getattr(self.executor, "_last_open_positions_ok", False)):
            self.logger.event("exchange sync skipped: could not read backend positions")
            return 0
        backend_symbols = {str(p.get("symbol")) for p in backend}
        cleared = 0
        with self._position_lock:
            for key, pos in list(self.positions.open_positions.items()):
                if pos.symbol in backend_symbols:
                    continue
                self.positions.close(key, 0.0, now_dt)
                self.logger.event(f"exchange_closed {pos.symbol}: backend no longer reports open; local leg cleared")
                self.logger.log_trade({"event": "exchange_close", "engine": pos.engine,
                                       "symbol": pos.symbol, "model": pos.model, "side": pos.side,
                                       "horizon": pos.horizon,
                                       "entry_price": pos.entry_price,
                                       "exit_price": "",
                                       "size_usd": pos.size_usd,
                                       "pnl_pct": "",
                                       "outcome": "exchange"})
                cleared += 1
        return cleared

    def scan_once(self, now: pd.Timestamp | None = None) -> list[Position]:
        if now is None:
            interval_sec = max(60, int(round(float(self.scan_interval_min) * 60.0)))
            anchor_sec = (int(dt.datetime.now(dt.timezone.utc).timestamp()) // interval_sec) * interval_sec
            now = pd.Timestamp.fromtimestamp(anchor_sec, tz="UTC")
        now = self._utc(now)
        if self.fetcher is not None:
            for sym in self.symbols:
                try:
                    self.fetcher.update_recent(sym, C.LIVE_UPDATE_LOOKBACK_MIN)
                except Exception as e:
                    self.logger.event(f"update {sym} failed: {e}")

        now = self._select_scan_anchor(now)
        snap = self._snapshot(now)
        if snap.empty:
            self.logger.event("empty snapshot")
            return []

        now_dt = now.to_pydatetime()
        synced = self.sync_exchange_positions(now_dt)

        if self.trust_engine is not None:
            self.check_exits(now, snap)
            harvested = self.harvest_exits(snap, now_dt) if self.green_harvest else 0
            opened = self._scan_trust(snap, now, now_dt)
            near_misses = getattr(self.trust_engine, "last_near_misses", [])
            if not opened and near_misses:
                self.logger.event("near_miss: " + " | ".join(str(x) for x in near_misses))
            with self._position_lock:
                open_count = len(self.positions.open_positions)
            self.logger.event(f"scan @ {now}: {len(snap)} symbols, {len(opened)} opened, "
                              f"{harvested} harvested, {synced} synced, "
                              f"{open_count} open "
                              f"[{self._engine_label()}]")
            return opened

        opened: list[Position] = []
        prob_cols = [c for c in snap.columns if c.startswith("prob_")]
        for _, row in snap.iterrows():
            sym = row["symbol"]
            entry = float(row["entry_price"])
            candles = self.store.load(sym)
            ts = index_to_ns(candles.index)
            close = candles["close"].to_numpy(float)
            reg = self.regime.detect(ts, close, ts_to_ns(now), entry)
            probs = {c[5:]: float(row[c]) for c in prob_cols}
            for sig in self.strategy.entries(sym, probs, reg):
                with self._position_lock:
                    ok, why = self.positions.can_open(sym, now_dt)
                if not ok:
                    continue
                fill = self.executor.enter(sym, sig.side, entry,
                                           self.trade_size_usd, move_pct=sig.move_pct)
                if not fill.ok:
                    self.logger.log_decision(symbol=sym, model=sig.model, side=sig.side,
                                             prob=sig.prob, threshold=sig.threshold,
                                             action="skip", reason=fill.info)
                    continue
                fill_time = self._fill_timestamp(fill, now)
                actual_entry = float(fill.entry_price) if float(fill.entry_price or 0) > 0 else float(entry)
                pos = Position(symbol=sym, side=sig.side, model=sig.model,
                               entry_price=actual_entry, size_usd=self.trade_size_usd,
                               move_pct=sig.move_pct, horizon=sig.horizon,
                               opened_at=fill_time.isoformat())
                with self._position_lock:
                    self.positions.open(pos, fill_time.to_pydatetime())
                opened.append(pos)
                self.logger.log_decision(symbol=sym, model=sig.model, side=sig.side,
                                         prob=sig.prob, threshold=sig.threshold,
                                         action="open", reason=f"{self.executor.mode}:fill_at={fill_time.isoformat()} {fill.info}")
                self.logger.log_trade({"event": "open", "symbol": sym, "model": sig.model,
                                       "side": sig.side, "horizon": sig.horizon,
                                       "entry_price": actual_entry, "size_usd": self.trade_size_usd,
                                       "opened_at": fill_time.isoformat()})
        self.check_exits(now, snap)
        with self._position_lock:
            open_count = len(self.positions.open_positions)
        self.logger.event(f"scan @ {now}: {len(snap)} symbols, {len(opened)} opened, "
                          f"{open_count} open")
        return opened

    def check_exits(self, now: pd.Timestamp, snap: pd.DataFrame | None = None) -> None:
        now = self._utc(now)
        now_dt = now.to_pydatetime()
        price = {}
        if snap is not None and not snap.empty and "entry_price" in snap.columns:
            price = dict(zip(snap["symbol"], snap["entry_price"]))
        with self._position_lock:
            for key, pos in list(self.positions.open_positions.items()):
                opened = self._utc(pos.opened_at)
                if now < opened + pd.Timedelta(minutes=self._horizon_min[pos.horizon]):
                    continue
                cur = price.get(pos.symbol)
                if cur is None:
                    cur = self._store_price_at(pos.symbol, now)
                pnl_pct = 0.0
                if cur is not None and pos.entry_price > 0:
                    sgn = 1 if pos.side == "long" else -1
                    pnl_pct = sgn * (float(cur) / pos.entry_price - 1.0) * 100
                try:
                    if self.max_legs > 1 and hasattr(self.executor, "partial_close"):
                        self.executor.partial_close(pos.symbol, pos.side, pos.sz)
                    elif hasattr(self.executor, "force_close"):
                        self.executor.force_close(pos.symbol)
                except Exception as e:
                    self.logger.event(f"deadline close {pos.symbol} failed: {e}")
                self.positions.close(key, pnl_pct, now_dt)
                self.logger.log_trade({"event": "deadline_close", "engine": pos.engine,
                                       "symbol": pos.symbol, "model": pos.model, "side": pos.side,
                                       "horizon": pos.horizon,
                                       "entry_price": pos.entry_price,
                                       "exit_price": "" if cur is None else cur,
                                       "size_usd": pos.size_usd,
                                       "pnl_pct": round(pnl_pct, 3),
                                       "outcome": "deadline",
                                       "closed_at": now.isoformat()})

    def reconcile(self, now: pd.Timestamp | None = None) -> None:
        """Adopt positions already open on the exchange so the loop won't re-open
        them and will still apply a deadline safety net (OCO usually exits first).
        """
        now = self._utc(now or pd.Timestamp.now(tz="UTC").floor("1min"))
        now_dt = now.to_pydatetime()
        horizons = self._horizon_min
        if self.trust_engine is not None and hasattr(self.trust_engine, "horizon_minutes"):
            horizons = getattr(self.trust_engine, "horizon_minutes")
        longest = max(horizons, key=lambda h: horizons[h])
        for p in self.executor.open_positions():
            sym = p["symbol"]
            with self._position_lock:
                if sym in self.positions.open_positions:
                    continue
                self.positions.open(Position(symbol=sym, side=p["side"], model="reconciled",
                                             entry_price=0.0, size_usd=self.trade_size_usd,
                                             move_pct=0.0, horizon=longest,
                                             opened_at=now.isoformat()), now_dt)
        with self._position_lock:
            n = len(self.positions.open_positions)
        if n:
            self.logger.event(f"reconciled {n} existing position(s) from exchange")

    def _deadline_loop(self) -> None:
        self.logger.event(f"deadline loop start: every {self.deadline_check_sec:.1f}s")
        while not self._deadline_stop.wait(self.deadline_check_sec):
            try:
                self.check_exits(pd.Timestamp.now(tz="UTC"))
            except Exception as e:
                self.logger.event(f"deadline loop error: {e}")

    def _start_deadline_loop(self) -> None:
        if self._deadline_thread is not None and self._deadline_thread.is_alive():
            return
        self._deadline_stop.clear()
        self._deadline_thread = threading.Thread(
            target=self._deadline_loop,
            name="live-deadline-loop",
            daemon=True,
        )
        self._deadline_thread.start()

    def _stop_deadline_loop(self) -> None:
        self._deadline_stop.set()
        thread = self._deadline_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def run(self) -> None:
        self.reconcile()
        self.logger.event(f"LIVE start: backend={self.executor.mode} "
                          f"symbols={len(self.symbols)} interval={self.scan_interval_min}m "
                          f"confirm={self.scan_confirmation_sec:.1f}s "
                          f"engine={self._engine_label()}")
        self._start_deadline_loop()
        try:
            while True:
                anchor, sleep_sec = self._next_scan_slot()
                self.logger.event(f"next scan anchor={anchor} in {sleep_sec:.1f}s")
                time.sleep(sleep_sec)
                try:
                    self.scan_once(anchor)
                except Exception as e:
                    self.logger.event(f"scan error: {e}")
        finally:
            self._stop_deadline_loop()
