"""Portfolio live engine: run SEVERAL saved explorer builds as ONE risk book.

Each build = the explorer JSON {sim, levels:[{conditions:[{regulator,value}]}],
banned}. This engine scores every distinct model once per scan, applies each
build's level filters (OR within a level, AND across levels — identical to
src/run_hc_build.py and the explorer), then CROSS-DEDUPS across builds so there
is never more than one position per (symbol, scan) — the higher p_dir wins.

LiveTrader then provides the shared risk book for free: one PositionManager =
shared max_concurrent + per-symbol cooldown + one-position-per-symbol. Real
orders only happen when LiveTrader is given an OKXExecutor(live=True); with
Shadow/Paper executors this is a dry run on live data.

The translation of a build's leg predicate matches the explorer exactly:
  side / p_dir / p_opp / spread / horizon-set / hour-of-day(Kyiv) / asset / lean.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..hc import config as HC
from ..hc_model_registry import SIM_TO_DIR, model_schema as _model_schema
from ..markets import is_equity
from .hc_live_engine import HCLiveEngine, HCLiveSignal
from .hc_v4_live_engine import HCV4LiveEngine

# default horizon grid when a build has no explicit horizon condition
_DEFAULT_HORIZONS = (30, 60, 90)


def _cond(leg: dict, c: dict) -> bool:
    """One condition vs one leg. Mirror of run_hc_build._cond + explorer cond()."""
    r = c.get("regulator") or c.get("reg")
    v = c.get("value", c.get("val"))
    pd_ = leg["pd"]; po = leg["po"]; sp = leg["sp"]; h = leg["h"]
    side = leg["side"]; eq = leg["eq"]; lean = leg.get("lean", 0.0); hod = leg["hod"]
    if r == "p_dir":    return pd_ >= v
    if r == "p_dir_max": return pd_ <= v
    if r == "p_dir_sides": return pd_ >= v[0] if side > 0 else pd_ >= v[1]
    if r == "p_dir_long":  return side < 0 or pd_ >= v
    if r == "p_dir_short": return side > 0 or pd_ >= v
    if r == "p_opp":    return po <= v
    if r == "spread":   return sp >= v
    if r == "wday":     return leg["wd"] in set(v or [])
    if r == "lean_min": return lean >= v
    if r == "lean_max": return lean <= v
    if r == "hmin":     return h >= v
    if r == "hmax":     return h <= v
    if r == "horizon":  return h in set(v or [])
    if r == "hour":     return hod in set(v or [])
    if r == "asset":    return v == "both" or (eq == 1 if v == "equity" else eq == 0)
    if r == "side":     return v == "both" or (side > 0 if v == "long" else side < 0)
    if r == "cost_max":
        raise NotImplementedError("cost_max is explorer/offline-only for now — live legs carry no per-symbol cost")
    # unknown regulator must FAIL LOUDLY, not silently loosen the filter
    raise ValueError(f"unknown build regulator: {r!r}")


def _level_conds(level: dict) -> list[dict]:
    return level.get("conditions") or level.get("conds") or []


def _apply(legs: list[dict], levels: list[dict]) -> list[dict]:
    """OR within a level, AND across levels (sequential narrowing)."""
    cur = legs
    for L in levels:
        conds = _level_conds(L)
        if not conds:
            continue
        cur = [l for l in cur if any(_cond(l, c) for c in conds)]
    return cur


def _build_horizons(build: dict) -> list[int]:
    hs: set[int] = set()
    for L in build.get("levels", []):
        for c in _level_conds(L):
            if (c.get("regulator") or c.get("reg")) == "horizon":
                for x in (c.get("value", c.get("val")) or []):
                    hs.add(int(x))
    return sorted(hs) if hs else list(_DEFAULT_HORIZONS)


class HCPortfolioEngine:
    """Trust-engine duck-type for LiveTrader, combining N builds into one book."""

    horizon_exit_only = True
    default_system_name = "Dancing Horizon"

    def __init__(
        self,
        builds: list[dict],
        *,
        notional_usd: float,
        entry_delay_min: int = HC.EXEC_ENTRY_DELAY_MIN,
        universe_path: Path = Path("configs/hc_universe_full.json"),
        system_name: str = default_system_name,
        profile: str = "portfolio",
        min_p_dir: float = 0.70,
        slots_per_engine: int = 0,
        consensus_boost: dict | None = None,
        max_stake_mult: float = 3.0,
    ) -> None:
        if not builds:
            raise ValueError("portfolio engine needs at least one build")
        self.system_name = str(system_name)
        self.profile = str(profile)
        self.notional_usd = float(notional_usd)
        self.entry_delay_min = int(entry_delay_min)
        self.universe_path = Path(universe_path)
        # global safety floor = the explorer data.js floor every build was tuned under;
        # restores the implicit p_dir gate for builds that have no explicit p_dir level (d8).
        self.min_p_dir = float(min_p_dir)
        # per-engine slot quota so one engine can't crowd out the others (0 = no cap).
        self.slots_per_engine = int(slots_per_engine)
        # consensus boost: {votes: stake_mult} — when >=N builds want the same
        # (symbol, side) this scan, the winning order's stake is multiplied.
        # JSON keys arrive as strings; normalise. votes=1 is implicitly 1.0.
        self.consensus_boost = {int(k): float(v) for k, v in (consensus_boost or {}).items()}
        # hard cap on the combined multiplier (build x sizing x consensus) — the
        # "nothing's on fire" guard: no config combination can exceed it.
        self.max_stake_mult = float(max_stake_mult)
        self.last_near_misses: list[str] = []

        # normalise builds
        self.builds = []
        all_h: set[int] = set()
        for b in builds:
            sim = b.get("sim", "OLD")
            mdir = SIM_TO_DIR.get(sim, sim)  # allow raw path too
            horizons = _build_horizons(b)
            all_h.update(horizons)
            sizing = b.get("sizing") if isinstance(b.get("sizing"), dict) else None
            if sizing is not None and not sizing.get("on", True):
                sizing = None
            self.builds.append({
                "name": b.get("name") or b.get("description") or sim,
                "sim": sim,
                "model_dir": str(mdir),
                "levels": b.get("levels", []),
                "banned": set(b.get("banned", [])),
                "horizons": horizons,
                "stake_mult": float(b.get("stake_mult", 1.0)),
                "sizing": sizing,
            })
        self.union_horizons = tuple(sorted(all_h))
        # label->minutes map LiveTrader needs for deadline exits
        self.horizon_minutes = {f"{h}m": h for h in self.union_horizons}

        # horizons per distinct model dir (v4 scorers query explicit sets)
        md_horizons: dict[str, set[int]] = {}
        for b in self.builds:
            md_horizons.setdefault(b["model_dir"], set()).update(b["horizons"])

        # one scorer per DISTINCT model dir: 302 -> HCLiveEngine, v4(1-min) -> HCV4LiveEngine
        self._scorers: dict[str, object] = {}
        self._v4_keys: set[str] = set()
        self._scored: dict[str, pd.DataFrame] = {}
        for md, hset in md_horizons.items():
            if not Path(md).exists():
                raise ValueError(f"model dir not found: {md!r} — unknown sim "
                                 f"(add it to SIM_TO_DIR or fix the build)")
            schema = _model_schema(md)
            if schema == "v4":
                self._scorers[md] = HCV4LiveEngine(
                    model_dir=Path(md),
                    high=self.min_p_dir,
                    horizons=tuple(sorted(hset)) or self.union_horizons,
                    entry_delay_min=self.entry_delay_min,
                    notional_usd=self.notional_usd,
                    universe_path=self.universe_path,
                    system_name=self.system_name,
                )
                self._v4_keys.add(self._key(md))
            elif schema == "legacy":
                self._scorers[md] = HCLiveEngine(
                    model_dir=Path(md),
                    horizons=self.union_horizons,
                    horizon_min=min(self.union_horizons),
                    horizon_max=max(self.union_horizons),
                    entry_delay_min=self.entry_delay_min,
                    notional_usd=self.notional_usd,
                    system_name=self.system_name,
                )
            else:  # v2 (5m, no-BTC) / v5 (regime block) — no live feature builder wired
                raise ValueError(
                    f"model {md!r} has schema '{schema}' — live scoring NOT wired "
                    f"(v5 regime features are dataset-only so far). Remove builds on "
                    f"this model from the portfolio or wire the scorer first.")

    # ---- info ----
    def describe(self) -> str:
        parts = []
        for b in self.builds:
            parts.append(f"{b['name']}[{b['sim']}|h{min(b['horizons'])}-{max(b['horizons'])}]")
        return (f"{self.system_name} / portfolio({len(self.builds)}): " + " + ".join(parts) +
                f" | union_h={len(self.union_horizons)} | ${self.notional_usd:.2f}/pos | "
                f"models={len(self._scorers)}")

    # ---- watchlist ----
    def build_watchlist(self, store, top_n: int = 0, logger=None) -> list[str]:
        data = json.loads(self.universe_path.read_text(encoding="utf-8"))
        universe = data.get("symbols", data) if isinstance(data, dict) else data
        blacklist = set(C.hc_blacklist_symbols())
        # a build's ban list filters THAT build's legs; only symbols banned by
        # every build leave the shared watchlist (union muted symbols for all)
        banned_all = (set.intersection(*[set(b["banned"]) for b in self.builds])
                      if self.builds else set())
        skip = blacklist | banned_all
        vols: list[tuple[str, float]] = []
        missing = 0
        for sym in sorted(str(s) for s in universe):
            if sym in skip:
                continue
            candles = store.load(sym)
            if candles is None or candles.empty:
                missing += 1
                continue
            tail = candles.iloc[-1440:]
            vol = float((tail["close"] * tail["volume"]).sum())
            if np.isfinite(vol):
                vols.append((sym, vol))
        vols.sort(key=lambda item: item[1], reverse=True)
        limit = int(top_n or 0)
        watch = [s for s, _ in (vols[:limit] if limit > 0 else vols)]
        if logger is not None:
            scope = f"top {limit}" if limit > 0 else "all"
            logger.event(f"watchlist: portfolio universe {scope}: {len(watch)} symbols "
                         f"(universe={len(universe)}, missing={missing}, "
                         f"blacklist={len(blacklist)}, banned={len(banned_all)})")
        return watch

    # ---- snapshot: score each distinct model once; stash per-model frames ----
    # (no cross-model merge — each build reads its OWN model's frame in decide,
    #  so 302 and v4(1-min) models with different horizon grids coexist.)
    _META_COLS = ["symbol", "anchor_time", "base_time", "entry_price",
                  "entry_source_time", "horizon_minutes"]

    def snapshot(self, store, symbols: list[str], now: pd.Timestamp) -> pd.DataFrame:
        self._scored = {}
        metas = []
        for md, scorer in self._scorers.items():
            s = scorer.snapshot(store, symbols, now)
            if s is None or s.empty:
                continue
            self._scored[self._key(md)] = s
            metas.append(s[self._META_COLS])
        if not metas:
            return pd.DataFrame()
        # union meta frame for LiveTrader (entry_price for exits; symbol set)
        return pd.concat(metas, ignore_index=True).drop_duplicates(
            ["symbol", "horizon_minutes"]).reset_index(drop=True)

    @staticmethod
    def _key(model_dir: str) -> str:
        return Path(model_dir).name

    # ---- legs for one build from its OWN model's scored frame ----
    def _legs_for_build(self, build: dict) -> list[dict]:
        key = self._key(build["model_dir"])
        sf = self._scored.get(key)
        if sf is None or sf.empty:
            return []
        banned = build["banned"]
        # base_time -> Kyiv hour-of-day (matches explorer: (epoch_min+180)//60 %24)
        bt = pd.to_datetime(sf["base_time"], utc=True)
        epoch_min = ((bt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy()
        hod_arr = ((epoch_min + 180) // 60) % 24
        wd_arr = ((epoch_min + 180) // 1440 + 4) % 7
        legs: list[dict] = []
        sym = sf["symbol"].to_numpy()
        hm = sf["horizon_minutes"].to_numpy()
        up = sf["up_prob"].to_numpy(dtype=float)
        dn = sf["down_prob"].to_numpy(dtype=float)
        floor = self.min_p_dir
        for i in range(len(sf)):
            s = str(sym[i])
            if s in banned:
                continue
            h = int(hm[i]); hod = int(hod_arr[i]); wd = int(wd_arr[i]); eq = 1 if is_equity(s) else 0
            u = float(up[i]); d = float(dn[i])
            # long leg (global p_dir floor = explorer data floor)
            if u >= floor:
                legs.append({"sym": s, "h": h, "hod": hod, "wd": wd, "eq": eq, "side": 1,
                             "pd": u, "po": d, "sp": u - d})
            # short leg
            if d >= floor:
                legs.append({"sym": s, "h": h, "hod": hod, "wd": wd, "eq": eq, "side": -1,
                             "pd": d, "po": u, "sp": d - u})
        # per-scan lean (regime): avg(p_dir|long) - avg(p_dir|short) over candidate legs
        lp = [l["pd"] for l in legs if l["side"] > 0]
        sp = [l["pd"] for l in legs if l["side"] < 0]
        lean = (sum(lp) / len(lp) if lp else 0.0) - (sum(sp) / len(sp) if sp else 0.0)
        for l in legs:
            l["lean"] = lean
        return legs

    # ---- stake multiplier: build base x p_dir-tier x consensus, hard-capped ----
    @staticmethod
    def _sizing_mult(sizing: dict | None, p_dir: float) -> float:
        """Mirror of explorer szFn(): pd>=t1 -> m1, pd>=t2 -> m2, else m3."""
        if not sizing:
            return 1.0
        if p_dir >= float(sizing.get("t1", 0.85)):
            return float(sizing.get("m1", 1.5))
        if p_dir >= float(sizing.get("t2", 0.75)):
            return float(sizing.get("m2", 1.0))
        return float(sizing.get("m3", 0.5))

    def _consensus_mult(self, votes: int) -> float:
        """Largest configured threshold <= votes wins; 1 vote = flat 1.0."""
        mult = 1.0
        for need in sorted(self.consensus_boost):
            if votes >= need:
                mult = self.consensus_boost[need]
        return mult

    # ---- decide: filter each build, cross-dedup, return signals ----
    def decide(self, snap: pd.DataFrame, top_n: int = 10) -> list[HCLiveSignal]:
        self.last_near_misses = []
        if snap is None or snap.empty:
            return []
        # symbol -> best (leg, build) by p_dir
        best: dict[str, tuple[dict, dict]] = {}
        # (symbol, side) -> set of DISTINCT model dirs that want it. A vote is
        # "an independent MODEL agrees", not "a build agrees": 6 builds on the
        # same d8 model are ONE opinion, not six. Counting builds let same-family
        # builds fake a consensus and boosted ~the whole book (2026-06-13 night).
        votes: dict[tuple[str, int], set[str]] = {}
        per_build_counts: dict[str, int] = {}
        cap = self.slots_per_engine if self.slots_per_engine > 0 else None
        for build in self.builds:
            legs = self._legs_for_build(build)
            sel = _apply(legs, build["levels"])
            # risk-unit: one best leg per symbol within this build
            by_sym: dict[str, dict] = {}
            for l in sel:
                cur = by_sym.get(l["sym"])
                if cur is None or l["pd"] > cur["pd"]:
                    by_sym[l["sym"]] = l
            per_build_counts[build["name"]] = len(by_sym)
            fam = build["model_dir"]
            for l in by_sym.values():
                k = (l["sym"], l["side"])
                votes.setdefault(k, set()).add(fam)
            # per-engine slot quota: this build only offers its top `cap` by p_dir
            offered = sorted(by_sym.values(), key=lambda l: l["pd"], reverse=True)
            if cap is not None:
                offered = offered[:cap]
            for l in offered:
                s = l["sym"]
                prev = best.get(s)
                if prev is None or l["pd"] > prev[0]["pd"]:
                    best[s] = (l, build)

        rows = sorted(best.values(), key=lambda kv: kv[0]["pd"], reverse=True)[: int(top_n)]
        signals: list[HCLiveSignal] = []
        for leg, build in rows:
            bname = build["name"]
            h = int(leg["h"])
            side = "long" if leg["side"] > 0 else "short"
            # distinct MODELS that agreed on this (symbol, side); >=1 by construction
            n_votes = len(votes.get((leg["sym"], leg["side"]), {build["model_dir"]}))
            mult = (build["stake_mult"]
                    * self._sizing_mult(build["sizing"], float(leg["pd"]))
                    * self._consensus_mult(n_votes))
            mult = min(self.max_stake_mult, mult)
            if mult <= 0:   # stake_mult=0 = "vote-only" build: counts votes, never orders
                continue
            signals.append(HCLiveSignal(
                symbol=leg["sym"],
                model=f"{bname}_{h}m",
                side=side,
                horizon=f"{h}m",
                move_pct=HC.threshold_pct(h) / 100.0,
                prob=float(leg["pd"]),
                score=float(leg["sp"]),
                spread=float(leg["sp"]),
                agree=n_votes,
                size_mult=mult,
                source=(f"engine={bname} p_dir={leg['pd']:.4f} opp={leg['po']:.4f} "
                        f"hod={leg['hod']} votes={n_votes} mult={mult:.2f}"),
                engine=bname,
                size_usd=self.notional_usd * mult,
                threshold=0.0,
            ))
        self.last_near_misses = [f"{b}:{n} cand" for b, n in per_build_counts.items()]
        return signals
