"""Local control-panel server for the HC Sim Explorer (stdlib only, localhost).

Run:  python -m dh.webapp.server   (then open http://127.0.0.1:8765)

Serves reports/sim_explorer/* and a small JSON API that launches the existing
Python tools as background jobs:
  POST /api/fetch   {lookback}                  -> run_fetcher (refresh candles)
  POST /api/data    {hours,floor}               -> run_hc_export_html (regen data.js)
  POST /api/train   {name,days,depth,iters,...} -> dataset build + prod train
  POST /api/livestop {id}                       -> stop a standing job
  GET  /api/jobs                                -> all jobs (status)
  GET  /api/job?id=                             -> one job + log tail
  GET  /api/models                              -> model metadata
  POST /api/portfolio {portfolio|build_names,mode,...} -> run a portfolio of builds
  GET/POST/DELETE /api/builds                   -> saved builds (with description)
/api/portfolio supports mode=live
(real OKX orders) — the runner self-guards on credentials; use small stakes.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[2]
EXPLORER = ROOT / "reports" / "sim_explorer"
BUILDS_DIR = ROOT / "configs" / "builds"
LOG_DIR = ROOT / "outputs" / "webapp_jobs"
PY = sys.executable
HOST, PORT = "127.0.0.1", 8765

MODELS = [
    ("d7 (hc_final)", "models/hc_final", "depth7 · dense 5–180 · 319 syms · cutoff Jun5"),
    ("d8 (hc_final_d8)", "models/hc_final_d8", "depth8 · conservative sniper"),
    ("OLD", "models/hc_exec_stride120_nonoverlap", "trained to 2026-05-26"),
    ("NEW", "models/hc_exec_to20260604_prod", "trained to 2026-06-04 · live bad_day_worker"),
]

JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()

AUTO_DATA_REFRESH = False   # button-only by default (explorer-only; live fetches its own candles)
AUTO_DATA_CHECK_SEC = 60
AUTO_DATA_MIN_LAUNCH_GAP_SEC = 120
AUTO_DATA_HOURS = 64
AUTO_DATA_FLOOR = 0.70
AUTO_DATA_DENSE = ",".join(str(x) for x in range(20, 181, 5))
_AUTO_DATA = {"last_check": 0.0, "last_launch": 0.0, "last_job": None, "last_error": ""}


def _set_autodata(enabled: bool) -> bool:
    """Toggle the explorer auto-refresh loop at runtime (no restart needed)."""
    global AUTO_DATA_REFRESH
    AUTO_DATA_REFRESH = bool(enabled)
    return AUTO_DATA_REFRESH


def _launch(name: str, args: list[str], standing: bool = False, kind: str = "") -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jid = uuid.uuid4().hex[:8]
    logp = LOG_DIR / f"{jid}.log"
    f = logp.open("w", encoding="utf-8")
    cmd = [PY, "-c", args[1]] if args and args[0] == "__pyc__" else [PY, "-m", *args]
    f.write(f"$ {' '.join(cmd[:2])} ...\n\n"); f.flush()
    # children must not die on emoji/cyrillic prints under cp1251 consoles
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, env=env)
    with _LOCK:
        JOBS[jid] = {"id": jid, "name": name, "status": "running", "started": time.time(),
                     "log": str(logp), "standing": standing, "kind": kind,
                     "_proc": proc, "_f": f, "_cmd": cmd}

    def waiter():
        rc = proc.wait()
        f.flush(); f.close()
        with _LOCK:
            j = JOBS.get(jid)
            if j:
                j["status"] = "done" if rc == 0 else f"failed({rc})"
                j["ended"] = time.time()
    threading.Thread(target=waiter, daemon=True).start()
    return jid


def _job_view(j: dict) -> dict:
    return {k: v for k, v in j.items() if not k.startswith("_")}


def _running_job(kind: str) -> dict | None:
    with _LOCK:
        for j in sorted(JOBS.values(), key=lambda x: -x["started"]):
            if j.get("kind") == kind and j.get("status") == "running":
                return _job_view(j)
    return None


def _model_info(name: str, rel: str, blurb: str) -> dict:
    d = ROOT / rel
    out = {"name": name, "dir": rel, "blurb": blurb, "exists": d.exists()}
    snap = d / "config_snapshot.json"
    if snap.exists():
        try:
            s = json.loads(snap.read_text(encoding="utf-8"))
            mp = s.get("actual_model_params") or s.get("model_params") or {}
            out.update({
                "cutoff": s.get("cutoff_utc"),
                "depth": mp.get("depth"),
                "iterations": mp.get("iterations"),
                "features": s.get("feature_count") or s.get("expected_feature_count"),
                "symbols": s.get("symbols"),
                "base_min": s.get("base_time_min"), "base_max": s.get("base_time_max"),
                "tag": s.get("tag_feature"),
                "folds": [f.get("name") for f in s.get("folds", [])],
            })
        except Exception as e:
            out["error"] = str(e)
    met = d / "metrics.json"
    if met.exists():
        try:
            m = json.loads(met.read_text(encoding="utf-8"))
            rec = m[0] if isinstance(m, list) and m else m
            mods = rec.get("models", {}) if isinstance(rec, dict) else {}
            out["val_auc"] = {k: (v.get("val_auc") if isinstance(v, dict) else None) for k, v in mods.items()}
        except Exception:
            pass
    return out


def _num(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _csv_rows(fp: Path) -> list[dict]:
    if not fp.exists():
        return []
    try:
        with fp.open("r", newline="", encoding="utf-8", errors="replace") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _shadow_run_summary(run_dir: Path) -> dict | None:
    events_fp = run_dir / "events.log"
    trades_fp = run_dir / "trades.csv"
    events = []
    if events_fp.exists():
        try:
            events = [x for x in events_fp.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        except Exception:
            events = []
    backend = ""
    for line in events:
        marker = "backend="
        if marker in line:
            backend = line.split(marker, 1)[1].split()[0].strip()
            break
    if backend and backend != "shadow":
        return None

    trades = _csv_rows(trades_fp)
    if not backend:
        shadowish = any("shadow" in str(r).lower() for r in trades) or any("shadow" in x.lower() for x in events)
        if not shadowish:
            return None
        backend = "shadow"

    opens = [r for r in trades if r.get("event") == "open"]
    closes = [r for r in trades if r.get("event") and r.get("event") != "open"]
    pnl_usd = sum(_num(r.get("size_usd")) * _num(r.get("pnl_pct")) / 100.0 for r in closes)
    pnl_pcts = [_num(r.get("pnl_pct")) for r in closes if str(r.get("pnl_pct", "")).strip()]
    wins = sum(1 for x in pnl_pcts if x > 0)
    stat_files = [p for p in (events_fp, trades_fp, run_dir / "decisions.csv") if p.exists()]
    last_ts = max((p.stat().st_mtime for p in stat_files), default=run_dir.stat().st_mtime)
    return {
        "dir": run_dir.name,
        "backend": backend,
        "started": (events[0][:19] if events else run_dir.name.replace("live_", "")),
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_ts)),
        "opens": len(opens),
        "closes": len(closes),
        "open_now": max(0, len(opens) - len(closes)),
        "realized_usd": round(pnl_usd, 2),
        "avg_pnl_pct": round(sum(pnl_pcts) / len(pnl_pcts), 3) if pnl_pcts else 0.0,
        "winrate_pct": round(wins * 100.0 / len(pnl_pcts), 1) if pnl_pcts else 0.0,
        "last_event": events[-1] if events else "",
    }


def _shadow_runs() -> list[dict]:
    root = ROOT / "outputs" / "trading_logs"
    if not root.exists():
        return []
    out = []
    for run_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        item = _shadow_run_summary(run_dir)
        if item is not None:
            out.append(item)
        if len(out) >= 25:
            break
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, obj=None, raw=None, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        body = raw if raw is not None else json.dumps(obj or {}).encode("utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # ---- static ----
    def _serve_static(self, path):
        rel = path.lstrip("/") or "index.html"
        fp = (EXPLORER / rel).resolve()
        if not str(fp).startswith(str(EXPLORER.resolve())) or not fp.is_file():
            return self._send(404, {"error": "not found"})
        ctype = ("text/html" if fp.suffix == ".html" else
                 "application/javascript" if fp.suffix == ".js" else
                 "text/markdown" if fp.suffix == ".md" else "text/plain")
        self._send(200, raw=fp.read_bytes(), ctype=ctype + "; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        p, q = u.path, parse_qs(u.query)
        if p == "/api/jobs":
            with _LOCK:
                return self._send(200, {"jobs": [_job_view(j) for j in sorted(JOBS.values(), key=lambda x: -x["started"])]})
        if p == "/api/job":
            jid = (q.get("id") or [""])[0]
            with _LOCK:
                j = JOBS.get(jid)
            if not j:
                return self._send(404, {"error": "no job"})
            tail = ""
            try:
                tail = Path(j["log"]).read_text(encoding="utf-8", errors="replace")[-6000:]
            except Exception:
                pass
            return self._send(200, {**_job_view(j), "tail": tail})
        if p == "/api/status":
            with _LOCK:
                running = sum(1 for j in JOBS.values() if j["status"] == "running")
            candle_edge = _btc_edge()
            data_window = _data_window()
            return self._send(200, {"now": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "candle_edge": candle_edge, "data_window": data_window,
                                    "data_refresh": _maybe_auto_data_refresh(candle_edge, data_window),
                                    "running_jobs": running})
        if p == "/api/models":
            return self._send(200, {"models": [_model_info(*m) for m in MODELS]})
        if p == "/api/shadowruns":
            return self._send(200, {"runs": _shadow_runs()})
        if p == "/api/builds":
            BUILDS_DIR.mkdir(parents=True, exist_ok=True)
            out = []
            for fp in sorted(BUILDS_DIR.glob("*.json")):
                try:
                    cfg = json.loads(fp.read_text(encoding="utf-8"))
                    cfg.setdefault("name", fp.stem)
                    cfg["is_portfolio"] = bool(cfg.get("builds"))
                    out.append(cfg)
                except Exception:
                    pass
            return self._send(200, {"builds": out})
        if p.startswith("/api/"):
            return self._send(404, {"error": "unknown api"})
        return self._serve_static(p)

    def do_DELETE(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/api/builds":
            nm = (q.get("name") or [""])[0]
            fp = _find_build_file(nm)
            if fp is not None:
                fp.unlink()
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "?"})

    def do_POST(self):
        u = urlparse(self.path); p = u.path; b = self._body()
        if p == "/api/fetch":
            lb = int(b.get("lookback", 1500))
            if b.get("then_regen"):
                hours = str(b.get("hours", 64)); floor = str(b.get("floor", 0.70))
                dense = b.get("dense", ",".join(str(x) for x in range(20, 181, 5)))
                drv = ("import subprocess,sys;"
                       f"subprocess.run([sys.executable,'-m','src.run_fetcher','--once','--universe','store','--workers','10','--lookback-min',{str(lb)!r}]);"
                       f"subprocess.run([sys.executable,'-m','src.run_hc_export_html','--hours',{hours!r},'--floor',{floor!r},'--dense',{dense!r}])")
                jid = _launch("fetch + update data", _pyc(drv), kind="data")
            else:
                jid = _launch("fetch candles", ["src.run_fetcher", "--once", "--universe", "store",
                                                "--workers", "10", "--lookback-min", str(lb)])
            return self._send(200, {"id": jid})
        if p == "/api/refreshall":
            # "idiot-proof" button: fetch candles to now -> regenerate the explorer stats.
            # v4/flat models are scored DENSELY on fresh candles inside the export
            # (no dataset to go stale), so this is just fetch + regen.
            lb = int(b.get("lookback", 1500)); hours = str(b.get("hours", 48))
            floor = str(b.get("floor", 0.65))
            drv = ("import subprocess,sys;"
                   f"subprocess.run([sys.executable,'-m','src.run_fetcher','--once','--universe','store','--workers','10','--lookback-min',{str(lb)!r}]);"
                   f"subprocess.run([sys.executable,'-m','src.run_hc_export_html','--hours',{hours!r},'--floor',{floor!r}])")
            jid = _launch("🙋 fetch to now + refresh stats", _pyc(drv), kind="data")
            return self._send(200, {"id": jid})
        if p == "/api/runbuild":
            fp = _find_build_file(b.get("name", ""))
            if fp is None:
                return self._send(404, {"error": "no build"})
            nm = fp.stem
            if any(k in b for k in ("hours", "from_ago_h", "to_ago_h", "floor")):
                args = ["src.run_hc_build", "--build", str(fp), "--json-out"]
                if b.get("hours") is not None:
                    args += ["--hours", str(b.get("hours"))]
                if b.get("from_ago_h") is not None:
                    args += ["--from-ago-h", str(b.get("from_ago_h"))]
                if b.get("to_ago_h") is not None:
                    args += ["--to-ago-h", str(b.get("to_ago_h"))]
                if b.get("floor") is not None:
                    args += ["--floor", str(b.get("floor"))]
                jid = _launch(f"sim build: {nm}", args)
                return self._send(200, {"id": jid})
                jid = _launch(f"sim build: {nm}", args)
                return self._send(200, {"id": jid})
            jid = _launch(f"sim build: {nm}", ["src.run_hc_build", "--build", str(fp), "--json-out"])
            return self._send(200, {"id": jid})
        if p == "/api/data":
            hours = str(b.get("hours", 64)); floor = str(b.get("floor", 0.70))
            dense = b.get("dense", ",".join(str(x) for x in range(20, 181, 5)))
            jid = _launch_data_refresh("data refresh", hours, floor, dense)
            return self._send(200, {"id": jid})
        if p == "/api/train":
            jid = _train(b)
            return self._send(200, {"id": jid})
        if p == "/api/portfolio":
            try:
                jid = _portfolio(b)
            except RuntimeError as e:
                return self._send(409, {"error": str(e)})
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            return self._send(200, {"id": jid})
        if p == "/api/binanceportfolio":
            try:
                out = _binance_portfolio(b)
            except RuntimeError as e:
                return self._send(409, {"error": str(e)})
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            return self._send(200, out)
        if p == "/api/v4live":
            return self._send(200, {"id": _v4live(b)})
        if p == "/api/ttexport":
            # rebuild the up-to-now TT window (reuses the cached market frame, fast)
            # then score the curve into window.TT. The sterile training set is untouched.
            days = str(int(b.get("days", 18)))
            drv = ("import subprocess,sys;"
                   "b=subprocess.run([sys.executable,'-m','src.run_tt_dataset','--out-dir','data/tt_now/dataset',"
                   f"'--days',{days!r},'--holdout-days','0','--market-cache','data/tt_curve/market_frame.parquet',"
                   "'--workers','8','--fresh']);"
                   "sys.exit(b.returncode) if b.returncode else None;"
                   "e=subprocess.run([sys.executable,'-m','src.run_tt_export_html']);"
                   "sys.exit(e.returncode)")
            jid = _launch("🌀 TT: compute to now → explorer", _pyc(drv), kind="ttexport")
            return self._send(200, {"id": jid})
        if p == "/api/autodata":
            return self._send(200, {"enabled": _set_autodata(bool(b.get("enabled", False)))})
        if p == "/api/binancenow":
            return self._send(200, {"id": _binance_now(b)})
        if p == "/api/exittest":
            try:
                jid = _exittest(b)
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            return self._send(200, {"id": jid})
        if p == "/api/enginetest":
            try:
                jid = _enginetest(b)
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            return self._send(200, {"id": jid})
        if p == "/api/livestop":
            jid = b.get("id", "")
            with _LOCK:
                j = JOBS.get(jid)
            if j and j.get("_proc") and j["status"] == "running":
                try:
                    j["_proc"].terminate()
                except Exception:
                    pass
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "no running job"})
        if p == "/api/builds":
            nm = _safe(b.get("name", ""))
            if not nm:
                return self._send(400, {"error": "no name"})
            BUILDS_DIR.mkdir(parents=True, exist_ok=True)
            (BUILDS_DIR / f"{nm}.json").write_text(json.dumps(b, indent=2, ensure_ascii=False), encoding="utf-8")
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "unknown api"})


def _safe(s: str) -> str:
    return "".join(c for c in str(s) if c.isalnum() or c in "-_ ").strip()[:60]


def _find_build_file(name: str) -> "Path | None":
    """Resolve a saved build by name, tolerating legacy filenames.

    Older _safe() versions allowed chars (e.g. parentheses) the current one
    strips, so disk names and sanitized names drifted. Try: exact filename ->
    sanitized filename -> the cfg's own "name" field inside each json.
    """
    raw = str(name).strip()
    if raw and not any(ch in raw for ch in "\\/:*?\"<>|"):
        fp = BUILDS_DIR / f"{raw}.json"
        if fp.exists():
            return fp
    nm = _safe(raw)
    if nm:
        fp = BUILDS_DIR / f"{nm}.json"
        if fp.exists():
            return fp
    for fp in sorted(BUILDS_DIR.glob("*.json")):
        try:
            cfg = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if cfg.get("name") == raw:
            return fp
    return None


_EDGE_CACHE = {"t": 0, "v": None}
def _btc_edge():
    if time.time() - _EDGE_CACHE["t"] < 8 and _EDGE_CACHE["v"]:
        return _EDGE_CACHE["v"]
    try:
        import pandas as pd
        fp = ROOT / "data" / "candles" / "BTC_USDT_SWAP.parquet"
        v = str(pd.read_parquet(fp, columns=["timestamp"])["timestamp"].max())
    except Exception as e:
        v = f"?({e})"
    _EDGE_CACHE.update(t=time.time(), v=v)
    return v


def _data_window():
    try:
        prefix = "window.SIM_META="
        with (EXPLORER / "data.js").open("r", encoding="utf-8") as f:
            line = f.readline().strip()  # SIM_META is line 1 (any length)
        if not line.startswith(prefix):
            return {}
        m = json.loads(line[len(prefix):].rstrip().rstrip(";"))
        return {"start": m.get("window_start"), "end": m.get("window_end"),
                "floor": m.get("floor"), "horizons": len(m.get("horizons", []))}
    except Exception:
        return {}


def _parse_utc_ts(value):
    if not value or str(value).startswith("?"):
        return None
    try:
        import pandas as pd
        ts = pd.Timestamp(value)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except Exception:
        return None


def _data_refresh_status(candle_edge=None, data_window=None) -> dict:
    edge = _parse_utc_ts(candle_edge or _btc_edge())
    window = data_window if data_window is not None else _data_window()
    data_end = _parse_utc_ts(window.get("end") if isinstance(window, dict) else None)
    running = _running_job("data")
    stale = bool(edge is not None and (data_end is None or edge.floor("h") > data_end.floor("h")))
    lag_min = None
    if edge is not None and data_end is not None:
        lag_min = round((edge - data_end).total_seconds() / 60.0, 1)
    return {
        "enabled": AUTO_DATA_REFRESH,
        "stale": stale,
        "running": bool(running),
        "job_id": running.get("id") if running else _AUTO_DATA.get("last_job"),
        "lag_minutes": lag_min,
        "last_error": _AUTO_DATA.get("last_error", ""),
    }


def _launch_data_refresh(name: str, hours=64, floor=0.70, dense=AUTO_DATA_DENSE) -> str:
    running = _running_job("data")
    if running:
        return running["id"]
    jid = _launch(str(name), ["src.run_hc_export_html", "--hours", str(hours),
                              "--floor", str(floor), "--dense", str(dense)],
                  kind="data")
    _AUTO_DATA.update(last_launch=time.time(), last_job=jid, last_error="")
    return jid


def _maybe_auto_data_refresh(candle_edge=None, data_window=None) -> dict:
    status = _data_refresh_status(candle_edge, data_window)
    now = time.time()
    if not AUTO_DATA_REFRESH or not status["stale"] or status["running"]:
        return status
    if now - float(_AUTO_DATA.get("last_check", 0.0)) < AUTO_DATA_CHECK_SEC:
        return status
    _AUTO_DATA["last_check"] = now
    if now - float(_AUTO_DATA.get("last_launch", 0.0)) < AUTO_DATA_MIN_LAUNCH_GAP_SEC:
        return status
    try:
        jid = _launch_data_refresh(
            "auto data refresh", AUTO_DATA_HOURS, AUTO_DATA_FLOOR, AUTO_DATA_DENSE
        )
        status.update(running=True, job_id=jid)
    except Exception as exc:
        _AUTO_DATA["last_error"] = f"{type(exc).__name__}: {exc}"
        status["last_error"] = _AUTO_DATA["last_error"]
    return status


def _auto_data_refresh_loop() -> None:
    while True:
        try:
            _maybe_auto_data_refresh(_btc_edge(), _data_window())
        except Exception as exc:
            _AUTO_DATA["last_error"] = f"{type(exc).__name__}: {exc}"
        time.sleep(AUTO_DATA_CHECK_SEC)


def _train(b: dict) -> str:
    """dataset build + prod train, chained in one job via a tiny shell-ish wrapper."""
    name = _safe(b.get("name", "hc_custom")) or "hc_custom"
    days = str(int(b.get("days", 14)))
    dense_step = str(int(b.get("dense_step", 5)))
    rcount = str(int(b.get("random_count", 30)))
    depth = str(int(b.get("depth", 7)))
    iters = str(int(b.get("iterations", 6000)))
    cutoff = b.get("cutoff_local") or "2026-06-06 18:00"
    universe = b.get("universe", "configs/hc_universe_full.json")
    ds = f"data/{name}/dataset"
    mdir = f"models/{name}"
    # run_hc_dataset then run_hc_prod_train, sequentially, in one python -c driver
    driver = (
        "import subprocess,sys;"
        f"a=subprocess.run([sys.executable,'-m','src.run_hc_dataset','--stage','dataset','--exec',"
        f"'--universe',{universe!r},'--days',{days!r},'--random-count',{rcount!r},'--random-step-min',{dense_step!r},"
        f"'--out-dir',{ds!r}]);"
        "sys.exit(a.returncode) if a.returncode else None;"
        f"b=subprocess.run([sys.executable,'-m','src.run_hc_prod_train','--dataset-dir',{ds!r},'--model-dir',{mdir!r},"
        f"'--cutoff-local',{cutoff!r},'--depth',{depth!r},'--iterations',{iters!r},'--no-early-stop','--task-type','GPU']);"
        "sys.exit(b.returncode)"
    )
    return _launch(f"training {name}", _pyc(driver), standing=False)


def _pyc(code: str) -> list[str]:
    return ["__pyc__", code]


def _load_saved_build(name: str) -> dict:
    fp = _find_build_file(name)
    if fp is None:
        raise ValueError(f"unknown build: {name}")
    nm = fp.stem
    cfg = json.loads(fp.read_text(encoding="utf-8"))
    if cfg.get("builds"):
        raise ValueError(f"{nm} is a portfolio config, not a single build")
    if not cfg.get("sim") or not cfg.get("levels"):
        raise ValueError(f"{nm} is not a runnable explorer build")
    if cfg.get("reverse"):
        raise ValueError(f"{nm} has reverse=true; live portfolio cannot mirror reverse sims")
    cfg["name"] = cfg.get("name") or nm
    return cfg


def _portfolio_config_from_saved_builds(b: dict) -> Path | None:
    names = b.get("build_names") or b.get("builds")
    if not names:
        return None
    if not isinstance(names, list):
        raise ValueError("build_names must be a list")
    seen: set[str] = set()
    builds = []
    for raw in names:
        # no pre-sanitizing here: _load_saved_build resolves legacy filenames
        nm = str(raw).strip()
        if not nm or nm in seen:
            continue
        seen.add(nm)
        builds.append(_load_saved_build(nm))
    if not builds:
        raise ValueError("select at least one saved build")

    # per-build stake multipliers from the UI ({build name: mult}); 0 = vote-only
    stake_mults = b.get("stake_mults") or {}
    if not isinstance(stake_mults, dict):
        raise ValueError("stake_mults must be {build_name: multiplier}")
    for bd in builds:
        if bd["name"] in stake_mults:
            bd["stake_mult"] = float(stake_mults[bd["name"]])

    # default exit policy: applied to every build that has no exit_policy of its own
    exit_default = b.get("exit_default")
    if exit_default:
        for bd in builds:
            if not bd.get("exit_policy"):
                bd["exit_policy"] = exit_default

    name = _safe(b.get("portfolio_name", "")) or ("picked_" + uuid.uuid4().hex[:6])
    cfg = {
        "name": name,
        "description": "Generated by HC Control from selected saved builds.",
        "stake_margin": float(b.get("stake_margin", 5.0)),
        "leverage": int(b.get("leverage", 3)),
        "max_concurrent": int(b.get("max_concurrent", 12)),
        "cooldown_min": int(b.get("cooldown_min", 30)),
        "top_per_scan": int(b.get("top_per_scan", 12)),
        "min_p_dir": float(b.get("min_p_dir", 0.70)),
        "slots_per_engine": int(b.get("slots_per_engine", 4)),
        "universe": b.get("universe", "configs/hc_universe_full.json"),
        "builds": builds,
    }
    # consensus boost {votes: stake mult} + the hard cap on the combined mult
    if b.get("consensus_boost"):
        cb = b["consensus_boost"]
        if not isinstance(cb, dict):
            raise ValueError("consensus_boost must be {votes: multiplier}")
        cfg["consensus_boost"] = {str(int(k)): float(v) for k, v in cb.items()}
    if b.get("max_stake_mult") is not None:
        cfg["max_stake_mult"] = float(b.get("max_stake_mult"))
    if b.get("save_as"):
        # persist as a reusable portfolio config next to single builds
        sname = _safe(str(b["save_as"]))
        if not sname:
            raise ValueError("bad save_as name")
        cfg["name"] = sname
        cfg["is_portfolio"] = True
        fp = BUILDS_DIR / f"{sname}.json"
        fp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        return fp
    out_dir = LOG_DIR / "generated_portfolios"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{name}_{uuid.uuid4().hex[:8]}.json"
    fp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return fp


def _portfolio(b: dict) -> str:
    """Launch a SAVED PORTFOLIO of builds as one risk book.

    mode: 'shadow' (default, no orders) | 'demo' (OKX sandbox) | 'live' (REAL $).
    The runner self-guards: --live exits if OKX credentials are missing.
    """
    portfolio = _safe(b.get("portfolio", "portfolio_5x3")) or "portfolio_5x3"
    fp = BUILDS_DIR / f"{portfolio}.json"
    if not fp.exists():
        # allow passing a full relative path too
        alt = ROOT / str(b.get("portfolio", ""))
        fp = alt if alt.exists() else fp
    mode = str(b.get("mode", "shadow")).lower()
    if mode not in {"shadow", "demo", "live"}:
        raise ValueError("mode must be shadow, demo, or live")
    if mode == "live":
        running = _running_job("portfolio_live")
        if running:
            raise RuntimeError(f"live portfolio already running: {running['id']}")
    generated = _portfolio_config_from_saved_builds(b)
    if generated is not None:
        fp = generated
        portfolio = fp.stem
    elif not fp.exists():
        raise ValueError(f"portfolio not found: {portfolio}")
    args = ["src.run_hc_portfolio_live", "--portfolio", str(fp)]
    kind = "portfolio_shadow"
    if mode == "live":
        args.append("--live")
        kind = "portfolio_live"
        label = f"LIVE 🔴 portfolio: {portfolio}"
    elif mode == "demo":
        args += ["--live", "--demo"]
        kind = "portfolio_demo"
        label = f"demo portfolio: {portfolio}"
    else:
        args.append("--shadow")
        label = f"shadow portfolio: {portfolio}"
    if b.get("stake_margin") is not None:
        args += ["--stake-margin", str(b.get("stake_margin"))]
    if b.get("leverage") is not None:
        args += ["--leverage", str(b.get("leverage"))]
    if b.get("watchlist_size") is not None:
        args += ["--watchlist-size", str(int(b.get("watchlist_size")))]
    if b.get("once"):
        args.append("--once")
    return _launch(label, args, standing=not b.get("once"), kind=kind)


def _exittest(b: dict) -> str:
    """Backtest an exit policy for one build over the live pool (last N days to now).

    from_min/to_min (epoch minutes) optionally narrow to a sub-window. Writes
    exit_result.json.
    """
    sim = str(b.get("sim", ""))
    if "binance" not in sim:
        raise ValueError("exit-test needs a binance model")
    ep = b.get("exit_policy") or {}
    if not ep:
        raise ValueError("no exit policy enabled")
    spec = {"sim": sim, "levels": b.get("levels", []), "exit_policy": ep}
    if b.get("from_min") is not None and b.get("to_min") is not None:
        spec["from_min"] = int(b["from_min"]); spec["to_min"] = int(b["to_min"])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    spec_fp = LOG_DIR / "exittest_spec.json"
    spec_fp.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    out_fp = EXPLORER / "exit_result.json"
    cmd = ["src.run_binance_exittest", "--spec", str(spec_fp), "--out", str(out_fp)]
    return _launch("🚪 exit test", cmd, kind="exittest")


def _enginetest(b: dict) -> str:
    """Engine backtest WITH exits for a set of builds; writes engine_result.json."""
    names = b.get("build_names") or []
    if not isinstance(names, list) or not names:
        raise ValueError("pass build_names[]")
    builds = []
    for nm in names:
        cfg = _load_saved_build(str(nm).strip())
        builds.append({"sim": cfg.get("sim"), "levels": cfg.get("levels", []),
                       "banned": cfg.get("banned", []), "exit_policy": cfg.get("exit_policy"),
                       "name": cfg.get("name")})
    spec = {"builds": builds, "exit_default": b.get("exit_default") or {},
            "book": b.get("book") or {}, "notional": float(b.get("notional", 15))}
    if b.get("from_min") is not None and b.get("to_min") is not None:
        spec["from_min"] = int(b["from_min"]); spec["to_min"] = int(b["to_min"])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    spec_fp = LOG_DIR / "enginetest_spec.json"
    spec_fp.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    out_fp = EXPLORER / "engine_result.json"
    cmd = ["src.run_binance_engine_exittest", "--spec", str(spec_fp), "--out", str(out_fp)]
    return _launch("🚪 engine test with exits", cmd, kind="enginetest")


def _binance_portfolio(b: dict) -> dict:
    """Assemble a portfolio from saved builds and run it on BINANCE.

    mode: 'save' (only write the config, no run) | 'shadow' (default, no orders)
    | 'testnet' (Binance sandbox). REAL --live is deliberately NOT exposed here:
    it stays gated behind forward-shadow validation (go live small via the runner).
    """
    mode = str(b.get("mode", "shadow")).lower()
    if mode not in {"save", "shadow", "testnet"}:
        raise ValueError("mode must be save, shadow, or testnet (live is gated to the runner)")
    fp = _portfolio_config_from_saved_builds(b)
    if fp is None:
        # no build_names -> run an existing portfolio config by name
        fp = _find_build_file(b.get("portfolio", ""))
        if fp is None:
            raise ValueError("pass build_names[] or an existing portfolio name")
    if mode == "save":
        return {"config": str(fp)}
    args = ["src.run_binance_live", "--portfolio", str(fp)]
    if mode == "testnet":
        args.append("--testnet")
        label = f"binance TESTNET portfolio: {fp.stem}"
    else:
        args.append("--shadow")
        label = f"binance shadow portfolio: {fp.stem}"
    if b.get("stake_margin") is not None:
        args += ["--stake-margin", str(b.get("stake_margin"))]
    if b.get("leverage") is not None:
        args += ["--leverage", str(b.get("leverage"))]
    if b.get("scan_interval_min") is not None:
        args += ["--scan-interval-min", str(int(b.get("scan_interval_min")))]
    jid = _launch(label, args, standing=True, kind="binance_portfolio")
    return {"id": jid, "config": str(fp)}


def _binance_now(b: dict) -> str:
    """Refresh the single Binance data pool (data.js) up to NOW.

    Chains, stop-on-failure: fetch Binance candles to now -> rebuild the recent
    dataset (data/binance_now/dataset) -> export the standard model set fresh into
    data.js over the last N days up to now. One source, no sealed windows.
    Single-flight: returns the running job if one is already going.
    """
    running = _running_job("binance_now")
    if running:
        return running["id"]
    fetch_days = str(int(b.get("fetch_days", 4)))
    build_days = str(int(b.get("build_days", 12)))
    workers = str(int(b.get("workers", 8)))
    uni = b.get("universe", "configs/binance_train_universe.json")
    ds = "data/binance_now/dataset"
    drv = (
        "import subprocess,sys;"
        f"a=subprocess.run([sys.executable,'-m','src.binance_fetcher','--universe',{uni!r},"
        f"'--days',{fetch_days!r},'--workers',{workers!r}]);"
        "sys.exit(a.returncode) if a.returncode else None;"
        f"c=subprocess.run([sys.executable,'-m','src.run_binance_dataset','--out-dir',{ds!r},"
        f"'--universe',{uni!r},'--days',{build_days!r},'--workers',{workers!r},'--fresh']);"
        "sys.exit(c.returncode) if c.returncode else None;"
        f"e=subprocess.run([sys.executable,'-m','src.run_binance_export','--all','--fresh','--dataset',{ds!r}]);"
        "sys.exit(e.returncode)"
    )
    return _launch("🟢 Binance: fetch + rebuild to now", _pyc(drv), kind="binance_now")


def _v4live(b: dict) -> str:
    """Launch the v4 (1-min-horizon) model min1_2to120 live.

    mode: 'shadow' (default, no orders) | 'demo' (OKX sandbox) | 'live' (REAL $).
    """
    mode = str(b.get("mode", "shadow")).lower()
    if mode not in {"shadow", "demo", "live"}:
        raise ValueError("mode must be shadow, demo, or live")
    high = str(b.get("high", 0.85))
    horizons = str(b.get("horizons", "60,75,90,105,120"))
    args = ["src.run_hc_v4_live", "--high", high, "--horizons", horizons]
    if b.get("model_dir"):
        args += ["--model-dir", str(b.get("model_dir"))]
    kind = "v4_shadow"
    if mode == "live":
        if _running_job("v4_live"):
            raise RuntimeError("live v4 already running")
        args.append("--live"); kind = "v4_live"; label = "LIVE 🔴 v4 min1_2to120"
    elif mode == "demo":
        args += ["--live", "--demo"]; kind = "v4_demo"; label = "demo v4 min1_2to120"
    else:
        args.append("--shadow"); label = "shadow v4 min1_2to120"
    if b.get("stake_margin") is not None:
        args += ["--stake-margin", str(b.get("stake_margin"))]
    if b.get("leverage") is not None:
        args += ["--leverage", str(b.get("leverage"))]
    if b.get("watchlist_size") is not None:
        args += ["--watchlist-size", str(int(b.get("watchlist_size")))]
    if b.get("once"):
        args.append("--once")
    return _launch(label, args, standing=not b.get("once"), kind=kind)


def main():
    BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((HOST, PORT), H)
    # always start the loop thread; it idles unless AUTO_DATA_REFRESH is toggled on
    threading.Thread(target=_auto_data_refresh_loop, daemon=True,
                     name="auto-data-refresh").start()
    print(f"HC control panel -> http://{HOST}:{PORT}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
