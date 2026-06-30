"""Variant 1 weighted exit test + high threshold slices for up models."""
import argparse
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from src import config as C
from src.fast import config as FC
from src.trading.fast_combo_engine import FastComboEngine, WORTHY
from src.trading.timeutil import index_to_ns
from src.run_okx_liquid_sim import OkxLiquidPriceBook
from src.run_test_engine_harvest_sim import simulate_engine
from src.run_test_engines_compare import EXIT_MIN

EVAL = FC.EVAL_COST; NS = 60_000_000_000; NOTIONAL = 50.0
OKX = Path('data/okx_liquid/candles_mixed')
BASES = {
    'fast_v2_up_10m': 0.77, 'fast_v2_up_8m': 0.77, 'fast_v2_up_2m': 0.92,
    'fast_v2_down_10m': 0.82, 'fast_v2_down_8m': 0.83, 'fast_v2_down_2m': 0.92,
}
WORTHY_MAP = {
    'fast_v2_up_10m': 'up_10m', 'fast_v2_up_8m': 'up_8m', 'fast_v2_up_2m': 'up_2m',
    'fast_v2_down_10m': 'down_10m', 'fast_v2_down_8m': 'down_8m', 'fast_v2_down_2m': 'down_2m',
}
HMIN = {'2m': 2, '5m': 5, '6m': 6, '8m': 8, '9m': 9, '10m': 10}
EXIT_MIN.update({'6m': 6, '9m': 9})


def median_exit(voted_horizons):
    if not voted_horizons:
        return '8m'
    m = int(np.median(voted_horizons))
    for h in [2, 5, 6, 8, 9, 10]:
        if m <= h:
            return f'{h}m'
    return '10m'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--top-per-scan", type=int, default=3)
    ap.add_argument("--max-open", type=int, default=5)
    ap.add_argument("--cooldown-min", type=int, default=10)
    ap.add_argument("--apply-blacklist", action="store_true",
                    help="exclude C.BLACKLIST_SYMBOLS; default is all okx_liquid files")
    args = ap.parse_args()

    eng = FastComboEngine('pulse00')
    all_syms = {p.stem for p in OKX.glob('*.parquet')}
    blacklist = set(C.BLACKLIST_SYMBOLS) if args.apply_blacklist else set()
    syms = sorted(all_syms - blacklist)
    now = pd.Timestamp.now(tz='UTC').floor('1min')
    end = now - pd.Timedelta(minutes=11)
    start = now - pd.Timedelta(days=float(args.days))
    anch = pd.date_range(start.ceil('2min'), end.floor('2min'), freq='2min')
    ans = anch.as_unit('ns').asi8
    window_days = max(1e-9, (end - start).total_seconds() / 86400.0)
    print(f'window {start:%m-%d} -> {end:%m-%d %H:%M}  anchors={len(anch)}  syms={len(syms)}')
    print(f'scope: all_okx={len(all_syms)} blacklist_applied={args.apply_blacklist} excluded={len(blacklist)}')

    rows = []
    for sym in syms:
        try:
            df = pd.read_parquet(OKX / f'{sym}.parquet')
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df = df.set_index('timestamp')
            df = df.sort_index()
            if len(df) < 300:
                continue
        except Exception:
            continue
        ts = index_to_ns(df.index); cl = df['close'].to_numpy('float64')
        ff, fv = eng.curve.build_matrix(ts, cl, ans)
        if fv.sum() == 0:
            continue
        idx = np.where(fv)[0]
        X = pd.DataFrame(ff[idx], columns=eng.columns)
        probs = {}
        for nm in WORTHY_MAP:
            key = WORTHY_MAP[nm]; m, cols = eng._models[key]
            probs[nm] = m.predict_proba(X[cols])[:, 1]
        rets = {}
        for h, hm in HMIN.items():
            a_ns = ans[idx]
            ei = np.searchsorted(ts, a_ns, 'right') - 1
            xj = np.searchsorted(ts, a_ns + hm * NS, 'right') - 1
            r = np.full(len(idx), np.nan)
            for i in range(len(idx)):
                if ei[i] >= 0 and xj[i] > ei[i] and cl[ei[i]] > 0:
                    r[i] = cl[xj[i]] / cl[ei[i]] - 1.0
            rets[h] = r
        for i in range(len(idx)):
            up_v = []; dn_v = []
            for nm in WORTHY_MAP:
                if probs[nm][i] >= BASES[nm]:
                    h = int(nm.split('_')[-1].replace('m', ''))
                    if WORTHY[nm][2] == 1:
                        up_v.append(h)
                    else:
                        dn_v.append(h)
            row = {
                'sym': sym, 'anchor': anch[idx[i]],
                'up_n': len(up_v), 'dn_n': len(dn_v),
                'up_voted': sorted(up_v), 'dn_voted': sorted(dn_v),
            }
            for nm in WORTHY_MAP:
                row['p_' + WORTHY_MAP[nm]] = float(probs[nm][i])
            for h in HMIN:
                row['r' + h] = float(rets[h][i])
            rows.append(row)

    d = pd.DataFrame(rows)
    d['anchor'] = pd.to_datetime(d['anchor'], utc=True)
    print(f'Total rows: {len(d)}\n')

    # ====================================================
    # 1. HIGH THRESHOLD SLICES
    # ====================================================
    print('=' * 72)
    print('### SLICES: up_2m / up_8m / up_10m — from 0.70 to 0.98 (exit=8m)')
    print('=' * 72)
    print(f'  {"model@exit":16} {"thr":>5} {"n":>6} {"n/day":>6} {"win":>5} {"avg%":>9} {"total%":>9}')
    print('  ' + '-' * 62)
    configs = [
        ('up_2m', 'r2m', 'exits IMMEDIATELY'),
        ('up_2m', 'r8m', 'standard exit'),
        ('up_8m', 'r8m', ''),
        ('up_10m', 'r8m', ''),
        ('up_10m', 'r10m', 'at its own horizon'),
    ]
    for model, ret_col, note in configs:
        note_str = f'  <- {note}' if note else ''
        for thr in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]:
            msk = (d[f'p_{model}'] >= thr) & np.isfinite(d[ret_col])
            n = int(msk.sum())
            if n < 3:
                continue
            ret = d.loc[msk, ret_col].to_numpy()
            pnl = ret - EVAL
            label = f'{model}@{ret_col[1:]}'
            marker = ' <-- POSITIVE' if pnl.mean() > 0 else ''
            print(f'  {label:16} {thr:>5.2f} {n:>6d} {n/window_days:>6.0f} '
                  f'{float((pnl>0).mean()):>5.3f} {pnl.mean()*100:>+9.4f} '
                  f'{pnl.sum()*100:>+9.1f}{marker}')
        print()

    # ====================================================
    # 2. VARIANT 1 — weighted exit (min_count=2)
    # ====================================================
    print('=' * 72)
    print('### VARIANT 1 — weighted exit (min_count=2, exit=median(voted horizons))')
    print('  Example: up_8m+up_10m (no 2m) -> median(8,10)=9m exit')
    print('           up_2m+up_10m (no 8m) -> median(2,10)=6m exit')
    print('           up_2m+up_8m  (no 10m)-> median(2,8)=5m exit')
    print('           all three               -> median(2,8,10)=8m exit')
    print('=' * 72)

    def make_cands(df, engine_name, min_count=3, weighted=False, fixed_exit='8m'):
        rows_c = []
        for _, row in df.iterrows():
            for side_v, voted, opp in [(1, row.up_voted, row.dn_n),
                                       (-1, row.dn_voted, row.up_n)]:
                if len(voted) >= min_count and opp == 0:
                    if weighted:
                        exit_h = median_exit(voted)
                    else:
                        exit_h = fixed_exit
                    rows_c.append({
                        'engine': engine_name, 'family': 'f', 'source': 'f',
                        'signal_model': 'P', 'symbol': row.sym,
                        'anchor_time': row.anchor,
                        'day': row.anchor.strftime('%m-%d'),
                        'side': side_v, 'exit': exit_h,
                        'threshold': np.nan, 'leverage': 1.0, 'score': 1.0,
                    })
        return pd.DataFrame(rows_c)

    book = OkxLiquidPriceBook()

    configs_sim = [
        ('Pulse3 exit=8m  (CURRENT)', make_cands(d, 'p3_8m',  3, False, '8m')),
        ('Pulse3 Variant1 weighted',  make_cands(d, 'p3_v1',  3, True,  None)),
        ('Pulse3 exit=10m (OLD)',     make_cands(d, 'p3_10m', 3, False, '10m')),
        ('Pulse2 exit=8m  (more signals)', make_cands(d, 'p2_8m', 2, False, '8m')),
        ('Pulse2 exit=10m (more signals)', make_cands(d, 'p2_10m', 2, False, '10m')),
        ('Pulse2 Variant1 weighted', make_cands(d, 'p2_v1',  2, True,  None)),
    ]

    print(f'\n  {"engine":35} {"cands":>6} {"trades":>6} {"win":>5} '
          f'{"avg%":>8} {"total%":>8} {"total$":>8}')
    print('  ' + '-' * 78)
    for label, cand in configs_sim:
        if cand.empty:
            print(f'  {label}: no candidates'); continue
        st = sorted(pd.Timestamp(t) for t in cand['anchor_time'].drop_duplicates())
        tr, _ = simulate_engine('x', cand.copy(), st, book,
                                harvest=False, top_per_scan=args.top_per_scan,
                                max_open=args.max_open, cooldown_min=args.cooldown_min)
        if tr.empty:
            print(f'  {label:35} {len(cand):>6}  no trades'); continue
        n = len(tr); win = tr.won.mean()
        avg = tr.net_pnl_pct.mean(); tot = tr.net_pnl_pct.sum()
        usd = NOTIONAL * tot / 100
        days_g = tr.groupby('open_day')['net_pnl_pct'].sum()
        day_str = ' | '.join(f'{k}:{v:+.0f}%' for k, v in days_g.items())
        print(f'  {label:35} {len(cand):>6} {n:>6} {win:>5.3f} '
              f'{avg:>+8.4f} {tot:>+8.2f}% ${usd:>+6.2f}')
        print(f'    per day: {day_str}')

    # Variant 1 exit distribution
    v1_cand = make_cands(d, 'p2_v1', 2, True, None)
    if not v1_cand.empty:
        print(f'\n  Variant1 exit distribution:')
        print('  ' + v1_cand.groupby('exit').size().to_string())


if __name__ == '__main__':
    main()
