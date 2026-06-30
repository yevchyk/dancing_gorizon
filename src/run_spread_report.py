"""Spread metric analysis: p_up - p_down, top-50, simulation in holdout."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from src import config as C
from src.database import CandleStore
from src.trading.fast_combo_engine import FastComboEngine
from src.trading.timeutil import index_to_ns

EVAL = 0.0015; EDGE = 0.001

def main():
    lc = pd.read_parquet('outputs/analysis/latest_crisis/holdout_scores.parquet')
    lc['anchor_time'] = pd.to_datetime(lc['anchor_time'], utc=True)

    eng = FastComboEngine('pulse00'); store = CandleStore(C.CANDLES_DIR)
    base = lc[lc.horizon=='2m'][['symbol','anchor_time']].drop_duplicates()
    recs = []
    for sym, g in base.groupby('symbol'):
        c = store.load(sym)
        if c is None or c.empty: continue
        c = c.sort_index()
        ans = pd.DatetimeIndex(g.anchor_time).as_unit('ns').asi8
        ff, fv = eng.curve.build_matrix(index_to_ns(c.index), c['close'].to_numpy('float64'), ans)
        if fv.sum() == 0: continue
        idx = np.where(fv)[0]; X = pd.DataFrame(ff[idx], columns=eng.columns)
        out = pd.DataFrame({'symbol': sym, 'anchor_time': pd.DatetimeIndex(g.anchor_time)[idx]})
        for nm in ['up_2m','down_2m','up_8m','down_8m']:
            m, cols = eng._models[nm]; out['v2_'+nm] = m.predict_proba(X[cols])[:, 1]
        recs.append(out)
    v2 = pd.concat(recs, ignore_index=True)

    d = lc[lc.horizon=='8m'].merge(v2, on=['symbol','anchor_time']).copy()
    d = d[np.isfinite(d.real_ret)].copy()

    d['v2_spread_up']  = d.v2_up_8m - d.v2_down_8m
    d['cr_spread_up']  = d.p_up - d.p_down
    d['v2_spread_abs'] = d.v2_spread_up.abs()
    d['cr_spread_abs'] = d.cr_spread_up.abs()
    d['v2_side']       = np.where(d.v2_spread_up > 0, 1, -1)
    d['cr_side']       = np.where(d.cr_spread_up > 0, 1, -1)
    d['v2_pnl']        = d.v2_side * d.real_ret - EVAL
    d['cr_pnl']        = d.cr_side * d.real_ret - EVAL

    print('='*80)
    print('SPREAD = p_up_8m - p_down_8m')
    print('Приклад: p_up=0.80, p_down=0.50 -> spread=+0.30 (LONG, впевненість 0.30)')
    print('Краще за просто p_up: показує ПЕРЕВАГУ однієї сторони над іншою')
    print(f'holdout: {d.anchor_time.min():%H:%M} -> {d.anchor_time.max():%H:%M} UTC  n={len(d)}')
    print('='*80)

    # ===== TOP-50 BY V2 SPREAD =====
    print()
    print('### ТОП-50 за |v2_spread| (8m) ###')
    top50 = d.nlargest(50, 'v2_spread_abs').copy()
    top50['direction'] = np.where(top50.v2_spread_up > 0, 'LONG', 'SHORT')
    top50['pnl'] = top50.v2_side * top50.real_ret - EVAL
    print(f'  {"#":>2} {"symbol":16} {"time":>5} {"dir":>5} {"p_up":>6} {"p_dn":>6} {"spread":>7} {"ret%":>8} {"pnl%":>7} {"W/L":>4}')
    print('  ' + '-'*76)
    for i, (_, r) in enumerate(top50.iterrows(), 1):
        wl = 'WIN' if r.pnl > 0 else 'loss'
        print(f'  {i:>2d} {r.symbol:16} {r.anchor_time.strftime("%H:%M"):>5} {r.direction:>5} '
              f'{r.v2_up_8m:>6.3f} {r.v2_down_8m:>6.3f} {r.v2_spread_up:>+7.3f} '
              f'{r.real_ret*100:>+8.3f}% {r.pnl*100:>+7.3f}% {wl:>4}')
    w50 = int((top50.pnl > 0).sum())
    print(f'\n  TOP-50 разом: win={w50}/50 ({w50*2:.0f}%) avg={top50.pnl.mean()*100:+.3f}% total={top50.pnl.sum()*100:+.1f}%')

    # ===== SPREAD BINS =====
    print()
    print('### WIN RATE ПО ШИРИНІ СПРЕДУ v2 ###')
    print(f'  {"spread_bin":>12} {"n":>5} {"long%":>6} {"win":>6} {"avg_pnl%":>9} {"total_pnl%":>10}')
    bins = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 1.0]
    for lo, hi in zip(bins, bins[1:]):
        msk = (d.v2_spread_abs >= lo) & (d.v2_spread_abs < hi)
        s = d[msk]; n = len(s)
        if n == 0: continue
        lp = float((s.v2_spread_up > 0).mean())
        win = float((s.v2_pnl > 0).mean())
        avg = s.v2_pnl.mean() * 100; tot = s.v2_pnl.sum() * 100
        print(f'  {lo:.2f}-{hi:.2f}      {n:>5d} {lp:>6.2f} {win:>6.3f} {avg:>+9.4f} {tot:>+10.1f}')

    # ===== V2 TOP-N (no live sim, just pure ranking) =====
    print()
    print('### V2 up_8m: top-N по p_up_8m (все лонг) — без cooldown/cap ###')
    print(f'  {"topN":>5} {"win":>5} {"avg_pnl%":>9} {"total_pnl%":>10}')
    dup = d[d.v2_spread_up > 0].sort_values('v2_up_8m', ascending=False).copy()
    dup['pnl_up'] = dup.real_ret - EVAL
    for n in [5, 10, 15, 20, 25, 30, 40, 50]:
        x = dup.head(n)
        print(f'  {n:>5d} {float((x.pnl_up>0).mean()):>5.3f} {x.pnl_up.mean()*100:>+9.4f} {x.pnl_up.sum()*100:>+10.2f}')

    # ===== SIMULATION =====
    print()
    print('### СИМУЛЯЦІЯ в холдауті (live-like, top-N/scan по spread, 8m exit) ###')
    print(f'  {"strategy":42} {"n":>4} {"win":>5} {"avg%":>8} {"total%":>8}')
    print('  ' + '-'*72)

    def simulate(df, n_top, cooldown_min=10, max_open=8,
                 spread_col='v2_spread_abs', side_col='v2_side', label=''):
        x = df.sort_values(['anchor_time', spread_col], ascending=[True, False]).copy()
        open_pos = {}; last_trade = {}; trades = []
        per_scan_count = {}
        for row in x.itertuples(index=False):
            now = pd.Timestamp(row.anchor_time)
            for sym in list(open_pos.keys()):
                if now >= open_pos[sym]['deadline']:
                    p = open_pos.pop(sym)
                    trades.append({'won': p['won'], 'pnl': p['pnl']})
            cnt = per_scan_count.get(now, 0)
            if cnt >= n_top: continue
            if len(open_pos) >= max_open: continue
            sym = row.symbol
            if sym in open_pos: continue
            last = last_trade.get(sym)
            if last and (now - last).total_seconds() < cooldown_min * 60: continue
            side = int(getattr(row, side_col))
            ret = float(row.real_ret)
            if not np.isfinite(ret): continue
            pnl = side * ret - EVAL
            open_pos[sym] = {'deadline': now + pd.Timedelta(minutes=8),
                             'won': int(pnl > 0), 'pnl': pnl, 'opened': now}
            last_trade[sym] = now
            per_scan_count[now] = cnt + 1
        for p in open_pos.values():
            trades.append({'won': p['won'], 'pnl': p['pnl']})
        if not trades:
            print(f'  {label:42} n=  0')
            return
        t = pd.DataFrame(trades)
        n = len(t); win = t.won.mean(); avg = t.pnl.mean()*100; tot = t.pnl.sum()*100
        print(f'  {label:42} n={n:3d} win={win:.3f} avg={avg:+.4f}% total={tot:+.2f}%')

    simulate(d, n_top=1,  label='v2 spread top-1/scan  cd10m')
    simulate(d, n_top=3,  label='v2 spread top-3/scan  cd10m')
    simulate(d, n_top=5,  label='v2 spread top-5/scan  cd10m')
    simulate(d, n_top=10, label='v2 spread top-10/scan cd10m')
    simulate(d, n_top=3, cooldown_min=2, label='v2 spread top-3/scan  cd2m')

    simulate(d, n_top=3, spread_col='cr_spread_abs', side_col='cr_side',
             label='crisis spread top-3/scan cd10m')

    d['combined_spread'] = np.where(
        d.v2_side == d.cr_side,
        (d.v2_spread_abs + d.cr_spread_abs) / 2, 0.0)
    d['combined_side'] = np.where(d.v2_side == d.cr_side, d.v2_side, 0)
    agree = d[d.combined_side != 0].copy()
    simulate(agree, n_top=3, spread_col='combined_spread', side_col='combined_side',
             label='AGREE v2+crisis combined spread top-3')
    simulate(agree, n_top=5, spread_col='combined_spread', side_col='combined_side',
             label='AGREE v2+crisis combined spread top-5')

    print()
    print(f'Ринок ці 3 год: avg 8m ret={d.real_ret.mean()*100:+.4f}%  '
          f'down={(d.real_ret<0).mean():.2f}  std={d.real_ret.std()*100:.4f}%')

if __name__ == '__main__':
    main()
