"""
Coarse hyperparameter sweep for the ML Strategy C prototype.
- Imports walk_forward_evaluate from `scripts/retrain_c_ml.py`
- Grid-searches a coarse set of params (prob_buy, thr, hold_bars, allocation)
- Runs across top N cached symbols (by file size) to be fast by default
- Writes `storage/retrain_c_ml_sweep_results.csv` with best params per symbol

Usage:
.venv\Scripts\python.exe scripts\retrain_c_ml_sweep.py --max-symbols 10
"""

import os
import argparse
import pandas as pd
import json
from glob import glob

import importlib.util
from importlib.machinery import SourceFileLoader

# Load scripts/retrain_c_ml.py as module `base`
spec = importlib.util.spec_from_loader('retrain_c_ml', SourceFileLoader('retrain_c_ml', os.path.join('scripts','retrain_c_ml.py')))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

OUT = 'storage'
CACHE = 'data_cache'


def list_symbols(max_symbols=None):
    files = glob(os.path.join(CACHE, '*_15m.csv'))
    # sort by filesize desc
    files = sorted(files, key=lambda p: os.path.getsize(p), reverse=True)
    syms = [os.path.basename(p).split('_15m.csv')[0] for p in files]
    if max_symbols:
        syms = syms[:max_symbols]
    return syms


def sweep_symbol(symbol, grid, params_base):
    fn = os.path.join(CACHE, f'{symbol}_15m.csv')
    if not os.path.exists(fn):
        return None
    df = pd.read_csv(fn, parse_dates=['Datetime'], index_col='Datetime')
    best = None
    for prob in grid['prob_buy']:
        for thr in grid['thr']:
            for hold in grid['hold_bars']:
                for alloc in grid['allocation']:
                    p = params_base.copy()
                    p.update({'prob_buy': prob, 'thr': thr, 'hold_bars': hold, 'allocation': alloc})
                    real, trades, final_cash, details = base.walk_forward_evaluate(df, p)
                    pct = (final_cash - p['initial_cash']) / p['initial_cash'] * 100
                    # keep best by final_cash (could also use Sharpe or risk-adjusted)
                    if best is None or final_cash > best['final_cash']:
                        best = {'symbol': symbol, 'prob_buy': prob, 'thr': thr, 'hold_bars': hold, 'allocation': alloc, 'realized': real, 'trades': trades, 'final_cash': final_cash, 'pct_return': pct}
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-symbols', type=int, default=30)
    parser.add_argument('--initial_cash', type=float, default=100000)
    parser.add_argument('--fee', type=float, default=0.0005)
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    symbols = list_symbols(args.max_symbols)
    print('Running sweep on symbols:', symbols)

    # Expanded grid: allow lower prob threshold and smaller allocations to reduce churn
    grid = {
        'prob_buy': [0.45, 0.5, 0.55, 0.6],
        'thr': [0.001, 0.002, 0.003, 0.004],
        'hold_bars': [6, 12, 24],
        'allocation': [0.005, 0.01, 0.03]
    }

    params_base = {
        'train_size': 2000,
        'test_size': 336,
        'horizon': 12,
        'thr': 0.003,
        'prob_buy': 0.6,
        'allocation': 0.03,
        'fee': args.fee,
        'hold_bars': 12,
        'initial_cash': args.initial_cash
    }

    results = []
    for s in symbols:
        print('\nSweeping', s)
        best = sweep_symbol(s, grid, params_base)
        if best:
            results.append(best)
            print('Best for', s, json.dumps(best, default=str))
        else:
            print('No result for', s)

    out_fn = os.path.join(OUT, 'retrain_c_ml_sweep_results.csv')
    pd.DataFrame(results).to_csv(out_fn, index=False)
    print('\nWrote sweep results to', out_fn)

if __name__ == '__main__':
    main()
