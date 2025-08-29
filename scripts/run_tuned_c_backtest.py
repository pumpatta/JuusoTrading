"""
Run tuned rule-based Strategy C backtests using per-symbol tuned RSI thresholds.
- Reads `storage/retrain_c_tuning_per_symbol.csv` (expects columns: symbol,best_params,...)
- Loads cached 15m data from `data_cache/{SYMBOL}_15m.csv`
- Simulates trades using the tuned buy/sell RSI thresholds
- Tests scenarios: fees [0.0005, 0.001], allocations [0.01, 0.025], stop_loss [None, 0.01]
- Writes detailed results to `storage/tuned_c_backtest_results.csv` and aggregated summary to `storage/tuned_c_backtest_summary.csv`

Run:
.venv\Scripts\python.exe scripts\run_tuned_c_backtest.py
"""

import os
import ast
import pandas as pd
import numpy as np

CACHE = 'data_cache'
OUT = 'storage'
TUNING = os.path.join(OUT, 'retrain_c_tuning_per_symbol.csv')


def load_symbol_df(symbol):
    fn = os.path.join(CACHE, f'{symbol}_15m.csv')
    if not os.path.exists(fn):
        return None
    return pd.read_csv(fn, parse_dates=['Datetime'], index_col='Datetime')


def simulate_c(df, rsi_buy, rsi_sell, allocation=0.05, initial_cash=100000, fee=0.0005, stop_loss=None):
    cash = initial_cash
    positions = []
    realized = []
    for t, row in df.iterrows():
        rsi = float(row.get('RSI', np.nan)) if 'RSI' in row.index else np.nan
        price = float(row['Close'])
        if np.isnan(rsi):
            continue
        # buy
        if rsi < rsi_buy and not positions:
            qty = int((cash * allocation) / price)
            if qty > 0:
                cost = qty * price * (1 + fee)
                cash -= cost
                positions.append({'qty': qty, 'price': price, 'entry_idx': t})
        # manage positions
        if positions:
            pos = positions[0]
            # stop-loss check
            if stop_loss is not None and (price - pos['price'])/pos['price'] <= -stop_loss:
                pl = (price - pos['price']) * pos['qty'] - (pos['price']*pos['qty']*fee + price*pos['qty']*fee)
                realized.append(pl)
                cash += pos['qty'] * price * (1 - fee)
                positions.pop(0)
                continue
            # sell signal
            if rsi > rsi_sell:
                pos = positions.pop(0)
                pl = (price - pos['price']) * pos['qty'] - (pos['price']*pos['qty']*fee + price*pos['qty']*fee)
                realized.append(pl)
                cash += pos['qty'] * price * (1 - fee)
    # mark-to-market any remaining positions
    if positions:
        pos = positions[0]
        price = float(df.iloc[-1]['Close'])
        pl = (price - pos['price']) * pos['qty'] - (pos['price']*pos['qty']*fee)
        realized.append(pl)
    return sum(realized), len(realized), cash


def parse_best_params(val):
    # `best_params` may be like "(20, 80)" or "[20, 80]" or empty
    if pd.isna(val):
        return None
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, (tuple, list)) and len(parsed) >= 2:
            return int(parsed[0]), int(parsed[1])
    except Exception:
        pass
    # try split
    try:
        parts = val.strip('()[]').split(',')
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def bench_return(symbol):
    fn = os.path.join(CACHE, f'{symbol}_15m.csv')
    if not os.path.exists(fn):
        return np.nan
    df = pd.read_csv(fn, parse_dates=['Datetime'], index_col='Datetime')
    if df.empty:
        return np.nan
    return (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(TUNING):
        print('Tuning CSV not found:', TUNING)
        return
    tuning = pd.read_csv(TUNING)
    scenarios = []
    fees = [0.0005, 0.001]
    allocations = [0.01, 0.025]
    stop_losses = [None, 0.01]
    rows = []
    for _, r in tuning.iterrows():
        sym = r['symbol']
        bp = parse_best_params(r.get('best_params', ''))
        if bp is None:
            print('No best params for', sym)
            continue
        buy, sell = bp
        df = load_symbol_df(sym)
        if df is None or df.empty:
            print('Missing data for', sym)
            continue
        # ensure RSI exists; if not compute quickly
        if 'RSI' not in df.columns:
            # compute simple RSI
            delta = df['Close'].diff()
            up = delta.clip(lower=0).rolling(14).mean()
            down = -delta.clip(upper=0).rolling(14).mean()
            rs = up/(down + 1e-9)
            df['RSI'] = 100 - (100/(1+rs))
        for fee in fees:
            for alloc in allocations:
                for sl in stop_losses:
                    realized, trades, final_cash = simulate_c(df, buy, sell, allocation=alloc, initial_cash=100000, fee=fee, stop_loss=sl)
                    pct = (final_cash - 100000)/100000*100
                    rows.append({'symbol': sym, 'buy': buy, 'sell': sell, 'fee': fee, 'allocation': alloc, 'stop_loss': sl, 'realized_pl': realized, 'trades': trades, 'final_cash': final_cash, 'pct_return': pct})
    out_df = pd.DataFrame(rows)
    out_fn = os.path.join(OUT, 'tuned_c_backtest_results.csv')
    out_df.to_csv(out_fn, index=False)
    print('Wrote detailed results to', out_fn)

    # aggregate summary per scenario across all symbols
    agg = out_df.groupby(['fee','allocation','stop_loss']).agg({'realized_pl':'sum','trades':'sum'}).reset_index()
    agg['final_cash'] = 100000 + agg['realized_pl']
    agg['pct_return'] = agg['realized_pl']/ (100000 * len(tuning)) * 100
    # benchmark returns (SPY/QQQ) for same caches
    spy_ret = bench_return('SPY')
    qqq_ret = bench_return('QQQ')
    summary_fn = os.path.join(OUT, 'tuned_c_backtest_summary.csv')
    agg.to_csv(summary_fn, index=False)
    print('Wrote summary to', summary_fn)
    print('\nBenchmarks: SPY pct=', spy_ret, 'QQQ pct=', qqq_ret)

if __name__ == '__main__':
    main()
