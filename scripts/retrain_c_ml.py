"""
Per-symbol ML prototype for Strategy C (CPU prototype -> later GPU port).
- Walk-forward train/test using rolling windows
- Fast model: SGDClassifier (log loss)
- Features: RSI, returns, vol, macd
- Label: forward return over horizon > threshold
- Simulate simple trading, include per-trade fee
- Writes CSV results to storage/retrain_c_ml_results.csv

Run (example):
.venv\Scripts\python.exe scripts\retrain_c_ml.py

"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CACHE = 'data_cache'
OUT = 'storage'


def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig


def features_from_df(df):
    df = df.copy()
    df['ret1'] = df['Close'].pct_change().fillna(0)
    df['rsi'] = compute_rsi(df['Close']).fillna(50)
    macd, sig = compute_macd(df['Close'])
    df['macd'] = (macd - sig).fillna(0)
    df['vol20'] = df['Close'].pct_change().rolling(20).std().fillna(0)
    # lagged features
    df['ret_1'] = df['ret1'].shift(1).fillna(0)
    df['ret_2'] = df['ret1'].shift(2).fillna(0)
    return df[['rsi','macd','vol20','ret_1','ret_2','ret1']]


def label_forward(df, horizon=12, thr=0.003):
    # label 1 if forward return over horizon > thr
    future = df['Close'].shift(-horizon)
    fwdret = (future - df['Close']) / df['Close']
    return (fwdret > thr).astype(int)


def simulate_trades(df, preds, allocation=0.03, initial_cash=100000, fee=0.0005, hold_bars=12):
    cash = initial_cash
    pos = 0
    entry_price = 0.0
    realized = []
    hold_count = 0
    for i, (ts, row) in enumerate(df.iterrows()):
        price = float(row['Close'])
        p = preds[i]
        # entry
        if pos == 0 and p == 1:
            qty = int((cash * allocation) / price)
            if qty > 0:
                cost = qty * price * (1 + fee)
                cash -= cost
                pos = qty
                entry_price = price
                hold_count = 0
        # when holding, either exit after hold_bars or if model says 0
        if pos > 0:
            hold_count += 1
            if p == 0 or hold_count >= hold_bars:
                proceeds = pos * price * (1 - fee)
                pl = (price - entry_price) * pos - (entry_price * pos * fee + price * pos * fee)
                realized.append(pl)
                cash += proceeds
                pos = 0
                entry_price = 0.0
                hold_count = 0
    # mark-to-market of open position
    if pos > 0:
        pl = (float(df.iloc[-1]['Close']) - entry_price) * pos - (entry_price * pos * fee)
        realized.append(pl)
    return sum(realized), len(realized), cash


def walk_forward_evaluate(df, params):
    features = features_from_df(df)
    labels = label_forward(df, horizon=params['horizon'], thr=params['thr'])
    X = features.values
    y = labels.values
    n = len(df)
    train_size = params['train_size']
    test_size = params['test_size']
    step = test_size
    results = []
    idx = train_size
    # scaler + sgd pipe
    for start in range(idx, n, step):
        train_start = start - train_size
        if train_start < 0:
            continue
        train_end = start
        test_end = min(start + test_size, n)
        if train_end >= test_end:
            break
        X_train, y_train = X[train_start:train_end], y[train_start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]
        if len(np.unique(y_train)) < 2:
            # skip fold if no positive examples
            continue
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3))])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs > params['prob_buy']).astype(int)
        # simulate trades on this test window
        df_test = df.iloc[train_end:test_end].reset_index(drop=True)
        realized, trades, cash = simulate_trades(df_test, preds, allocation=params['allocation'], initial_cash=params['initial_cash'], fee=params['fee'], hold_bars=params['hold_bars'])
        # benchmark return over same period
        bench_return = (df_test['Close'].iloc[-1] - df_test['Close'].iloc[0]) / df_test['Close'].iloc[0]
        results.append({'start_idx': train_end, 'end_idx': test_end, 'realized': realized, 'trades': trades, 'bench_return': bench_return})
    # aggregate
    total_realized = sum(r['realized'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    # final cash estimate: initial_cash + total_realized
    final_cash = params['initial_cash'] + total_realized
    return total_realized, total_trades, final_cash, results


def run_symbols(symbols, params):
    summary = []
    for s in symbols:
        fn = os.path.join(CACHE, f'{s}_15m.csv')
        if not os.path.exists(fn):
            print('Missing cache for', s)
            continue
        df = pd.read_csv(fn, parse_dates=['Datetime'], index_col='Datetime')
        n = len(df)
        if df.empty or n < 100:
            print('Insufficient data for', s)
            continue

        # Auto-adjust train/test sizes if requested sizes exceed available data
        adjusted = params.copy()
        req_total = params['train_size'] + params['test_size']
        if req_total >= n:
            # use 60% train, 20% test, leave some headroom
            adj_train = max(200, int(n * 0.6))
            adj_test = max(50, int(n * 0.2))
            # ensure we have at least a small validation step
            if adj_train + adj_test >= n:
                adj_train = max(100, int(n * 0.5))
                adj_test = max(50, n - adj_train - 1)
            adjusted['train_size'] = adj_train
            adjusted['test_size'] = adj_test
            print(f'Adjusted train/test for {s}: train={adj_train}, test={adj_test} (n={n})')

        real, trades, final_cash, details = walk_forward_evaluate(df, adjusted)
        pct = (final_cash - params['initial_cash']) / params['initial_cash'] * 100
        summary.append({'symbol': s, 'realized_pl': real, 'trades': trades, 'final_cash': final_cash, 'pct_return': pct})
        print(f'{s}: P&L={real:.2f}, trades={trades}, final_cash={final_cash:.2f}, return={pct:.2f}%')
    out = pd.DataFrame(summary)
    out_fn = os.path.join(OUT, 'retrain_c_ml_results.csv')
    out.to_csv(out_fn, index=False)
    print('Wrote results to', out_fn)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='*', default=['AAPL','MSFT','GOOGL','TSLA','AMZN'])
    parser.add_argument('--train_size', type=int, default=2000)
    parser.add_argument('--test_size', type=int, default=336)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--thr', type=float, default=0.003)
    parser.add_argument('--prob_buy', type=float, default=0.6)
    parser.add_argument('--allocation', type=float, default=0.03)
    parser.add_argument('--fee', type=float, default=0.0005)
    parser.add_argument('--hold_bars', type=int, default=12)
    parser.add_argument('--initial_cash', type=float, default=100000)
    args = parser.parse_args()

    params = {
        'train_size': args.train_size,
        'test_size': args.test_size,
        'horizon': args.horizon,
        'thr': args.thr,
        'prob_buy': args.prob_buy,
        'allocation': args.allocation,
        'fee': args.fee,
        'hold_bars': args.hold_bars,
        'initial_cash': args.initial_cash
    }
    os.makedirs(OUT, exist_ok=True)
    run_symbols(args.symbols, params)

if __name__ == '__main__':
    main()
