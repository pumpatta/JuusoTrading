#!/usr/bin/env python3
"""Intraday multi-instrument backtest with online learning (1h/15m data)

- Downloads intraday OHLCV data for a list of tickers (yfinance)
- Computes simple technical features (EMA, RSI, MACD, returns)
- Uses an online SGDClassifier (scikit-learn) with partial_fit to learn continuously
- Simulates trading across instruments with a single account

Note: For large instrument lists (>100) and long ranges, this will take time and memory.
"""

import argparse
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json

import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import joblib

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
except Exception:
    SGDClassifier = None


def calculate_indicators(df):
    df = df.copy()
    # Helper to extract a single-column Series for common fields even if yfinance returns
    # a DataFrame with MultiIndex columns.
    def _get_series(frame, name):
        if name in frame.columns:
            s = frame[name]
        else:
            # MultiIndex or strange column names: pick first column that contains the name
            found = None
            for c in frame.columns:
                try:
                    cname = c if isinstance(c, str) else ' '.join(map(str, c))
                except Exception:
                    cname = str(c)
                if name.lower() in cname.lower():
                    found = c
                    break
            if found is None:
                return pd.Series(dtype=float, index=frame.index)
            s = frame[found]
        # if we got a DataFrame slice, take first column
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.astype(float)

    close = _get_series(df, 'Close')
    open_s = _get_series(df, 'Open')
    high = _get_series(df, 'High')
    low = _get_series(df, 'Low')
    vol = _get_series(df, 'Volume')

    df = pd.DataFrame({'Open': open_s, 'High': high, 'Low': low, 'Close': close, 'Volume': vol})

    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Return_1'] = df['Close'].pct_change()
    df['Vol_Change'] = df['Volume'].pct_change()
    df = df.dropna()
    return df


def build_feature_matrix(df):
    feats = df[['RSI', 'MACD', 'Return_1', 'Vol_Change']].fillna(0)
    return feats.values


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download_and_cache(symbol, start, end, interval, cache_dir, force=False):
    cache_file = os.path.join(cache_dir, f"{symbol}_{interval}.csv")
    if os.path.exists(cache_file) and not force:
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return symbol, df
        except Exception:
            pass

    try:
        # yfinance only serves intraday for limited lookback (~60 days). Download in <=60d chunks.
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        max_days = 60
        parts = []
        cur = start_ts
        while cur < end_ts:
            chunk_end = min(cur + pd.Timedelta(days=max_days), end_ts)
            # yfinance expects strings
            # retry per-chunk with exponential backoff
            part = None
            err_msg = None
            for attempt in range(3):
                try:
                    part = yf.download(symbol, start=cur.strftime('%Y-%m-%d'), end=(chunk_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), interval=interval, progress=False)
                    if part is not None and not part.empty:
                        parts.append(part)
                    break
                except Exception as e:
                    err_msg = str(e)
                    sleep_time = 1 + 2 ** attempt
                    time.sleep(sleep_time)
                    part = None
            if part is None or (hasattr(part, 'empty') and part.empty):
                # continue to next chunk, but record the last error if any
                if err_msg:
                    # attach small warning to cache file name later via returned status
                    pass
            # advance
            cur = chunk_end

        if parts:
            try:
                df = pd.concat(parts).sort_index()
                df = df[~df.index.duplicated(keep='first')]
            except Exception as e:
                return symbol, None, f"concat_error: {e}"
        else:
            # If no intraday parts, fallback to daily->intraday generation
            try:
                df_daily = yf.download(symbol, start=start, end=end, interval='1d', progress=False)
            except Exception as e:
                return symbol, None, f"daily_download_error: {e}"
            if df_daily is None or df_daily.empty:
                return symbol, None, 'no_data'
            df = generate_intraday_from_daily(df_daily, interval)

        if df is None or df.empty:
            return symbol, None, 'no_data'

        try:
            df = calculate_indicators(df)
        except Exception as e:
            return symbol, None, f'indicator_error: {e}'
        # Save to cache
        try:
            ensure_dir(cache_dir)
            df.to_csv(cache_file)
        except Exception:
            pass
        return symbol, df, 'ok'
    except Exception as e:
        print(f"  Failed download {symbol}: {e}")
        return symbol, None, f'exception: {e}'


def run_backtest(symbols, start, end, interval, initial_capital=100000, max_instruments=10, cache_dir='data_cache', parallel_workers=6, force_download=False):
    print(f"Starting intraday multi backtest: {len(symbols)} symbols, interval={interval}, {start}..{end}")

    # Limit instruments
    symbols = symbols[:max_instruments]

    # Download in parallel with caching
    data_store = {}
    ensure_dir(cache_dir)
    download_report = []
    with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
        futures = {ex.submit(download_and_cache, s, start, end, interval, cache_dir, force_download): s for s in symbols}
        for f in as_completed(futures):
            s = futures[f]
            try:
                sym, df, status = f.result()
                status = status or 'ok'
                if df is None or (hasattr(df, 'empty') and df.empty):
                    print(f"  No data for {sym}, skipping ({status})")
                    download_report.append((sym, status, 0))
                    continue
                data_store[sym] = df
                download_report.append((sym, status, len(df)))
                print(f"  {sym}: {len(df)} rows (cached)")
            except Exception as e:
                print(f"  Error for {s}: {e}")
                download_report.append((s, f'exception: {e}', 0))

    if not data_store:
        print("No data available for any symbols. Exiting.")
        return {'final_value': initial_capital, 'trades': []}

    # Save download report
    try:
        ensure_dir('storage')
        rep_df = pd.DataFrame(download_report, columns=['symbol', 'status', 'rows'])
        rep_df['rows'] = rep_df['rows'].astype(int)
        rep_df.to_csv('storage/download_report.csv', index=False)
    except Exception:
        pass

    # Directory for model checkpoints
    models_dir = 'models'
    ensure_dir(models_dir)

    # Build a sorted list of all timestamps across symbols
    timestamps = sorted({ts for df in data_store.values() for ts in df.index})

    # Support multiple strategies (silos). Default strategies: A,B,C
    strategies = ['A', 'B', 'C']

    # Setup per-strategy per-symbol models/scalers/buffers where applicable
    if SGDClassifier is None:
        print("scikit-learn not available. Please pip install scikit-learn to use online learning.")
        return

    # Structures: models[strategy][symbol]
    models = {st: {} for st in strategies}
    scalers = {st: {} for st in strategies}
    buffers = {st: {} for st in strategies}
    for st in strategies:
        for s in data_store.keys():
            if st == 'B':
                # online SGD for strategy B
                models[st][s] = SGDClassifier(loss='log_loss', max_iter=1000)
                scalers[st][s] = StandardScaler()
                buffers[st][s] = {'X': [], 'y': []}
            else:
                # A and C are rule-based and don't need models/scalers
                models[st][s] = None
                scalers[st][s] = None
                buffers[st][s] = {'X': [], 'y': []}

    classes = np.array([-1, 1])

    # Per-strategy portfolios and positions
    cash = {st: initial_capital for st in strategies}
    positions = {st: {} for st in strategies}  # positions[st][symbol] = {...}
    trade_log = []

    # Loop through timestamps
    for idx, t in enumerate(timestamps):
        for s, df in data_store.items():
            if t not in df.index:
                continue
            row = df.loc[t]
            feats = np.array([row[['RSI', 'MACD', 'Return_1', 'Vol_Change']].fillna(0).values]).astype(float)

            # For each strategy independently
            for st in strategies:
                # prepare buffer labeling for trainable strategy B
                buf = buffers[st][s]
                pos_in_df = df.index.get_loc(t)
                if pos_in_df + 1 < len(df):
                    future_ret = df['Return_1'].iloc[pos_in_df + 1]
                    label = 1 if future_ret > 0 else -1
                    buf['X'].append(feats.flatten())
                    buf['y'].append(label)

                # Train for strategy B
                if st == 'B' and len(buf['X']) >= 50 and (len(buf['X']) % 25 == 0):
                    batch_X = np.vstack(buf['X'][-50:])
                    batch_y = np.array(buf['y'][-50:])
                    try:
                        scalers[st][s].partial_fit(batch_X)
                        Xb = scalers[st][s].transform(batch_X)
                        models[st][s].partial_fit(Xb, batch_y, classes=classes)
                    except Exception:
                        try:
                            models[st][s].fit(batch_X, batch_y)
                        except Exception:
                            pass
                    # checkpoint
                    try:
                        joblib.dump(models[st][s], os.path.join(models_dir, f"{st}_{s}_model.pkl"))
                        joblib.dump(scalers[st][s], os.path.join(models_dir, f"{st}_{s}_scaler.pkl"))
                    except Exception:
                        pass

                # Prediction or rule evaluation
                pred = 0
                prob = 0.0
                if st == 'A':
                    # EMA crossover
                    try:
                        if df.loc[t, 'EMA_12'] > df.loc[t, 'EMA_26']:
                            pred = 1
                        else:
                            pred = -1
                        prob = 0.8
                    except Exception:
                        pred = 0
                elif st == 'B':
                    # online classifier
                    if hasattr(models[st][s], 'coef_'):
                        try:
                            Xs = scalers[st][s].transform(feats) if hasattr(scalers[st][s], 'mean_') else feats
                            pred = models[st][s].predict(Xs)[0]
                            prob = max(models[st][s].predict_proba(Xs)[0]) if hasattr(models[st][s], 'predict_proba') else 0.6
                        except Exception:
                            pred = 0
                elif st == 'C':
                    # RSI mean-reversion
                    try:
                        rsi = float(df.loc[t, 'RSI'])
                        if rsi < 30:
                            pred = 1
                        elif rsi > 70:
                            pred = -1
                        else:
                            pred = 0
                        prob = 0.75
                    except Exception:
                        pred = 0

                # Trading rule per-strategy (isolated accounts)
                conf_thresh = 0.55
                acct_cash = cash[st]
                acct_positions = positions[st]

                if pred == 1 and prob >= conf_thresh and s not in acct_positions:
                    allocation = acct_cash * 0.05
                    qty = int(allocation / row['Close'])
                    if qty > 0:
                        cost = qty * row['Close']
                        cash[st] -= cost
                        positions[st][s] = {'qty': qty, 'entry_price': row['Close'], 'entry_time': t}
                        trade_log.append((t, s, 'BUY', qty, row['Close'], st))

                if s in acct_positions:
                    held_hours = (t - acct_positions[s]['entry_time']) / pd.Timedelta(hours=1)
                    sell_on_pred = (pred == -1 and prob >= conf_thresh)
                    if sell_on_pred or held_hours >= 24:
                        qty = acct_positions[s]['qty']
                        proceeds = qty * row['Close']
                        cash[st] += proceeds
                        trade_log.append((t, s, 'SELL', qty, row['Close'], st))
                        del acct_positions[s]

        # progress
        if idx % 200 == 0:
            print(f"Processed timestamp {idx+1}/{len(timestamps)}: {t}")

    # Close remaining positions for each strategy
    for st in strategies:
        for s, pos in list(positions[st].items()):
            last_price = data_store[s]['Close'].iloc[-1]
            proceeds = pos['qty'] * last_price
            cash[st] += proceeds
            trade_log.append((data_store[s].index[-1], s, 'SELL_FINAL', pos['qty'], last_price, st))
            del positions[st][s]

    # compute portfolio values per strategy (cash only + closed positions already added)
    portfolio_value = {st: cash[st] for st in strategies}

    print("\n--- BACKTEST SUMMARY ---")
    print(f"Initial capital: {initial_capital}")
    print("Final portfolio values per strategy:")
    for st in strategies:
        print(f"  {st}: {portfolio_value[st]:.2f} (net P&L: {portfolio_value[st] - initial_capital:.2f})")
    print(f"Number of trades: {len(trade_log)}")

    print("\nSample trades:")
    for t in trade_log[:20]:
        print(t)

    # final checkpoint save for all models/scalers
    try:
        for s in models.keys():
            try:
                joblib.dump(models[s], os.path.join(models_dir, f"{s}_model.pkl"))
                joblib.dump(scalers[s], os.path.join(models_dir, f"{s}_scaler.pkl"))
            except Exception:
                pass
    except Exception:
        pass

    return {'final_value': portfolio_value, 'trades': trade_log}

def generate_intraday_from_daily(daily_df, interval):
    """Create synthetic intraday bars from daily OHLC by linear interpolation.
    interval examples: '60m', '15m' -> minutes
    """
    minutes = int(interval.replace('m', ''))
    rows = []
    for date, r in daily_df.iterrows():
        # trading hours assume 9:30-16:00 -> 6.5 hours = 390 minutes
        start = pd.Timestamp(date.date()) + pd.Timedelta(hours=9, minutes=30)
        end = start + pd.Timedelta(minutes=390)
        times = pd.date_range(start=start, end=end, freq=f"{minutes}min")
        if len(times) == 0:
            continue
        # linear interpolate from Open to Close across times for price
        opens = np.linspace(r['Open'], r['Close'], len(times))
        highs = np.linspace(r['High'], r['High'], len(times))
        lows = np.linspace(r['Low'], r['Low'], len(times))
        closes = np.linspace(r['Open'], r['Close'], len(times))
        # Ensure volume is a scalar number (Series.item() if a single-element Series)
        vol_total = r.get('Volume', 1)
        try:
            vol_total = int(float(vol_total))
        except Exception:
            try:
                vol_total = int(float(vol_total.item()))
            except Exception:
                vol_total = 1
        vols = np.full(len(times), max(1, int(max(1, vol_total) / len(times))))
        for t, o, h, l, c, v in zip(times, opens, highs, lows, closes, vols):
            rows.append((t, o, h, l, c, v))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']).set_index('Datetime')
    return df


    # The main processing (moved into run_backtest below)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, default='SPY,AAPL,MSFT,TSLA,QQQ', help='Comma-separated ticker list')
    parser.add_argument('--start', type=str, default='2024-07-01')
    parser.add_argument('--end', type=str, default='2024-08-01')
    parser.add_argument('--interval', type=str, default='60m')
    parser.add_argument('--max-instruments', type=int, default=5)
    parser.add_argument('--parallel-workers', type=int, default=6)
    parser.add_argument('--force-download', action='store_true')
    parser.add_argument('--symbols-file', type=str, default=None, help='File with one ticker per line')
    args = parser.parse_args()

    if args.symbols_file:
        with open(args.symbols_file, 'r') as fh:
            symbols = [l.strip().upper() for l in fh if l.strip()]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    start = args.start
    end = args.end
    interval = args.interval

    res = run_backtest(symbols, start, end, interval, max_instruments=args.max_instruments, cache_dir='data_cache', parallel_workers=args.parallel_workers, force_download=args.force_download)
    
    # Save results
    trades = res.get('trades', []) if isinstance(res, dict) else []
    ensure_dir('storage')
    if trades:
        # detect if per-trade strategy included (6th element)
        has_strategy = len(trades[0]) >= 6
        if has_strategy:
            out_df = pd.DataFrame(trades, columns=['timestamp', 'symbol', 'side', 'qty', 'price', 'strategy'])
        else:
            out_df = pd.DataFrame(trades, columns=['timestamp', 'symbol', 'side', 'qty', 'price'])
    else:
        out_df = pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'qty', 'price'])
    out_df.to_csv('storage/intraday_backtest_trades.csv', index=False)
    print('\nSaved trade log to storage/intraday_backtest_trades.csv')
