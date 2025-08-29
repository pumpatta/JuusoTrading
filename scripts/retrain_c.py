import pandas as pd
import numpy as np
import os

CACHE = 'data_cache'
OUT = 'storage'

def load_symbol(symbol):
    fn = os.path.join(CACHE, f'{symbol}_15m.csv')
    if not os.path.exists(fn):
        return None
    df = pd.read_csv(fn, parse_dates=['Datetime'], index_col='Datetime')
    return df

def simulate_c_with_params(df, rsi_buy, rsi_sell, allocation=0.05, initial_cash=100000):
    cash = initial_cash
    positions = []
    realized = []
    for t, row in df.iterrows():
        rsi = float(row.get('RSI', np.nan))
        price = float(row['Close'])
        if np.isnan(rsi):
            continue
        # buy signal
        if rsi < rsi_buy and not positions:
            qty = int((cash * allocation) / price)
            if qty > 0:
                cash -= qty * price
                positions.append({'qty': qty, 'price': price})
        # sell signal
        if positions and rsi > rsi_sell:
            pos = positions.pop(0)
            pl = (price - pos['price']) * pos['qty']
            realized.append(pl)
            cash += pos['qty'] * price
    return sum(realized), len(realized)

def tune_for_symbol(symbol):
    df = load_symbol(symbol)
    if df is None or df.empty:
        return None
    # grid for RSI buy in [20,35], sell in [60,80]
    best = (None, -1, None)
    for buy in range(20, 36, 5):
        for sell in range(60, 81, 5):
            if buy >= sell:
                continue
            pl, trades = simulate_c_with_params(df, buy, sell)
            if pl > best[1]:
                best = ((buy, sell), pl, trades)
    return {'symbol': symbol, 'best_params': best[0], 'best_pl': best[1], 'trades': best[2]}

def main():
    syms = [fn.split('_15m.csv')[0] for fn in os.listdir(CACHE) if fn.endswith('_15m.csv')]
    results = []
    for s in syms:
        r = tune_for_symbol(s)
        if r:
            results.append(r)
    out = pd.DataFrame(results)
    out_fn = os.path.join(OUT, 'retrain_c_tuning_per_symbol.csv')
    out.to_csv(out_fn, index=False)
    print('Saved tuning results to', out_fn)

if __name__ == '__main__':
    main()
