import sys
sys.path.insert(0, '.')
from engine.datafeed import _load_samples

symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA']
samples = _load_samples(symbols)
print('Loaded samples for:', list(samples.keys()))
for sym, df in samples.items():
    print(f'{sym}: {len(df)} rows')
    if len(df) > 0:
        print(f'  Date range: {df["ts"].min()} to {df["ts"].max()}')
