import sys
sys.path.insert(0, '.')
from engine.datafeed import get_bars
from datetime import datetime, timedelta, timezone

symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA']
end = datetime.now(timezone.utc)
start = end - timedelta(minutes=2000)

print('Testing data fetch with 2000-minute window...')
bars = get_bars(symbols, start, end, timeframe='1Min')

for sym, df in bars.items():
    print(f'{sym}: {len(df)} rows')
    if len(df) > 0:
        print(f'  Date range: {df["ts"].min()} to {df["ts"].max()}')
        print(f'  Sample data points: {len(df)} total')
