"""Small CLI to probe market-data REST fallback (bars + latest-trade) using engine.datafeed.get_bars

Usage: run from project root, for example:
    .venv/Scripts/python.exe scripts/rest_probe.py --symbols SPY,AAPL --minutes 10

It prints datafeed messages and a short summary of returned frames.
"""
import sys
from pathlib import Path
# Ensure project root is on sys.path so `import engine` works when running from scripts/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datetime import datetime, timedelta
import argparse
import pandas as pd

from engine import datafeed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbols', default='SPY', help='Comma-separated symbols')
    p.add_argument('--minutes', type=int, default=10, help='Lookback minutes for bars')
    p.add_argument('--timeframe', default='1Min', help='Bar timeframe')
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    end = datetime.utcnow()
    start = end - timedelta(minutes=args.minutes)

    print(f'Probing REST bars for: {symbols}  start={start.isoformat()}  end={end.isoformat()} timeframe={args.timeframe}')

    try:
        out = datafeed.get_bars(symbols, start, end, timeframe=args.timeframe, prefer_samples=False)
    except Exception as e:
        print('get_bars() raised:', type(e).__name__, e)
        return

    if not out:
        print('No data returned')
        return

    for s, df in out.items():
        print('-' * 60)
        print(f'Symbol: {s}  rows: {len(df)}')
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(df.head().to_string(index=False))
        else:
            print(repr(df))


if __name__ == '__main__':
    main()
