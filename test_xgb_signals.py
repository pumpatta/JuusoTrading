import sys
sys.path.append('.')
from strategies.xgb_classifier import XgbSignal
from engine.datafeed import get_bars
from datetime import datetime, timedelta, timezone
import pandas as pd

# Create XGB strategy
xgb = XgbSignal(account='B')
print(f"XGB fitted: {xgb.fitted}")

# Get some recent data
end = datetime.now(timezone.utc)
start = end - timedelta(hours=24)
bars = get_bars(['SPY', 'QQQ', 'AAPL', 'TSLA'], start, end, timeframe='1Min')

if bars:
    print(f"Received bars for {len(bars)} symbols")
    for sym, df in bars.items():
        print(f"{sym}: {len(df)} rows")
        if len(df) > 0:
            print(f"  Latest price: {df.iloc[-1]['close']}")
            print(f"  Data range: {df.index[0]} to {df.index[-1]}")

    print('\nTesting XGB strategy signals...')
    signals = xgb.on_bar(bars)
    print(f'Generated {len(signals)} signals')
    for sig in signals:
        print(f'  {sig}')

    # Check probabilities for a few symbols
    for sym in ['SPY', 'AAPL', 'TSLA']:
        if sym in bars and len(bars[sym]) > 300:
            df = bars[sym]
            print(f"\n{sym} analysis:")
            feats = xgb._features(df).iloc[-1:][['ret1','vol','mom','rsi']].fillna(0)
            print(f"Features: {feats.iloc[0].to_dict()}")
            proba = float(xgb.model.predict_proba(feats)[0,1])
            price = float(df.iloc[-1]['close'])
            print(f"Probability: {proba:.3f}, Price: {price:.2f}")
            print(f"Would generate: {'BUY' if proba > 0.56 else 'SELL' if proba < 0.44 else 'HOLD'}")
else:
    print("No bars data received")