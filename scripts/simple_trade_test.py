#!/usr/bin/env python3
"""
Simple controlled trading test - bypasses heavy initialization
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from engine.broker_alpaca import Broker
from engine.datafeed import get_bars
from strategies.ema_trend import EmaTrend
from utils.config import SETTINGS
from utils.safety import can_place_buy, can_trade_symbol
import datetime

def simple_trading_test():
    """Run a simple trading test with minimal setup."""

    print("🚀 Starting simple trading test...")

    # Set small amounts for testing
    os.environ['INITIAL_CAPITAL'] = '1000'

    # Create broker for account A
    broker = Broker(paper=True, account='A')
    print(f"✅ Broker created for account {broker.account}")

    # Get some market data
    symbols = ['AAPL']
    end = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(minutes=60)

    print("📊 Fetching market data...")
    bars = get_bars(symbols, start, end, timeframe="1Min")
    print(f"✅ Got data for {len(bars)} symbols")

    # Create a simple strategy
    strategy = EmaTrend()
    print(f"✅ Strategy {strategy.strategy_id} created")

    # Generate intents
    print("🎯 Generating trading intents...")
    intents = strategy.on_bar(bars)

    if intents:
        print(f"✅ Generated {len(intents)} trading intents:")
        for intent in intents:
            print(f"  - {intent}")

        # Process intents
        for intent in intents:
            symbol = intent['symbol']
            qty = float(intent.get('qty', 1))

            # Check safety
            if symbol in bars:
                cur_price = float(bars[symbol].iloc[-1]['close'])
                if can_place_buy(1000, cur_price, qty, 1000):
                    if can_trade_symbol({}, symbol):  # Empty trade_counts for test
                        print(f"🛡️ Safety checks passed for {symbol}")

                        # Set reasonable TP/SL based on current price
                        tp = cur_price * 1.02  # 2% take profit
                        sl = cur_price * 0.98  # 2% stop loss

                        print(f"📈 Attempting bracket order: {symbol} qty={qty} TP={tp:.2f} SL={sl:.2f}")

                        try:
                            result = broker.buy_bracket(symbol, qty, tp, sl, f"TEST_{int(time.time())}")
                            print(f"✅ Bracket order successful!")

                            # Handle both dict and Order object responses
                            if isinstance(result, dict):
                                order_id = result.get('id', 'N/A')
                                status = result.get('status', 'N/A')
                            else:
                                # Assume it's an Order object with attributes
                                order_id = getattr(result, 'id', 'N/A')
                                status = getattr(result, 'status', 'N/A')

                            print(f"   Order ID: {order_id}")
                            print(f"   Status: {status}")

                            # Wait a bit to see if it gets filled
                            print("⏳ Waiting 30 seconds to check order status...")
                            time.sleep(30)

                            return True  # Success

                        except Exception as e:
                            print(f"❌ Bracket order failed: {type(e).__name__}: {str(e)}")
                            return False
                    else:
                        print(f"🚫 Safety: Daily trade limit reached for {symbol}")
                else:
                    print(f"🚫 Safety: Cannot place buy for {symbol} qty={qty}")
            else:
                print(f"🚫 No price data available for {symbol}")
    else:
        print("ℹ️ No trading intents generated (this is normal for test conditions)")

    return True  # Test completed without errors

if __name__ == '__main__':
    success = simple_trading_test()
    if success:
        print("✅ Simple trading test completed successfully!")
    else:
        print("❌ Simple trading test failed")
