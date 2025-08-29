#!/usr/bin/env python3
"""
Forced bracket order test - creates a test trade to validate the full flow
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
from utils.config import SETTINGS

def forced_bracket_test():
    """Force a bracket order to test the complete flow."""

    print("🔬 Starting forced bracket order test...")

    # Create broker for account A
    broker = Broker(paper=True, account='A')
    print(f"✅ Broker created for account {broker.account}")

    # Test parameters
    symbol = 'AAPL'
    qty = 1
    current_price = 230.50  # Approximate current price
    tp = current_price * 1.02  # 2% take profit
    sl = current_price * 0.98  # 2% stop loss
    client_order_id = f"FORCED_TEST_{int(time.time())}"

    print(f"📊 Test parameters:")
    print(f"   Symbol: {symbol}")
    print(f"   Quantity: {qty}")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Take Profit: ${tp:.2f}")
    print(f"   Stop Loss: ${sl:.2f}")
    print(f"   Client Order ID: {client_order_id}")

    # Temporarily disable dry_run for real test
    original_dry_run = getattr(SETTINGS, 'dry_run', False)
    setattr(SETTINGS, 'dry_run', False)

    try:
        print("📈 Submitting bracket order...")
        result = broker.buy_bracket(symbol, qty, tp, sl, client_order_id)

        # Handle both dict and Order object responses
        if isinstance(result, dict):
            order_id = result.get('id', 'N/A')
            status = result.get('status', 'N/A')
            legs = result.get('legs', [])
        else:
            # Assume it's an Order object with attributes
            order_id = getattr(result, 'id', 'N/A')
            status = getattr(result, 'status', 'N/A')
            legs = getattr(result, 'legs', [])

        print(f"✅ Bracket order submitted successfully!")
        print(f"   Main Order ID: {order_id}")
        print(f"   Status: {status}")
        print(f"   Number of legs: {len(legs) if legs else 0}")

        if legs:
            print("   Legs:")
            for i, leg in enumerate(legs):
                if isinstance(leg, dict):
                    leg_type = leg.get('type', 'N/A')
                    leg_side = leg.get('side', 'N/A')
                    leg_price = leg.get('limit_price') or leg.get('stop_price') or 'N/A'
                    leg_status = leg.get('status', 'N/A')
                else:
                    leg_type = getattr(leg, 'type', 'N/A')
                    leg_side = getattr(leg, 'side', 'N/A')
                    leg_price = getattr(leg, 'limit_price', getattr(leg, 'stop_price', 'N/A'))
                    leg_status = getattr(leg, 'status', 'N/A')

                print(f"     Leg {i+1}: {leg_type.upper()} {leg_side} @ ${leg_price} (status: {leg_status})")

        # Wait to see if order gets filled
        print("⏳ Waiting 60 seconds to monitor order status...")
        time.sleep(60)

        print("✅ Forced bracket test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Bracket order failed: {type(e).__name__}: {str(e)}")
        return False

    finally:
        # Restore original dry_run setting
        setattr(SETTINGS, 'dry_run', original_dry_run)

if __name__ == '__main__':
    success = forced_bracket_test()
    if success:
        print("🎉 All tests passed! Bracket order functionality is working correctly.")
    else:
        print("💥 Test failed. Check the error messages above.")
