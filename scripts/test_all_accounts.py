#!/usr/bin/env python3
"""
Test all three paper accounts (A, B, C) for bracket order functionality
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

def test_all_accounts():
    """Test bracket orders on all three accounts A, B, C."""

    accounts = ['A', 'B', 'C']
    results = {}

    print("🔬 Testing all paper accounts (A, B, C)...")
    print("=" * 50)

    # Temporarily disable dry_run for real testing
    original_dry_run = getattr(SETTINGS, 'dry_run', False)
    setattr(SETTINGS, 'dry_run', False)

    try:
        for account in accounts:
            print(f"\n📊 Testing Account {account}")
            print("-" * 30)

            try:
                # Create broker for this account
                broker = Broker(paper=True, account=account)
                print(f"✅ Broker created for account {broker.account}")
                print(f"   Has SDK client: {bool(broker.client)}")

                # Test parameters
                symbol = 'AAPL'
                qty = 1
                current_price = 230.50
                tp = current_price * 1.01  # 1% take profit (smaller for testing)
                sl = current_price * 0.99  # 1% stop loss (smaller for testing)
                client_order_id = f"ACCOUNT_TEST_{account}_{int(time.time())}"

                print(f"   📈 Submitting bracket order...")
                print(f"      Symbol: {symbol}")
                print(f"      Quantity: {qty}")
                print(f"      Take Profit: ${tp:.2f}")
                print(f"      Stop Loss: ${sl:.2f}")

                # Submit bracket order
                result = broker.buy_bracket(symbol, qty, tp, sl, client_order_id)

                # Handle both dict and Order object responses
                if isinstance(result, dict):
                    order_id = result.get('id', 'N/A')
                    status = result.get('status', 'N/A')
                    legs = result.get('legs', [])
                else:
                    order_id = getattr(result, 'id', 'N/A')
                    status = getattr(result, 'status', 'N/A')
                    legs = getattr(result, 'legs', [])

                print(f"   ✅ Order submitted successfully!")
                print(f"      Order ID: {order_id}")
                print(f"      Status: {status}")
                print(f"      Legs: {len(legs) if legs else 0}")

                results[account] = {
                    'success': True,
                    'order_id': order_id,
                    'status': status,
                    'legs_count': len(legs) if legs else 0,
                    'error': None
                }

            except Exception as e:
                print(f"   ❌ Account {account} failed: {type(e).__name__}: {str(e)}")
                results[account] = {
                    'success': False,
                    'order_id': None,
                    'status': None,
                    'legs_count': 0,
                    'error': str(e)
                }

        # Summary
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)

        all_success = True
        for account, result in results.items():
            status_icon = "✅" if result['success'] else "❌"
            print(f"{status_icon} Account {account}: {'SUCCESS' if result['success'] else 'FAILED'}")
            if result['success']:
                print(f"   Order ID: {result['order_id']}")
                print(f"   Status: {result['status']}")
                print(f"   Legs: {result['legs_count']}")
            else:
                print(f"   Error: {result['error']}")
            print()

        if all(result['success'] for result in results.values()):
            print("🎉 ALL ACCOUNTS WORKING! Ready for multi-account paper trading.")
        else:
            failed_accounts = [acc for acc, res in results.items() if not res['success']]
            print(f"⚠️ Some accounts failed: {', '.join(failed_accounts)}")
            all_success = False

        return all_success

    finally:
        # Restore original dry_run setting
        setattr(SETTINGS, 'dry_run', original_dry_run)

if __name__ == '__main__':
    success = test_all_accounts()
    if success:
        print("\n🚀 All paper accounts are ready for automated trading!")
    else:
        print("\n💥 Some accounts need attention before proceeding.")
