#!/usr/bin/env python3
"""
Debug bracket order errors - capture full Alpaca API error responses
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from engine.broker_alpaca import Broker
from utils.config import SETTINGS

def debug_bracket_errors():
    """Debug bracket order errors by capturing full API responses."""

    print("🔍 Debugging bracket order errors...")

    # Test with account A first
    broker = Broker(paper=True, account='A')
    print(f"Broker created for account {broker.account}")

    # Test parameters
    symbol = 'AAPL'
    qty = 1
    current_price = 230.50
    tp = round(current_price * 1.01, 2)  # 1% take profit, rounded to penny
    sl = round(current_price * 0.99, 2)  # 1% stop loss, rounded to penny
    client_order_id = f"DEBUG_TEST_{int(__import__('time').time())}"

    print(f"Test parameters:")
    print(f"  Symbol: {symbol}")
    print(f"  Qty: {qty}")
    print(f"  TP: ${tp:.2f}")
    print(f"  SL: ${sl:.2f}")
    print(f"  Client ID: {client_order_id}")

    # Temporarily disable dry_run
    original_dry_run = getattr(SETTINGS, 'dry_run', False)
    setattr(SETTINGS, 'dry_run', False)

    try:
        print("\n📡 Testing REST bracket order directly...")

        # Build the payload manually to debug
        body = {
            'symbol': symbol,
            'qty': qty,
            'side': 'buy',
            'type': 'market',
            'time_in_force': 'day',
            'order_class': 'bracket',
            'take_profit': {'limit_price': float(tp)},
            'stop_loss': {'stop_price': float(sl)},
        }
        if client_order_id:
            body['client_order_id'] = client_order_id

        print(f"Request payload: {json.dumps(body, indent=2)}")

        # Make the request directly
        import requests
        url = f"{broker.base_url.rstrip('/')}/v2/orders"
        headers = broker._order_headers()

        print(f"URL: {url}")
        print(f"Headers: {headers}")

        r = requests.post(url, json=body, headers=headers, timeout=10)

        if r.status_code == 422:
            print(f"❌ HTTP 422 Error")
            print(f"Response text: {r.text}")
            try:
                error_json = r.json()
                print(f"Error details: {json.dumps(error_json, indent=2)}")
            except:
                print("Could not parse error as JSON")
        elif r.status_code == 200:
            print("✅ Success!")
            result = r.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ HTTP {r.status_code}")
            print(f"Response: {r.text}")

    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {str(e)}")

    finally:
        setattr(SETTINGS, 'dry_run', original_dry_run)

if __name__ == '__main__':
    debug_bracket_errors()
