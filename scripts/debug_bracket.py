#!/usr/bin/env python3
"""
Detailed bracket order test to capture Alpaca API error responses.
"""

import sys
import json
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from engine.broker_alpaca import Broker
from utils.config import SETTINGS

def test_bracket_order():
    """Test bracket order submission and capture detailed error responses."""

    # Create broker for account A
    broker = Broker(paper=True, account='A')
    print(f"Broker created for account {broker.account}")
    print(f"Has SDK client: {bool(broker.client)}")

    # Test parameters
    symbol = 'AAPL'
    qty = 1
    take_profit = 250.0
    stop_loss = 200.0
    client_order_id = 'TEST_BRACKET_001'

    print(f"\nTesting bracket order: {symbol} qty={qty} TP={take_profit} SL={stop_loss}")

    # Temporarily disable dry_run to test real API
    original_dry_run = getattr(SETTINGS, 'dry_run', False)
    setattr(SETTINGS, 'dry_run', False)

    try:
        # Try to submit bracket order
        print("Submitting bracket order...")
        result = broker.buy_bracket(symbol, qty, take_profit, stop_loss, client_order_id)
        print(f"Success! Result: {result}")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")

        # If it's an HTTP error, try to get the response details
        try:
            import requests
            if isinstance(e, requests.HTTPError) and hasattr(e, 'response'):
                try:
                    error_details = e.response.json()
                    print(f"HTTP Error Details: {json.dumps(error_details, indent=2)}")
                except:
                    print(f"Raw response text: {e.response.text}")
        except ImportError:
            pass

    finally:
        # Restore original dry_run setting
        setattr(SETTINGS, 'dry_run', original_dry_run)

if __name__ == '__main__':
    test_bracket_order()
