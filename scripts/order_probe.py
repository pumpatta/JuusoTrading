"""Safe order probe for Alpaca paper trading.

Usage:
  .venv\Scripts\python.exe scripts/order_probe.py --symbol AAPL --qty 1 [--account A] [--confirm]

Without --confirm the script will only show what it would do.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
from datetime import datetime

from engine.broker_alpaca import Broker
from utils.config import SETTINGS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='AAPL')
    p.add_argument('--qty', type=float, default=1.0)
    p.add_argument('--account', default=None, help='Account letter A/B/C or none to use default')
    p.add_argument('--confirm', action='store_true', help='Actually place the order (paper).')
    args = p.parse_args()

    # create broker for account
    b = Broker(paper=True, account=args.account)
    print('Using account:', getattr(b, 'account', None))

    # show account summary via REST (safe)
    try:
        # prefer SDK if available
        if b.client and hasattr(b.client, 'get_account'):
            acc = b.client.get_account()
            print('Account via SDK:', getattr(acc, 'status', None), 'cash=', getattr(acc, 'cash', None))
        else:
            # REST GET /v2/account
            import requests
            base = b.base_url.rstrip('/')
            if not base.endswith('/v2'):
                base = f"{base}/v2"
            url = f"{base}/account"
            r = requests.get(url, headers=b._order_headers(), timeout=10)
            r.raise_for_status()
            acc = r.json()
            print('Account via REST:', acc.get('status'), 'cash=', acc.get('cash'))
    except Exception as e:
        print('Failed to fetch account:', type(e).__name__, e)

    # check market clock
    try:
        base = b.base_url.rstrip('/')
        if not base.endswith('/v2'):
            base = f"{base}/v2"
        url = f"{base}/clock"
        import requests
        r = requests.get(url, headers=b._order_headers(), timeout=5)
        r.raise_for_status()
        clock = r.json()
        print('Market open:', clock.get('is_open'), 'next_open:', clock.get('next_open'))
    except Exception as e:
        print('Failed to fetch clock:', type(e).__name__, e)

    # show existing positions
    try:
        if b.client and hasattr(b.client, 'get_all_positions'):
            pos = b.client.get_all_positions()
            print('Positions via SDK:', pos)
        else:
            base = b.base_url.rstrip('/')
            if not base.endswith('/v2'):
                base = f"{base}/v2"
            url = f"{base}/positions"
            r = requests.get(url, headers=b._order_headers(), timeout=10)
            if r.status_code == 200:
                print('Positions via REST:', r.json())
            else:
                print('Positions REST returned:', r.status_code, r.text)
    except Exception as e:
        print('Failed to fetch positions:', type(e).__name__, e)

    # show what would happen
    print('\nPlanned action:')
    print(f" - {'PLACE' if args.confirm else 'DRY-RUN'} market BUY {args.qty} {args.symbol} on account {b.account}")

    if args.confirm:
        # submit via broker convenience method, which uses SDK when available otherwise REST
        try:
            res = b.buy_market(args.symbol, args.qty)
            print('Order response:', res)
        except Exception as e:
            print('Order placement failed:', type(e).__name__, e)
    else:
        print('No order placed. Re-run with --confirm to place a paper order.')


if __name__ == '__main__':
    main()
