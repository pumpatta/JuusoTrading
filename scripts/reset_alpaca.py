"""Reset Alpaca paper account: list -> cancel orders -> close positions -> verify.

This script reads credentials from `utils.config.SETTINGS` (which loads .env).
Run this from the project root inside the activated `.venv`.
"""
import json
import sys
from pprint import pprint

try:
    import requests
except Exception:
    requests = None

from utils.config import SETTINGS

HEADERS = {
    'APCA-API-KEY-ID': SETTINGS.key_id,
    'APCA-API-SECRET-KEY': SETTINGS.secret_key,
}

BASE = SETTINGS.base_url.rstrip('/')

def _get(url):
    if requests:
        r = requests.get(url, headers=HEADERS)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text
    else:
        # fallback using urllib
        from urllib.request import Request, urlopen
        req = Request(url)
        req.add_header('APCA-API-KEY-ID', HEADERS['APCA-API-KEY-ID'])
        req.add_header('APCA-API-SECRET-KEY', HEADERS['APCA-API-SECRET-KEY'])
        with urlopen(req) as resp:
            data = resp.read().decode('utf-8')
            try:
                return resp.status, json.loads(data)
            except Exception:
                return resp.status, data

def _delete(url):
    if requests:
        r = requests.delete(url, headers=HEADERS)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text
    else:
        from urllib.request import Request, urlopen
        req = Request(url, method='DELETE')
        req.add_header('APCA-API-KEY-ID', HEADERS['APCA-API-KEY-ID'])
        req.add_header('APCA-API-SECRET-KEY', HEADERS['APCA-API-SECRET-KEY'])
        with urlopen(req) as resp:
            data = resp.read().decode('utf-8')
            try:
                return resp.status, json.loads(data)
            except Exception:
                return resp.status, data

def main():
    if not SETTINGS.key_id or not SETTINGS.secret_key:
        print('No Alpaca keys found in SETTINGS (.env). Aborting.')
        sys.exit(1)

    print('Base URL:', BASE)

    print('\n1) Listing open orders...')
    status, orders = _get(f'{BASE}/v2/orders?status=open')
    print('HTTP', status)
    pprint(orders)

    print('\n2) Listing positions...')
    status, positions = _get(f'{BASE}/v2/positions')
    print('HTTP', status)
    pprint(positions)

    print('\n3) Canceling all open orders...')
    status, resp = _delete(f'{BASE}/v2/orders?status=open')
    print('HTTP', status)
    pprint(resp)

    print('\n4) Closing all positions...')
    status, resp = _delete(f'{BASE}/v2/positions')
    print('HTTP', status)
    pprint(resp)

    print('\n5) Verifying...')
    status, orders = _get(f'{BASE}/v2/orders?status=open')
    print('Orders HTTP', status, 'count=', len(orders) if isinstance(orders, list) else 'N/A')
    status, positions = _get(f'{BASE}/v2/positions')
    print('Positions HTTP', status, 'count=', len(positions) if isinstance(positions, list) else 'N/A')

    print('\nReset complete.')


if __name__ == '__main__':
    main()
