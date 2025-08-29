import requests
import os
from dotenv import load_dotenv
load_dotenv()

key_id = os.getenv('ALPACA_KEY_ID')
secret_key = os.getenv('ALPACA_SECRET_KEY')
base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

print(f'Testing REST API with key_id: {(key_id or "")[:8]}...')
print(f'Base URL: {base_url}')

url = f'{base_url}/v2/stocks/SPY/bars'
headers = {
    'APCA-API-KEY-ID': key_id,
    'APCA-API-SECRET-KEY': secret_key,
}
params = {'timeframe': '1Min', 'limit': 10}

try:
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    print(f'Response status: {resp.status_code}')
    if resp.status_code == 200:
        data = resp.json()
        bars = data.get('bars', [])
        print(f'Success! Got {len(bars)} bars')
        if bars:
            print(f'First bar: {bars[0]}')
    else:
        print(f'Error: {resp.text}')
except Exception as e:
    print(f'Exception: {type(e).__name__}: {e}')
