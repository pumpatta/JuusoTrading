# Diagnostic for Alpaca TradingClient instantiation (safe - no secrets printed)
import os
from importlib import import_module

def try_instantiation(TradingClient, kid, sk, base_url, paper=True):
    attempts = []
    # Attempt 1: positional (old style)
    try:
        obj = TradingClient(kid, sk, paper=paper)
        return True, 'positional'
    except Exception as e:
        attempts.append(('positional', type(e).__name__, str(e)[:300]))
    # Attempt 2: keyword args api_key/api_secret/base_url
    try:
        obj = TradingClient(api_key=kid, api_secret=sk, base_url=base_url)
        return True, 'kw_api_key'
    except Exception as e:
        attempts.append(('kw_api_key', type(e).__name__, str(e)[:300]))
    # Attempt 3: kw with paper flag
    try:
        obj = TradingClient(api_key=kid, api_secret=sk, paper=paper)
        return True, 'kw_api_key_paper'
    except Exception as e:
        attempts.append(('kw_api_key_paper', type(e).__name__, str(e)[:300]))
    return False, attempts


def main():
    print('Diagnostic start')
    try:
        m = import_module('alpaca.trading.client')
        TradingClient = getattr(m, 'TradingClient', None)
    except Exception as e:
        print('IMPORT_ERROR', type(e).__name__, str(e)[:300])
        return
    if TradingClient is None:
        print('TradingClient not found in alpaca.trading.client')
        return
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    for acc in ['A','B','C']:
        kid = os.getenv(f'ALPACA_KEY_ID_{acc}') or os.getenv('ALPACA_KEY_ID')
        sk = os.getenv(f'ALPACA_SECRET_KEY_{acc}') or os.getenv('ALPACA_SECRET_KEY')
        present = bool(kid and sk)
        print(f'Account {acc}: keys_present={present}')
        if not present:
            continue
        ok, info = try_instantiation(TradingClient, kid, sk, base_url)
        if ok:
            print(f'  Instantiation succeeded using pattern: {info}')
        else:
            print('  Instantiation failed. Attempts:')
            for att in info:
                print('   -', att[0], att[1], att[2])

if __name__ == "__main__":
    main()
