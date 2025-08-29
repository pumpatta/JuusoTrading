import traceback
from datetime import datetime, timedelta

print('=== datafeed probe ===')
try:
    from engine import datafeed
    res = datafeed.get_bars(['SPY'], datetime.utcnow() - timedelta(minutes=30), datetime.utcnow(), timeframe='1Min', prefer_samples=False)
    print('Datafeed probe keys:', list(res.keys()))
except Exception as e:
    print('Datafeed probe failed:', type(e).__name__, str(e))
    traceback.print_exc()

print('\n=== broker connectivity probe ===')
try:
    from engine.broker_alpaca import Broker
    for acct in ('A', 'B', 'C'):
        try:
            b = Broker(paper=True, account=acct)
            if b.client is None:
                print(f'Broker {acct}: client not instantiated (SDK/constructor mismatch)')
            else:
                try:
                    get_acc = getattr(b.client, 'get_account', None)
                    if callable(get_acc):
                        acc = get_acc()
                        st = getattr(acc, 'status', getattr(acc, 'account_status', 'N/A'))
                        cash = getattr(acc, 'cash', getattr(acc, 'cash_balance', 'N/A'))
                        print(f'Broker {acct}: get_account -> status={st}, cash={cash}')
                    else:
                        print(f'Broker {acct}: client instantiated but get_account not available on this SDK')
                except Exception as e:
                    print(f'Broker {acct}: get_account call failed: {type(e).__name__} {str(e)[:200]}')
        except Exception as e:
            print('Broker wrapper init failed for', acct, type(e).__name__, str(e)[:200])
except Exception:
    print('Failed to import Broker wrapper')
    traceback.print_exc()
