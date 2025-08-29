from datetime import datetime, timedelta
import traceback

print('=== load bars (datafeed.get_bars) ===')
try:
    from engine import datafeed
    end = datetime.utcnow()
    start = end - timedelta(minutes=60)
    bars = datafeed.get_bars(['SPY'], start, end, timeframe='1Min', prefer_samples=False)
    print('bars keys:', list(bars.keys()))
    spy = bars.get('SPY')
    if spy is not None:
        print('latest bar sample:', spy.tail(3).to_dict(orient='records'))
    else:
        print('no SPY bars returned')
except Exception as e:
    print('datafeed error:', type(e).__name__, e)
    traceback.print_exc()

print('\n=== strategy signal probe ===')
try:
    from strategies.ema_trend import EmaTrend
    from strategies.xgb_classifier import XgbSignal
    from strategies.ensemble import Ensemble
    
    # instantiate strategies for A, B, C as configured in engine
    sA = EmaTrend(account='A')
    sB = XgbSignal(account='B')
    sC = Ensemble(account='C')
    
    latest_bars = {'SPY': spy} if spy is not None else {}
    print('Running EmaTrend (A).on_bar')
    try:
        sigA = sA.on_bar(latest_bars)
        print('EmaTrend signals:', sigA)
    except Exception as e:
        print('EmaTrend on_bar error:', type(e).__name__, e)

    print('Running XgbSignal (B).on_bar')
    try:
        sigB = sB.on_bar(latest_bars)
        print('XgbSignal signals:', sigB)
    except Exception as e:
        print('XgbSignal on_bar error:', type(e).__name__, e)

    print('Running Ensemble (C).on_bar')
    try:
        sigC = sC.on_bar(latest_bars)
        print('Ensemble signals:', sigC)
    except Exception as e:
        print('Ensemble on_bar error:', type(e).__name__, e)

except Exception as e:
    print('Strategy probe failed:', type(e).__name__, e)
    traceback.print_exc()

print('\n=== check open orders via broker wrapper for A ===')
try:
    from engine.broker_alpaca import Broker
    b = Broker(paper=True, account='A')
    if b.client is None:
        print('Broker A: client None')
    else:
        # try multiple ways to list orders depending on SDK
        try:
            if hasattr(b.client, 'get_orders'):
                orders = b.client.get_orders()
                print('get_orders returned:', type(orders), getattr(orders, '__len__', lambda: 'len?')())
            elif hasattr(b.client, 'list_orders'):
                orders = b.client.list_orders()
                print('list_orders returned:', type(orders), getattr(orders, '__len__', lambda: 'len?')())
            else:
                print('No orders-listing API detected on client')
        except Exception as e:
            print('orders listing failed:', type(e).__name__, str(e)[:200])
except Exception as e:
    print('Broker probe failed:', type(e).__name__, e)
    traceback.print_exc()
