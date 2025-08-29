"""Place a tiny paper market buy on account A and immediately close if filled.
Non-destructive for real funds (paper environment). Prints only non-sensitive fields.
"""
import time
from utils.config import get_settings_for
from importlib import import_module

def main():
    s = get_settings_for('A')
    print('Using account', s.account)
    try:
        m = import_module('alpaca.trading.client')
        TradingClient = getattr(m, 'TradingClient')
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
    except Exception as e:
        print('Import error:', type(e).__name__, e)
        return

    try:
        client = TradingClient(s.key_id, s.secret_key, paper=True)
    except Exception as e:
        print('Client instantiation error:', type(e).__name__, e)
        return

    try:
        # place buy
        req = MarketOrderRequest(symbol='SPY', qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        order = client.submit_order(req)
        oid = getattr(order, 'id', None)
        status = getattr(order, 'status', None)
        print('Order submitted:', oid, 'status=', status)

        # wait a short moment for paper fill
        time.sleep(1)
        try:
            order = client.get_order(oid)
        except Exception:
            pass
        status = getattr(order, 'status', None)
        filled_qty = float(getattr(order, 'filled_qty', 0) or 0)
        print('Order status after wait:', status, 'filled_qty=', filled_qty)

        if filled_qty > 0:
            # close position by selling filled_qty
            print('Closing position: selling', filled_qty)
            req2 = MarketOrderRequest(symbol='SPY', qty=filled_qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
            close_order = client.submit_order(req2)
            print('Close order id=', getattr(close_order,'id',None), 'status=', getattr(close_order,'status',None))
        else:
            # cancel if not filled
            try:
                client.cancel_order(oid)
                print('Cancelled order', oid)
            except Exception as e:
                print('Cancel error:', type(e).__name__, e)
    except Exception as e:
        print('Order error:', type(e).__name__, e)

if __name__ == '__main__':
    main()
