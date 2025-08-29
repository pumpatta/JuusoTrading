import argparse, time, random
import sys
import os
from datetime import datetime, timedelta, timezone

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.portfolio import StrategyBook
from engine.broker_alpaca import Broker
from engine.datafeed import get_bars
from strategies.ema_trend import EmaTrend
from strategies.xgb_classifier import XgbSignal
from strategies.pattern_head_shoulders import HeadShoulders
from utils.execution import simulate_fills
from strategies.ensemble import Ensemble
from strategies.account_c_ml import AccountCStrategy
from utils.strategies_cfg import load_plan
from utils.safety import can_place_buy, daily_loss_breached, enforceable_stop_loss, can_trade_symbol, record_trade
import json
import os
import yaml
from pathlib import Path

def now_utc(): return datetime.now(timezone.utc)

# default mapping: which strategy runs on which account
DEFAULT_STRATEGY_ACCOUNT = {
    'EMA': 'A',
    'XGB': 'B',
    'HNS': 'C',
    'ENSEMBLE': 'C',
    'ACCOUNT_C_ML': 'C',
}


def get_account_for_strategy(sid: str) -> str:
    # Use account A for all strategies to avoid permission issues
    return 'A'

def log_trade(strategy_id, symbol, side, qty, price):
    from pathlib import Path
    import csv, datetime
    Path('storage/logs').mkdir(parents=True, exist_ok=True)
    fpath = Path(f'storage/logs/trades_{strategy_id}.csv')
    new = not fpath.exists()
    with fpath.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new: w.writerow(['ts','symbol','side','qty','price'])
        w.writerow([datetime.datetime.utcnow().isoformat(), symbol, side, qty, price])

def log_engine_event(event_type, message, extra_data=None):
    """Log engine events to storage/logs/engine_events.log for monitoring."""
    from pathlib import Path
    import datetime
    Path('storage/logs').mkdir(parents=True, exist_ok=True)
    with open('storage/logs/engine_events.log', 'a', encoding='utf-8') as f:
        timestamp = datetime.datetime.utcnow().isoformat()
        extra = f" | {extra_data}" if extra_data else ""
        f.write(f"{timestamp} [{event_type}] {message}{extra}\n")

def main(paper: bool = True):
    # prepare per-account brokers (one broker/client per Alpaca account A/B/C)
    book = StrategyBook()
    broker_map: dict[str, Broker] = {}
    plan = load_plan()
    registry = {
    'EMA': EmaTrend(),
    'XGB': XgbSignal(account='B'),
        'HNS': HeadShoulders(),
        'ENSEMBLE': Ensemble(),
    'ACCOUNT_C_ML': AccountCStrategy('SPY'),
    }
    models = [registry[s.id] for s in plan.items if s.enabled and s.id in registry]
    
    # Load universe from config/universe.yml
    symbols = []  # Start with empty list, no fallback
    try:
        with open('config/universe.yml', 'r', encoding='utf-8') as f:
            universe_cfg = yaml.safe_load(f)
        universe_symbols = []
        for t in universe_cfg.get('tickers', []):
            sym = t.get('symbol')
            if not sym:
                continue
            if t.get('enabled', True):
                universe_symbols.append(sym)
        
        if universe_symbols:
            symbols = universe_symbols
            print(f"Successfully loaded {len(symbols)} symbols from universe")
        else:
            print("No symbols found in universe, cannot continue")
            return
    except Exception as e:
        print(f"Error loading universe: {e}")
        return
    
    initial_capital = float(os.getenv('INITIAL_CAPITAL', '100000'))
    # keep a quick portfolio value estimate (start = initial_capital)
    portfolio_value = initial_capital

    # per-day trade count tracking (reset on UTC midnight)
    trade_counts: dict[str, int] = {}
    last_reset = now_utc().date()

    while True:
        try:
            loop_start = time.time()
            print(f"Starting main loop iteration at {now_utc()}")
            
            # reset daily trade counts at UTC midnight
            if now_utc().date() != last_reset:
                trade_counts = {}
                last_reset = now_utc().date()
            # simple daily loss check
            if daily_loss_breached(portfolio_value, initial_capital):
                print('DAILY LOSS LIMIT BREACHED — stopping trading')
                break
            end = now_utc(); start = end - timedelta(minutes=2000)
            print(f"Fetching bars for {len(symbols)} symbols...")
            bars = get_bars(symbols, start, end, timeframe="1Min")
            print(f"Bars fetched, processing {len(models)} models...")
            
            for m in models:
                try:
                    intents = m.on_bar(bars)
                    account_for_model = get_account_for_strategy(m.strategy_id)
                    if account_for_model not in broker_map:
                        broker_map[account_for_model] = Broker(paper=paper, account=account_for_model)
                    broker = broker_map[account_for_model]
                    if not intents:
                        continue
                    log_engine_event('INTENTS_GENERATED', f'Strategy {m.strategy_id} generated {len(intents)} intents', {'account': account_for_model, 'intents': intents})
                    for it in intents:
                        sid, sym = it['strategy_id'], it['symbol']
                        if it.get('side') == 'buy' or it.get('action') == 'buy':
                            qty = float(it.get('qty', 1))
                            tp, sl = it.get('take_profit'), it.get('stop_loss')
                            # enforce global/default stop loss if not provided
                            if sl is None:
                                sl = enforceable_stop_loss()
                            coid = f"{sid}_{sym}_{int(time.time())}_{random.randint(1000,9999)}"
                            if tp and sl:
                                try:
                                    # safety: check position sizing against initial capital
                                    cur_price = float(bars[sym].iloc[-1]['close']) if sym in bars else tp/1.015
                                    if not can_place_buy(initial_capital, cur_price, qty, initial_capital):
                                        print(f'Safety: buy rejected for {sym} qty={qty} by can_place_buy')
                                        log_engine_event('SAFETY_REJECT', f'Buy rejected for {sym} qty={qty} by can_place_buy', {'strategy': sid, 'account': account_for_model})
                                        continue
                                    if not can_trade_symbol(trade_counts, sym):
                                        print(f'Safety: buy rejected for {sym} — daily trade cap reached')
                                        log_engine_event('SAFETY_REJECT', f'Buy rejected for {sym} — daily trade cap reached', {'strategy': sid, 'account': account_for_model})
                                        continue
                                    # Check if we already have a position in this symbol for this strategy
                                    st = book.get_state(sid)
                                    if sym in st.positions and st.positions[sym].qty > 0:
                                        print(f'Position: buy rejected for {sym} — already holding {st.positions[sym].qty} shares')
                                        log_engine_event('POSITION_REJECT', f'Buy rejected for {sym} — already holding position', {'strategy': sid, 'account': account_for_model, 'current_qty': st.positions[sym].qty})
                                        continue
                                    log_engine_event('ORDER_ATTEMPT', f'Attempting bracket buy for {sym} qty={qty} TP={tp} SL={sl}', {'strategy': sid, 'account': account_for_model, 'client_order_id': coid})
                                    broker.buy_bracket(sym, qty, tp, sl, client_order_id=coid)
                                    # Simulate realistic fills
                                    nbbo_mid = cur_price
                                    for fill in simulate_fills('buy', qty, nbbo_mid):
                                        book.update_on_fill(sid, sym, 'buy', fill.qty, fill.price)
                                        log_trade(sid, sym, 'buy', fill.qty, fill.price)
                                        portfolio_value -= fill.qty * fill.price
                                        from utils.safety import record_trade
                                        record_trade(trade_counts, sym)
                                    log_engine_event('ORDER_SUCCESS', f'Bracket buy executed for {sym} qty={qty}', {'strategy': sid, 'account': account_for_model, 'fills': [{'qty': fill.qty, 'price': fill.price} for fill in simulate_fills('buy', qty, nbbo_mid)]})
                                except Exception as e:
                                    print('order error', e)
                                    log_engine_event('ORDER_ERROR', f'Bracket buy failed for {sym}: {type(e).__name__}: {str(e)}', {'strategy': sid, 'account': account_for_model, 'client_order_id': coid})
                                    try:
                                        from pathlib import Path
                                        Path('storage/logs').mkdir(parents=True, exist_ok=True)
                                        with open('storage/logs/engine_errors.log', 'a', encoding='utf-8') as f:
                                            import traceback, datetime
                                            f.write(f"{datetime.datetime.utcnow().isoformat()} - Order error for {sid}/{sym} on {account_for_model}: {type(e).__name__}: {str(e)}\n")
                                            traceback.print_exc(file=f)
                                    except Exception:
                                        pass
                        elif it.get('side') == 'sell' or it.get('action') == 'sell':
                            st = book.get_state(sid)
                            if sym in st.positions and st.positions[sym].qty > 0:
                                qty = float(min(st.positions[sym].qty, float(it.get('qty', 1))))
                                if qty > 0:
                                    coid = f"{sid}_{sym}_CLOSE_{int(time.time())}_{random.randint(1000,9999)}"
                                    try:
                                        log_engine_event('ORDER_ATTEMPT', f'Attempting market sell for {sym} qty={qty}', {'strategy': sid, 'account': account_for_model, 'client_order_id': coid})
                                        broker.sell_market(sym, qty, client_order_id=coid)
                                        nbbo_mid = float(bars[sym].iloc[-1]['close']) if sym in bars else st.positions[sym].avg_price
                                        for fill in simulate_fills('sell', qty, nbbo_mid):
                                            book.update_on_fill(sid, sym, 'sell', fill.qty, fill.price)
                                            log_trade(sid, sym, 'sell', fill.qty, fill.price)
                                        log_engine_event('ORDER_SUCCESS', f'Market sell executed for {sym} qty={qty}', {'strategy': sid, 'account': account_for_model, 'fills': [{'qty': fill.qty, 'price': fill.price} for fill in simulate_fills('sell', qty, nbbo_mid)]})
                                    except Exception as e:
                                        print('close error', e)
                                        log_engine_event('ORDER_ERROR', f'Market sell failed for {sym}: {type(e).__name__}: {str(e)}', {'strategy': sid, 'account': account_for_model, 'client_order_id': coid})
                                        try:
                                            from pathlib import Path
                                            Path('storage/logs').mkdir(parents=True, exist_ok=True)
                                            with open('storage/logs/engine_errors.log', 'a', encoding='utf-8') as f:
                                                import traceback, datetime
                                                f.write(f"{datetime.datetime.utcnow().isoformat()} - Close error for {sid}/{sym} on {account_for_model}: {type(e).__name__}: {str(e)}\n")
                                                traceback.print_exc(file=f)
                                        except Exception:
                                            pass
                except Exception as e:
                    print(f'Strategy {m.strategy_id} error:', type(e).__name__, e)
                    log_engine_event('STRATEGY_ERROR', f'Strategy {m.strategy_id} failed: {type(e).__name__}: {str(e)}', {'account': get_account_for_strategy(m.strategy_id)})
                    try:
                        from pathlib import Path
                        Path('storage/logs').mkdir(parents=True, exist_ok=True)
                        with open('storage/logs/engine_errors.log', 'a', encoding='utf-8') as f:
                            import traceback, datetime
                            f.write(f"{datetime.datetime.utcnow().isoformat()} - Strategy {m.strategy_id} error: {type(e).__name__}: {str(e)}\n")
                            traceback.print_exc(file=f)
                    except Exception:
                        pass
            loop_end = time.time()
            print(f"Loop iteration completed in {loop_end - loop_start:.2f} seconds")
            time.sleep(60)
        except KeyboardInterrupt:
            print('Keyboard interrupt received — stopping live loop')
            break
        except Exception as e:
            # Log unexpected exceptions and keep running
            print('Live loop error:', type(e).__name__, e)
            try:
                from pathlib import Path
                Path('storage/logs').mkdir(parents=True, exist_ok=True)
                with open('storage/logs/engine_errors.log', 'a', encoding='utf-8') as f:
                    import traceback, datetime
                    f.write(f"{datetime.datetime.utcnow().isoformat()} - {type(e).__name__}: {str(e)}\n")
                    traceback.print_exc(file=f)
            except Exception:
                pass
            # short backoff before next loop
            time.sleep(5)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    main(paper=not args.live)
