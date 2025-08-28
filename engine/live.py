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
    # allow override via environment variable STRATEGY_ACCOUNT_{SID}=A|B|C
    env = os.getenv(f'STRATEGY_ACCOUNT_{sid.upper()}')
    if env:
        return env.upper()
    return DEFAULT_STRATEGY_ACCOUNT.get(sid.upper(), os.getenv('ALPACA_ACCOUNT', 'A'))

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

def main(paper: bool = True):
    # prepare per-account brokers (one broker/client per Alpaca account A/B/C)
    book = StrategyBook()
    broker_map: dict[str, Broker] = {}
    plan = load_plan()
    registry = {
        'EMA': EmaTrend(),
        'XGB': XgbSignal(),
        'HNS': HeadShoulders(),
        'ENSEMBLE': Ensemble(),
        'ACCOUNT_C_ML': AccountCStrategy('SPY'),
    }
    models = [registry[s.id] for s in plan.items if s.enabled and s.id in registry]
    symbols = ['SPY','QQQ','AAPL','MSFT','NVDA','AMZN','META','TSLA']
    while True:
        end = now_utc(); start = end - timedelta(minutes=400)
        bars = get_bars(symbols, start, end, timeframe="1Min")
        for m in models:
            intents = m.on_bar(bars)
            account_for_model = get_account_for_strategy(m.strategy_id)
            if account_for_model not in broker_map:
                broker_map[account_for_model] = Broker(paper=paper, account=account_for_model)
            broker = broker_map[account_for_model]
            for it in intents:
                sid, sym = it['strategy_id'], it['symbol']
                if it['side'] == 'buy':
                    qty = float(it.get('qty', 1))
                    tp, sl = it.get('take_profit'), it.get('stop_loss')
                    coid = f"{sid}_{sym}_{int(time.time())}_{random.randint(1000,9999)}"
                    if tp and sl:
                        try:
                            broker.buy_bracket(sym, qty, tp, sl, client_order_id=coid)
                            # Simuloidaan realistinen täyttöhinta nbbo_mid ~ current price
                            nbbo_mid = float(bars[sym].iloc[-1]['close']) if sym in bars else tp/1.015
                            for fill in simulate_fills('buy', qty, nbbo_mid):
                                book.update_on_fill(sid, sym, 'buy', fill.qty, fill.price)
                                log_trade(sid, sym, 'buy', fill.qty, fill.price)
                        except Exception as e:
                            print('order error', e)
                elif it['side'] == 'sell':
                    st = book.get_state(sid)
                    if sym in st.positions and st.positions[sym].qty > 0:
                        qty = float(min(st.positions[sym].qty, float(it.get('qty', 1))))
                        if qty > 0:
                            coid = f"{sid}_{sym}_CLOSE_{int(time.time())}_{random.randint(1000,9999)}"
                            try:
                                broker.sell_market(sym, qty, client_order_id=coid)
                                nbbo_mid = float(bars[sym].iloc[-1]['close']) if sym in bars else st.positions[sym].avg_price
                                for fill in simulate_fills('sell', qty, nbbo_mid):
                                    book.update_on_fill(sid, sym, 'sell', fill.qty, fill.price)
                                    log_trade(sid, sym, 'sell', fill.qty, fill.price)
                            except Exception as e:
                                print('close error', e)
        time.sleep(60)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    main(paper=not args.live)
