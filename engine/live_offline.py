"""
Offline Live Engine for JuusoTrader
Runs strategies with sample data when markets are closed
"""

import argparse
import time
import random
import sys
import os
from datetime import datetime, timedelta, timezone

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.portfolio import StrategyBook
from engine.broker_alpaca import Broker
from strategies.ema_trend import EmaTrend
from strategies.xgb_classifier import XgbSignal
from strategies.account_c_ml import AccountCStrategy
from utils.execution import simulate_fills
from utils.strategies_cfg import load_plan
import pandas as pd
from pathlib import Path

def now_utc(): 
    return datetime.now(timezone.utc)

def load_sample_data(symbols=['SPY']):
    """Load sample data for offline testing"""
    data = {}
    base = Path('storage/sample_bars')
    
    for symbol in symbols:
        csv_path = base / f"{symbol}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=['ts'])
                # Take recent data (last 200 bars)
                df = df.tail(200).copy()
                data[symbol] = df
                print(f"Loaded {len(df)} bars for {symbol}")
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
    
    return data

def get_account_for_strategy(sid: str) -> str:
    DEFAULT_STRATEGY_ACCOUNT = {
        'EMA': 'A',
        'XGB': 'B', 
        'ACCOUNT_C_ML': 'C',
    }
    return DEFAULT_STRATEGY_ACCOUNT.get(sid.upper(), 'A')

def log_trade(strategy_id, symbol, side, qty, price):
    from pathlib import Path
    import csv, datetime
    Path('storage/logs').mkdir(parents=True, exist_ok=True)
    fpath = Path(f'storage/logs/trades_{strategy_id}.csv')
    new = not fpath.exists()
    with fpath.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new: 
            w.writerow(['ts','symbol','side','qty','price'])
        w.writerow([datetime.datetime.utcnow().isoformat(), symbol, side, qty, price])

def main_offline():
    """Run trading engine with sample data (offline mode)"""
    print("🔄 Starting JuusoTrader Offline Engine...")
    print("📊 Using sample data (markets may be closed)")
    
    # Initialize
    book = StrategyBook()
    broker_map = {}
    plan = load_plan()
    
    # Strategy registry
    registry = {
        'EMA': EmaTrend(),
        'XGB': XgbSignal(),
        'ACCOUNT_C_ML': AccountCStrategy('SPY'),
    }
    
    # Active strategies from config
    models = [registry[s.id] for s in plan.items if s.enabled and s.id in registry]
    print(f"📋 Active strategies: {[s.strategy_id for s in models]}")
    
    # Load sample data
    bars = load_sample_data(['SPY'])
    if not bars:
        print("❌ No sample data available!")
        return
    
    print("🚀 Starting trading loop...")
    cycle = 0
    
    while True:
        cycle += 1
        print(f"\n🔄 Trading Cycle #{cycle} - {now_utc().strftime('%H:%M:%S')}")
        
        try:
            # Collect signals from all active strategies
            all_intents = []
            
            for model in models:
                try:
                    # Get signals from strategy
                    intents = model.on_bar(bars)
                    
                    for intent in intents:
                        if 'strategy_id' not in intent:
                            intent['strategy_id'] = model.strategy_id
                        all_intents.append(intent)
                        
                        print(f"📡 {intent['strategy_id']}: {intent.get('action', 'N/A')} {intent.get('symbol', 'N/A')} "
                              f"(conf: {intent.get('confidence', 0):.3f})")
                        
                except Exception as e:
                    print(f"❌ Strategy {model.strategy_id} error: {e}")
            
            # Process trading intents by account
            accounts_active = {}
            
            for intent in all_intents:
                sid = intent['strategy_id']
                account = get_account_for_strategy(sid)
                
                if account not in accounts_active:
                    accounts_active[account] = []
                accounts_active[account].append(intent)
                
                # Initialize broker if needed
                if account not in broker_map:
                    broker_map[account] = Broker(paper=True, account=account)
                
                # Execute trade (simplified for offline)
                symbol = intent.get('symbol', 'SPY')
                action = intent.get('action', 'hold')
                
                if action == 'buy':
                    qty = intent.get('position_size', 0.1) * 100  # Convert to shares
                    if qty > 0:
                        # Simulate fill
                        price = float(bars[symbol].iloc[-1]['close'])
                        book.update_on_fill(sid, symbol, 'buy', qty, price)
                        log_trade(sid, symbol, 'buy', qty, price)
                        print(f"  ✅ BUY {qty} {symbol} @ ${price:.2f}")
                
                elif action == 'sell':
                    # Check if we have position
                    state = book.get_state(sid)
                    if symbol in state.positions and state.positions[symbol].qty > 0:
                        qty = min(state.positions[symbol].qty, 100)
                        price = float(bars[symbol].iloc[-1]['close'])
                        book.update_on_fill(sid, symbol, 'sell', qty, price)
                        log_trade(sid, symbol, 'sell', qty, price)
                        print(f"  ✅ SELL {qty} {symbol} @ ${price:.2f}")
            
            # Account summary
            print(f"\n📈 Account Activity:")
            for account in ['A', 'B', 'C']:
                if account in accounts_active:
                    signals = len(accounts_active[account])
                    print(f"  Account {account}: {signals} signals")
                else:
                    print(f"  Account {account}: No active signals")
            
            # Portfolio summary
            print(f"\n💰 Portfolio Status:")
            for model in models:
                state = book.get_state(model.strategy_id)
                total_value = sum(pos.qty * pos.avg_price for pos in state.positions.values())
                print(f"  {model.strategy_id}: ${total_value:.2f}")
            
        except Exception as e:
            print(f"❌ Cycle error: {e}")
        
        print(f"⏰ Waiting 30 seconds until next cycle...")
        time.sleep(30)  # Shorter cycle for demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run offline with sample data")
    args = parser.parse_args()
    
    if args.offline:
        main_offline()
    else:
        print("Use --offline flag to run offline mode")
        print("Example: python engine/live_offline.py --offline")
