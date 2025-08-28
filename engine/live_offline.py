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
import shutil

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

def load_sample_data(symbols=['SPY'], max_bars=500):
    """Load sample data for offline testing. Load up to `max_bars` recent bars if available.
    Returns dict[symbol] = DataFrame. Prints diagnostic if files missing or short."""
    data = {}
    base = Path('storage/sample_bars')

    for symbol in symbols:
        csv_path = base / f"{symbol}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=['ts'])
                # If file has more than max_bars, use only the most recent ones
                if len(df) > max_bars:
                    df = df.tail(max_bars).copy()
                else:
                    df = df.copy()
                data[symbol] = df
                print(f"Loaded {len(df)} bars for {symbol} (path={csv_path})")
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
        else:
            print(f"Sample bars not found for {symbol}: {csv_path}")

    return data

def get_account_for_strategy(sid: str) -> str:
    """Map strategy id to account. Supports strategy ids with suffixes like ACCOUNT_C_ML_SPY."""
    s = sid.upper()
    if s.startswith('EMA'):
        return 'A'
    if s.startswith('XGB'):
        return 'B'
    if s.startswith('ACCOUNT_C_ML') or 'ACCOUNT_C_ML' in s:
        return 'C'
    # default
    return 'A'

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

def main_offline(symbols: list[str], cycles: int = 5, sleep_sec: int = 30, seed: str | None = None, accel: bool = False, step: int = 1, clear_logs: bool = False, persist_seed: bool = False):
    """Run trading engine with sample data (offline mode).

    symbols: list of ticker strings to load from storage/sample_bars
    cycles: number of cycles to run (default 5). Use 0 for infinite.
    sleep_sec: seconds to wait between cycles.
    """
    print("🔄 Starting JuusoTrader Offline Engine...")
    print("📊 Using sample data (markets may be closed)")

    # Initialize
    book = StrategyBook()
    broker_map = {}
    plan = load_plan()

    # Optionally archive existing CSV logs to avoid confusing persisted seed/fill history
    if clear_logs:
        logs_dir = Path('storage/logs')
        if logs_dir.exists():
            csvs = list(logs_dir.glob('*.csv'))
            if csvs:
                timestamp = now_utc().strftime('%Y%m%d_%H%M%S')
                backup_dir = Path(f'storage/logs_backup_{timestamp}')
                backup_dir.mkdir(parents=True, exist_ok=True)
                for f in csvs:
                    try:
                        shutil.move(str(f), str(backup_dir / f.name))
                    except Exception as e:
                        print(f"⚠️ Failed to move {f}: {e}")
                print(f"📦 Archived {len(csvs)} log files to {backup_dir}")
            else:
                print("ℹ️ No CSV logs found to archive")
        else:
            print("ℹ️ No logs directory found to archive")

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
    bars = load_sample_data(symbols, max_bars=500)
    if not bars:
        print("❌ No sample data available!")
        return

    print("🚀 Starting trading loop...")
    cycle = 0

    # Prepare replay pointers per symbol for accelerated replay
    pointers: dict[str, int] = {}
    # choose a safe warmup start so strategies that require history have data
    warmup = 60
    for sym, df in bars.items():
        pointers[sym] = min(max(warmup, 1), max(len(df) - 1, 1))

    # Optionally seed an initial long position so sell signals can close it.
    # When using accelerated replay, seed at the current pointer price instead of the absolute last.
    if seed:
        try:
            parts = seed.split(':')
            seed_symbol = parts[0].upper()
            seed_qty = float(parts[1]) if len(parts) > 1 else 100.0
            if seed_symbol in bars:
                ptr = pointers.get(seed_symbol, len(bars[seed_symbol]) - 1)
                seed_price = float(bars[seed_symbol].iloc[ptr]['close'])
                # create a synthetic buy fill for strategy 'XGB' to seed inventory
                book.update_on_fill('XGB', seed_symbol, 'buy', seed_qty, seed_price)
                # By default the seed fill is synthetic and NOT written to persistent trade logs.
                # If the user explicitly requests persistence (for debugging), use --persist-seed.
                if persist_seed:
                    log_trade('XGB', seed_symbol, 'buy', seed_qty, seed_price)
                    print(f"🪴 Seeded {seed_qty} {seed_symbol} @ ${seed_price:.2f} for strategy XGB (ptr={ptr}) — persisted to logs")
                else:
                    print(f"🪴 Seeded {seed_qty} {seed_symbol} @ ${seed_price:.2f} for strategy XGB (ptr={ptr}) — synthetic only (not persisted)")
            else:
                print(f"⚠️ Cannot seed, no data for {seed_symbol}")
        except Exception as e:
            print(f"⚠️ Failed to parse seed argument '{seed}': {e}")

    while True:
        cycle += 1
        print(f"\n🔄 Trading Cycle #{cycle} - {now_utc().strftime('%H:%M:%S')}")

        try:
            # Collect signals from all active strategies
            all_intents = []
            model_signal_counts = {}

            # Build the per-cycle bars slice according to pointers (simulate time progression)
            current_bars: dict[str, pd.DataFrame] = {}
            for sym, df in bars.items():
                ptr = pointers.get(sym, len(df) - 1)
                # ensure ptr is within range
                ptr = max(1, min(ptr, len(df) - 1))
                current_bars[sym] = df.iloc[:ptr + 1].copy()

            for model in models:
                try:
                    # Get signals from strategy using the moving window
                    intents = model.on_bar(current_bars)
                    model_signal_counts[model.strategy_id] = len(intents) if intents is not None else 0

                    if intents:
                        for intent in intents:
                            if 'strategy_id' not in intent:
                                intent['strategy_id'] = model.strategy_id
                            all_intents.append(intent)
                    else:
                        print(f"ℹ️ {model.strategy_id} returned no signals this cycle")

                except Exception as e:
                    print(f"❌ Strategy {getattr(model,'strategy_id',str(model))} error: {e}")

            if not all_intents:
                # Print diagnostics per model and data lengths so user can see why no trades
                print("\n⚠️ No intents generated this cycle. Diagnostics:")
                for model in models:
                    lengths = {sym: len(df) for sym, df in bars.items()}
                    print(f"  - {model.strategy_id}: signals={model_signal_counts.get(model.strategy_id,0)}, data_lengths={lengths}")

            # Normalize intents: accept 'side' as 'action', 'qty' as position/qty, and 'ticker' as 'symbol'
            for intent in all_intents:
                # normalize action
                if 'action' not in intent and 'side' in intent:
                    intent['action'] = intent.pop('side')
                if 'action' in intent and isinstance(intent['action'], str):
                    intent['action'] = intent['action'].lower()
                # normalize symbol
                if 'symbol' not in intent and 'ticker' in intent:
                    intent['symbol'] = intent.pop('ticker')
                # normalize position_size from qty if present
                if 'position_size' not in intent:
                    if 'qty' in intent:
                        try:
                            intent['position_size'] = float(intent.get('qty', 0)) / 100.0
                        except Exception:
                            intent['position_size'] = 0.0
                    else:
                        intent['position_size'] = intent.get('position_size', 0.0)

            # Print normalized intents for visibility
            if all_intents:
                print('\n📡 Normalized intents:')
                for intent in all_intents:
                    print(f"  - {intent.get('strategy_id')} -> {intent.get('action','N/A')} {intent.get('symbol','N/A')} (pos_size: {intent.get('position_size',0):.3f})")

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
                        book.update_on_fill(intent.get('strategy_id', model.strategy_id), symbol, 'buy', qty, price)
                        log_trade(intent.get('strategy_id', model.strategy_id), symbol, 'buy', qty, price)
                        print(f"  ✅ BUY {qty} {symbol} @ ${price:.2f}")

                elif action == 'sell':
                    # Check if we have position
                    state = book.get_state(intent.get('strategy_id', model.strategy_id))
                    if symbol in state.positions and state.positions[symbol].qty > 0:
                        qty = min(state.positions[symbol].qty, 100)
                        price = float(bars[symbol].iloc[-1]['close'])
                        book.update_on_fill(intent.get('strategy_id', model.strategy_id), symbol, 'sell', qty, price)
                        log_trade(intent.get('strategy_id', model.strategy_id), symbol, 'sell', qty, price)
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

        # advance pointers for next cycle (accelerated replay)
        for sym in list(pointers.keys()):
            if accel:
                pointers[sym] = min(pointers[sym] + step, len(bars[sym]) - 1)
            else:
                # default behaviour: advance by 1
                pointers[sym] = min(pointers[sym] + 1, len(bars[sym]) - 1)

        # Exit if we've run the requested number of cycles
        if cycles > 0 and cycle >= cycles:
            print(f"✅ Completed {cycle} cycles; exiting offline run")
            break

        print(f"⏰ Waiting {sleep_sec} seconds until next cycle...")
        # Sleep only when not accelerated; when accel is enabled we still optionally sleep a tiny amount
        if not accel:
            time.sleep(sleep_sec)  # Shorter cycle for demo
        else:
            # tiny sleep to yield CPU and allow terminal responsiveness
            time.sleep(0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run offline with sample data")
    parser.add_argument("--symbols", type=str, default="SPY", help="Comma-separated symbols to load from storage/sample_bars")
    parser.add_argument("--cycles", type=int, default=5, help="Number of cycles to run (0 = infinite)")
    parser.add_argument("--sleep", type=int, default=30, help="Seconds to sleep between cycles")
    parser.add_argument("--seed", type=str, default=None, help="Optional seed position as SYMBOL:QTY (e.g. SPY:100)")
    parser.add_argument("--accel", action="store_true", help="Run accelerated replay (advance pointers without sleeping)")
    parser.add_argument("--step", type=int, default=1, help="Number of bars to advance per cycle when accelerated")
    parser.add_argument("--clear-logs", action="store_true", help="Archive existing storage/logs/*.csv before running to avoid persisted fills showing up")
    parser.add_argument("--persist-seed", action="store_true", help="If set, write synthetic seed fills to storage/logs (default: do not persist seeds)")
    args = parser.parse_args()

    if args.offline:
        symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
        main_offline(symbols=symbols, cycles=args.cycles, sleep_sec=args.sleep, seed=args.seed, accel=args.accel, step=args.step, clear_logs=args.clear_logs, persist_seed=args.persist_seed)
    else:
        print("Use --offline flag to run offline mode")
        print("Example: python engine/live_offline.py --offline --symbols SPY --cycles 3")
