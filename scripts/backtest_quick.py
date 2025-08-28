from pathlib import Path
import json
import math
import pandas as pd
import sys

# Ensure project root is on sys.path so `python scripts/backtest_quick.py` can import top-level packages
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from engine.datafeed import get_bars
from strategies.xgb_classifier import XgbSignal
from strategies.tcn_torch import TcnSignal
from strategies.ema_trend import EmaTrend


def max_drawdown(equity):
    peak = -math.inf
    dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = max(dd, (peak - v) / peak if peak>0 else 0)
    return dd


def run_backtest():
    print('Starting multi-strategy backtest simulation...')
    
    # Use sample data for consistent testing
    bars = get_bars(['SPY'], None, None, timeframe='1Min', prefer_samples=True)
    if 'SPY' not in bars:
        print('No SPY bars; aborting')
        return
    df = bars['SPY'].sort_values('ts').reset_index(drop=True)
    n = len(df)
    split = int(n * 0.75)
    train = df.iloc[:split]
    test = df.iloc[split:]
    
    print(f'Data: {n} bars, training on {len(train)}, testing on {len(test)}')

    # Test strategies with different accounts
    strategies_to_test = [
        ('A', 'EMA', EmaTrend()),
        ('B', 'XGB', XgbSignal(account='B')), 
        ('B', 'TCN', TcnSignal(account='B'))
    ]
    
    all_results = {}
    
    for account, strategy_name, strat in strategies_to_test:
        print(f'\n=== Testing Account {account}: {strategy_name} ===')
        result = run_single_strategy(strat, strategy_name, train, test, n)
        all_results[f'{account}_{strategy_name}'] = result
        
        print(f'{strategy_name} Results: {result["n_trades"]} trades, {result["returns_pct"]:.2f}% return, {result["max_drawdown_pct"]:.2f}% MDD')
    
    # Save combined results  
    final_report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_info': f'{n} total bars, {len(train)} train / {len(test)} test',
        'strategies': all_results
    }
    
    p = Path('storage')
    p.mkdir(exist_ok=True)
    (p / 'backtest_report.json').write_text(json.dumps(final_report, indent=2))
    
    print('\n=== BACKTEST SUMMARY ===')
    for strategy_key, result in all_results.items():
        print(f'{strategy_key:8}: {result["n_trades"]:3d} trades, {result["returns_pct"]:7.2f}% return, {result["max_drawdown_pct"]:6.2f}% MDD')
    print(f'\nResults saved to: storage/backtest_report.json')


def run_single_strategy(strat, strategy_name, train, test, n):
    """Run backtest for a single strategy"""
    # Train if needed (EMA doesn't need training)
    if hasattr(strat, 'fit') and strategy_name != 'EMA':
        print(f'Training {strategy_name}...')
        strat.fit(train)
        if not strat.fitted:
            print(f'{strategy_name} training failed, using defaults')
        else:
            print(f'{strategy_name} training completed')
    else:
        print(f'{strategy_name} ready (rule-based strategy)')
        
    # Run the backtest simulation
    cash = 100_000.0
    position = 0  # number of shares
    entry_price = 0.0
    equity_curve = []
    trades = []

    window_df = train.copy()
    for i, row in test.iterrows():
        window_df = pd.concat([window_df, row.to_frame().T], ignore_index=True)
        bars_dict = {'SPY': window_df}
        signals = strat.on_bar(bars_dict)
        price = float(row['close'])
        # process signals (simple: act on first signal only)
        if signals:
            sig = signals[0]
            if sig['side'] == 'buy' and position == 0:
                qty = int(sig.get('qty', 1))
                cost = qty * price
                if cost <= cash:
                    cash -= cost
                    position += qty
                    entry_price = price
                    trades.append({'side':'buy','price':price,'qty':qty,'ts':row['ts']})
            elif sig['side'] == 'sell' and position > 0:
                qty = position
                proceeds = qty * price
                cash += proceeds
                pnl = (price - entry_price) * qty
                trades.append({'side':'sell','price':price,'qty':qty,'ts':row['ts'],'pnl':pnl})
                position = 0
                entry_price = 0.0

        nav = cash + position * price
        equity_curve.append(nav)

    # Close any remaining position at last price
    if position > 0:
        last_price = float(test.iloc[-1]['close'])
        cash += position * last_price
        pnl = (last_price - entry_price) * position
        trades.append({'side':'sell','price':last_price,'qty':position,'ts':test.iloc[-1]['ts'],'pnl':pnl})
        position = 0

    final_nav = cash
    returns = (final_nav - 100_000.0) / 100_000.0
    wins = [t['pnl'] for t in trades if 'pnl' in t and t['pnl']>0]
    losses = [t['pnl'] for t in trades if 'pnl' in t and t['pnl']<=0]
    win_rate = len(wins) / (len(wins)+len(losses)) if (wins or losses) else None
    mdd = max_drawdown(equity_curve) if equity_curve else 0.0

    # Ensure all objects in `trades` are JSON-serializable (convert Timestamps)
    serial_trades = []
    for t in trades[:50]:  # Limit to first 50 trades
        tt = t.copy()
        if 'ts' in tt:
            try:
                # pandas Timestamp -> ISO format string
                tt['ts'] = tt['ts'].isoformat()
            except Exception:
                tt['ts'] = str(tt['ts'])
        serial_trades.append(tt)

    return {
        'n_rows': n,
        'n_trades': len([t for t in trades if t['side']=='sell']),
        'final_nav': final_nav,
        'returns_pct': returns * 100,
        'win_rate': win_rate,
        'max_drawdown_pct': mdd * 100,
        'trades': serial_trades
    }


if __name__ == '__main__':
    run_backtest()
