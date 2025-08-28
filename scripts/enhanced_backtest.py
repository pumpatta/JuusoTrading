#!/usr/bin/env python3
"""
Enhanced Backtest with Optimized Signal Filtering
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import json

from engine.datafeed import get_bars
from strategies.xgb_classifier import XgbSignal
from strategies.tcn_torch import TcnSignal
from strategies.ema_trend import EmaTrend

def enhanced_signal_filter(strategy, bars, signals, regime_info):
    """Apply enhanced filtering to strategy signals"""
    if not signals:
        return []
    
    filtered_signals = []
    
    for signal in signals:
        # Basic signal validation
        if signal['side'] not in ['buy', 'sell']:
            continue
        
        # Market regime filters
        if regime_info['volatility'] == 'high' and regime_info['vol_ratio'] > 2.0:
            # Skip signals in extremely high volatility
            continue
        
        if regime_info['momentum'] == 'weak' and abs(regime_info['roc_10']) < 0.3:
            # Skip signals in very low momentum markets
            continue
        
        # Strategy-specific filters
        if isinstance(strategy, XgbSignal):
            # For XGB, be more selective in neutral trends
            if regime_info['trend'] == 'neutral' and abs(regime_info['roc_10']) < 1.0:
                continue
        
        elif isinstance(strategy, TcnSignal):
            # For TCN, require stronger momentum
            if regime_info['momentum'] == 'weak':
                continue
        
        # Position sizing based on volatility
        base_qty = signal['qty']
        if regime_info['vol_ratio'] > 1.5:
            signal['qty'] = max(1, int(base_qty * 0.7))  # Reduce size in high vol
        elif regime_info['vol_ratio'] < 0.8:
            signal['qty'] = int(base_qty * 1.2)  # Increase size in low vol
        
        filtered_signals.append(signal)
    
    return filtered_signals

def market_regime_analysis(bars):
    """Analyze market regime for signal filtering"""
    if len(bars) < 50:
        return {'trend': 'neutral', 'volatility': 'normal', 'momentum': 'weak', 'vol_ratio': 1.0, 'roc_10': 0.0}
    
    close = bars['close']
    
    # Trend
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    current_price = close.iloc[-1]
    
    if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
        trend = 'bullish'
    elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
        trend = 'bearish'
    else:
        trend = 'neutral'
    
    # Volatility
    returns = close.pct_change()
    vol_20 = returns.rolling(20).std()
    vol_60 = returns.rolling(60).std()
    vol_ratio = vol_20.iloc[-1] / vol_60.iloc[-1] if vol_60.iloc[-1] > 0 else 1.0
    
    if vol_ratio > 1.5:
        volatility = 'high'
    elif vol_ratio < 0.7:
        volatility = 'low'
    else:
        volatility = 'normal'
    
    # Momentum
    roc_10 = (close.iloc[-1] / close.iloc[-11] - 1) * 100 if len(close) > 10 else 0
    if abs(roc_10) > 2:
        momentum = 'strong'
    elif abs(roc_10) > 0.5:
        momentum = 'moderate'
    else:
        momentum = 'weak'
    
    return {
        'trend': trend,
        'volatility': volatility,
        'momentum': momentum,
        'vol_ratio': vol_ratio,
        'roc_10': roc_10
    }

def main():
    print("Enhanced Backtest with Optimization")
    print("=" * 50)
    
    # Get data
    all_bars = get_bars(['SPY'], None, None, prefer_samples=True)
    if not all_bars or 'SPY' not in all_bars:
        print("No data available")
        return
    
    bars = all_bars['SPY']
    print(f"Loaded {len(bars)} bars")
    
    # Split data
    n_total = len(bars)
    split_point = int(n_total * 0.75)
    
    train_data = bars.iloc[:split_point].copy()
    test_data = bars.iloc[split_point:].copy()
    
    print(f"Training: {len(train_data)} bars, Testing: {len(test_data)} bars")
    
    # Initialize strategies
    strategies = {
        'A_EMA': EmaTrend(),
        'B_XGB': XgbSignal(account='B'),
        'B_TCN': TcnSignal(account='B')
    }
    
    # Results tracking
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\nTesting {name}...")
        
        # Skip fit for EMA (doesn't need training)
        if hasattr(strategy, 'fit') and name != 'A_EMA':
            try:
                print(f"  Loading existing model...")
                # Models should already be trained
            except Exception as e:
                print(f"  Warning: {e}")
        
        # Simulation
        nav = 100000.0
        position = 0
        trades = []
        consecutive_losses = 0
        max_dd = 0
        peak_nav = nav
        
        for i in range(len(test_data)):
            current_bars = {'SPY': test_data.iloc[:i+50]}  # More history for regime analysis
            
            if len(current_bars['SPY']) < 50:
                continue
            
            # Market regime analysis
            regime_info = market_regime_analysis(current_bars['SPY'])
            
            # Get signals
            try:
                signals = strategy.on_bar(current_bars)
                
                # Apply enhanced filtering
                if signals:
                    signals = enhanced_signal_filter(strategy, current_bars['SPY'], signals, regime_info)
                
            except Exception as e:
                print(f"  Error getting signals: {e}")
                signals = []
            
            # Risk management: stop if max drawdown exceeded
            current_value = nav + (position * test_data.iloc[i]['close'] if position > 0 else 0)
            peak_nav = max(peak_nav, current_value)
            drawdown = (peak_nav - current_value) / peak_nav
            max_dd = max(max_dd, drawdown)
            
            if drawdown > 0.05:  # 5% max drawdown limit
                print(f"  {name}: Stopping due to max drawdown ({drawdown:.2%})")
                break
            
            if consecutive_losses >= 4:  # Stop after 4 consecutive losses
                print(f"  {name}: Stopping due to consecutive losses")
                break
            
            # Execute signals
            for signal in signals:
                price = test_data.iloc[i]['close']
                
                if signal['side'] == 'buy' and position <= 0:
                    qty = signal['qty']
                    cost = price * qty
                    if cost <= nav:
                        position += qty
                        nav -= cost
                        trades.append({
                            'side': 'buy',
                            'price': price,
                            'qty': qty,
                            'nav_before': nav + cost,
                            'reason': signal.get('reason', 'buy'),
                            'regime': regime_info['trend']
                        })
                
                elif signal['side'] == 'sell' and position > 0:
                    qty = min(signal['qty'], position)
                    proceeds = price * qty
                    position -= qty
                    nav += proceeds
                    
                    # Track consecutive losses
                    if len(trades) > 0:
                        last_buy = None
                        for t in reversed(trades):
                            if t['side'] == 'buy':
                                last_buy = t
                                break
                        
                        if last_buy and proceeds < last_buy['price'] * qty:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0
                    
                    trades.append({
                        'side': 'sell',
                        'price': price,
                        'qty': qty,
                        'nav_after': nav,
                        'reason': signal.get('reason', 'sell'),
                        'regime': regime_info['trend']
                    })
        
        # Final liquidation
        if position > 0:
            final_price = test_data.iloc[-1]['close']
            nav += position * final_price
            trades.append({
                'side': 'sell',
                'price': final_price,
                'qty': position,
                'nav_after': nav,
                'reason': 'final_liquidation',
                'regime': 'end'
            })
        
        # Calculate results
        final_return = (nav - 100000) / 100000 * 100
        n_trades = len([t for t in trades if t['side'] == 'buy'])
        
        # Win rate calculation
        wins = 0
        for i, trade in enumerate(trades):
            if trade['side'] == 'sell' and i > 0:
                buy_price = None
                for j in range(i-1, -1, -1):
                    if trades[j]['side'] == 'buy':
                        buy_price = trades[j]['price']
                        break
                if buy_price and trade['price'] > buy_price:
                    wins += 1
        
        win_rate = (wins / max(1, n_trades)) * 100
        
        results[name] = {
            'final_nav': nav,
            'return_pct': final_return,
            'max_drawdown_pct': max_dd * 100,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'consecutive_losses': consecutive_losses,
            'trades': trades[-5:]  # Last 5 trades
        }
        
        print(f"  Final NAV: ${nav:,.2f}")
        print(f"  Return: {final_return:.3f}%")
        print(f"  Max Drawdown: {max_dd*100:.2f}%")
        print(f"  Trades: {n_trades}")
        print(f"  Win Rate: {win_rate:.1f}%")
    
    # Save enhanced results
    report = {
        'timestamp': datetime.now().isoformat(),
        'optimization': True,
        'enhanced_filtering': True,
        'test_period': f"{len(test_data)} bars",
        'results': results
    }
    
    with open('storage/enhanced_backtest_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nEnhanced backtest results saved to storage/enhanced_backtest_report.json")
    
    # Summary
    print("\n" + "=" * 50)
    print("ENHANCED BACKTEST SUMMARY")
    print("=" * 50)
    
    for name, result in results.items():
        print(f"{name:6} | Return: {result['return_pct']:6.2f}% | "
              f"Drawdown: {result['max_drawdown_pct']:5.2f}% | "
              f"Trades: {result['n_trades']:3d} | "
              f"Win Rate: {result['win_rate']:5.1f}%")

if __name__ == '__main__':
    main()
