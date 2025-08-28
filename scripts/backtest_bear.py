#!/usr/bin/env python3
"""
Market Regime Testing Script
Tests strategies across different market conditions: bull, bear, sideways, and volatile markets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from engine.datafeed import get_bars
from strategies.xgb_classifier import XgbSignal
from strategies.tcn_torch import TcnSignal
from strategies.ema_trend import EmaTrend

def generate_synthetic_market_data(scenario='bull', n_bars=1000, start_price=450.0):
    """Generate synthetic market data for different scenarios"""
    
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2024-01-01 09:30:00', periods=n_bars, freq='1min')
    
    # Base parameters
    price = start_price
    prices = [price]
    volumes = []
    
    if scenario == 'bull':
        # Bull market: upward trend with moderate volatility
        trend = 0.0002  # 0.02% per bar average gain
        volatility = 0.008  # 0.8% volatility
        vol_base = 1000000
        
    elif scenario == 'bear':
        # Bear market: downward trend with higher volatility
        trend = -0.0003  # -0.03% per bar average loss
        volatility = 0.015  # 1.5% volatility
        vol_base = 1500000  # Higher volume in bear markets
        
    elif scenario == 'sideways':
        # Sideways market: mean reverting with low trend
        trend = 0.0  # No trend
        volatility = 0.006  # Lower volatility
        vol_base = 800000
        
    elif scenario == 'volatile':
        # High volatility market: alternating direction
        trend = 0.0001
        volatility = 0.025  # Very high volatility
        vol_base = 2000000
        
    elif scenario == 'crash':
        # Market crash scenario
        trend = -0.001  # Sharp decline
        volatility = 0.03  # Extreme volatility
        vol_base = 3000000
    
    for i in range(1, n_bars):
        # Add regime changes for realism
        if scenario == 'volatile' and i % 100 == 0:
            trend *= -1  # Flip direction periodically
        
        if scenario == 'crash' and i > 200:
            # Recovery after crash
            trend = max(trend + 0.0001, 0.0002)
            volatility = max(volatility - 0.0005, 0.01)
        
        # Generate price movement
        return_shock = np.random.normal(trend, volatility)
        
        # Add mean reversion for sideways markets
        if scenario == 'sideways':
            deviation = (price - start_price) / start_price
            return_shock -= deviation * 0.1  # Mean reversion factor
        
        price *= (1 + return_shock)
        prices.append(price)
        
        # Generate volume (correlated with volatility)
        vol_shock = np.random.normal(1, 0.3)
        volume = int(vol_base * abs(return_shock) * 10 + vol_base * vol_shock)
        volume = max(100000, volume)  # Minimum volume
        volumes.append(volume)
    
    # Add initial volume
    volumes.insert(0, vol_base)
    
    # Create OHLC data
    df = pd.DataFrame({
        'ts': dates,
        'close': prices
    })
    
    # Generate OHLC from close prices
    df['open'] = df['close'].shift(1).fillna(start_price)
    
    # High/Low based on intrabar volatility
    intrabar_vol = volatility * 0.5
    df['high'] = df['close'] + np.abs(np.random.normal(0, intrabar_vol, len(df))) * df['close']
    df['low'] = df['close'] - np.abs(np.random.normal(0, intrabar_vol, len(df))) * df['close']
    
    # Ensure OHLC consistency
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    df['volume'] = volumes
    
    return df

def analyze_market_regime(df):
    """Analyze the market regime characteristics"""
    returns = df['close'].pct_change().dropna()
    
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    volatility = returns.std() * np.sqrt(252 * 390) * 100  # Annualized volatility
    max_dd = ((df['close'] / df['close'].expanding().max()) - 1).min() * 100
    
    # Trend strength
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    trend_periods = (sma_20 > sma_50).sum() / len(sma_20) * 100
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'max_drawdown': max_dd,
        'bullish_periods': trend_periods,
        'avg_daily_return': returns.mean() * 390 * 100,  # Convert to daily %
        'return_skew': returns.skew(),
        'worst_day': returns.min() * 100,
        'best_day': returns.max() * 100
    }

def test_strategy_performance(strategy, market_data, regime_name):
    """Test strategy performance on specific market data"""
    
    print(f"\nTesting {strategy.__class__.__name__} on {regime_name} market...")
    
    # Split data for training/testing
    n_total = len(market_data)
    train_size = int(n_total * 0.7)
    
    train_data = market_data.iloc[:train_size].copy()
    test_data = market_data.iloc[train_size:].copy()
    
    # Train strategy if it has fit method
    if hasattr(strategy, 'fit'):
        try:
            print(f"  Training on {len(train_data)} bars...")
            strategy.fit(train_data)
        except Exception as e:
            print(f"  Training failed: {e}")
            return None
    
    # Test strategy
    nav = 100000.0
    position = 0
    trades = []
    nav_history = [nav]
    
    print(f"  Testing on {len(test_data)} bars...")
    
    for i in range(len(test_data)):
        current_bars = {'SPY': test_data.iloc[:i+1]}
        
        if len(current_bars['SPY']) < 20:  # Need some history
            nav_history.append(nav)
            continue
        
        try:
            signals = strategy.on_bar(current_bars)
        except Exception as e:
            signals = []
        
        # Execute signals
        for signal in signals:
            price = test_data.iloc[i]['close']
            
            if signal['side'] == 'buy' and position <= 0:
                qty = signal.get('qty', 1)
                cost = price * qty
                if cost <= nav:
                    position += qty
                    nav -= cost
                    trades.append({
                        'side': 'buy',
                        'price': price,
                        'qty': qty,
                        'bar': i,
                        'reason': signal.get('reason', 'buy')
                    })
            
            elif signal['side'] == 'sell' and position > 0:
                qty = min(signal.get('qty', 1), position)
                proceeds = price * qty
                position -= qty
                nav += proceeds
                trades.append({
                    'side': 'sell',
                    'price': price,
                    'qty': qty,
                    'bar': i,
                    'reason': signal.get('reason', 'sell')
                })
        
        # Mark to market
        current_value = nav + (position * test_data.iloc[i]['close'] if position > 0 else 0)
        nav_history.append(current_value)
    
    # Final liquidation
    if position > 0:
        final_price = test_data.iloc[-1]['close']
        nav += position * final_price
        trades.append({
            'side': 'sell',
            'price': final_price,
            'qty': position,
            'bar': len(test_data) - 1,
            'reason': 'final_liquidation'
        })
    
    # Calculate performance metrics
    final_return = (nav - 100000) / 100000 * 100
    
    nav_series = pd.Series(nav_history)
    peak = nav_series.expanding().max()
    drawdown = (nav_series - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    n_trades = len([t for t in trades if t['side'] == 'buy'])
    
    # Calculate win rate
    wins = 0
    buy_trades = [t for t in trades if t['side'] == 'buy']
    sell_trades = [t for t in trades if t['side'] == 'sell']
    
    for buy in buy_trades:
        # Find corresponding sell
        corresponding_sells = [s for s in sell_trades if s['bar'] > buy['bar']]
        if corresponding_sells:
            sell = min(corresponding_sells, key=lambda x: x['bar'])
            if sell['price'] > buy['price']:
                wins += 1
    
    win_rate = (wins / max(1, n_trades)) * 100
    
    # Sharpe ratio approximation
    nav_returns = nav_series.pct_change().dropna()
    sharpe = nav_returns.mean() / (nav_returns.std() + 1e-9) * np.sqrt(252 * 390) if len(nav_returns) > 1 else 0
    
    result = {
        'final_nav': nav,
        'return_pct': final_return,
        'max_drawdown_pct': max_drawdown,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'trades_per_day': n_trades / (len(test_data) / 390),
        'avg_trade_return': final_return / max(1, n_trades),
        'recent_trades': trades[-3:] if trades else []
    }
    
    print(f"  Results: {final_return:.2f}% return, {n_trades} trades, {win_rate:.1f}% win rate")
    
    return result

def main():
    print("Market Regime Testing")
    print("=" * 50)
    
    # Define market scenarios
    scenarios = {
        'bull': 'Bull Market (+15% trend, moderate vol)',
        'bear': 'Bear Market (-20% trend, high vol)', 
        'sideways': 'Sideways Market (0% trend, low vol)',
        'volatile': 'Volatile Market (high vol, regime changes)',
        'crash': 'Market Crash (-50% then recovery)'
    }
    
    # Initialize strategies
    strategies = {
        'EMA': EmaTrend(),
        'XGB': XgbSignal(account='B'),
        'TCN': TcnSignal(account='B')
    }
    
    # Results storage
    all_results = {}
    
    for scenario_name, description in scenarios.items():
        print(f"\n{'='*20} {description} {'='*20}")
        
        # Generate market data
        market_data = generate_synthetic_market_data(scenario_name, n_bars=1500)
        
        # Analyze market characteristics
        regime_stats = analyze_market_regime(market_data)
        print(f"Market stats: {regime_stats['total_return']:.1f}% return, "
              f"{regime_stats['volatility']:.1f}% vol, "
              f"{regime_stats['max_drawdown']:.1f}% max DD")
        
        # Test each strategy
        scenario_results = {}
        for strategy_name, strategy in strategies.items():
            try:
                result = test_strategy_performance(strategy, market_data, scenario_name)
                if result:
                    scenario_results[strategy_name] = result
            except Exception as e:
                print(f"  {strategy_name} failed: {e}")
                scenario_results[strategy_name] = None
        
        all_results[scenario_name] = {
            'description': description,
            'market_stats': regime_stats,
            'strategy_results': scenario_results
        }
    
    # Save comprehensive results
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'market_regime_analysis',
        'scenarios_tested': len(scenarios),
        'strategies_tested': list(strategies.keys()),
        'results': all_results
    }
    
    with open('storage/market_regime_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("MARKET REGIME PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    for scenario, data in all_results.items():
        print(f"\n{scenario.upper()} MARKET ({data['market_stats']['total_return']:.1f}% return):")
        print("-" * 50)
        
        for strategy, result in data['strategy_results'].items():
            if result:
                print(f"{strategy:6} | Return: {result['return_pct']:6.2f}% | "
                      f"Trades: {result['n_trades']:3d} | "
                      f"Win Rate: {result['win_rate']:5.1f}% | "
                      f"Sharpe: {result['sharpe_ratio']:5.2f}")
            else:
                print(f"{strategy:6} | FAILED")
    
    # Best/worst performance analysis
    print(f"\n{'='*70}")
    print("STRATEGY ROBUSTNESS ANALYSIS")
    print(f"{'='*70}")
    
    for strategy in strategies.keys():
        strategy_performance = []
        for scenario, data in all_results.items():
            if data['strategy_results'][strategy]:
                strategy_performance.append({
                    'scenario': scenario,
                    'return': data['strategy_results'][strategy]['return_pct'],
                    'sharpe': data['strategy_results'][strategy]['sharpe_ratio'],
                    'trades': data['strategy_results'][strategy]['n_trades']
                })
        
        if strategy_performance:
            avg_return = np.mean([p['return'] for p in strategy_performance])
            avg_sharpe = np.mean([p['sharpe'] for p in strategy_performance])
            total_trades = sum([p['trades'] for p in strategy_performance])
            
            best_scenario = max(strategy_performance, key=lambda x: x['return'])
            worst_scenario = min(strategy_performance, key=lambda x: x['return'])
            
            print(f"\n{strategy} STRATEGY:")
            print(f"  Average Return: {avg_return:.2f}%")
            print(f"  Average Sharpe: {avg_sharpe:.2f}")
            print(f"  Total Trades: {total_trades}")
            print(f"  Best in: {best_scenario['scenario']} ({best_scenario['return']:.2f}%)")
            print(f"  Worst in: {worst_scenario['scenario']} ({worst_scenario['return']:.2f}%)")
    
    print(f"\nReport saved to storage/market_regime_report.json")

if __name__ == '__main__':
    main()
