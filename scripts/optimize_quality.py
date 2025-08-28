#!/usr/bin/env python3
"""
Signal Quality Optimization
Fixes the specific issues identified: too many low-quality signals
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

def create_optimized_xgb():
    """Create XGB strategy with better signal filtering"""
    
    class OptimizedXgbSignal(XgbSignal):
        def __init__(self, account=None):
            super().__init__(account)
            self.min_probability = 0.65  # Higher threshold
            self.cooldown_bars = 5       # Wait between signals
            self.last_signal_bar = -999
        
        def on_bar(self, bars: dict[str, pd.DataFrame]) -> list[dict]:
            if not self.fitted or 'SPY' not in bars:
                return []
            
            df = bars['SPY']
            if len(df) < 100:
                return []
            
            # Cooldown logic
            current_bar = len(df) - 1
            if current_bar - self.last_signal_bar < self.cooldown_bars:
                return []
            
            # Get features and prediction
            features = self._features(df)
            if len(features) == 0:
                return []
            
            X = features.iloc[-1:].fillna(0)
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(X)[0, 1]
            else:
                return []  # No probability, skip
            
            # Market condition filters
            close = df['close']
            returns = close.pct_change()
            current_vol = returns.rolling(20).std().iloc[-1]
            avg_vol = returns.rolling(60).std().iloc[-1]
            
            # Skip high volatility periods
            if current_vol > avg_vol * 1.8:
                return []
            
            # Price momentum filter
            momentum_5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100
            
            signals = []
            
            # Strong buy signal: high confidence + positive momentum
            if prob > self.min_probability and momentum_5 > 0.2:
                signals.append({
                    'symbol': 'SPY',
                    'side': 'buy',
                    'qty': 1,
                    'reason': f'XGB_buy_p{prob:.2f}_m{momentum_5:.1f}'
                })
                self.last_signal_bar = current_bar
            
            # Strong sell signal: low confidence + negative momentum  
            elif prob < (1 - self.min_probability) and momentum_5 < -0.2:
                signals.append({
                    'symbol': 'SPY',
                    'side': 'sell',
                    'qty': 1,
                    'reason': f'XGB_sell_p{prob:.2f}_m{momentum_5:.1f}'
                })
                self.last_signal_bar = current_bar
            
            return signals
    
    return OptimizedXgbSignal(account='B')

def create_optimized_tcn():
    """Create TCN strategy with proper signal generation"""
    
    class OptimizedTcnSignal(TcnSignal):
        def __init__(self, horizon=20, device=None, account=None):
            super().__init__(horizon, device, account)
            self.signal_threshold = 0.7   # Higher threshold for quality
            self.cooldown_bars = 8        # More spacing between signals
            self.last_signal_bar = -999
        
        def on_bar(self, bars: dict[str, pd.DataFrame]) -> list[dict]:
            if not self.fitted or 'SPY' not in bars:
                return []
            
            df = bars['SPY']
            if len(df) < 100:
                return []
            
            # Cooldown logic
            current_bar = len(df) - 1
            if current_bar - self.last_signal_bar < self.cooldown_bars:
                return []
            
            # Get features and prediction
            features = self._features(df)
            if len(features) == 0:
                return []
            
            import torch
            X = torch.tensor(features.values.T[None, ...], dtype=torch.float32, device=self.device)
            
            self.model.eval()
            with torch.no_grad():
                logit = self.model(X).item()
                prob = torch.sigmoid(torch.tensor(logit)).item()
            
            # Market condition checks
            close = df['close']
            returns = close.pct_change()
            
            # Trend strength
            sma_20 = close.rolling(20).mean()
            trend_strength = abs(close.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
            
            # Volatility check
            current_vol = returns.rolling(10).std().iloc[-1]
            avg_vol = returns.rolling(50).std().iloc[-1]
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            signals = []
            
            # Quality buy signal: high probability + decent trend + normal volatility
            if (prob > self.signal_threshold and 
                trend_strength > 0.005 and 
                vol_ratio < 1.5):
                
                signals.append({
                    'symbol': 'SPY',
                    'side': 'buy',
                    'qty': 1,
                    'reason': f'TCN_buy_p{prob:.2f}_t{trend_strength:.3f}'
                })
                self.last_signal_bar = current_bar
            
            # Quality sell signal: low probability + decent trend
            elif (prob < (1 - self.signal_threshold) and 
                  trend_strength > 0.005 and
                  vol_ratio < 1.5):
                
                signals.append({
                    'symbol': 'SPY',
                    'side': 'sell',
                    'qty': 1,
                    'reason': f'TCN_sell_p{prob:.2f}_t{trend_strength:.3f}'
                })
                self.last_signal_bar = current_bar
            
            return signals
    
    return OptimizedTcnSignal(account='B')

def run_optimized_backtest():
    """Run backtest with quality-optimized strategies"""
    print("=== Quality-Optimized Backtest ===")
    
    # Get data
    all_bars = get_bars(['SPY'], None, None, prefer_samples=True)
    if not all_bars or 'SPY' not in all_bars:
        print("No data available")
        return
    
    bars = all_bars['SPY']
    print(f"Loaded {len(bars)} bars")
    
    # Split data: 75% train, 25% test
    split_point = int(len(bars) * 0.75)
    test_data = bars.iloc[split_point:].copy()
    print(f"Testing on {len(test_data)} bars")
    
    # Initialize optimized strategies
    strategies = {
        'XGB_Optimized': create_optimized_xgb(),
        'TCN_Optimized': create_optimized_tcn(),
        'EMA_Baseline': EmaTrend()
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\n--- Testing {name} ---")
        
        nav = 100000.0
        position = 0
        trades = []
        signals_generated = 0
        
        for i in range(len(test_data)):
            # Use full history up to current bar
            current_bars = {'SPY': bars.iloc[:split_point + i + 1]}
            
            try:
                signals = strategy.on_bar(current_bars)
                
                if signals:
                    signals_generated += len(signals)
                    print(f"  Bar {i}: {signals}")
                
                # Execute signals
                for signal in signals:
                    price = test_data.iloc[i]['close']
                    
                    if signal['side'] == 'buy' and position == 0:
                        qty = signal['qty']
                        cost = price * qty
                        if cost <= nav:
                            position = qty
                            nav -= cost
                            trades.append({
                                'side': 'buy',
                                'price': price,
                                'qty': qty,
                                'bar': i,
                                'reason': signal.get('reason', 'buy')
                            })
                    
                    elif signal['side'] == 'sell' and position > 0:
                        qty = min(signal['qty'], position)
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
                        
            except Exception as e:
                print(f"  Bar {i}: Error - {e}")
        
        # Final liquidation
        if position > 0:
            final_price = test_data.iloc[-1]['close']
            nav += position * final_price
            trades.append({
                'side': 'sell',
                'price': final_price,
                'qty': position,
                'bar': len(test_data)-1,
                'reason': 'final_liquidation'
            })
        
        # Calculate metrics
        final_return = (nav - 100000) / 100000 * 100
        n_buy_trades = len([t for t in trades if t['side'] == 'buy'])
        
        # Win rate calculation
        wins = 0
        total_trades = 0
        for i, trade in enumerate(trades):
            if trade['side'] == 'sell' and i > 0:
                # Find matching buy
                buy_trade = None
                for j in range(i-1, -1, -1):
                    if trades[j]['side'] == 'buy':
                        buy_trade = trades[j]
                        break
                
                if buy_trade:
                    total_trades += 1
                    profit = trade['price'] - buy_trade['price']
                    if profit > 0:
                        wins += 1
        
        win_rate = (wins / max(1, total_trades)) * 100
        
        # Max drawdown calculation
        nav_history = []
        running_nav = 100000.0
        running_position = 0
        
        for i in range(len(test_data)):
            # Apply trades that occurred at this bar
            for trade in trades:
                if trade['bar'] == i:
                    if trade['side'] == 'buy':
                        running_position += trade['qty']
                        running_nav -= trade['price'] * trade['qty']
                    else:
                        running_position -= trade['qty']
                        running_nav += trade['price'] * trade['qty']
            
            # Mark to market
            current_value = running_nav + running_position * test_data.iloc[i]['close']
            nav_history.append(current_value)
        
        # Calculate max drawdown
        peak = nav_history[0]
        max_dd = 0
        for value in nav_history:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
        
        results[name] = {
            'final_nav': nav,
            'return_pct': final_return,
            'max_drawdown_pct': max_dd * 100,
            'n_trades': n_buy_trades,
            'win_rate': win_rate,
            'signals_generated': signals_generated,
            'trades': trades[-3:]  # Last 3 trades
        }
        
        print(f"  Final NAV: ${nav:,.2f}")
        print(f"  Return: {final_return:.3f}%")
        print(f"  Max Drawdown: {max_dd*100:.2f}%")
        print(f"  Trades: {n_buy_trades}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Signals Generated: {signals_generated}")
    
    # Save results
    report = {
        'timestamp': datetime.now().isoformat(),
        'optimization_type': 'signal_quality',
        'test_period': f"{len(test_data)} bars",
        'results': results
    }
    
    with open('storage/optimized_backtest_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nOptimized results saved to storage/optimized_backtest_report.json")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("QUALITY OPTIMIZATION RESULTS")
    print("=" * 60)
    
    for name, result in results.items():
        print(f"{name:15} | Return: {result['return_pct']:6.2f}% | "
              f"Drawdown: {result['max_drawdown_pct']:5.2f}% | "
              f"Trades: {result['n_trades']:3d} | "
              f"Win Rate: {result['win_rate']:5.1f}% | "
              f"Signals: {result['signals_generated']:3d}")
    
    return results

def main():
    print("JuusoTrader Signal Quality Optimization")
    print("=" * 50)
    print("Fixing issues identified in signal analysis:")
    print("- XGB: Too many noisy signals → Add quality filters")
    print("- TCN: Generates every bar → Add threshold logic")
    print("- Both: Add cooldown periods and market condition checks")
    print()
    
    results = run_optimized_backtest()
    
    if results:
        print("\n" + "=" * 50)
        print("OPTIMIZATION SUMMARY")
        print("=" * 50)
        
        # Find best strategy
        best_strategy = max(results.keys(), key=lambda k: results[k]['return_pct'])
        best_return = results[best_strategy]['return_pct']
        
        print(f"Best performing strategy: {best_strategy}")
        print(f"Best return: {best_return:.3f}%")
        
        # Check if optimization improved things
        if best_return > 0.05:  # At least 0.05% positive
            print("✅ Optimization successful - positive returns achieved!")
        elif best_return > -0.01:  # Better than -0.01%
            print("⚠️  Optimization partially successful - reduced losses")
        else:
            print("❌ Further optimization needed")
        
        print("\nNext steps:")
        print("1. Review optimized_backtest_report.json for detailed analysis")
        print("2. If results are good, deploy optimized strategies")
        print("3. If results need improvement, adjust thresholds further")

if __name__ == '__main__':
    main()
