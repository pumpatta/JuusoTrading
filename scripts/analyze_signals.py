#!/usr/bin/env python3
"""
Signal Analysis and Tuning Script
Analyzes raw signals from strategies to understand and optimize performance
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

def analyze_raw_signals():
    """Analyze what signals each strategy is generating"""
    print("=== Analyzing Raw Strategy Signals ===")
    
    # Get data
    all_bars = get_bars(['SPY'], None, None, prefer_samples=True)
    if not all_bars or 'SPY' not in all_bars:
        print("No data available")
        return
    
    bars = all_bars['SPY']
    print(f"Analyzing {len(bars)} bars")
    
    # Initialize strategies
    strategies = {
        'A_EMA': EmaTrend(),
        'B_XGB': XgbSignal(account='B'),
        'B_TCN': TcnSignal(account='B')
    }
    
    # Analyze signals for each strategy
    analysis_results = {}
    
    for name, strategy in strategies.items():
        print(f"\n--- Analyzing {name} ---")
        
        signals_log = []
        bars_window = 200  # Look at last 200 bars for signal analysis
        
        for i in range(max(100, len(bars) - bars_window), len(bars)):
            current_bars = {'SPY': bars.iloc[:i+1]}
            
            try:
                signals = strategy.on_bar(current_bars)
                
                if signals:
                    for signal in signals:
                        signals_log.append({
                            'bar_index': i,
                            'price': bars.iloc[i]['close'],
                            'signal': signal
                        })
                        print(f"  Bar {i}: {signal}")
                
            except Exception as e:
                print(f"  Bar {i}: Error - {e}")
        
        print(f"  Total signals: {len(signals_log)}")
        analysis_results[name] = signals_log
    
    return analysis_results

def tune_signal_thresholds():
    """Test different threshold values to find optimal settings"""
    print("\n=== Tuning Signal Thresholds ===")
    
    # Get data
    all_bars = get_bars(['SPY'], None, None, prefer_samples=True)
    bars = all_bars['SPY']
    
    # Test different thresholds for XGB
    print("\nTesting XGB thresholds:")
    xgb_strategy = XgbSignal(account='B')
    
    test_bars = {'SPY': bars}
    
    # Let's look at what the XGB strategy is actually doing
    try:
        # Test with recent data
        signals = xgb_strategy.on_bar(test_bars)
        print(f"XGB raw signals: {signals}")
        
        # Look at XGB internals if possible
        if hasattr(xgb_strategy, 'model') and xgb_strategy.model:
            print("XGB model is loaded")
            
            # Get features to see what the model sees
            features = xgb_strategy._features(bars)
            print(f"XGB features shape: {features.shape}")
            print(f"XGB features (last 5 rows):")
            print(features.tail())
            
            # Get predictions for last few bars
            if not features.empty:
                for i in range(max(0, len(features)-5), len(features)):
                    try:
                        X = features.iloc[i:i+1].fillna(0)
                        if hasattr(xgb_strategy.model, 'predict_proba'):
                            prob = xgb_strategy.model.predict_proba(X)[0, 1]
                            print(f"  Bar {i}: Probability = {prob:.3f}")
                        else:
                            pred = xgb_strategy.model.predict(X)[0]
                            print(f"  Bar {i}: Prediction = {pred}")
                    except Exception as e:
                        print(f"  Bar {i}: Prediction error - {e}")
        
    except Exception as e:
        print(f"XGB analysis error: {e}")
    
    # Test TCN
    print("\nTesting TCN:")
    tcn_strategy = TcnSignal(account='B')
    
    try:
        signals = tcn_strategy.on_bar(test_bars)
        print(f"TCN raw signals: {signals}")
        
        if hasattr(tcn_strategy, 'model') and tcn_strategy.model:
            print("TCN model is loaded")
            
            # Get features
            features = tcn_strategy._features(bars)
            print(f"TCN features shape: {features.shape}")
            print(f"TCN features (last 5 rows):")
            print(features.tail())
    
    except Exception as e:
        print(f"TCN analysis error: {e}")

def create_simple_backtest():
    """Create a simple backtest with minimal filtering to see baseline performance"""
    print("\n=== Simple Backtest (Minimal Filtering) ===")
    
    # Get data
    all_bars = get_bars(['SPY'], None, None, prefer_samples=True)
    bars = all_bars['SPY']
    
    # Use last 30% for testing
    split_point = int(len(bars) * 0.7)
    test_data = bars.iloc[split_point:].copy()
    
    print(f"Testing on {len(test_data)} bars")
    
    # Initialize strategies
    strategies = {
        'B_XGB': XgbSignal(account='B'),
        'B_TCN': TcnSignal(account='B')
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\nTesting {name}...")
        
        nav = 100000.0
        position = 0
        trades = []
        signals_generated = 0
        
        for i in range(len(test_data)):
            # Use full history for each bar
            current_bars = {'SPY': bars.iloc[:split_point + i + 1]}
            
            try:
                signals = strategy.on_bar(current_bars)
                
                if signals:
                    signals_generated += len(signals)
                    print(f"  Bar {i}: {len(signals)} signals")
                
                # Execute all signals without filtering
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
                                'bar': i
                            })
                            print(f"    BUY: {qty} @ ${price:.2f}")
                    
                    elif signal['side'] == 'sell' and position > 0:
                        qty = min(signal['qty'], position)
                        proceeds = price * qty
                        position -= qty
                        nav += proceeds
                        trades.append({
                            'side': 'sell',
                            'price': price,
                            'qty': qty,
                            'bar': i
                        })
                        print(f"    SELL: {qty} @ ${price:.2f}")
                        
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
                'bar': len(test_data)-1
            })
            print(f"    FINAL SELL: {position} @ ${final_price:.2f}")
        
        # Calculate results
        final_return = (nav - 100000) / 100000 * 100
        n_trades = len([t for t in trades if t['side'] == 'buy'])
        
        results[name] = {
            'final_nav': nav,
            'return_pct': final_return,
            'n_trades': n_trades,
            'signals_generated': signals_generated,
            'trades': trades
        }
        
        print(f"  Final NAV: ${nav:,.2f}")
        print(f"  Return: {final_return:.3f}%")
        print(f"  Trades: {n_trades}")
        print(f"  Signals generated: {signals_generated}")
    
    # Save results
    with open('storage/signal_analysis_report.json', 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        import json
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nSignal analysis results saved to storage/signal_analysis_report.json")
    return results

def suggest_optimizations(analysis_results, backtest_results):
    """Suggest specific optimizations based on analysis"""
    print("\n=== Optimization Suggestions ===")
    
    for strategy_name in analysis_results:
        signals = analysis_results[strategy_name]
        print(f"\n{strategy_name}:")
        
        if len(signals) == 0:
            print("  ❌ No signals generated - strategy may be too conservative")
            print("  💡 Suggestions:")
            print("     - Lower prediction thresholds")
            print("     - Check model loading and feature computation")
            print("     - Verify input data format")
        
        elif len(signals) < 5:
            print("  ⚠️  Very few signals generated - may be overly conservative")
            print("  💡 Suggestions:")
            print("     - Slightly lower thresholds")
            print("     - Add more sensitive indicators")
        
        elif len(signals) > 50:
            print("  ⚠️  Many signals generated - may be too noisy")
            print("  💡 Suggestions:")
            print("     - Increase thresholds")
            print("     - Add signal filtering")
            print("     - Require confirmation from multiple indicators")
        
        else:
            print("  ✅ Signal frequency seems reasonable")
    
    if backtest_results:
        print("\nPerformance Analysis:")
        for strategy_name, results in backtest_results.items():
            return_pct = results['return_pct']
            n_trades = results['n_trades']
            
            if return_pct < -0.1:
                print(f"  {strategy_name}: Poor performance ({return_pct:.2f}%)")
                print("    💡 Consider: Feature engineering, different time horizons")
            elif return_pct > 0.1:
                print(f"  {strategy_name}: Good performance ({return_pct:.2f}%)")
                print("    💡 Consider: Increase position sizes, add leverage")
            else:
                print(f"  {strategy_name}: Neutral performance ({return_pct:.2f}%)")
                print("    💡 Consider: Adjust signal sensitivity")

def main():
    print("JuusoTrader Signal Analysis and Tuning")
    print("=" * 50)
    
    # Step 1: Analyze raw signals
    analysis_results = analyze_raw_signals()
    
    # Step 2: Tune thresholds
    tune_signal_thresholds()
    
    # Step 3: Simple backtest
    backtest_results = create_simple_backtest()
    
    # Step 4: Suggestions
    suggest_optimizations(analysis_results, backtest_results)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("Check storage/signal_analysis_report.json for detailed results")

if __name__ == '__main__':
    main()
