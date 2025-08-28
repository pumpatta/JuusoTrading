#!/usr/bin/env python3
"""
Performance Optimization Script
Focuses on improving signal quality and reducing noise in existing models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from engine.datafeed import get_bars

def analyze_current_performance():
    """Analyze current backtest results to identify optimization opportunities"""
    print("=== Analyzing Current Performance ===")
    
    # Load recent backtest results
    report_path = Path('storage/backtest_report.json')
    if not report_path.exists():
        print("No backtest report found. Please run backtest first.")
        return None
    
    with open(report_path) as f:
        report = json.load(f)
    
    print("\nCurrent Performance Summary:")
    for account, data in report['strategies'].items():
        trades = data.get('n_trades', 0)
        returns = (data.get('returns_pct', 0) or 0) * 100  # Convert to percentage
        win_rate = data.get('win_rate', 0) or 0
        print(f"{account}: {trades} trades, {returns:.3f}% return, {win_rate:.1f}% win rate")
    
    return report

def optimize_signal_filters():
    """Create optimized strategy configurations with better signal filtering"""
    print("\n=== Creating Optimized Signal Filters ===")
    
    # Enhanced market regime detection
    def market_regime_filter(bars: pd.DataFrame) -> dict:
        """Detect market conditions for better signal filtering"""
        if len(bars) < 50:
            return {'trend': 'neutral', 'volatility': 'normal', 'momentum': 'weak'}
        
        close = bars['close']
        
        # Trend detection
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        current_price = close.iloc[-1]
        
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend = 'bullish'
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Volatility regime
        returns = close.pct_change()
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        current_vol = vol_20.iloc[-1]
        avg_vol = vol_60.iloc[-1]
        
        if current_vol > avg_vol * 1.5:
            volatility = 'high'
        elif current_vol < avg_vol * 0.7:
            volatility = 'low'
        else:
            volatility = 'normal'
        
        # Momentum strength
        roc_10 = (close.iloc[-1] / close.iloc[-11] - 1) * 100
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
            'vol_ratio': current_vol / avg_vol if avg_vol > 0 else 1.0,
            'roc_10': roc_10
        }
    
    # Create configuration for signal filtering
    config = {
        'market_regime_filter': market_regime_filter,
        'signal_thresholds': {
            'xgb_min_prob': 0.6,      # Minimum probability for XGB signals
            'tcn_min_prob': 0.65,     # Minimum probability for TCN signals
            'min_momentum': 0.5,      # Minimum momentum strength
            'max_volatility': 2.0,    # Maximum volatility ratio
        },
        'risk_filters': {
            'max_drawdown_stop': 0.02,  # Stop trading if drawdown > 2%
            'consecutive_loss_limit': 3, # Stop after 3 consecutive losses
            'position_size_vol_adj': True, # Adjust position size based on volatility
        }
    }
    
    # Save configuration
    with open('config/optimization.yml', 'w') as f:
        # Convert to YAML-like format
        f.write("# Optimization Configuration\n")
        f.write("signal_thresholds:\n")
        for key, value in config['signal_thresholds'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\nrisk_filters:\n")
        for key, value in config['risk_filters'].items():
            f.write(f"  {key}: {value}\n")
    
    print("Created optimization configuration in config/optimization.yml")
    return config

def create_enhanced_backtest():
    """Create an enhanced backtest with better signal filtering"""
    print("\n=== Creating Enhanced Backtest Script ===")
    
    enhanced_script = '''#!/usr/bin/env python3
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
        print(f"\\nTesting {name}...")
        
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
    
    print(f"\\nEnhanced backtest results saved to storage/enhanced_backtest_report.json")
    
    # Summary
    print("\\n" + "=" * 50)
    print("ENHANCED BACKTEST SUMMARY")
    print("=" * 50)
    
    for name, result in results.items():
        print(f"{name:6} | Return: {result['return_pct']:6.2f}% | "
              f"Drawdown: {result['max_drawdown_pct']:5.2f}% | "
              f"Trades: {result['n_trades']:3d} | "
              f"Win Rate: {result['win_rate']:5.1f}%")

if __name__ == '__main__':
    main()
'''
    
    with open('scripts/enhanced_backtest.py', 'w') as f:
        f.write(enhanced_script)
    
    print("Created enhanced backtest script: scripts/enhanced_backtest.py")

def main():
    print("JuusoTrader Performance Optimization")
    print("=" * 50)
    
    # Step 1: Analyze current performance
    current_report = analyze_current_performance()
    
    # Step 2: Create optimization configurations
    config = optimize_signal_filters()
    
    # Step 3: Create enhanced backtest
    create_enhanced_backtest()
    
    # Step 4: Run enhanced backtest
    print("\n=== Running Enhanced Backtest ===")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'scripts/enhanced_backtest.py'
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("Enhanced backtest completed successfully!")
            print(result.stdout)
        else:
            print("Enhanced backtest failed:")
            print(result.stderr)
    except Exception as e:
        print(f"Failed to run enhanced backtest: {e}")
    
    print("\nOptimization complete!")
    print("\nNext steps:")
    print("1. Review enhanced_backtest_report.json for improved performance")
    print("2. Adjust signal_thresholds in config/optimization.yml if needed")
    print("3. Run scripts/enhanced_backtest.py to test with different parameters")

if __name__ == '__main__':
    main()
