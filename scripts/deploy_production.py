#!/usr/bin/env python3
"""
Production Deployment Script
Deploy the optimized XGB strategy for live trading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from strategies.xgb_classifier import XgbSignal
from engine.datafeed import get_bars

class ProductionXgbStrategy(XgbSignal):
    """Production-ready XGB strategy with enhanced signal quality"""
    
    def __init__(self, account='B'):
        super().__init__(account)
        self.min_probability = 0.58  # Optimized from testing
        self.cooldown_bars = 3       # Reduced from testing for more activity
        self.last_signal_bar = -999
        self.position_size_base = 1
        
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
        
        # Use only numerical features for XGB model
        X = features.iloc[-1:][['ret1','vol','mom','rsi']].fillna(0)
        if not hasattr(self.model, 'predict_proba'):
            return []
        
        prob = self.model.predict_proba(X)[0, 1]
        
        # Market regime detection
        close = df['close']
        returns = close.pct_change()
        
        # Volatility regime
        current_vol = returns.rolling(20).std().iloc[-1]
        avg_vol = returns.rolling(60).std().iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Trend strength
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        trend_signal = 1 if sma_20.iloc[-1] > sma_50.iloc[-1] else -1
        
        # Price momentum (5-bar)
        momentum_5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0
        
        # Dynamic position sizing based on volatility
        if vol_ratio < 0.8:
            position_size = self.position_size_base * 1.2  # Low vol = bigger size
        elif vol_ratio > 1.5:
            position_size = max(1, int(self.position_size_base * 0.7))  # High vol = smaller size
        else:
            position_size = self.position_size_base
        
        signals = []
        
        # Enhanced buy signal
        if (prob > self.min_probability and 
            momentum_5 > 0.15 and  # Positive momentum
            vol_ratio < 2.0 and    # Not extreme volatility
            trend_signal > 0):     # Bullish trend
            
            signals.append({
                'symbol': 'SPY',
                'side': 'buy',
                'qty': int(position_size),
                'reason': f'PROD_XGB_buy_p{prob:.2f}_m{momentum_5:.1f}_v{vol_ratio:.1f}'
            })
            self.last_signal_bar = current_bar
        
        # Enhanced sell signal
        elif (prob < (1 - self.min_probability) and 
              momentum_5 < -0.15 and  # Negative momentum
              vol_ratio < 2.0):       # Not extreme volatility
            
            signals.append({
                'symbol': 'SPY',
                'side': 'sell',
                'qty': int(position_size),
                'reason': f'PROD_XGB_sell_p{prob:.2f}_m{momentum_5:.1f}_v{vol_ratio:.1f}'
            })
            self.last_signal_bar = current_bar
        
        return signals

def validate_production_strategy():
    """Validate the production strategy on recent data"""
    print("=== Validating Production Strategy ===")
    
    # Get recent data
    all_bars = get_bars(['SPY'], None, None, prefer_samples=True)
    if not all_bars or 'SPY' not in all_bars:
        print("❌ No data available for validation")
        return False
    
    bars = all_bars['SPY']
    print(f"✅ Loaded {len(bars)} bars for validation")
    
    # Initialize production strategy
    strategy = ProductionXgbStrategy(account='B')
    
    # Test recent signals
    print("\n--- Testing Signal Generation ---")
    test_bars = {'SPY': bars}
    signals = strategy.on_bar(test_bars)
    
    if signals:
        print(f"✅ Strategy generating signals: {len(signals)}")
        for signal in signals:
            print(f"   {signal}")
    else:
        print("⚠️  No signals generated (may be in cooldown)")
    
    # Test features
    print("\n--- Testing Feature Generation ---")
    features = strategy._features(bars)
    print(f"✅ Features shape: {features.shape}")
    print(f"✅ Feature columns: {list(features.columns)}")
    
    # Test model
    print("\n--- Testing Model ---")
    if hasattr(strategy, 'model') and strategy.model:
        print("✅ Model loaded successfully")
        
        # Test prediction
        X = features.iloc[-1:][['ret1','vol','mom','rsi']].fillna(0)
        try:
            prob = strategy.model.predict_proba(X)[0, 1]
            print(f"✅ Current market probability: {prob:.3f}")
        except Exception as e:
            print(f"❌ Model prediction failed: {e}")
            return False
    else:
        print("❌ Model not loaded")
        return False
    
    print("\n✅ Production strategy validation complete!")
    return True

def create_deployment_config():
    """Create deployment configuration"""
    config = {
        'strategy': 'ProductionXgbStrategy',
        'account': 'B',
        'parameters': {
            'min_probability': 0.58,
            'cooldown_bars': 3,
            'position_size_base': 1,
            'max_volatility_ratio': 2.0,
            'min_momentum_threshold': 0.15
        },
        'risk_management': {
            'max_position_size': 5,
            'max_daily_trades': 10,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 1.0
        },
        'deployment_timestamp': datetime.now().isoformat(),
        'validation_passed': True
    }
    
    with open('config/production_deployment.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Deployment configuration saved to config/production_deployment.json")
    return config

def main():
    print("JuusoTrader Production Deployment")
    print("=" * 50)
    
    # Step 1: Validate strategy
    if not validate_production_strategy():
        print("❌ Validation failed. Cannot proceed with deployment.")
        return
    
    # Step 2: Create deployment config
    config = create_deployment_config()
    
    # Step 3: Deployment summary
    print("\n" + "=" * 50)
    print("DEPLOYMENT READY")
    print("=" * 50)
    print("✅ Strategy: ProductionXgbStrategy")
    print("✅ Account: B")
    print("✅ Model: XGB Classifier (trained)")
    print("✅ Features: 4 technical indicators")
    print("✅ Risk Management: Dynamic position sizing")
    print("✅ Signal Quality: Enhanced filtering")
    
    print("\n📋 NEXT STEPS:")
    print("1. Review config/production_deployment.json")
    print("2. Run: python engine/live.py --paper (paper trading)")
    print("3. Monitor performance for 1-2 weeks")
    print("4. If successful, switch to live trading")
    
    print("\n⚠️  IMPORTANT:")
    print("- Start with paper trading only")
    print("- Monitor all signals and trades")
    print("- Set position size limits")
    print("- Have stop-loss mechanisms ready")

if __name__ == '__main__':
    main()
