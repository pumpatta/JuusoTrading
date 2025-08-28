#!/usr/bin/env python3
"""
Strategy Optimization Script
Optimizes model parameters, features, and signal filtering for better performance
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
from strategies.xgb_classifier import XgbSignal
from strategies.tcn_torch import TcnSignal
from strategies.ema_trend import EmaTrend
from utils.config import SETTINGS

def enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering with more predictive indicators"""
    data = df.copy()
    
    # Basic price features
    data['ret1'] = data['close'].pct_change()
    data['ret5'] = data['close'].pct_change(5)
    data['ret10'] = data['close'].pct_change(10)
    
    # Volatility features
    data['vol_5'] = data['ret1'].rolling(5).std()
    data['vol_20'] = data['ret1'].rolling(20).std()
    data['vol_ratio'] = data['vol_5'] / (data['vol_20'] + 1e-9)
    
    # Price action features
    data['hl_range'] = (data['high'] - data['low']) / data['close']
    data['oc_ratio'] = (data['close'] - data['open']) / (data['open'] + 1e-9)
    data['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['close']
    data['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['close']
    
    # Moving averages and trends
    data['sma_5'] = data['close'].rolling(5).mean()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['price_vs_sma5'] = (data['close'] - data['sma_5']) / data['sma_5']
    data['price_vs_sma20'] = (data['close'] - data['sma_20']) / data['sma_20']
    data['sma_slope'] = data['sma_20'].pct_change(5)
    
    # RSI improvements
    up = data['ret1'].clip(lower=0).rolling(14).mean()
    down = (-data['ret1'].clip(upper=0)).rolling(14).mean()
    rs = up / (down + 1e-9)
    data['rsi'] = 100 - (100 / (1 + rs))
    data['rsi_normalized'] = (data['rsi'] - 50) / 50  # Normalize to [-1, 1]
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    data['bb_middle'] = data['close'].rolling(bb_period).mean()
    bb_std_val = data['close'].rolling(bb_period).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * bb_std_val)
    data['bb_lower'] = data['bb_middle'] - (bb_std * bb_std_val)
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    
    # MACD
    ema12 = data['close'].ewm(span=12).mean()
    ema26 = data['close'].ewm(span=26).mean()
    data['macd'] = ema12 - ema26
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    
    # Volume features (if available)
    if 'volume' in data.columns:
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / (data['volume_sma'] + 1e-9)
        data['price_volume'] = data['ret1'] * np.log1p(data['volume_ratio'])
    
    # Regime detection
    data['vol_regime'] = (data['vol_20'] > data['vol_20'].rolling(60).quantile(0.75)).astype(int)
    data['trend_strength'] = np.abs(data['price_vs_sma20'])
    
    return data

def optimize_xgb_strategy():
    """Optimize XGB strategy with better features and parameters"""
    print("=== Optimizing XGB Strategy ===")
    
    # Get data
    bars = get_bars(['SPY'], None, None, prefer_samples=True)
    bars = bars.get('SPY') if bars else None
    if bars is None or bars.empty:
        print("No data available for optimization")
        return
    
    # Enhanced features
    enhanced_data = enhanced_features(bars)
    
    # Better feature selection for XGB
    feature_cols = [
        'ret1', 'ret5', 'vol_5', 'vol_20', 'vol_ratio',
        'price_vs_sma5', 'price_vs_sma20', 'sma_slope',
        'rsi_normalized', 'bb_position', 'bb_width',
        'macd', 'macd_histogram', 'hl_range', 'oc_ratio',
        'upper_shadow', 'lower_shadow', 'trend_strength', 'vol_regime'
    ]
    
    # Create optimized XGB strategy
    class OptimizedXgbSignal(XgbSignal):
        def _features(self, d: pd.DataFrame) -> pd.DataFrame:
            enhanced_data = enhanced_features(d)
            available_features = [col for col in feature_cols if col in enhanced_data.columns]
            return enhanced_data[available_features].fillna(0)
        
        def fit(self, d: pd.DataFrame):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
            
            df = self._features(d)
            # Better labeling: look for significant moves
            future_returns = d['close'].shift(-5) / d['close'] - 1
            # Target 1 if future return > 0.5% or < -0.5% (avoid noise)
            significant_moves = np.abs(future_returns) > 0.005
            y = ((future_returns > 0) & significant_moves).astype(int)
            
            X = df.fillna(0)
            mask = ~y.isna() & significant_moves  # Only train on significant moves
            X, y = X[mask], y[mask]
            
            if len(y) == 0 or y.nunique() < 2:
                print('XGB optimization: insufficient training data')
                self.fitted = False
                return
            
            # Train-test split for validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Use RandomForest with optimized parameters
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train, y_train)
            
            # Validation
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            print(f"XGB Training accuracy: {train_score:.3f}")
            print(f"XGB Test accuracy: {test_score:.3f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("Top 5 features:")
            print(feature_importance.head())
            
            self.fitted = True
            self.save_model()
        
        def on_bar(self, bars: dict[str, pd.DataFrame]) -> list[dict]:
            if not self.fitted or 'SPY' not in bars:
                return []
            
            df = bars['SPY']
            if len(df) < 100:  # Need enough history
                return []
            
            # Enhanced signal filtering
            features = self._features(df)
            if len(features) == 0:
                return []
            
            # Get prediction
            X = features.iloc[-1:].fillna(0)
            prob = self.model.predict_proba(X)[0, 1]  # Probability of positive class
            
            # More sophisticated signal generation
            current_data = enhanced_features(df)
            current_rsi = current_data['rsi'].iloc[-1]
            current_bb_pos = current_data['bb_position'].iloc[-1]
            vol_regime = current_data['vol_regime'].iloc[-1]
            trend_strength = current_data['trend_strength'].iloc[-1]
            
            signals = []
            
            # Buy signal: high probability + not overbought + strong trend
            if (prob > 0.65 and current_rsi < 70 and current_bb_pos < 0.8 and 
                trend_strength > 0.01 and vol_regime == 0):  # Low vol regime
                signals.append({
                    'symbol': 'SPY',
                    'side': 'buy',
                    'qty': 1,
                    'reason': f'OptXGB_buy_prob_{prob:.3f}_rsi_{current_rsi:.1f}'
                })
            
            # Sell signal: low probability + not oversold
            elif (prob < 0.35 and current_rsi > 30 and current_bb_pos > 0.2):
                signals.append({
                    'symbol': 'SPY', 
                    'side': 'sell',
                    'qty': 1,
                    'reason': f'OptXGB_sell_prob_{prob:.3f}_rsi_{current_rsi:.1f}'
                })
            
            return signals
    
    # Train optimized model
    strategy = OptimizedXgbSignal(account='B')
    strategy.fit(bars)
    return strategy

def optimize_tcn_strategy():
    """Optimize TCN strategy with better architecture and signal filtering"""
    print("=== Optimizing TCN Strategy ===")
    
    # Get data
    bars = get_bars(['SPY'], None, None, prefer_samples=True)
    bars = bars.get('SPY') if bars else None
    if bars is None or bars.empty:
        print("No data available for optimization")
        return
    
    class OptimizedTcnSignal(TcnSignal):
        def __init__(self, horizon=10, device=None, account=None):
            super().__init__(horizon, device, account)
            # Smaller horizon for more responsive signals
            self.horizon = horizon
            self.signal_threshold = 0.6  # Lower threshold for less conservative signals
        
        def _features(self, d: pd.DataFrame) -> pd.DataFrame:
            enhanced_data = enhanced_features(d)
            # More features for TCN
            feature_cols = ['ret1', 'vol_5', 'hl_range', 'oc_ratio', 'rsi_normalized', 
                          'bb_position', 'macd_histogram', 'price_vs_sma5']
            available_features = [col for col in feature_cols if col in enhanced_data.columns]
            return enhanced_data[available_features].fillna(0)
        
        def fit(self, d: pd.DataFrame):
            try:
                from utils.labels import triple_barrier_labels
                
                Xdf = self._features(d)
                y = triple_barrier_labels(d['close'], horizon=self.horizon, 
                                        up_mult=0.01, dn_mult=0.01)  # Tighter barriers
                
                if len(Xdf) == 0 or y.isna().all():
                    print('TCN optimization: insufficient training data')
                    self.fitted = False
                    return
                
                import torch
                import torch.nn as nn
                
                X = torch.tensor(Xdf.values.T[None, ...], dtype=torch.float32, device=self.device)
                yv = torch.tensor([y.iloc[-1]], dtype=torch.float32, device=self.device)
                
                # Enhanced training
                opt = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)
                lossf = nn.BCEWithLogitsLoss()
                
                self.model.train()
                best_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(100):  # More epochs
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=("cuda" if self.device=="cuda" else "cpu"), 
                                      enabled=(self.device=="cuda")):
                        out = self.model(X)
                        loss = lossf(out.squeeze(-1), yv)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                    opt.step()
                    scheduler.step(loss)
                    
                    # Early stopping
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter > 20:
                            break
                
                print(f"TCN training completed. Final loss: {best_loss:.4f}")
                self.fitted = True
                self.save_model()
                
            except Exception as e:
                print(f'TCN optimization failed: {type(e).__name__}: {e}')
                self.fitted = False
        
        def on_bar(self, bars: dict[str, pd.DataFrame]) -> list[dict]:
            if not self.fitted or 'SPY' not in bars:
                return []
            
            df = bars['SPY']
            if len(df) < 100:
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
            
            # Enhanced signal generation with market context
            current_data = enhanced_features(df)
            current_rsi = current_data['rsi'].iloc[-1]
            bb_position = current_data['bb_position'].iloc[-1]
            vol_regime = current_data['vol_regime'].iloc[-1]
            trend_strength = current_data['trend_strength'].iloc[-1]
            
            signals = []
            
            # More aggressive signal generation
            if (prob > self.signal_threshold and current_rsi < 75 and 
                bb_position < 0.9 and trend_strength > 0.005):
                signals.append({
                    'symbol': 'SPY',
                    'side': 'buy', 
                    'qty': 1,
                    'reason': f'OptTCN_buy_prob_{prob:.3f}_trend_{trend_strength:.3f}'
                })
            elif (prob < (1 - self.signal_threshold) and current_rsi > 25 and 
                  bb_position > 0.1):
                signals.append({
                    'symbol': 'SPY',
                    'side': 'sell',
                    'qty': 1, 
                    'reason': f'OptTCN_sell_prob_{prob:.3f}_trend_{trend_strength:.3f}'
                })
            
            return signals
    
    # Train optimized model
    strategy = OptimizedTcnSignal(horizon=10, account='B')
    strategy.fit(bars)
    return strategy

def run_optimization_backtest(strategies):
    """Run backtest with optimized strategies"""
    print("\n=== Running Optimization Backtest ===")
    
    bars = get_bars(['SPY'], None, None, prefer_samples=True)
    bars = bars.get('SPY') if bars else None
    if bars is None or bars.empty:
        print("No data for backtest")
        return
    
    n_total = len(bars)
    train_size = int(n_total * 0.75)
    
    train_data = bars.iloc[:train_size].copy()
    test_data = bars.iloc[train_size:].copy()
    
    # Train all strategies
    for name, strategy in strategies.items():
        print(f"\nTraining {name}...")
        strategy.fit(train_data)
    
    print(f"\nBacktesting on {len(test_data)} bars...")
    
    results = {}
    for name, strategy in strategies.items():
        print(f"\nTesting {name}...")
        
        nav = 100000.0
        position = 0
        trades = []
        nav_history = [nav]
        
        for i in range(len(test_data)):
            current_bars = {'SPY': test_data.iloc[:i+1]}
            signals = strategy.on_bar(current_bars)
            
            for signal in signals:
                if signal['side'] == 'buy' and position <= 0:
                    price = test_data.iloc[i]['close']
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
                            'reason': signal.get('reason', 'buy')
                        })
                
                elif signal['side'] == 'sell' and position > 0:
                    price = test_data.iloc[i]['close']
                    qty = min(signal['qty'], position)
                    proceeds = price * qty
                    position -= qty
                    nav += proceeds
                    trades.append({
                        'side': 'sell', 
                        'price': price,
                        'qty': qty,
                        'nav_after': nav,
                        'reason': signal.get('reason', 'sell')
                    })
            
            # Mark to market
            if position > 0:
                current_value = nav + position * test_data.iloc[i]['close']
            else:
                current_value = nav
            nav_history.append(current_value)
        
        # Final liquidation
        if position > 0:
            final_price = test_data.iloc[-1]['close']
            nav += position * final_price
            trades.append({
                'side': 'sell',
                'price': final_price, 
                'qty': position,
                'nav_after': nav,
                'reason': 'final_liquidation'
            })
        
        # Calculate metrics
        final_return = (nav - 100000) / 100000 * 100
        nav_series = pd.Series(nav_history)
        peak = nav_series.expanding().max()
        drawdown = (nav_series - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        n_trades = len([t for t in trades if t['side'] == 'buy'])
        
        results[name] = {
            'final_nav': nav,
            'return_pct': final_return,
            'max_drawdown_pct': max_drawdown,
            'n_trades': n_trades,
            'trades': trades[-10:]  # Last 10 trades
        }
        
        print(f"{name} Results:")
        print(f"  Final NAV: ${nav:,.2f}")
        print(f"  Return: {final_return:.2f}%")
        print(f"  Max Drawdown: {max_drawdown:.2f}%") 
        print(f"  Number of trades: {n_trades}")
    
    # Save results
    report = {
        'timestamp': datetime.now().isoformat(),
        'optimization': True,
        'test_period': f"{len(test_data)} bars",
        'strategies': results
    }
    
    with open('storage/optimization_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nOptimization results saved to storage/optimization_report.json")
    return results

def main():
    print("JuusoTrader Strategy Optimization")
    print("=" * 50)
    
    # Create optimized strategies
    strategies = {}
    
    # Keep original EMA as baseline
    strategies['EMA_Baseline'] = EmaTrend()
    
    # Add optimized strategies
    try:
        strategies['XGB_Optimized'] = optimize_xgb_strategy()
    except Exception as e:
        print(f"XGB optimization failed: {e}")
    
    try:
        strategies['TCN_Optimized'] = optimize_tcn_strategy()
    except Exception as e:
        print(f"TCN optimization failed: {e}")
    
    # Run comparative backtest
    if len(strategies) > 1:
        results = run_optimization_backtest(strategies)
        
        print("\n" + "=" * 50)
        print("OPTIMIZATION SUMMARY")
        print("=" * 50)
        
        if results:
            for name, result in results.items():
                print(f"{name:15} | Return: {result['return_pct']:6.2f}% | "
                      f"Drawdown: {result['max_drawdown_pct']:6.2f}% | "
                      f"Trades: {result['n_trades']:3d}")
        else:
            print("No results generated")
    
    print("\nOptimization complete!")

if __name__ == '__main__':
    main()
