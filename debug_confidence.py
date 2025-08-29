#!/usr/bin/env python3
"""Debug script to check confidence levels in ML strategies"""

import pandas as pd
import numpy as np
import yfinance as yf

def debug_confidence():
    # Get some real data
    spy = yf.download('SPY', start='2025-03-01', end='2025-08-27', progress=False)
    
    print("Debugging confidence calculations...")
    print(f"Data shape: {spy.shape}")
    
    # Add technical indicators like in backtest
    spy['EMA_12'] = spy['Close'].ewm(span=12).mean()
    spy['EMA_26'] = spy['Close'].ewm(span=26).mean()
    spy['MACD'] = spy['EMA_12'] - spy['EMA_26']
    
    # RSI calculation
    delta = spy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    spy['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    spy['Volatility'] = spy['Close'].rolling(window=20).std() / spy['Close'].rolling(window=20).mean()
    
    spy = spy.dropna()
    
    print(f"\nAfter indicators: {spy.shape}")
    print(f"Date range: {spy.index[0]} to {spy.index[-1]}")
    
    # Check first 10 days for XGB-style confidence
    print("\n=== XGB CONFIDENCE DEBUG ===")
    confidence_threshold = 0.50
    
    for i in range(10, 20):  # Skip first 10 for lookback
        row = spy.iloc[i]
        date = spy.index[i]
        
        # Same features as in backtest
        features = [
            row['RSI'] / 100,
            row['MACD'] / row['Close'],
            row['Volatility'],
            (row['Close'] - row['EMA_26']) / row['EMA_26']
        ]
        
        # Same confidence calculation
        rsi_strength = abs(row['RSI'] - 50) / 50
        macd_strength = abs(row['MACD']) / row['Close'] if row['Close'] > 0 else 0
        trend_strength = abs((row['Close'] - row['EMA_26']) / row['EMA_26'])
        
        base_confidence = (rsi_strength + min(macd_strength, 0.1) + min(trend_strength, 0.2)) / 3
        confidence = min(0.95, base_confidence * 0.7 + 0.4 + np.random.normal(0, 0.05))
        
        prediction_score = sum(features) / len(features)
        
        would_trade = confidence > confidence_threshold
        
        print(f"{date.strftime('%Y-%m-%d')}: confidence={confidence:.3f}, threshold={confidence_threshold}, would_trade={would_trade}")
        print(f"  RSI={row['RSI']:.1f}, MACD={row['MACD']:.4f}, features={[f'{f:.3f}' for f in features]}")
        print(f"  base_confidence={base_confidence:.3f}, prediction_score={prediction_score:.3f}")
    
    # Check Enhanced ML confidence
    print("\n=== ENHANCED ML CONFIDENCE DEBUG ===")
    confidence_threshold = 0.55
    
    for i in range(40, 50):  # Skip first 40 for enhanced ML
        row = spy.iloc[i]
        date = spy.index[i]
        
        # Same calculation as in Enhanced ML
        news_sentiment = np.random.normal(0, 0.3)  # Simulated news
        pattern_score = np.random.normal(0, 0.2)   # Simulated pattern
        
        technical_signal = (row['RSI'] - 50) / 50
        momentum_signal = row['MACD'] / row['Close'] if row['Close'] > 0 else 0
        
        signal_alignment = abs(news_sentiment) * abs(pattern_score) * abs(technical_signal)
        confidence = min(0.95, signal_alignment + 0.4)
        
        news_weight = 0.3
        pattern_weight = 0.4
        ensemble_weight = 0.3
        
        ensemble_prediction = (
            news_weight * news_sentiment +
            pattern_weight * pattern_score +
            ensemble_weight * (technical_signal + momentum_signal) / 2
        )
        
        would_trade = confidence > confidence_threshold
        strong_signal = ensemble_prediction > 0.1 or ensemble_prediction < -0.1
        
        print(f"{date.strftime('%Y-%m-%d')}: confidence={confidence:.3f}, threshold={confidence_threshold}")
        print(f"  ensemble_pred={ensemble_prediction:.3f}, strong_signal={strong_signal}, would_trade={would_trade}")
        print(f"  signal_alignment={signal_alignment:.3f}")

if __name__ == "__main__":
    debug_confidence()
