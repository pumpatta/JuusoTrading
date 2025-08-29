"""
Account C: ML + External Features Strategy
Integrates news sentiment, pattern recognition, and ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import Strategy
from strategies.pattern_head_shoulders import HeadShoulders
from strategies.ensemble import Ensemble
from features.enhanced_news_sentiment import EnhancedNewsStrategy
import time

class AccountCStrategy:
    """
    Account C: Advanced ML strategy combining:
    1. News sentiment analysis with FinBERT
    2. Technical pattern recognition (Head & Shoulders)
    3. Ensemble consensus (EMA + XGB)
    4. Continuous learning from outcomes
    """
    
    def __init__(self, symbol="SPY", account="C"):
        self.symbol = symbol
        self.account = account
        self.strategy_id = f"ACCOUNT_C_ML_{symbol}"
        
        # Initialize component strategies - simplified for now
        self.pattern_strategy = None  # Will integrate directly
        self.ensemble_strategy = None  # Will integrate directly
        self.news_strategy = EnhancedNewsStrategy()
        
        # Disable news analysis to prevent startup crashes
        self.news_disabled = True
        
        # Strategy weights (can be adaptive)
        self.weights = {
            'news_sentiment': 0.4,
            'patterns': 0.3,
            'ensemble': 0.3
        }
        
        # Minimum confidence thresholds
        self.min_confidence = {
            'news_sentiment': 0.7,
            'patterns': 0.6,
            'ensemble': 0.6
        }
        
        # Signal cache to avoid repeated calculations
        self.signal_cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Timeout tracking for news analysis
        self.news_timeout_count = 0
        self.max_news_timeouts = 3  # Disable news after 3 timeouts
        self.news_disabled = False
        
        print(f"Account C Strategy initialized for {symbol}")
        print(f"Components: News Sentiment + Pattern Recognition + Ensemble")
    
    def on_bar(self, bars: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Main entry point for live trading system"""
        signals = []
        
        try:
            # Get data for our symbol
            if self.symbol not in bars:
                return []
            
            data = bars[self.symbol]
            
            # Generate comprehensive signal
            signal_result = self.generate_signal(data)
            
            if signal_result['action'] != 'hold' and signal_result['confidence'] > 0.6:
                # Convert to live trading format
                live_signal = {
                    'symbol': self.symbol,
                    'action': signal_result['action'],
                    'confidence': signal_result['confidence'],
                    'reason': signal_result['reason'],
                    'strategy': 'ACCOUNT_C_ML',
                    'timestamp': signal_result['timestamp'],
                    'position_size': signal_result['position_size'],
                    'components': signal_result.get('components', {}),
                    'qualified_signals': signal_result.get('qualified_signals', 0)
                }
                signals.append(live_signal)
                
                print(f"Account C ML Signal: {signal_result['action']} {self.symbol} "
                      f"(conf: {signal_result['confidence']:.3f}, reason: {signal_result['reason']})")
            
        except Exception as e:
            print(f"Account C on_bar error: {e}")
        
        return signals
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive signal from all components"""
        
        if len(data) < 50:
            return self._no_signal("Insufficient data")
            
        try:
            # Get individual signals from each component
            news_signal = self._get_news_signal()
            pattern_signal = self._get_pattern_signal(data)
            ensemble_signal = self._get_ensemble_signal(data)
            
            # Combine signals using weighted approach
            combined_signal = self._combine_signals(news_signal, pattern_signal, ensemble_signal)
            
            return combined_signal
            
        except Exception as e:
            print(f"Account C signal generation error: {e}")
            return self._no_signal(f"Error: {e}")
    
    def _get_news_signal(self) -> Dict:
        """Get news sentiment signal with caching and timeout protection"""
        
        # Skip news analysis if disabled
        if self.news_disabled:
            return {
                'type': 'news_sentiment',
                'action': 'hold',
                'confidence': 0.0,
                'strength': 0.0,
                'position_size': 0.0,
                'details': {'reason': 'News analysis disabled due to timeouts'},
                'timestamp': time.time()
            }
        
        cache_key = 'news_sentiment'
        now = time.time()
        
        # Check cache
        if (cache_key in self.signal_cache and 
            now - self.signal_cache[cache_key]['timestamp'] < self.cache_duration):
            return self.signal_cache[cache_key]
        
        try:
            # Add timeout protection for news analysis using threading
            import threading
            
            result = {'data': None, 'error': None}  # type: ignore
            
            def fetch_news():
                try:
                    news_data = self.news_strategy.get_market_sentiment_signal()
                    result['data'] = news_data
                except Exception as e:
                    result['error'] = str(e)
            
            # Start news fetching in a separate thread
            news_thread = threading.Thread(target=fetch_news)
            news_thread.daemon = True
            news_thread.start()
            
            # Wait for up to 30 seconds
            news_thread.join(timeout=30)
            
            if news_thread.is_alive():
                print("News analysis timed out, returning neutral signal")
                self.news_timeout_count += 1
                if self.news_timeout_count >= self.max_news_timeouts:
                    self.news_disabled = True
                    print(f"News analysis disabled after {self.news_timeout_count} timeouts")
                news_data = {
                    'signal': 'hold',
                    'confidence': 0.0,
                    'sentiment_score': 0.0,
                    'reason': 'Analysis timeout'
                }
            elif result['error']:
                print(f"News analysis error: {result['error']}")
                news_data = {
                    'signal': 'hold',
                    'confidence': 0.0,
                    'sentiment_score': 0.0,
                    'reason': f'Analysis error: {result["error"]}'
                }
            else:
                # Reset timeout count on successful analysis
                self.news_timeout_count = 0
                news_data = result['data'] if result['data'] is not None else {
                    'signal': 'hold',
                    'confidence': 0.0,
                    'sentiment_score': 0.0,
                    'reason': 'No data returned'
                }
            
            # Convert to our signal format
            signal_data = {
                'type': 'news_sentiment',
                'action': news_data.get('signal', 'hold'),
                'confidence': news_data.get('confidence', 0.0),
                'strength': abs(news_data.get('sentiment_score', 0.0)),
                'position_size': news_data.get('position_size', 0.0),
                'details': {
                    'sentiment_score': news_data.get('sentiment_score', 0.0),
                    'predicted_impact': news_data.get('predicted_impact', 0.0),
                    'article_count': news_data.get('article_count', 0),
                    'reason': news_data.get('reason', '')
                },
                'timestamp': now
            }
            
            # Cache result
            self.signal_cache[cache_key] = signal_data
            
            return signal_data
            
        except Exception as e:
            print(f"News sentiment error: {e}")
            return {
                'type': 'news_sentiment',
                'action': 'hold',
                'confidence': 0.0,
                'strength': 0.0,
                'position_size': 0.0,
                'details': {'error': str(e)},
                'timestamp': now
            }
    
    def _get_pattern_signal(self, data: pd.DataFrame) -> Dict:
        """Get pattern recognition signal - placeholder for now"""
        
        try:
            # Simple pattern detection based on recent price action
            if len(data) < 20:
                return self._empty_signal('pattern_recognition', 'Insufficient data')
            
            # Try both uppercase and lowercase column names
            if 'Close' in data.columns:
                close = data['Close']
            elif 'close' in data.columns:
                close = data['close']
            else:
                return self._empty_signal('pattern_recognition', 'No close price column found')
            
            # Simple momentum check
            recent_return = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
            
            if recent_return > 0.02:
                action = 'buy'
                confidence = min(0.7, abs(recent_return) * 10)
            elif recent_return < -0.02:
                action = 'sell'
                confidence = min(0.7, abs(recent_return) * 10)
            else:
                action = 'hold'
                confidence = 0.0
            
            return {
                'type': 'pattern_recognition',
                'action': action,
                'confidence': confidence,
                'strength': confidence,
                'position_size': 1.0 if action != 'hold' else 0.0,
                'details': {'recent_return': recent_return, 'method': 'simple_momentum'},
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"Pattern recognition error: {e}")
            return self._empty_signal('pattern_recognition', str(e))
    
    def _get_ensemble_signal(self, data: pd.DataFrame) -> Dict:
        """Get ensemble strategy signal - placeholder for now"""
        
        try:
            # Simple ensemble logic placeholder
            if len(data) < 50:
                return self._empty_signal('ensemble', 'Insufficient data')
            
            # Try both uppercase and lowercase column names
            if 'Close' in data.columns:
                close = data['Close']
            elif 'close' in data.columns:
                close = data['close']
            else:
                return self._empty_signal('ensemble', 'No close price column found')
            
            # Simple EMA crossover
            ema_short = close.ewm(span=5).mean()
            ema_long = close.ewm(span=20).mean()
            
            if len(ema_short) >= 2 and len(ema_long) >= 2:
                if ema_short.iloc[-1] > ema_long.iloc[-1] and ema_short.iloc[-2] <= ema_long.iloc[-2]:
                    action = 'buy'
                    confidence = 0.6
                elif ema_short.iloc[-1] < ema_long.iloc[-1] and ema_short.iloc[-2] >= ema_long.iloc[-2]:
                    action = 'sell'
                    confidence = 0.6
                else:
                    action = 'hold'
                    confidence = 0.0
            else:
                action = 'hold'
                confidence = 0.0
            
            return {
                'type': 'ensemble',
                'action': action,
                'confidence': confidence,
                'strength': confidence,
                'position_size': 1.0 if action != 'hold' else 0.0,
                'details': {'method': 'ema_crossover'},
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"Ensemble strategy error: {e}")
            return self._empty_signal('ensemble', str(e))
    
    def _empty_signal(self, signal_type: str, reason: str) -> Dict:
        """Return empty signal"""
        return {
            'type': signal_type,
            'action': 'hold',
            'confidence': 0.0,
            'strength': 0.0,
            'position_size': 0.0,
            'details': {'error': reason},
            'timestamp': time.time()
        }
    
    def _combine_signals(self, news_signal: Dict, pattern_signal: Dict, ensemble_signal: Dict) -> Dict:
        """Combine signals from all components using sophisticated logic"""
        
        signals = [news_signal, pattern_signal, ensemble_signal]
        
        # Filter signals by minimum confidence
        qualified_signals = []
        for signal in signals:
            signal_type = signal['type']
            if signal['confidence'] >= self.min_confidence.get(signal_type, 0.6):
                qualified_signals.append(signal)
        
        if not qualified_signals:
            return self._no_signal("No signals meet minimum confidence threshold")
        
        # Count buy/sell/hold signals
        signal_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        weighted_strength = 0.0
        total_weight = 0.0
        signal_details = []
        
        for signal in qualified_signals:
            action = signal['action']
            signal_type = signal['type']
            weight = self.weights.get(signal_type, 1.0)
            
            signal_counts[action] += 1
            
            if action != 'hold':
                weighted_strength += signal['strength'] * weight
                total_weight += weight
            
            signal_details.append({
                'type': signal_type,
                'action': action,
                'confidence': signal['confidence'],
                'strength': signal['strength'],
                'weight': weight
            })
        
        # Decision logic
        if signal_counts['buy'] > signal_counts['sell'] and signal_counts['buy'] > signal_counts['hold']:
            final_action = 'buy'
        elif signal_counts['sell'] > signal_counts['buy'] and signal_counts['sell'] > signal_counts['hold']:
            final_action = 'sell'
        else:
            final_action = 'hold'
        
        # Calculate combined confidence
        if total_weight > 0:
            combined_confidence = weighted_strength / total_weight
            combined_strength = weighted_strength / total_weight
        else:
            combined_confidence = 0.0
            combined_strength = 0.0
        
        # Position sizing based on consensus and strength
        consensus_ratio = max(signal_counts.values()) / len(qualified_signals)
        position_size = combined_strength * consensus_ratio if final_action != 'hold' else 0.0
        position_size = min(position_size, 1.0)  # Cap at 100%
        
        # Generate reason
        active_signals = [s for s in signal_details if s['action'] != 'hold']
        if active_signals:
            signal_summary = ", ".join([f"{s['type']}({s['action']})" for s in active_signals])
            reason = f"Consensus: {signal_counts} - Active: {signal_summary}"
        else:
            reason = "No strong signals from any component"
        
        return {
            'action': final_action,
            'confidence': combined_confidence,
            'strength': combined_strength,
            'position_size': position_size,
            'reason': reason,
            'components': {
                'news_sentiment': news_signal,
                'pattern_recognition': pattern_signal,
                'ensemble': ensemble_signal
            },
            'signal_counts': signal_counts,
            'qualified_signals': len(qualified_signals),
            'consensus_ratio': consensus_ratio,
            'timestamp': time.time()
        }
    
    def _no_signal(self, reason: str) -> Dict:
        """Return a no-signal response"""
        return {
            'action': 'hold',
            'confidence': 0.0,
            'strength': 0.0,
            'position_size': 0.0,
            'reason': reason,
            'components': {},
            'signal_counts': {'buy': 0, 'sell': 0, 'hold': 0},
            'qualified_signals': 0,
            'consensus_ratio': 0.0,
            'timestamp': time.time()
        }
    
    def update_market_outcome(self, predicted_return: float, actual_return: float):
        """Update learning models with market outcome"""
        try:
            # Update news strategy learning
            if hasattr(self.news_strategy, 'update_with_market_outcome'):
                # Get last news signal for learning update
                if 'news_sentiment' in self.signal_cache:
                    news_data = self.signal_cache['news_sentiment']['details']
                    self.news_strategy.update_with_market_outcome(news_data, actual_return)
        except Exception as e:
            print(f"Error updating market outcome: {e}")

# Convenience function for testing
def test_account_c_strategy():
    """Test Account C strategy with sample data"""
    import yfinance as yf
    
    print("=== Testing Account C Strategy ===")
    
    # Get sample data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1mo", interval="1h")
    
    if data.empty:
        print("No market data available")
        return
    
    # Initialize strategy
    strategy = AccountCStrategy("SPY", "C")
    
    # Generate signal
    signal = strategy.generate_signal(data)
    
    print(f"\nAccount C Signal Results:")
    print(f"Action: {signal['action']}")
    print(f"Confidence: {signal['confidence']:.3f}")
    print(f"Strength: {signal['strength']:.3f}")
    print(f"Position Size: {signal['position_size']:.3f}")
    print(f"Reason: {signal['reason']}")
    print(f"Qualified Signals: {signal['qualified_signals']}")
    print(f"Consensus Ratio: {signal['consensus_ratio']:.3f}")
    
    # Component breakdown
    print(f"\nComponent Signals:")
    for component, comp_signal in signal.get('components', {}).items():
        if comp_signal:
            print(f"  {component}: {comp_signal['action']} (conf: {comp_signal['confidence']:.3f})")

if __name__ == "__main__":
    test_account_c_strategy()
