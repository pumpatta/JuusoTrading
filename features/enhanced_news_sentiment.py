"""
Comprehensive News & Sentiment Analysis System for Account C
Features:
- Real-time news feeds from multiple sources
- Advanced sentiment analysis with FinBERT
- Market correlation tracking
- Continuous learning from prediction outcomes
- Economic calendar integration
"""

import os
import time
import json
import feedparser
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
# Lazy import for transformers to avoid slow startup
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class NewsDataCollector:
    """Collects news from multiple free sources"""
    
    def __init__(self):
        self.sources = {
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/marketpulse/',
            'reuters_business': 'https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best',
            'bloomberg_markets': 'https://feeds.bloomberg.com/markets/news.rss',
            'cnbc_markets': 'https://www.cnbc.com/id/100003114/device/rss/rss.html'
        }
        self.cache = {}
        self.last_fetch = {}
        
    def fetch_rss_feed(self, url: str, source_name: str) -> List[Dict]:
        """Fetch and parse RSS feed with timeout"""
        try:
            # Check cache first
            now = time.time()
            if source_name in self.last_fetch and now - self.last_fetch[source_name] < 300:  # 5 min cache
                return self.cache.get(source_name, [])
                
            # Use requests with timeout to avoid hanging
            import requests
            response = requests.get(url, timeout=5)  # 5 second timeout
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            articles = []
            
            for entry in feed.entries[:10]:  # Reduced to 10 articles per source
                article = {
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', entry.get('description', '')),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': source_name,
                    'timestamp': now
                }
                articles.append(article)
                
            self.cache[source_name] = articles
            self.last_fetch[source_name] = now
            return articles
            
        except Exception as e:
            print(f"Error fetching {source_name}: {e}")
            return []
    
    def get_all_news(self) -> List[Dict]:
        """Fetch news from all sources with timeout protection"""
        all_news = []
        start_time = time.time()
        max_time = 15  # Maximum 15 seconds for all news fetching
        
        for source_name, url in self.sources.items():
            if time.time() - start_time > max_time:
                print(f"News fetching timeout reached ({max_time}s), stopping...")
                break
                
            try:
                articles = self.fetch_rss_feed(url, source_name)
                all_news.extend(articles)
                time.sleep(0.2)  # Reduced rate limiting
            except Exception as e:
                print(f"Failed to fetch from {source_name}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        all_news.sort(key=lambda x: x['timestamp'], reverse=True)
        return all_news[:30]  # Return top 30 most recent articles

class AdvancedSentimentAnalyzer:
    """Enhanced sentiment analysis with FinBERT and market correlation"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.scaler = StandardScaler()
        self.market_correlation_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.sentiment_history = []
        self.market_returns_history = []
        self.trained = False
        
    def load_finbert(self):
        """Load FinBERT model for financial sentiment"""
        if self.model is None:
            print("Loading FinBERT model...")
            # Lazy import to avoid slow startup
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            self.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            self.model.eval()
            print("FinBERT loaded successfully")
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of single text"""
        self.load_finbert()
        
        if not text or len(text.strip()) < 10:
            return {'sentiment': 0.0, 'confidence': 0.0, 'label': 'neutral'}
            
        try:
            with torch.no_grad():
                # Type checking bypass
                tokenizer_func = self.tokenizer  # type: ignore
                model_func = self.model  # type: ignore
                
                inputs = tokenizer_func(text, padding=True, truncation=True, 
                                      max_length=512, return_tensors="pt")
                outputs = model_func(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
                # Labels: [negative, neutral, positive]
                sentiment_score = float(probs[2] - probs[0])  # positive - negative
                confidence = float(max(probs))
                
                if sentiment_score > 0.1:
                    label = 'positive'
                elif sentiment_score < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'
                    
                return {
                    'sentiment': sentiment_score,
                    'confidence': confidence,
                    'label': label,
                    'probs': probs.tolist()
                }
                
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0, 'label': 'neutral'}
    
    def analyze_news_batch(self, articles: List[Dict]) -> Dict:
        """Analyze sentiment for batch of news articles"""
        if not articles:
            return {'overall_sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}
            
        sentiments = []
        confidences = []
        
        for article in articles:
            # Analyze title and summary
            title_sentiment = self.analyze_text_sentiment(article.get('title', ''))
            summary_sentiment = self.analyze_text_sentiment(article.get('summary', ''))
            
            # Weight title more heavily
            combined_sentiment = (title_sentiment['sentiment'] * 0.7 + 
                                summary_sentiment['sentiment'] * 0.3)
            combined_confidence = (title_sentiment['confidence'] * 0.7 + 
                                 summary_sentiment['confidence'] * 0.3)
            
            sentiments.append(combined_sentiment)
            confidences.append(combined_confidence)
            
            article['sentiment_analysis'] = {
                'title_sentiment': title_sentiment,
                'summary_sentiment': summary_sentiment,
                'combined_sentiment': combined_sentiment,
                'combined_confidence': combined_confidence
            }
        
        # Calculate weighted average (weight by confidence)
        if sentiments:
            weights = np.array(confidences)
            weights = weights / (weights.sum() + 1e-8)  # Normalize
            overall_sentiment = np.average(sentiments, weights=weights)
            avg_confidence = np.mean(confidences)
        else:
            overall_sentiment = 0.0
            avg_confidence = 0.0
            
        return {
            'overall_sentiment': float(overall_sentiment),
            'confidence': float(avg_confidence),
            'article_count': len(articles),
            'sentiment_distribution': {
                'positive': sum(1 for s in sentiments if s > 0.1),
                'neutral': sum(1 for s in sentiments if -0.1 <= s <= 0.1),
                'negative': sum(1 for s in sentiments if s < -0.1)
            },
            'articles': articles
        }

class MarketSentimentCorrelator:
    """Tracks correlation between news sentiment and market movements"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.sentiment_market_data = []
        self.correlation_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def add_data_point(self, sentiment_score: float, market_return: float, 
                      confidence: float, timestamp: float):
        """Add new sentiment-market correlation data point"""
        data_point = {
            'sentiment': sentiment_score,
            'market_return': market_return,
            'confidence': confidence,
            'timestamp': timestamp
        }
        
        self.sentiment_market_data.append(data_point)
        
        # Keep only recent data
        if len(self.sentiment_market_data) > self.max_history:
            self.sentiment_market_data = self.sentiment_market_data[-self.max_history:]
            
        # Retrain model if we have enough data
        if len(self.sentiment_market_data) >= 50:
            self._retrain_correlation_model()
    
    def _retrain_correlation_model(self):
        """Retrain the correlation prediction model"""
        if len(self.sentiment_market_data) < 20:
            return
            
        try:
            # Prepare features
            features = []
            targets = []
            
            for i, data in enumerate(self.sentiment_market_data[:-1]):
                # Features: sentiment, confidence, time since last, rolling averages
                sentiment = data['sentiment']
                confidence = data['confidence']
                
                # Rolling averages (last 5 and 10 data points)
                start_5 = max(0, i - 4)
                start_10 = max(0, i - 9)
                
                sentiment_5 = np.mean([d['sentiment'] for d in self.sentiment_market_data[start_5:i+1]])
                sentiment_10 = np.mean([d['sentiment'] for d in self.sentiment_market_data[start_10:i+1]])
                
                feature_vector = [sentiment, confidence, sentiment_5, sentiment_10]
                features.append(feature_vector)
                
                # Target: next market return
                next_return = self.sentiment_market_data[i + 1]['market_return']
                targets.append(next_return)
            
            if len(features) >= 10:
                X = np.array(features)
                y = np.array(targets)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train model
                self.correlation_model.fit(X_scaled, y)
                self.is_trained = True
                
                print(f"Correlation model retrained with {len(features)} data points")
                
        except Exception as e:
            print(f"Error retraining correlation model: {e}")
    
    def predict_market_impact(self, sentiment: float, confidence: float) -> float:
        """Predict market impact based on current sentiment"""
        if not self.is_trained or len(self.sentiment_market_data) < 10:
            # Simple heuristic if not enough data
            return sentiment * confidence * 0.001  # Very conservative
            
        try:
            # Calculate rolling averages from recent data
            recent_data = self.sentiment_market_data[-10:]
            sentiment_5 = np.mean([d['sentiment'] for d in recent_data[-5:]])
            sentiment_10 = np.mean([d['sentiment'] for d in recent_data])
            
            feature_vector = np.array([[sentiment, confidence, sentiment_5, sentiment_10]])
            feature_scaled = self.scaler.transform(feature_vector)
            
            prediction = self.correlation_model.predict(feature_scaled)[0]
            
            # Clip extreme predictions
            return np.clip(prediction, -0.05, 0.05)
            
        except Exception as e:
            print(f"Error predicting market impact: {e}")
            return sentiment * confidence * 0.001

class ContinuousLearningEngine:
    """Manages continuous learning from trading outcomes"""
    
    def __init__(self, storage_path: str = "storage/ml_learning_data.json"):
        self.storage_path = storage_path
        self.learning_data = self.load_learning_data()
        self.performance_tracker = {}
        
    def load_learning_data(self) -> Dict:
        """Load previous learning data"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading learning data: {e}")
            
        return {
            'sentiment_outcomes': [],
            'strategy_performance': {},
            'market_conditions': [],
            'adaptation_log': []
        }
    
    def save_learning_data(self):
        """Save learning data to disk"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.learning_data, f, indent=2)
        except Exception as e:
            print(f"Error saving learning data: {e}")
    
    def record_prediction_outcome(self, sentiment_data: Dict, actual_return: float, 
                                prediction: float, strategy: str):
        """Record outcome of sentiment-based prediction"""
        outcome = {
            'timestamp': time.time(),
            'sentiment_score': sentiment_data.get('overall_sentiment', 0),
            'confidence': sentiment_data.get('confidence', 0),
            'predicted_return': prediction,
            'actual_return': actual_return,
            'strategy': strategy,
            'prediction_error': abs(prediction - actual_return),
            'correct_direction': (prediction * actual_return) > 0
        }
        
        self.learning_data['sentiment_outcomes'].append(outcome)
        
        # Keep only recent outcomes (last 1000)
        if len(self.learning_data['sentiment_outcomes']) > 1000:
            self.learning_data['sentiment_outcomes'] = self.learning_data['sentiment_outcomes'][-1000:]
            
        self.save_learning_data()
    
    def get_sentiment_accuracy_metrics(self) -> Dict:
        """Calculate accuracy metrics for sentiment predictions"""
        outcomes = self.learning_data['sentiment_outcomes']
        
        if len(outcomes) < 10:
            return {'direction_accuracy': 0.5, 'avg_prediction_error': 0.01, 'total_samples': len(outcomes)}
            
        recent_outcomes = outcomes[-100:]  # Last 100 predictions
        
        direction_accuracy = np.mean([o['correct_direction'] for o in recent_outcomes])
        avg_error = np.mean([o['prediction_error'] for o in recent_outcomes])
        
        return {
            'direction_accuracy': direction_accuracy,
            'avg_prediction_error': avg_error,
            'total_samples': len(outcomes),
            'recent_samples': len(recent_outcomes)
        }
    
    def adapt_strategy_parameters(self) -> Dict:
        """Suggest strategy parameter adaptations based on learning"""
        metrics = self.get_sentiment_accuracy_metrics()
        
        # Adaptive confidence threshold
        if metrics['direction_accuracy'] > 0.6:
            confidence_threshold = 0.6  # More aggressive
        elif metrics['direction_accuracy'] > 0.55:
            confidence_threshold = 0.7  # Moderate
        else:
            confidence_threshold = 0.8  # Conservative
            
        # Adaptive position sizing
        if metrics['avg_prediction_error'] < 0.005:
            position_multiplier = 1.2  # Increase positions
        elif metrics['avg_prediction_error'] > 0.02:
            position_multiplier = 0.8  # Reduce positions
        else:
            position_multiplier = 1.0  # Normal
            
        adaptations = {
            'confidence_threshold': confidence_threshold,
            'position_multiplier': position_multiplier,
            'sentiment_weight': min(1.0, metrics['direction_accuracy'] * 1.5),
            'last_adapted': time.time()
        }
        
        # Log adaptation
        self.learning_data['adaptation_log'].append({
            'timestamp': time.time(),
            'metrics': metrics,
            'adaptations': adaptations
        })
        
        self.save_learning_data()
        return adaptations

class EnhancedNewsStrategy:
    """Main strategy class integrating news, sentiment, and continuous learning"""
    
    def __init__(self):
        self.news_collector = NewsDataCollector()
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.correlator = MarketSentimentCorrelator()
        self.learning_engine = ContinuousLearningEngine()
        
        self.last_analysis = None
        self.last_analysis_time = 0
        self.analysis_cache_duration = 300  # 5 minutes
        
    def get_market_sentiment_signal(self, current_price: Optional[float] = None) -> Dict:
        """Main method to get sentiment-based trading signal with timeout protection"""
        
        # Check cache
        now = time.time()
        if (self.last_analysis and 
            now - self.last_analysis_time < self.analysis_cache_duration):
            return self.last_analysis
            
        try:
            # Get latest news with timeout protection
            print("Fetching latest financial news...")
            news_articles = self.news_collector.get_all_news()
            
            if not news_articles:
                return {
                    'signal': 'hold',
                    'confidence': 0.0,
                    'sentiment_score': 0.0,
                    'reason': 'No news data available'
                }
            
            # Analyze sentiment with timeout
            print(f"Analyzing sentiment for {len(news_articles)} articles...")
            sentiment_analysis = self.sentiment_analyzer.analyze_news_batch(news_articles)
            
            # Get market correlation prediction
            predicted_impact = self.correlator.predict_market_impact(
                sentiment_analysis['overall_sentiment'],
                sentiment_analysis['confidence']
            )
            
            # Get adaptive parameters from learning engine
            adaptations = self.learning_engine.adapt_strategy_parameters()
            
            # Generate trading signal
            signal_data = self._generate_trading_signal(
                sentiment_analysis, predicted_impact, adaptations
            )
            
            # Cache result
            self.last_analysis = signal_data
            self.last_analysis_time = now
            
            return signal_data
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'sentiment_score': 0.0,
                'reason': f'Analysis error: {e}'
            }
    
    def _generate_trading_signal(self, sentiment_analysis: Dict, 
                               predicted_impact: float, adaptations: Dict) -> Dict:
        """Generate trading signal from sentiment analysis"""
        
        sentiment_score = sentiment_analysis['overall_sentiment']
        confidence = sentiment_analysis['confidence']
        
        # Apply adaptive thresholds
        confidence_threshold = adaptations.get('confidence_threshold', 0.7)
        position_multiplier = adaptations.get('position_multiplier', 1.0)
        sentiment_weight = adaptations.get('sentiment_weight', 1.0)
        
        # Weighted sentiment score
        weighted_sentiment = sentiment_score * sentiment_weight
        
        # Signal generation logic
        if confidence < confidence_threshold:
            signal = 'hold'
            reason = f'Low confidence ({confidence:.3f} < {confidence_threshold:.3f})'
        elif weighted_sentiment > 0.15 and predicted_impact > 0.002:
            signal = 'buy'
            reason = f'Strong positive sentiment ({weighted_sentiment:.3f}) with predicted impact {predicted_impact:.4f}'
        elif weighted_sentiment < -0.15 and predicted_impact < -0.002:
            signal = 'sell'
            reason = f'Strong negative sentiment ({weighted_sentiment:.3f}) with predicted impact {predicted_impact:.4f}'
        elif abs(weighted_sentiment) > 0.1:
            signal = 'buy' if weighted_sentiment > 0 else 'sell'
            reason = f'Moderate sentiment signal ({weighted_sentiment:.3f})'
        else:
            signal = 'hold'
            reason = f'Neutral sentiment ({weighted_sentiment:.3f})'
        
        # Calculate position size (relative to normal)
        if signal != 'hold':
            position_size = min(1.0, abs(weighted_sentiment) * 2.0) * position_multiplier
        else:
            position_size = 0.0
            
        return {
            'signal': signal,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
            'weighted_sentiment': weighted_sentiment,
            'predicted_impact': predicted_impact,
            'position_size': position_size,
            'reason': reason,
            'adaptations': adaptations,
            'article_count': sentiment_analysis.get('article_count', 0),
            'sentiment_distribution': sentiment_analysis.get('sentiment_distribution', {}),
            'timestamp': time.time()
        }
    
    def update_with_market_outcome(self, sentiment_data: Dict, actual_return: float):
        """Update learning models with actual market outcome"""
        if sentiment_data and 'predicted_impact' in sentiment_data:
            # Update correlation model
            self.correlator.add_data_point(
                sentiment_data['sentiment_score'],
                actual_return,
                sentiment_data['confidence'],
                sentiment_data['timestamp']
            )
            
            # Update learning engine
            self.learning_engine.record_prediction_outcome(
                sentiment_data,
                actual_return,
                sentiment_data['predicted_impact'],
                'enhanced_news_sentiment'
            )

# Convenience function for integration
def get_enhanced_sentiment_signal() -> Dict:
    """Get sentiment signal for Account C integration"""
    strategy = EnhancedNewsStrategy()
    return strategy.get_market_sentiment_signal()

if __name__ == "__main__":
    # Test the enhanced system
    print("=== Testing Enhanced News Sentiment Analysis ===")
    
    strategy = EnhancedNewsStrategy()
    signal = strategy.get_market_sentiment_signal()
    
    print(f"\nSentiment Analysis Results:")
    print(f"Signal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.3f}")
    print(f"Sentiment Score: {signal['sentiment_score']:.3f}")
    print(f"Predicted Impact: {signal.get('predicted_impact', 0):.4f}")
    print(f"Position Size: {signal.get('position_size', 0):.3f}")
    print(f"Reason: {signal['reason']}")
    print(f"Articles Analyzed: {signal.get('article_count', 0)}")
    
    if 'sentiment_distribution' in signal:
        dist = signal['sentiment_distribution']
        print(f"Sentiment Distribution: +{dist.get('positive', 0)} ={dist.get('neutral', 0)} -{dist.get('negative', 0)}")
