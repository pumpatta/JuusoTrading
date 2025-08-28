from strategies.base import Strategy
from strategies.ema_trend import EmaTrend
from strategies.xgb_classifier import XgbSignal
from utils.config import SETTINGS

class Ensemble(Strategy):
    strategy_id = "ENSEMBLE"
    
    def __init__(self, account=None):
        # Use account from parameter, SETTINGS, or default to 'C'
        self.account = account or getattr(SETTINGS, 'account', 'C')
        
        # Initialize component strategies
        self.ema_strategy = EmaTrend(account=self.account)
        self.xgb_strategy = XgbSignal(account=self.account)
        
    def on_bar(self, bars: dict) -> list[dict]:
        """
        Ensemble strategy that combines EMA and XGB signals
        Only generates signals when both strategies agree
        """
        signals = []
        
        # Get signals from component strategies
        ema_signals = self.ema_strategy.on_bar(bars)
        xgb_signals = self.xgb_strategy.on_bar(bars)
        
        # Create signal lookup by symbol
        ema_by_symbol = {s['symbol']: s for s in ema_signals}
        xgb_by_symbol = {s['symbol']: s for s in xgb_signals}
        
        # Find symbols where both strategies agree
        common_symbols = set(ema_by_symbol.keys()) & set(xgb_by_symbol.keys())
        
        for symbol in common_symbols:
            ema_signal = ema_by_symbol[symbol]
            xgb_signal = xgb_by_symbol[symbol]
            
            # Only generate signal if both agree on direction
            if ema_signal['side'] == xgb_signal['side']:
                # Use XGB's price levels but EMA's direction confirmation
                ensemble_signal = {
                    'strategy_id': self.strategy_id,
                    'symbol': symbol,
                    'side': ema_signal['side'],
                    'qty': min(ema_signal.get('qty', 1), xgb_signal.get('qty', 1)),
                    'reason': f"ENSEMBLE_{ema_signal['side']}_EMA+XGB_consensus"
                }
                
                # Add XGB's take_profit and stop_loss if available
                if 'take_profit' in xgb_signal:
                    ensemble_signal['take_profit'] = xgb_signal['take_profit']
                if 'stop_loss' in xgb_signal:
                    ensemble_signal['stop_loss'] = xgb_signal['stop_loss']
                
                signals.append(ensemble_signal)
        
        # If no consensus, try weaker signals (only XGB with high confidence)
        if not signals:
            for xgb_signal in xgb_signals:
                symbol = xgb_signal['symbol']
                # If XGB is very confident and no EMA conflict
                if symbol not in ema_by_symbol:
                    # Add ensemble signal with reduced position size
                    ensemble_signal = {
                        'strategy_id': self.strategy_id,
                        'symbol': symbol,
                        'side': xgb_signal['side'],
                        'qty': max(1, xgb_signal.get('qty', 1) // 2),  # Half size
                        'reason': f"ENSEMBLE_{xgb_signal['side']}_XGB_solo"
                    }
                    
                    if 'take_profit' in xgb_signal:
                        ensemble_signal['take_profit'] = xgb_signal['take_profit']
                    if 'stop_loss' in xgb_signal:
                        ensemble_signal['stop_loss'] = xgb_signal['stop_loss']
                    
                    signals.append(ensemble_signal)
        
        return signals
