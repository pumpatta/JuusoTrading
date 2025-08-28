import pandas as pd
from strategies.base import Strategy
from utils.regime import detect_regime
from utils.config import SETTINGS

class EmaTrend(Strategy):
    strategy_id = "EMA"

    def __init__(self, lookfast=50, lookslow=200, account=None):
        # Use account from parameter, SETTINGS, or default to 'A'
        self.account = account or getattr(SETTINGS, 'account', 'A')
        self.lookfast = lookfast
        self.lookslow = lookslow

    def on_bar(self, bars: dict[str, pd.DataFrame]) -> list[dict]:
        signals = []
        # Regime gating using SPY/QQQ if present
        try:
            spy = bars.get('SPY')
            qqq = bars.get('QQQ')
            if spy is not None and qqq is not None and len(spy) > 210 and len(qqq) > 210:
                regime, conf = detect_regime(spy, qqq, {k:v for k,v in bars.items() if k not in ('SPY','QQQ')})
            else:
                regime, conf = 'bull', 0.0
        except Exception:
            regime, conf = 'bull', 0.0

        for sym, df in bars.items():
            d = df.copy()
            d['ema_fast'] = d['close'].ewm(span=self.lookfast, adjust=False).mean()
            d['ema_slow'] = d['close'].ewm(span=self.lookslow, adjust=False).mean()
            if len(d) < self.lookslow + 5: continue
            last = d.iloc[-1]; prev = d.iloc[-2]
            cross_up = prev.ema_fast <= prev.ema_slow and last.ema_fast > last.ema_slow
            cross_down = prev.ema_fast >= prev.ema_slow and last.ema_fast < last.ema_slow
            price = float(last['close'])
            if cross_up and regime != 'bear':
                signals.append(dict(strategy_id=self.strategy_id, symbol=sym, side="buy", qty=1, take_profit=price*1.02, stop_loss=price*0.99))
            elif cross_down:
                signals.append(dict(strategy_id=self.strategy_id, symbol=sym, side="sell", qty=1))
        return signals
