import pandas as pd
import numpy as np
import os
from strategies.base import Strategy
from utils.regime import detect_regime

def pivots(price: pd.Series, pct: float = 3.0):
    piv = [0]
    direction = 0
    last_p = price.iloc[0]
    for i, p in enumerate(price.iloc[1:], start=1):
        change = (p - last_p) / last_p * 100.0
        if direction >= 0 and change <= -pct:
            piv.append(i-1); direction = -1; last_p = price.iloc[i-1]
        elif direction <= 0 and change >= pct:
            piv.append(i-1); direction = 1; last_p = price.iloc[i-1]
    piv.append(len(price)-1)
    return sorted(set(piv))

def is_head_shoulders(price: pd.Series, idxs) -> bool:
    if len(idxs) < 5: return False
    last5 = idxs[-5:]
    pts = price.iloc[last5].values
    head_i = np.argmax(pts)
    if head_i != 2: return False
    ls, head, rs = pts[1], pts[2], pts[3]
    if not (head > ls and head > rs): return False
    if abs(ls - rs) / head > 0.03: return False
    return True

class HeadShoulders(Strategy):
    strategy_id = "HNS"
    def on_bar(self, bars: dict[str, pd.DataFrame]) -> list[dict]:
        sigs = []
        # Prefer acting in bear regime
        try:
            spy = bars.get('SPY'); qqq = bars.get('QQQ')
            if spy is not None and qqq is not None and len(spy) > 210 and len(qqq) > 210:
                regime, conf = detect_regime(spy, qqq, {k:v for k,v in bars.items() if k not in ('SPY','QQQ')})
            else:
                regime, conf = 'bull', 0.0
        except Exception:
            regime, conf = 'bull', 0.0

        for sym, df in bars.items():
            if len(df) < 200: continue
            p = df['close']
            idxs = pivots(p, pct=2.0)  # Lowered from 3.0 to 2.0 for more sensitivity
            
            # Look for head and shoulders pattern
            if is_head_shoulders(p, idxs):
                price = float(p.iloc[-1])
                # Generate sell signal if pattern detected (regardless of regime for now)
                sigs.append(dict(strategy_id=self.strategy_id, symbol=sym, side="sell", qty=1))
            
            # Also look for inverted head and shoulders (buy signal)
            elif len(idxs) >= 5:
                last5 = idxs[-5:]
                pts = np.array(p.iloc[np.array(last5)].values)  # Ensure numpy array
                head_i = np.argmin(pts)  # Look for minimum instead of maximum
                if head_i == 2:  # Middle point is the lowest
                    ls, head, rs = pts[1], pts[2], pts[3]
                    if head < ls and head < rs:  # Inverted head and shoulders
                        if abs(ls - rs) / abs(head) < 0.03:  # Shoulders roughly equal
                            price = float(p.iloc[-1])
                            # Calculate position size based on available capital and limits
                            max_pos_value = float(os.getenv('INITIAL_CAPITAL', '100000')) * 0.03  # 3% of capital
                            qty = max(1, int(max_pos_value / price))  # At least 1 share
                            sigs.append(dict(strategy_id=self.strategy_id, symbol=sym, side="buy", qty=qty))
        return sigs
