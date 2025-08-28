import pandas as pd

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def breadth_above_ma(universe: dict[str, pd.DataFrame], ma: int = 50) -> float:
    if not universe: return 0.0
    cnt = 0
    tot = 0
    for sym, df in universe.items():
        if len(df) < ma + 1: 
            continue
        ma_val = df['close'].rolling(ma).mean().iloc[-1]
        if pd.notna(ma_val) and df['close'].iloc[-1] > ma_val:
            cnt += 1
        tot += 1
    return (cnt / tot) if tot else 0.0

def detect_regime(spy: pd.DataFrame, qqq: pd.DataFrame, universe: dict[str, pd.DataFrame]|None=None) -> tuple[str, float]:
    """Return ('bull'|'bear', confidence 0..1). Heuristic:
    - Bull if both SPY and QQQ above EMA200 and breadth>0.5
    - Bear if both below EMA200 and breadth<0.4
    - Else neutral; mapped to weaker bull/bear by EMA slope
    """
    assert len(spy) > 200 and len(qqq) > 200, "Need >200 bars for regime"
    spy_ema = ema(spy['close'], 200)
    qqq_ema = ema(qqq['close'], 200)
    b = breadth_above_ma(universe or {}, 50)
    spy_above = spy['close'].iloc[-1] > spy_ema.iloc[-1]
    qqq_above = qqq['close'].iloc[-1] > qqq_ema.iloc[-1]
    spy_slope = (spy_ema.iloc[-1] - spy_ema.iloc[-6]) / 5
    qqq_slope = (qqq_ema.iloc[-1] - qqq_ema.iloc[-6]) / 5

    if spy_above and qqq_above and b > 0.5:
        return 'bull', min(1.0, 0.6 + 0.4*b)
    if (not spy_above) and (not qqq_above) and b < 0.4:
        return 'bear', min(1.0, 0.6 + 0.4*(1-b))

    # fallback by slope
    slope_score = (spy_slope + qqq_slope) / max(abs(spy_ema.iloc[-1]) + abs(qqq_ema.iloc[-1]), 1e-9)
    if slope_score > 0:
        return 'bull', 0.5
    else:
        return 'bear', 0.5
