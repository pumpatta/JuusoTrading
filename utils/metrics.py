import numpy as np
import pandas as pd

def sharpe(returns, rf=0.0, scale=252):
    r = np.asarray(returns) - rf/scale
    if r.std() == 0: return 0.0
    return np.sqrt(scale) * r.mean() / r.std()

def sortino(returns, rf=0.0, scale=252):
    r = np.asarray(returns) - rf/scale
    downside = r[r<0]
    ds = downside.std() if len(downside)>0 else 0
    if ds == 0: return 0.0
    return np.sqrt(scale) * r.mean() / ds

def max_drawdown(equity: pd.Series):
    roll_max = equity.cummax()
    dd = equity/roll_max - 1.0
    return float(dd.min())
