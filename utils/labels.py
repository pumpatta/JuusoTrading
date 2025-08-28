import numpy as np
import pandas as pd

def triple_barrier_labels(close: pd.Series, horizon: int = 20, up_mult: float = 1.0, dn_mult: float = 1.0):
    vol = close.pct_change().rolling(50).std().bfill().replace(0, np.nan).ffill()
    up_th = close * (1 + up_mult * vol)
    dn_th = close * (1 - dn_mult * vol)
    n = len(close)
    y = pd.Series(index=close.index, dtype="float32")
    for i in range(n):
        end = min(i + horizon, n - 1)
        c0 = close.iloc[i]
        hit = 0
        for j in range(i+1, end+1):
            if close.iloc[j] >= up_th.iloc[i]:
                hit = 1; break
            if close.iloc[j] <= dn_th.iloc[i]:
                hit = -1; break
        if hit == 0:
            hit = np.sign(close.iloc[end] - c0)
        y.iloc[i] = int(hit > 0)  # Convert boolean to int directly
    return y

def purged_cv_folds(n, n_splits=5, embargo=10):
    idx = np.arange(n)
    fold_size = n // n_splits
    folds = []
    for k in range(n_splits):
        start = k*fold_size
        stop = (k+1)*fold_size if k < n_splits-1 else n
        test_idx = idx[start:stop]
        train_mask = np.ones(n, dtype=bool)
        train_mask[max(0, start-embargo):min(n, stop+embargo)] = False
        train_idx = idx[train_mask]
        folds.append((train_idx, test_idx))
    return folds
