import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from xgboost import XGBClassifier
from strategies.base import Strategy
from utils.regime import detect_regime
from utils.config import SETTINGS

class XgbSignal(Strategy):
    strategy_id = "XGB"

    def __init__(self, account=None):
        # Use account from parameter, SETTINGS, or default to 'A'
        self.account = account or getattr(SETTINGS, 'account', 'A')
        
        self.model = XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            device="cuda",
            use_label_encoder=False,
            eval_metric='logloss',
            objective='binary:logistic',
            base_score=0.5,
        )
        self.fitted = False
        
        # Try to load pre-trained model for this account
        self._load_model()

    def _load_model(self):
        """Load pre-trained model for this account if available"""
        model_file = Path('models') / f'{self.account}_{self.strategy_id}.model'
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                self.fitted = True
                print(f'Loaded pre-trained {self.strategy_id} model for account {self.account}')
            except Exception as e:
                print(f'Failed to load model {model_file}: {e}')
                self.fitted = False
        else:
            print(f'No pre-trained model found for account {self.account}, will train on first use')

    def save_model(self):
        """Save current model for this account"""
        if not self.fitted:
            print('Cannot save unfitted model')
            return
            
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        model_file = models_dir / f'{self.account}_{self.strategy_id}.model'
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            print(f'Model saved to: {model_file}')
        except Exception as e:
            print(f'Failed to save model: {e}')

    def _features(self, d: pd.DataFrame) -> pd.DataFrame:
        df = d.copy()
        df['ret1'] = df['close'].pct_change()
        df['vol'] = df['ret1'].rolling(20).std()
        df['mom'] = df['close'].pct_change(10)
        up = df['ret1'].clip(lower=0).rolling(14).mean()
        down = (-df['ret1'].clip(upper=0)).rolling(14).mean()
        rs = (up / (down + 1e-9)).fillna(1.0)
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def fit(self, d: pd.DataFrame):
        df = self._features(d)
        y = (df['close'].shift(-5) > df['close']).astype(int)
        X = df[['ret1','vol','mom','rsi']].fillna(0)
        mask = ~y.isna()
        X, y = X[mask], y[mask]
        if len(y) == 0:
            print('XgbSignal.fit: no training samples, skipping fit')
            self.fitted = False
            return
        if y.nunique() < 2:
            print('XgbSignal.fit: only one class present in y, skipping fit')
            self.fitted = False
            return
        try:
            self.model.fit(X, y)
            self.fitted = True
            # Auto-save model after successful training
            self.save_model()
        except Exception as e:
            print('XgbSignal.fit failed:', type(e).__name__, e)
            self.fitted = False

    def on_bar(self, bars: dict[str, pd.DataFrame]) -> list[dict]:
        signals = []
        # Regime gating
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
            if len(df) < 300: continue
            if not self.fitted: self.fit(df)
            feats = self._features(df).iloc[-1:][['ret1','vol','mom','rsi']].fillna(0)
            proba = float(self.model.predict_proba(feats)[0,1])
            price = float(df.iloc[-1]['close'])
            if proba > 0.56 and regime != 'bear':
                # Calculate position size based on available capital and limits
                max_pos_value = float(os.getenv('INITIAL_CAPITAL', '100000')) * 0.03  # 3% of capital
                qty = max(1, int(max_pos_value / price))  # At least 1 share
                signals.append(dict(strategy_id=self.strategy_id, symbol=sym, side="buy", qty=qty, take_profit=price*1.015, stop_loss=price*0.99))
            elif proba < 0.44:
                signals.append(dict(strategy_id=self.strategy_id, symbol=sym, side="sell", qty=1))
        return signals
