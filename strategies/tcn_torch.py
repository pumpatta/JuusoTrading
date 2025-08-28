import torch, torch.nn as nn
# Disable torch.compile for compatibility
torch._dynamo.config.disable = True
import pandas as pd
import numpy as np
from pathlib import Path
from strategies.base import Strategy
from utils.regime import detect_regime
from utils.labels import triple_barrier_labels
from utils.config import SETTINGS

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TCNModel(nn.Module):
    def __init__(self, in_ch=4, k=5, layers=3, p=0.1):
        super().__init__()
        ch = 32
        mods = []
        for i in range(layers):
            pad = k-1
            mods += [nn.Conv1d(in_ch if i==0 else ch, ch, k, padding=pad), Chomp1d(pad), nn.ReLU(), nn.Dropout(p)]
        self.tcn = nn.Sequential(*mods)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, 1))
    def forward(self, x):
        return self.head(self.tcn(x))

class TcnSignal(Strategy):
    strategy_id = "TCN"
    def __init__(self, horizon=20, device=None, account=None):
        # Use account from parameter, SETTINGS, or default to 'B' (TCN belongs to account B)
        self.account = account or getattr(SETTINGS, 'account', 'B')
        self.horizon = horizon
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TCNModel().to(self.device)
        self.fitted = False
        
        # Try to load pre-trained model for this account
        self._load_model()

    def _load_model(self):
        """Load pre-trained model for this account if available"""
        model_file = Path('models') / f'{self.account}_{self.strategy_id}.model'
        if model_file.exists():
            try:
                state_dict = torch.load(model_file, map_location=self.device)
                
                # Handle models saved with torch.compile (have _orig_mod prefix)
                if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                    # Remove _orig_mod prefix
                    cleaned_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('_orig_mod.'):
                            cleaned_key = key[10:]  # Remove '_orig_mod.' prefix
                            cleaned_state_dict[cleaned_key] = value
                        else:
                            cleaned_state_dict[key] = value
                    state_dict = cleaned_state_dict
                
                self.model.load_state_dict(state_dict)
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
            torch.save(self.model.state_dict(), model_file)
            print(f'Model saved to: {model_file}')
        except Exception as e:
            print(f'Failed to save model: {e}')
        
        # Try to compile model for optimization (optional)
        try: 
            self.model = torch.compile(self.model)
        except Exception: 
            pass
    def _features(self, d: pd.DataFrame) -> pd.DataFrame:
        df = d.copy()
        df['ret1'] = df['close'].pct_change()
        df['vol'] = df['ret1'].rolling(20).std()
        df['hl'] = (df['high'] - df['low']) / df['close']
        df['oc'] = (df['close'] - df['open']) / df['open']
        return df[['ret1','vol','hl','oc']].fillna(0)
    def fit(self, d: pd.DataFrame):
        try:
            Xdf = self._features(d)
            y = triple_barrier_labels(d['close'], horizon=self.horizon)
            X = torch.tensor(Xdf.values.T[None, ...], dtype=torch.float32, device=self.device)  # [1,C,T]
            yv = torch.tensor(y.values[-1:], dtype=torch.float32, device=self.device)  # [1]
            opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            lossf = nn.BCEWithLogitsLoss()
            self.model.train()
            for _ in range(50):
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=("cuda" if self.device=="cuda" else "cpu"), enabled=(self.device=="cuda")):
                    out = self.model(X)  # [1, 1]
                    loss = lossf(out.squeeze(-1), yv)  # Squeeze last dimension: [1, 1] -> [1]
                loss.backward(); opt.step()
            self.fitted = True
            # Auto-save model after successful training
            self.save_model()
            print(f'TCN training completed successfully for account {self.account}')
        except Exception as e:
            print(f'TCN fit failed: {type(e).__name__}: {e}')
            self.fitted = False
            import traceback
            traceback.print_exc()
    def on_bar(self, bars: dict[str, pd.DataFrame]) -> list[dict]:
        sigs = []
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
            if len(df) < 400: continue
            if not self.fitted: self.fit(df)
            Xdf = self._features(df).iloc[-256:]
            X = torch.tensor(Xdf.values.T[None, ...], dtype=torch.float32, device=self.device)
            self.model.eval()
            with torch.no_grad(), torch.autocast(device_type=("cuda" if self.device=="cuda" else "cpu"), enabled=(self.device=="cuda")):
                logit = self.model(X).item()
            prob = 1/(1+np.exp(-logit)); price = float(df.iloc[-1]['close'])
            if prob > 0.56: sigs.append(dict(strategy_id=self.strategy_id, symbol=sym, side="buy", qty=1, take_profit=price*1.015, stop_loss=price*0.99))
            elif prob < 0.44: sigs.append(dict(strategy_id=self.strategy_id, symbol=sym, side="sell", qty=1))
        return sigs
