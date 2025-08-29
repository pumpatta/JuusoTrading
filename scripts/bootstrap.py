from datetime import datetime, timedelta, timezone
import sys
import yaml
from pathlib import Path

# Ensure project root is on sys.path so `python scripts/bootstrap.py` can import top-level packages
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from engine.datafeed import get_bars
from strategies.xgb_classifier import XgbSignal

cfg = yaml.safe_load(Path('config/universe.yml').read_text(encoding='utf-8'))
# config/universe.yml uses a list of tickers with `symbol` and `enabled` keys
symbols = []
for t in cfg.get('tickers', []):
	sym = t.get('symbol')
	if not sym:
		continue
	if t.get('enabled', True):
		symbols.append(sym)

end = datetime.now(timezone.utc)
start = end - timedelta(days=365)

try:
	bars = get_bars(symbols, start, end, timeframe="1Min")
	anchor = 'SPY' if 'SPY' in bars else (list(bars.keys())[0] if bars else None)
	if anchor:
		xgb = XgbSignal(account='B')
		xgb.fit(bars[anchor])
	else:
		print('No bars returned; skipping model fit.')
except Exception as e:
	print('Warning: failed to download bars or fit model:', type(e).__name__, e)
	bars = {}
	anchor = None

Path('models').mkdir(exist_ok=True)
anchor_text = f'trained-on:{anchor}' if anchor else 'trained-on:unknown'
Path('models/xgb_anchor.txt').write_text(anchor_text, encoding='utf-8')

print('Bootstrap complete:', anchor_text, '-> models/xgb_anchor.txt')
