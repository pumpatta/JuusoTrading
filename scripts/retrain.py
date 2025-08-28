from pathlib import Path
import json
import argparse
import pickle
import sys
from datetime import datetime

# Ensure project root is on sys.path so `python scripts/retrain.py` can import top-level packages
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

try:
	import torch
	_HAS_TORCH = True
except ImportError:
	_HAS_TORCH = False

from engine.datafeed import get_bars
from strategies.xgb_classifier import XgbSignal
from strategies.tcn_torch import TcnSignal


def main():
	parser = argparse.ArgumentParser(description='Retrain models with per-account support')
	parser.add_argument('--account', choices=['A', 'B', 'C'], default='A', 
	                   help='Account to train model for (A, B, or C)')
	parser.add_argument('--strategy', choices=['XGB', 'TCN'], default='XGB', 
	                   help='Strategy to retrain (XGB or TCN)')
	args = parser.parse_args()

	out_dir = Path('models')
	out_dir.mkdir(exist_ok=True)

	# load data (prefer samples for training since market may be closed)
	bars = get_bars(['SPY'], None, None, timeframe='1Min', prefer_samples=True)
	if 'SPY' not in bars:
		print('No SPY data available; aborting retrain')
		return

	spy = bars['SPY']
	print(f'Training {args.strategy} model for account {args.account}...')
	
	# Initialize the correct strategy
	if args.strategy == 'XGB':
		strategy = XgbSignal(account=args.account)
	elif args.strategy == 'TCN':
		strategy = TcnSignal(account=args.account)
	else:
		print(f'Unknown strategy: {args.strategy}')
		return
		
	# fit the model
	try:
		strategy.fit(spy)
		print(f'{args.strategy} training completed')
	except Exception as e:
		print(f'{args.strategy} fit failed:', type(e).__name__, e)
		return

	# Check fitted status after training attempt
	if not hasattr(strategy, 'fitted') or not strategy.fitted:
		print('Model fitting failed - strategy not marked as fitted, aborting save')
		return
	
	print(f'Model training successful for account {args.account}, strategy {args.strategy}')

	# save model with account prefix - strategies handle their own saving during fit()
	model_file = out_dir / f'{args.account}_{args.strategy}.model'
	
	if args.strategy == 'XGB':
		print(f'XGB model auto-saved during fit to: {model_file}')
	elif args.strategy == 'TCN':
		print(f'TCN model auto-saved during fit to: {model_file}')
	
	# Note: No manual saving here since strategies save themselves during fit()

	# save a flag file for backward compatibility
	model_flag = out_dir / f'{args.account}_{args.strategy}.trained'
	model_flag.write_text(f'trained_on:SPY,account:{args.account},timestamp:{datetime.now().isoformat()}')

	# save detailed report
	report = {
		'account': args.account,
		'strategy': args.strategy,
		'model_file': str(model_file),
		'trained_on': 'SPY',
		'n_rows': len(spy),
		'fitted': bool(strategy.fitted),
		'timestamp': datetime.now().isoformat()
	}
	report_file = out_dir / f'{args.account}_{args.strategy}_report.json'
	report_file.write_text(json.dumps(report, indent=2))
	print('Retrain complete:', report)


if __name__ == '__main__':
	main()
