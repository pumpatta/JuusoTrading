"""Create per-account logging and metrics directories for JuusoTrader.

Usage:
    python scripts/create_account_dirs.py

This creates:
- storage/logs/A
- storage/logs/B
- storage/logs/C
- storage/metrics
"""
from pathlib import Path

def main():
    base = Path('storage')
    logs = base / 'logs'
    metrics = base / 'metrics'
    accounts = ['A','B','C']
    logs.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    for a in accounts:
        (logs / a).mkdir(parents=True, exist_ok=True)
    print('Created directories:')
    print(' -', logs)
    for a in accounts:
        print('   -', logs / a)
    print(' -', metrics)

if __name__ == '__main__':
    main()
