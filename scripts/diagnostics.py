"""Simple diagnostics harness.

Runs quick import checks, model loader checks, and an optional smoke test.
"""
import sys
import importlib
from pathlib import Path
import subprocess

MODULES = [
    'engine.live',
    'engine.live_offline',
    'engine.portfolio',
    'engine.broker_alpaca',
    'strategies.xgb_classifier',
    'strategies.tcn_torch',
    'utils.execution',
]

print('Running import checks...')
for m in MODULES:
    try:
        importlib.import_module(m)
        print('OK:', m)
    except Exception as e:
        print('FAIL:', m, '->', type(e).__name__, e)

print('\nRunning model loader check...')
try:
    subprocess.run([sys.executable, 'scripts/check_models_loadable.py'], check=False)
except Exception as e:
    print('Failed to run check_models_loadable:', e)

# optional: run one smoke test if pytest is available
print('\nOptional: run smoke tests (skipped if pytest not installed)')
try:
    import pytest
    print('pytest available; running tests/smoke/test_routing.py (quiet)')
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/smoke/test_routing.py'], check=False)
except Exception:
    print('pytest not available; skipping unit tests')

print('\nDiagnostics complete')
