import importlib, sys
mods = [
    'alpaca',
    'alpaca.data',
    'alpaca.trading',
    'alpaca.data.historical.stock',
    'alpaca.trading.client',
    'requests',
    'torch'
]
for m in mods:
    try:
        importlib.import_module(m)
        print(m + ': OK')
    except Exception as e:
        print(m + ': MISSING or failed -> ' + type(e).__name__ + ': ' + str(e))
