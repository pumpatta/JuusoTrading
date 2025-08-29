import yaml
try:
    with open('config/universe.yml', 'r', encoding='utf-8') as f:
        universe_cfg = yaml.safe_load(f)
    universe_symbols = []
    for t in universe_cfg.get('tickers', []):
        sym = t.get('symbol')
        if not sym:
            continue
        if t.get('enabled', True):
            universe_symbols.append(sym)
    print(f'Loaded {len(universe_symbols)} symbols: {universe_symbols[:10]}')
except Exception as e:
    print(f'Error: {e}')
