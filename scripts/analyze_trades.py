import pandas as pd
from collections import deque
import os
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

fn = 'storage/intraday_backtest_trades.csv'
try:
    df = pd.read_csv(fn, parse_dates=['timestamp'])
except Exception as e:
    print('Error reading', fn, e)
    raise SystemExit(1)

if df.empty:
    print('No trades found in', fn)
    raise SystemExit(0)

summary = []
all_realized = []

# Determine if strategy column exists
has_strategy = 'strategy' in df.columns

def compute_summary(group_df):
    open_positions = deque()
    realized = []
    for _, row in group_df.sort_values('timestamp').iterrows():
        side = row['side']
        qty = int(row['qty'])
        price = float(row['price'])
        if side.upper() == 'BUY':
            open_positions.append({'qty': qty, 'price': price})
        elif side.upper().startswith('SELL'):
            remaining = qty
            while remaining > 0 and open_positions:
                pos = open_positions[0]
                take = min(pos['qty'], remaining)
                pl = (price - pos['price']) * take
                realized.append(pl)
                remaining -= take
                pos['qty'] -= take
                if pos['qty'] == 0:
                    open_positions.popleft()
    total = sum(realized)
    count = len(realized)
    wins = sum(1 for r in realized if r > 0)
    losses = sum(1 for r in realized if r <= 0)
    win_rate = wins / count if count else 0.0
    avg_pl = total / count if count else 0.0
    gross_win = sum(r for r in realized if r > 0)
    gross_loss = sum(r for r in realized if r <= 0)
    return {'trades': count, 'realized_pl': total, 'avg_pl': avg_pl, 'win_rate': win_rate, 'gross_win': gross_win, 'gross_loss': gross_loss, 'realized_list': realized}

if has_strategy:
    strategies = sorted(df['strategy'].unique())
    per_strategy_summary = []
    ensure_dir = os.makedirs
    for st in strategies:
        sdf = df[df['strategy'] == st]
        res = compute_summary(sdf)
        per_strategy_summary.append({'strategy': st, **{k: res[k] for k in ['trades','realized_pl','avg_pl','win_rate','gross_win','gross_loss']}})
        all_realized += res['realized_list']
    pst = pd.DataFrame(per_strategy_summary).sort_values('realized_pl', ascending=False)
    pst.to_csv('storage/trade_summary_per_strategy.csv', index=False)
    print('Per-strategy summary:')
    print(pst.to_string(index=False))

    # per-symbol-per-strategy
    combos = []
    for (sym, st), g in df.groupby(['symbol', 'strategy']):
        res = compute_summary(g)
        combos.append({'symbol': sym, 'strategy': st, **{k: res[k] for k in ['trades','realized_pl','avg_pl','win_rate','gross_win','gross_loss']}})
        all_realized += res['realized_list']
    out = pd.DataFrame(combos).sort_values(['realized_pl'], ascending=False)
else:
    symbols = sorted(df['symbol'].unique())
    combos = []
    for sym in symbols:
        sdf = df[df['symbol'] == sym]
        res = compute_summary(sdf)
        combos.append({'symbol': sym, **{k: res[k] for k in ['trades','realized_pl','avg_pl','win_rate','gross_win','gross_loss']}})
        all_realized += res['realized_list']
    out = pd.DataFrame(combos).sort_values(['realized_pl'], ascending=False)

# 'out' was computed above (per-symbol or per-symbol-per-strategy). Save it.
out.to_csv('storage/trade_summary_per_symbol.csv', index=False)

if HAS_PLT:
    # Simple plots -> storage/plots
    plots_dir = 'storage/plots'
    os.makedirs(plots_dir, exist_ok=True)
    try:
        # top 10 symbols by realized P&L
        top = out.head(10)
        plt.figure(figsize=(8,4))
        plt.bar(top['symbol'].astype(str), top['realized_pl'])
        plt.title('Top 10 symbols by realized P&L')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'top10_symbols_realized_pl.png'))
        plt.close()
    except Exception:
        pass
else:
    print('matplotlib not available - skipping plots')

overall = {'total_trades': len(all_realized), 'total_realized_pl': sum(all_realized), 'avg_pl': (sum(all_realized)/len(all_realized)) if all_realized else 0.0}

print('Per-symbol top 20 by realized P&L:')
print(out.head(20).to_string(index=False))
print('\nPer-symbol top 20 by trade count:')
print(out.sort_values('trades', ascending=False).head(20).to_string(index=False))
print('\nOverall:', overall)
print('\nSaved summary to storage/trade_summary_per_symbol.csv')
