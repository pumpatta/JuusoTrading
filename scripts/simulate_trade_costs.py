import pandas as pd
import os

FN = 'storage/intraday_backtest_trades.csv'
OUT_DIR = 'storage'

def compute_realized_with_fee(df, fee_rate):
    # group by symbol and strategy if present
    by_cols = ['symbol'] + (['strategy'] if 'strategy' in df.columns else [])
    summary = []
    all_realized = []
    for name, g in df.groupby(by_cols):
        open_positions = []
        realized = []
        for _, row in g.sort_values('timestamp').iterrows():
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
                    buy_price = pos['price']
                    sell_price = price
                    pl = (sell_price - buy_price) * take
                    # subtract fees on both sides
                    fee = fee_rate * (buy_price + sell_price) * take
                    pl_after_fee = pl - fee
                    realized.append(pl_after_fee)
                    remaining -= take
                    pos['qty'] -= take
                    if pos['qty'] == 0:
                        open_positions.pop(0)
        total = sum(realized)
        count = len(realized)
        summary.append((name, count, total))
        all_realized += realized
    return summary, all_realized

def main():
    if not os.path.exists(FN):
        print('No trades file:', FN)
        return
    df = pd.read_csv(FN, parse_dates=['timestamp'])
    fee_rates = [0.0005, 0.001, 0.002]  # 0.05%, 0.1%, 0.2%
    results = []
    for fr in fee_rates:
        summary, all_realized = compute_realized_with_fee(df, fr)
        total_realized = sum(all_realized)
        results.append({'fee_rate': fr, 'total_realized_pl': total_realized, 'total_trades': len(all_realized)})
        # save per-strategy summary if strategy exists
        has_strategy = 'strategy' in df.columns
        rows = []
        for name, count, total in summary:
            if has_strategy:
                sym, st = name
                rows.append({'symbol': sym, 'strategy': st, 'trades': count, 'realized_pl_after_fees': total})
            else:
                sym = name
                rows.append({'symbol': sym, 'trades': count, 'realized_pl_after_fees': total})
        out_df = pd.DataFrame(rows)
        out_fn = os.path.join(OUT_DIR, f'trade_summary_with_fees_{int(fr*10000)}bps.csv')
        out_df.to_csv(out_fn, index=False)
        print('Saved', out_fn)

    print('\nFee scenarios summary:')
    for r in results:
        print(f"fee {r['fee_rate']*100:.3f}% -> total_realized_pl {r['total_realized_pl']:.2f} over {r['total_trades']} trades")

if __name__ == '__main__':
    main()
