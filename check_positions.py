from engine.broker_alpaca import Broker

b = Broker(paper=True, account='B')
print('Checking positions for account B...')
try:
    if b.client:
        positions = b.client.get_all_positions()
        print('Positions via SDK:', len(positions) if positions else 0)
        for pos in positions[:5]:  # Show first 5
            avg_price = getattr(pos, 'avg_entry_price', getattr(pos, 'avg_price', 'N/A'))
            print(f'  {pos.symbol}: {pos.qty} @ {avg_price}')
            
        # Check if NVDA is in positions
        nvda_pos = next((p for p in positions if p.symbol == 'NVDA'), None)
        if nvda_pos:
            print(f'NVDA position found: {nvda_pos.qty} shares')
        else:
            print('NVDA position not found')
except Exception as e:
    print('Error getting positions:', type(e).__name__, str(e))
