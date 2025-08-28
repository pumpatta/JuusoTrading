from datetime import datetime, timedelta
import random
from pathlib import Path
import csv

def generate_spy(path: Path, rows: int = 2000, start_price: float = 450.0):
    end = datetime.utcnow().replace(second=0, microsecond=0)
    start = end - timedelta(minutes=rows-1)
    ts = start
    price = start_price
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ts','open','high','low','close','volume'])
        for _ in range(rows):
            # simple random walk with small drift and intraminute range
            open_p = price
            change = random.normalvariate(0, 0.05)
            close_p = max(0.1, open_p + change)
            high_p = max(open_p, close_p) + abs(random.normalvariate(0, 0.03))
            low_p = min(open_p, close_p) - abs(random.normalvariate(0, 0.03))
            vol = random.randint(200000, 2000000)
            writer.writerow([ts.strftime('%Y-%m-%d %H:%M:%S'), f"{open_p:.4f}", f"{high_p:.4f}", f"{low_p:.4f}", f"{close_p:.4f}", vol])
            price = close_p
            ts += timedelta(minutes=1)

if __name__ == '__main__':
    p = Path('storage/sample_bars')
    p.mkdir(parents=True, exist_ok=True)
    out = p / 'SPY.csv'
    print('Generating', out)
    generate_spy(out, rows=2000, start_price=450.0)
    print('Done')
