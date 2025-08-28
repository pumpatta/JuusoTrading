#!/usr/bin/env python3
"""
Generate sample trade data for dashboard testing
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_sample_trades():
    """Generate sample trades for testing dashboard"""
    
    # Ensure logs directory exists
    logs_dir = Path('storage/logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample symbols and strategies
    symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']
    strategies = ['EMA', 'XGB', 'ACCOUNT_C_ML']
    
    base_time = datetime.now() - timedelta(days=7)
    
    for strategy in strategies:
        trades = []
        current_time = base_time
        
        # Generate 20-30 trades per strategy
        num_trades = random.randint(20, 30)
        
        for i in range(num_trades):
            # Random time increment (minutes to hours)
            time_delta = timedelta(
                minutes=random.randint(30, 480)  # 30 min to 8 hours
            )
            current_time += time_delta
            
            symbol = random.choice(symbols)
            side = random.choice(['buy', 'sell'])
            qty = random.randint(1, 10)
            
            # Price based on symbol (roughly realistic)
            base_prices = {
                'SPY': 450,
                'QQQ': 380,
                'AAPL': 180,
                'MSFT': 340,
                'NVDA': 460
            }
            
            base_price = base_prices.get(symbol, 100)
            price = base_price + random.uniform(-5, 5)
            
            trades.append({
                'ts': current_time.isoformat(),
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'price': round(price, 2)
            })
        
        # Write to CSV
        trade_file = logs_dir / f'trades_{strategy}.csv'
        with open(trade_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ts', 'symbol', 'side', 'qty', 'price'])
            writer.writeheader()
            writer.writerows(trades)
        
        print(f'✅ Luotu {len(trades)} testikauppaa strategialle {strategy}')

if __name__ == "__main__":
    print("📊 Generoidaan testikauppoja dashboardia varten...")
    generate_sample_trades()
    print("🎉 Testikaupat luotu! Voit nyt testata dashboardia.")
    print("Käynnistä dashboard: python launch_dashboard.py")
