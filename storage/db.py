from sqlalchemy import create_engine, text
from pathlib import Path

DB_PATH = Path("storage/trader.sqlite").as_posix()
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

def init_db():
    with engine.begin() as con:
        con.execute(text('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL,
            avg_price REAL NOT NULL,
            opened_at TEXT NOT NULL
        );
        '''))
        con.execute(text('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            ts TEXT NOT NULL,
            client_order_id TEXT
        );
        '''))
        con.execute(text('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            date TEXT NOT NULL,
            pnl REAL,
            equity REAL
        );
        '''))
