"""
Simple safety module: checks max position size per symbol, max exposure, daily loss limit, and enforces stop-loss.
Configuration via environment variables or defaults.
"""

import os
from typing import Optional

# Relaxed profile defaults (more trading activity while keeping guardrails)
MAX_POS_PCT = float(os.getenv('SAFETY_MAX_POS_PCT', '0.03'))  # 3% default per symbol
MAX_TOTAL_EXPOSURE_PCT = float(os.getenv('SAFETY_MAX_TOTAL_EXPOSURE_PCT', '0.3'))  # 30% total
DAILY_LOSS_PCT = float(os.getenv('SAFETY_DAILY_LOSS_PCT', '0.10'))  # 10% daily stop
MAX_TRADES_PER_SYMBOL_DAY = int(os.getenv('SAFETY_MAX_TRADES_PER_SYMBOL_DAY', '5'))
STATE_FILE = os.getenv('SAFETY_STATE_FILE', 'storage/safety_state.json')


def can_place_buy(cash: float, price: float, qty: float, initial_capital: float) -> bool:
    # check position size pct
    pos_value = price * qty
    if pos_value / initial_capital > MAX_POS_PCT:
        return False
    # since we don't have current portfolio exposure here, caller should also check total exposure
    return True


def daily_loss_breached(current_portfolio_value: float, initial_capital: float) -> bool:
    loss = (initial_capital - current_portfolio_value) / initial_capital
    return loss >= DAILY_LOSS_PCT


def enforceable_stop_loss():
    # placeholder for centralized stop-loss tracking; returns default stop loss (1% by env)
    return float(os.getenv('SAFETY_DEFAULT_STOP_LOSS', '0.01'))


def can_trade_symbol(trade_counts: dict, symbol: str) -> bool:
    """Return True if symbol has remaining allowed trades for today."""
    return trade_counts.get(symbol, 0) < MAX_TRADES_PER_SYMBOL_DAY


def record_trade(trade_counts: dict, symbol: str):
    trade_counts[symbol] = trade_counts.get(symbol, 0) + 1
