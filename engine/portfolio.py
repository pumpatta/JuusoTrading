from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Position:
    qty: float
    avg_price: float

@dataclass
class StrategyState:
    positions: Dict[str, Position] = field(default_factory=dict)

class StrategyBook:
    def __init__(self):
        self._state: Dict[str, StrategyState] = {}

    def get_state(self, strategy_id: str) -> StrategyState:
        if strategy_id not in self._state:
            self._state[strategy_id] = StrategyState()
        return self._state[strategy_id]

    def update_on_fill(self, strategy_id: str, symbol: str, side: str, qty: float, price: float):
        st = self.get_state(strategy_id)
        pos = st.positions.get(symbol, Position(0, 0.0))
        if side.lower() == "buy":
            new_qty = pos.qty + qty
            pos.avg_price = (pos.avg_price * pos.qty + price * qty) / new_qty if new_qty else price
            pos.qty = new_qty
        else:
            pos.qty -= qty
            if pos.qty <= 0:
                st.positions.pop(symbol, None)
                return
        st.positions[symbol] = pos
