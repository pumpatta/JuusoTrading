from abc import ABC, abstractmethod
from typing import Dict, Any

class Strategy(ABC):
    strategy_id: str

    @abstractmethod
    def on_bar(self, bars: Dict[str, Any]) -> list[dict]:
        ...
