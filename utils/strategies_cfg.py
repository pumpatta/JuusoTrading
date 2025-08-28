import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class StratCfg:
    id: str
    label: str  # A/B/C
    enabled: bool
    capital_pct: float

@dataclass
class StratPlan:
    items: List[StratCfg]
    nav_start: float

def load_plan(path: str = 'config/strategies.yml') -> StratPlan:
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    items = [StratCfg(**s) for s in data['strategies']]
    return StratPlan(items=items, nav_start=float(data.get('nav_start', 100000)))
