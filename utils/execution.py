import random
from dataclasses import dataclass
from pathlib import Path
import yaml

CFG = yaml.safe_load(Path('config/execution.yml').read_text(encoding='utf-8'))

@dataclass
class Fill:
    qty: float
    price: float

def _spread(price: float) -> float:
    mode = CFG.get('spread_mode', 'fixed_cents')
    val = float(CFG.get('spread_value', 0.01))
    if mode == 'bps_of_price':
        return price * (val / 1e4)  # bps -> fraction
    return val  # cents

def _extra_slippage(price: float) -> float:
    bps = float(CFG.get('extra_slippage_bps', 0.0))
    return price * (bps / 1e4)

def simulate_fills(side: str, wanted_qty: float, nbbo_mid: float) -> list[Fill]:
    """Palauttaa listan simuloituja fillejä (qty, price).
    - Lisää spread: buy maksaa (mid + spread/2 + extra), sell saa (mid - spread/2 - extra)
    - Osittaisuus: pienen todennäköisyyden mukaan jaetaan kahteen erään
    """
    sp = _spread(nbbo_mid)
    ex = _extra_slippage(nbbo_mid)
    if side.lower() == 'buy':
        px = nbbo_mid + sp/2 + ex
    else:
        px = nbbo_mid - sp/2 - ex

    fills = []
    if random.random() < float(CFG.get('partial_fill_prob', 0.0)):
        min_frac = float(CFG.get('partial_fill_min_frac', 0.3))
        f1 = max(min_frac, random.random())
        q1 = max(1.0, round(wanted_qty * f1))
        q2 = max(0.0, wanted_qty - q1)
        if q2 > 0: 
            fills.append(Fill(qty=q1, price=px))
            # toiseen erään lisätään pieni lisäliuku suuntaan
            px2 = px * (1 + (0.0005 if side.lower() == 'buy' else -0.0005))
            fills.append(Fill(qty=q2, price=px2))
        else:
            fills.append(Fill(qty=wanted_qty, price=px))
    else:
        fills.append(Fill(qty=wanted_qty, price=px))
    return fills
