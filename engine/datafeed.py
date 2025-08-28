from datetime import datetime
from typing import List, Optional
import pandas as pd
from pathlib import Path
import time

from utils.config import SETTINGS

# Alpaca imports are optional for offline/testing. Import lazily to allow running without credentials.
try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    _HAS_ALPACA = True
except Exception:
    _HAS_ALPACA = False

# Global state to avoid repeated SIP errors
_SIP_ERROR_DETECTED = False
_SAMPLE_CACHE = {}
_CACHE_TIMESTAMP = 0
_CACHE_DURATION = 300  # 5 minutes


def _load_samples(symbols: List[str]) -> dict[str, pd.DataFrame]:
    out = {}
    base = Path('storage/sample_bars')
    for s in symbols:
        p = base / f"{s}.csv"
        if p.exists():
            try:
                df = pd.read_csv(p, parse_dates=['ts'])
                out[s] = df
            except Exception:
                continue
    return out


def get_bars(symbols: List[str], start: Optional[datetime], end: Optional[datetime], timeframe: str = "1Min", prefer_samples: bool = False) -> dict[str, pd.DataFrame]:
    """Return a dict of symbol->DataFrame with a `ts` column.

    If prefer_samples=True, tries sample data first (useful for training/backtesting).
    Otherwise tries Alpaca first (if available), then falls back to CSVs in `storage/sample_bars/`.
    
    Uses caching and SIP error detection to avoid repeated API failures.
    """
    global _SIP_ERROR_DETECTED, _SAMPLE_CACHE, _CACHE_TIMESTAMP
    
    current_time = time.time()
    
    # If we already detected SIP error, use samples directly
    if _SIP_ERROR_DETECTED:
        # Check cache first
        if _SAMPLE_CACHE and (current_time - _CACHE_TIMESTAMP) < _CACHE_DURATION:
            print('Using cached sample bars (SIP unavailable):', ','.join(_SAMPLE_CACHE.keys()))
            return _SAMPLE_CACHE
            
        samples = _load_samples(symbols)
        if samples:
            _SAMPLE_CACHE = samples
            _CACHE_TIMESTAMP = current_time
            print('Loading fresh sample bars (SIP unavailable):', ','.join(samples.keys()))
            return samples
    
    # If prefer_samples is True, try samples first (good for training when market is closed)
    if prefer_samples:
        samples = _load_samples(symbols)
        if samples:
            print('Using sample bars for:', ','.join(samples.keys()))
            return samples
    
    # Try Alpaca if client is available and seems configured
    if _HAS_ALPACA and SETTINGS.key_id and not prefer_samples and not _SIP_ERROR_DETECTED:
        try:
            tf = TimeFrame.Minute if timeframe == "1Min" else TimeFrame.Day
            client = StockHistoricalDataClient(SETTINGS.key_id, SETTINGS.secret_key)
            req = StockBarsRequest(symbol_or_symbols=symbols, start=start, end=end, timeframe=tf)
            bars = client.get_stock_bars(req)
            out = {}
            
            # Modern Alpaca API: response has .data attribute with symbol->list mapping
            # Using getattr to handle dynamic API responses safely
            if hasattr(bars, 'data') and getattr(bars, 'data', None):
                data = getattr(bars, 'data')  # type: ignore
                for sym in symbols:
                    if sym in data and data[sym]:
                        # Convert list of Bar objects to DataFrame
                        bar_data = []
                        for bar in data[sym]:
                            bar_data.append({
                                'ts': getattr(bar, 'timestamp', None),
                                'open': getattr(bar, 'open', None),
                                'high': getattr(bar, 'high', None),
                                'low': getattr(bar, 'low', None),
                                'close': getattr(bar, 'close', None),
                                'volume': getattr(bar, 'volume', None)
                            })
                        if bar_data:
                            df = pd.DataFrame(bar_data)
                            out[sym] = df
            
            # Only return Alpaca data if we got results for requested symbols
            if out and any(sym in out for sym in symbols):
                print('Using live Alpaca data for:', ','.join(out.keys()))
                return out
            else:
                print('Alpaca returned no data for requested symbols, falling back to samples')
        except Exception as e:
            error_msg = str(e)
            # Detect SIP subscription error and avoid retrying
            if "subscription does not permit querying recent SIP data" in error_msg:
                print('SIP data subscription unavailable - switching to sample data mode')
                _SIP_ERROR_DETECTED = True
            else:
                print('Alpaca historical bars failed:', type(e).__name__, e)

    # Fallback: load sample CSVs from storage/sample_bars/
    samples = _load_samples(symbols)
    if samples:
        # Cache the samples if SIP error detected
        if _SIP_ERROR_DETECTED:
            _SAMPLE_CACHE = samples
            _CACHE_TIMESTAMP = current_time
        print('Loaded sample bars for:', ','.join(samples.keys()))
        return samples

    # Nothing available
    raise RuntimeError('No market data available: Alpaca failed or not configured, and no sample bars found.')
