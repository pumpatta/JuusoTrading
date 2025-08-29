from datetime import datetime
from typing import List, Optional
import pandas as pd
from pathlib import Path
import time
import requests
from typing import Dict

from utils.config import SETTINGS

# Alpaca imports are optional for offline/testing. Import lazily to allow running without credentials.
try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    _HAS_ALPACA = True
except Exception:
    _HAS_ALPACA = False

# Optional yfinance fallback for near-live bars when Alpaca SIP is not available
try:
    import yfinance as _yfinance
    _HAS_YFINANCE = True
except Exception:
    _HAS_YFINANCE = False

# Global state to avoid repeated SIP errors
_SIP_ERROR_DETECTED = False
_SAMPLE_CACHE = {}
_CACHE_TIMESTAMP = 0
_CACHE_DURATION = 300  # 5 minutes


def _load_samples(symbols: List[str]) -> dict[str, pd.DataFrame]:
    out = {}
    # First try storage/sample_bars
    base = Path('storage/sample_bars')
    for s in symbols:
        p = base / f"{s}.csv"
        if p.exists():
            try:
                df = pd.read_csv(p, parse_dates=['ts'])
                out[s] = df
            except Exception:
                continue
    
    # For symbols not found in sample_bars, try data_cache with 15m data
    data_cache_base = Path('data_cache')
    for s in symbols:
        if s not in out:  # Only if not already loaded
            cache_file = data_cache_base / f"{s}_15m.csv"
            if cache_file.exists():
                try:
                    df = pd.read_csv(cache_file, parse_dates=['Datetime'])
                    # Rename columns to match expected format
                    df = df.rename(columns={
                        'Datetime': 'ts',
                        'Open': 'open',
                        'High': 'high', 
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    # Keep only the basic OHLCV columns
                    if 'ts' in df.columns:
                        out[s] = df[['ts', 'open', 'high', 'low', 'close', 'volume']].copy()
                except Exception as e:
                    print(f'Failed to load cached data for {s}: {e}')
                    continue
    
    return out


def _iso_or_none(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    # Prefer ISO format; ensure UTC 'Z' suffix if naive
    try:
        s = dt.isoformat()
    except Exception:
        return None
    if s.endswith('+00:00'):
        s = s.replace('+00:00', 'Z')
    return s


def _get_bars_rest(symbol: str, start: Optional[datetime], end: Optional[datetime], timeframe: str = "1Min", limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetch bars for a single symbol from Alpaca market data REST API.

    Uses the configured SETTINGS.key_id / secret_key. Returns a DataFrame or None.
    """
    base = getattr(SETTINGS, 'market_data_base', 'https://data.alpaca.markets')
    url = f"{base}/v2/stocks/{symbol}/bars"
    headers = {
        'APCA-API-KEY-ID': SETTINGS.key_id,
        'APCA-API-SECRET-KEY': SETTINGS.secret_key,
    }
    params = {'timeframe': '1Min' if timeframe == '1Min' else timeframe, 'limit': limit}
    s = _iso_or_none(start)
    e = _iso_or_none(end)
    if s:
        params['start'] = s
    if e:
        params['end'] = e
    # Removed IEX exchange parameter logic - causes 400 errors
    # Simple retry/backoff and response parsing
    j = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            j = resp.json()
            break
        except requests.HTTPError as http_err:
            # If the server returned 400 Bad Request and we included an 'exchange'
            # hint (IEX), retry once without that param — some endpoints reject the
            # exchange hint for certain accounts/plans.
            try:
                status = getattr(http_err.response, 'status_code', None)
            except Exception:
                status = None
            if status == 400 and 'exchange' in params:
                try:
                    params.pop('exchange')
                    resp = requests.get(url, headers=headers, params=params, timeout=10)
                    resp.raise_for_status()
                    j = resp.json()
                    break
                except Exception:
                    # fall through to retry/backoff logic
                    pass
            # For other HTTP errors, propagate so caller can detect auth/entitlement issues
            # but treat it as a failed attempt for retry/backoff
            if attempt >= 2:
                # re-raise the last HTTPError so callers can handle it
                raise
        except Exception:
            import time as _time
            if attempt < 2:
                _time.sleep(0.5 * (attempt + 1))
            else:
                return None

    # Typical response: { 'bars': [ {t,o,h,l,c,v}, ... ], 'symbol': 'AAPL' }
    bars = None
    if isinstance(j, dict):
        bars = j.get('bars')
    elif isinstance(j, list):
        bars = j
    if not bars:
        return None
    rows = []
    for b in bars:
        # Alpaca returns keys like t (timestamp), o,h,l,c,v
        ts = b.get('t') or b.get('timestamp') or b.get('time')
        rows.append({
            'ts': ts,
            'open': b.get('o') or b.get('open'),
            'high': b.get('h') or b.get('high'),
            'low': b.get('l') or b.get('low'),
            'close': b.get('c') or b.get('close'),
            'volume': b.get('v') or b.get('volume')
        })
    if rows:
        df = pd.DataFrame(rows)
        # normalize ts to datetime if numeric/iso
        try:
            df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        except Exception:
            try:
                df['ts'] = pd.to_datetime(df['ts'], utc=True)
            except Exception:
                pass
        return df
    return None


def _get_latest_trade_rest(symbol: str) -> Optional[pd.DataFrame]:
    base = getattr(SETTINGS, 'market_data_base', 'https://data.alpaca.markets')
    url = f"{base}/v2/stocks/{symbol}/trades/latest"
    headers = {
        'APCA-API-KEY-ID': SETTINGS.key_id,
        'APCA-API-SECRET-KEY': SETTINGS.secret_key,
    }
    # retry a few times for transient network issues
    j = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            j = resp.json()
            break
        except Exception:
            import time as _time
            if attempt < 2:
                _time.sleep(0.25 * (attempt + 1))
            else:
                return None

    # Typical payload: { 'trade': { 't':..., 'p': price, 's': size } }
    trade = None
    if isinstance(j, dict):
        trade = j.get('trade') or j.get('data') or j
    else:
        trade = j
    if not trade:
        return None
    if isinstance(trade, dict):
        t = trade.get('t') or trade.get('timestamp') or trade.get('time')
        p = trade.get('p') or trade.get('price') or trade.get('c')
        v = trade.get('s') or trade.get('size') or trade.get('v')
        if p is None:
            return None
        df = pd.DataFrame([{'ts': t, 'open': p, 'high': p, 'low': p, 'close': p, 'volume': v}])
        try:
            df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        except Exception:
            try:
                df['ts'] = pd.to_datetime(df['ts'], utc=True)
            except Exception:
                pass
        return df
    return None


def _expand_latest_to_bars(df: pd.DataFrame, periods: int = 60, timeframe: str = '1Min') -> pd.DataFrame:
    """Expand a single-row latest-trade DataFrame into a small synthetic history.

    The synthetic bars repeat the latest price backwards in time at `timeframe` intervals.
    This is a best-effort compatibility layer so strategies that require history can run.
    """
    if df is None or df.empty:
        return df
    try:
        ts = pd.to_datetime(df.iloc[0]['ts'], utc=True)
    except Exception:
        try:
            ts = pd.to_datetime(df.iloc[0]['ts'])
        except Exception:
            ts = pd.Timestamp.now(tz='UTC')
    price = float(df.iloc[0]['close'])
    vol = int(df.iloc[0].get('volume', 0) or 0)
    # create a datetime index ending at ts, stepping backwards by 1 minute
    idx = pd.date_range(end=ts, periods=periods, freq='1min', tz='UTC')
    rows = []
    for t in idx:
        rows.append({'ts': t, 'open': price, 'high': price, 'low': price, 'close': price, 'volume': vol})
    out = pd.DataFrame(rows)
    return out


def get_bars(symbols: List[str], start: Optional[datetime], end: Optional[datetime], timeframe: str = "1Min", prefer_samples: bool = False) -> dict[str, pd.DataFrame]:
    """Return a dict of symbol->DataFrame with a `ts` column.

    If prefer_samples=True, tries sample data first (useful for training/backtesting).
    Otherwise tries Alpaca first (if available), then falls back to CSVs in `storage/sample_bars/`.
    
    Uses caching and SIP error detection to avoid repeated API failures.
    """
    global _SIP_ERROR_DETECTED, _SAMPLE_CACHE, _CACHE_TIMESTAMP
    
    current_time = time.time()
    
    # If the project is configured to use samples only, skip any external
    # data provider attempts (Alpaca / yfinance) to avoid switching feeds.
    if getattr(SETTINGS, 'data_feed', '').lower() in ('samples', 'sample', 'local', 'csv'):
        samples = _load_samples(symbols)
        if samples:
            print('DATA_FEED configured for samples only - using samples for:', ','.join(samples.keys()))
            return samples

    # If we already detected SIP error, use samples directly
    if _SIP_ERROR_DETECTED:
        # Check cache first
        if _SAMPLE_CACHE and (current_time - _CACHE_TIMESTAMP) < _CACHE_DURATION:
            print('Using cached sample bars (SIP unavailable):', ','.join(_SAMPLE_CACHE.keys()))
            return _SAMPLE_CACHE

        # Aggressive mode: if REST fallback is enabled, try REST pulls even when SIP error detected.
        if getattr(SETTINGS, 'use_rest_fallback', True) and getattr(SETTINGS, 'key_id', None):
            try:
                rest_out: Dict[str, pd.DataFrame] = {}
                for s in symbols:
                    df = _get_bars_rest(s, None, None, timeframe=timeframe)
                    if df is not None and not df.empty:
                        rest_out[s] = df
                if rest_out:
                    print('Using Alpaca REST-pulled data (SIP previously errored) for:', ','.join(rest_out.keys()))
                    return rest_out
            except Exception as e:
                print('Alpaca REST (after SIP error) failed:', type(e).__name__, e)

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
    
    # If Alpaca SDK is not available but API keys are set, try REST pull per-symbol
    if (not _HAS_ALPACA) and getattr(SETTINGS, 'key_id', None) and not prefer_samples and not _SIP_ERROR_DETECTED:
        try:
            rest_out: Dict[str, pd.DataFrame] = {}
            for s in symbols:
                df = _get_bars_rest(s, start, end, timeframe=timeframe)
                if df is not None and not df.empty:
                    rest_out[s] = df
            if rest_out:
                print('Using Alpaca REST-pulled data for:', ','.join(rest_out.keys()))
                return rest_out
        except Exception as e:
            print('Alpaca REST pull failed:', type(e).__name__, e)

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
                # If the SDK returned no bars but REST fallback is enabled, try REST per-symbol
                if getattr(SETTINGS, 'use_rest_fallback', True):
                    try:
                        rest_out2: Dict[str, pd.DataFrame] = {}
                        for s in symbols:
                            df = _get_bars_rest(s, start, end, timeframe=timeframe)
                            if df is not None and not df.empty:
                                rest_out2[s] = df
                        if rest_out2:
                            print('Using Alpaca REST-pulled data (SDK empty) for:', ','.join(rest_out2.keys()))
                            return rest_out2
                    except Exception as e:
                        print('Alpaca REST fallback after SDK empty failed:', type(e).__name__, e)
        except Exception as e:
            error_msg = str(e)
            # Detect SIP subscription error and avoid retrying
            if "subscription does not permit querying recent SIP data" in error_msg:
                print('SIP data subscription unavailable - switching to sample data mode')
                _SIP_ERROR_DETECTED = True
            else:
                print('Alpaca historical bars failed:', type(e).__name__, e)

        # As a last-resort near-live probe, try to fetch the latest trade/quote
        # using available Alpaca client methods (non-streaming, single-sample).
        try:
            latest_out = {}
            # probe method name candidates
            latest_methods = [
                'get_stock_latest_trade',
                'get_latest_trade',
                'get_latest_trades',
                'get_stock_latest_quote',
                'get_latest_quote',
                'get_last_trade',
            ]
            for sym in symbols:
                val = None
                for m in latest_methods:
                    fn = getattr(client, m, None)
                    if callable(fn):
                        try:
                            # different SDKs use different signatures
                            try:
                                res = fn(sym)
                            except TypeError:
                                res = fn(symbol=sym)
                            # inspect common attributes
                            price = getattr(res, 'price', None) or getattr(res, 'ask_price', None) or getattr(res, 'p', None)
                            ts = getattr(res, 'timestamp', None) or getattr(res, 't', None) or getattr(res, 'time', None)
                            if price is not None:
                                import pandas as _pd
                                df = _pd.DataFrame([{
                                    'ts': ts,
                                    'open': price,
                                    'high': price,
                                    'low': price,
                                    'close': price,
                                    'volume': getattr(res, 'size', getattr(res, 'v', None))
                                }])
                                latest_out[sym] = df
                                val = True
                                break
                        except Exception:
                            continue
                # end methods loop
            if latest_out:
                # Check if this is just synthetic data (all rows have same price)
                is_synthetic = True
                for sym, df in latest_out.items():
                    if len(df) > 1:
                        # Check if all prices are the same (synthetic data)
                        prices = df['close'].values
                        if not all(p == prices[0] for p in prices):
                            is_synthetic = False
                            break
                
                if not is_synthetic or len(latest_out) < len(symbols):
                    print('Using Alpaca latest-trade probe for:', ','.join(latest_out.keys()))
                    return latest_out
                else:
                    print('Latest-trade probe returned synthetic data only, trying samples instead')
                    # Fall through to sample loading
            # If SDK probe returned nothing and SETTINGS allows REST fallback, try REST latest-trade
            if not latest_out and getattr(SETTINGS, 'use_rest_fallback', True):
                rest_latest = {}
                for s in symbols:
                    df = _get_latest_trade_rest(s)
                    if df is not None and not df.empty:
                        # If caller requested 1Min timeframe and only a single latest trade
                        # is available, expand it into a short synthetic history so
                        # indicators/strategies that require a window can operate.
                        if timeframe and (str(timeframe).lower().startswith('1') or str(timeframe).lower() == '1min'):
                            try:
                                df_exp = _expand_latest_to_bars(df, periods=60, timeframe='1Min')
                                rest_latest[s] = df_exp
                            except Exception:
                                rest_latest[s] = df
                        else:
                            rest_latest[s] = df
                if rest_latest:
                    # Check if this is just synthetic data (all rows have same price)
                    is_synthetic = True
                    for sym, df in rest_latest.items():
                        if len(df) > 1:
                            # Check if all prices are the same (synthetic data)
                            prices = df['close'].values
                            if not all(p == prices[0] for p in prices):
                                is_synthetic = False
                                break
                    
                    if not is_synthetic or len(rest_latest) < len(symbols):
                        print('Using Alpaca REST latest-trade fallback for:', ','.join(rest_latest.keys()))
                        return rest_latest
                    else:
                        print('REST latest-trade fallback returned synthetic data only, trying samples instead')
                        # Fall through to sample loading
        except Exception as e:
            print('Alpaca latest-probe failed:', type(e).__name__, e)

    # Try yfinance as a best-effort near-live fallback when Alpaca is unavailable
    if _HAS_YFINANCE and not prefer_samples:
        try:
            out = {}
            for s in symbols:
                # yfinance accepts ticker symbols like 'SPY'
                # Use period covering start->end if start/end provided; else default to 1d
                period = None
                if start and end:
                    # yfinance uses start/end strings
                    df = _yfinance.download(s, start=start, end=end, interval='1m', progress=False)
                else:
                    df = _yfinance.download(s, period='1d', interval='1m', progress=False)
                if df is not None and not df.empty:
                    df = df.reset_index()
                    # Normalize column names to match expected schema
                    df = df.rename(columns={
                        'Datetime': 'ts',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    if 'ts' not in df.columns and 'Date' in df.columns:
                        df = df.rename(columns={'Date': 'ts'})
                    out[s] = df[['ts', 'open', 'high', 'low', 'close', 'volume']]
            if out:
                print('Using yfinance data for:', ','.join(out.keys()))
                return out
        except Exception as e:
            print('yfinance fallback failed:', type(e).__name__, e)

    # Fallback: load sample CSVs from storage/sample_bars/
    samples = _load_samples(symbols)
    if samples:
        # Cache the samples if SIP error detected
        if _SIP_ERROR_DETECTED:
            _SAMPLE_CACHE = samples
            _CACHE_TIMESTAMP = current_time
        print('Loaded sample bars for:', ','.join(samples.keys()))
        return samples

    # --- FORCE: try REST pulls first (aggressive fallback) when API keys exist ---
    if getattr(SETTINGS, 'key_id', None) and not prefer_samples:
        try:
            rest_out_forced: Dict[str, pd.DataFrame] = {}
            for s in symbols:
                try:
                    df = _get_bars_rest(s, start, end, timeframe=timeframe)
                except Exception:
                    df = None
                if df is not None and not df.empty:
                    rest_out_forced[s] = df
            if rest_out_forced:
                print('Using Alpaca REST-pulled data (forced attempt) for:', ','.join(rest_out_forced.keys()))
                return rest_out_forced
        except Exception as e:
            print('Forced Alpaca REST attempt failed:', type(e).__name__, e)

    # Nothing available
    raise RuntimeError('No market data available: Alpaca failed or not configured, and no sample bars found.')

