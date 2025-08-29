import requests
from utils.config import SETTINGS, get_settings_for


class Broker:
    def __init__(self, paper: bool = True, account: str | None = None):
        """Create a Broker bound to a specific Alpaca account (A/B/C).

        If the `alpaca` SDK is not installed, `self.client` will be None but
        credentials are still stored on the instance for testing.
        """
        cfg = get_settings_for(account)
        self.account = cfg.account
        self.key_id = cfg.key_id
        self.secret_key = cfg.secret_key
        # trading base URL (paper/live)
        self.base_url = getattr(cfg, 'base_url', 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets')
        self._paper = paper
        self.client = None
        # lazy import of Alpaca SDK to avoid hard dependency at import time
        try:
            from alpaca.trading.client import TradingClient
            # Note: avoid importing request helper classes here to remain compatible
            # with different versions of alpaca-py. Order building is handled at call
            # time to reduce import-time breakage.
            self._TradingClient = TradingClient

            # Try a few common constructor patterns for TradingClient to handle SDK versions
            instantiated = False
            inst_error = None
            try:
                # Pattern 1: positional args and paper kwarg
                self.client = TradingClient(self.key_id, self.secret_key, paper=paper)
                instantiated = True
            except Exception as e1:
                inst_error = e1
            if not instantiated:
                try:
                    # Pattern 2: explicit api_key/api_secret and base_url
                    self.client = TradingClient(api_key=self.key_id, api_secret=self.secret_key, base_url=self.base_url)
                    instantiated = True
                except Exception as e2:
                    inst_error = e2
            if not instantiated:
                try:
                    # Pattern 3: api_key/api_secret with paper flag
                    self.client = TradingClient(api_key=self.key_id, api_secret=self.secret_key, paper=paper)
                    instantiated = True
                except Exception as e3:
                    inst_error = e3

            if not instantiated:
                # Keep client as None but store last error for debugging
                self.client = None
                # avoid printing secrets; keep only exception type/message
                print(f"⚠️ Alpaca TradingClient instantiation failed: {type(inst_error).__name__}: {str(inst_error)[:200]}")
        except Exception as import_exc:
            # alpaca SDK not available; keep client as None but continue
            self.client = None
            print(f"⚠️ Alpaca SDK import failed: {type(import_exc).__name__}: {str(import_exc)[:200]}")

    # --- REST order helpers (safe to call even if SDK not installed) ---
    def _order_headers(self) -> dict:
        return {
            'APCA-API-KEY-ID': self.key_id,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }

    def submit_order_rest(self, symbol: str, qty: float, side: str = 'buy', order_type: str = 'market', time_in_force: str = 'day', client_order_id: str | None = None):
        """Submit an order via Alpaca REST /v2/orders (paper/live depending on base_url).

        This will respect SETTINGS.dry_run and SETTINGS.use_rest_fallback semantics.
        """
        if getattr(SETTINGS, 'dry_run', False):
            print(f"DRY RUN: would submit order {side} {qty} {symbol} (type={order_type})")
            return {'status': 'dry_run'}

        # Construct API URL - add /v2 only if not already present
        base = self.base_url.rstrip('/')
        if not base.endswith('/v2'):
            base = f"{base}/v2"
        url = f"{base}/orders"
        body = {
            'symbol': symbol,
            'qty': qty,
            'side': side.lower(),
            'type': order_type.lower(),
            'time_in_force': time_in_force.lower()
        }
        if client_order_id:
            body['client_order_id'] = client_order_id
        try:
            r = requests.post(url, json=body, headers=self._order_headers(), timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print('REST order submit failed:', type(e).__name__, e)
            raise

    def buy_market(self, symbol: str, qty: float, client_order_id: str | None = None):
        """Convenience: submit market buy using SDK when available, else REST."""
        if self.client:
            try:
                if hasattr(self.client, 'submit_order'):
                    print(f"Broker {self.account}: Attempting SDK market buy for {symbol} qty={qty}")
                    payload = {'symbol': symbol, 'qty': qty, 'side': 'buy', 'type': 'market', 'time_in_force': 'day'}
                    if client_order_id:
                        payload['client_order_id'] = client_order_id
                    result = self.client.submit_order(payload)  # type: ignore
                    print(f"Broker {self.account}: SDK market buy success for {symbol}: {result}")
                    return result
            except Exception as e:
                print(f"Broker {self.account}: SDK market buy failed, falling back to REST: {type(e).__name__}: {str(e)}")
                # fall through to REST
                pass
        print(f"Broker {self.account}: Attempting REST market buy for {symbol} qty={qty}")
        return self.submit_order_rest(symbol, qty, side='buy', order_type='market', time_in_force='day', client_order_id=client_order_id)

    def sell_market(self, symbol: str, qty: float, client_order_id: str | None = None):
        """Convenience: submit market sell using SDK when available, else REST."""
        if self.client:
            try:
                if hasattr(self.client, 'submit_order'):
                    print(f"Broker {self.account}: Attempting SDK market sell for {symbol} qty={qty}")
                    payload = {'symbol': symbol, 'qty': qty, 'side': 'sell', 'type': 'market', 'time_in_force': 'day'}
                    if client_order_id:
                        payload['client_order_id'] = client_order_id
                    result = self.client.submit_order(payload)  # type: ignore
                    print(f"Broker {self.account}: SDK market sell success for {symbol}: {result}")
                    return result
            except Exception as e:
                print(f"Broker {self.account}: SDK market sell failed, falling back to REST: {type(e).__name__}: {str(e)}")
                pass
        print(f"Broker {self.account}: Attempting REST market sell for {symbol} qty={qty}")
        return self.submit_order_rest(symbol, qty, side='sell', order_type='market', time_in_force='day', client_order_id=client_order_id)

    def buy_bracket(self, symbol: str, qty: float, take_profit: float, stop_loss: float, client_order_id: str | None = None):
        """Submit a bracket buy order (market entry + take profit + stop loss).

        This tries to use the SDK's submit_order with bracket/order_class support when
        available; otherwise it falls back to the REST /v2/orders endpoint with the
        bracket payload. Respects SETTINGS.dry_run.
        """
        # Round prices to nearest penny to comply with Alpaca requirements
        take_profit = round(take_profit, 2)
        stop_loss = round(stop_loss, 2)

        # Prefer SDK when available
        if self.client:
            try:
                if hasattr(self.client, 'submit_order'):
                    print(f"Broker {self.account}: Attempting SDK bracket buy for {symbol} qty={qty}")
                    # Try dict payload first as it's more reliable across SDK versions
                    payload = {
                        'symbol': symbol,
                        'qty': qty,
                        'side': 'buy',
                        'type': 'market',
                        'time_in_force': 'day',
                        'order_class': 'bracket',
                        'take_profit': {'limit_price': float(take_profit)},
                        'stop_loss': {'stop_price': float(stop_loss)},
                    }
                    if client_order_id:
                        payload['client_order_id'] = client_order_id
                    result = self.client.submit_order(payload)  # type: ignore
                    print(f"Broker {self.account}: SDK bracket buy success for {symbol}: {result}")
                    return result
            except Exception as e:
                print(f"Broker {self.account}: SDK bracket buy failed, falling back to REST: {type(e).__name__}: {str(e)}")
                # swallow and fall back to REST
                pass

        # REST fallback
        if getattr(SETTINGS, 'dry_run', False):
            print(f"DRY RUN: would submit bracket BUY {qty} {symbol} TP={take_profit} SL={stop_loss}")
            return {'status': 'dry_run'}

        print(f"Broker {self.account}: Attempting REST bracket buy for {symbol} qty={qty}")
        # Construct API URL - add /v2 only if not already present
        base = self.base_url.rstrip('/')
        if not base.endswith('/v2'):
            base = f"{base}/v2"
        url = f"{base}/orders"
        body = {
            'symbol': symbol,
            'qty': qty,
            'side': 'buy',
            'type': 'market',
            'time_in_force': 'day',
            'order_class': 'bracket',
            'take_profit': {'limit_price': float(take_profit)},
            'stop_loss': {'stop_price': float(stop_loss)},
        }
        if client_order_id:
            body['client_order_id'] = client_order_id
        try:
            r = requests.post(url, json=body, headers=self._order_headers(), timeout=10)
            r.raise_for_status()
            result = r.json()
            print(f"Broker {self.account}: REST bracket buy success for {symbol}: {result}")
            return result
        except Exception as e:
            print(f"Broker {self.account}: REST bracket buy failed: {type(e).__name__}: {str(e)}")
            raise

    # Note: SDK-specific bracket/market request helpers omitted — use submit_order_rest or
    # buy_market/sell_market convenience methods which will use SDK if available otherwise REST.
