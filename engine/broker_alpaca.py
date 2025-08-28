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
        self._paper = paper
        self.client = None
        # lazy import of Alpaca SDK to avoid hard dependency at import time
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest, BracketOrderRequest, TakeProfitRequest, StopLossRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            # store classes for later use
            self._TradingClient = TradingClient
            self._MarketOrderRequest = MarketOrderRequest
            self._BracketOrderRequest = BracketOrderRequest
            self._TakeProfitRequest = TakeProfitRequest
            self._StopLossRequest = StopLossRequest
            self._OrderSide = OrderSide
            self._TimeInForce = TimeInForce
            # instantiate client
            self.client = TradingClient(self.key_id, self.secret_key, paper=paper)
        except Exception:
            # alpaca SDK not available; keep client as None but continue
            self.client = None

    def buy_bracket(self, symbol: str, qty: float, take_profit: float, stop_loss: float, client_order_id: str):
        if not self.client:
            raise RuntimeError('Alpaca client not available (alpaca SDK not installed)')
        req = self._BracketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=self._OrderSide.BUY,
            time_in_force=self._TimeInForce.DAY,
            take_profit=self._TakeProfitRequest(limit_price=take_profit),
            stop_loss=self._StopLossRequest(stop_price=stop_loss)
        )
        order = self.client.submit_order(req, client_order_id=client_order_id)
        return order

    def sell_market(self, symbol: str, qty: float, client_order_id: str):
        if not self.client:
            raise RuntimeError('Alpaca client not available (alpaca SDK not installed)')
        req = self._MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=self._OrderSide.SELL,
            time_in_force=self._TimeInForce.DAY
        )
        return self.client.submit_order(req, client_order_id=client_order_id)
