from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


def _select_account(account_letter: str):
    # account_letter expected 'A', 'B', or 'C'
    account_letter = (account_letter or '').upper()
    if account_letter not in ('A', 'B', 'C'):
        return os.getenv('ALPACA_KEY_ID', ''), os.getenv('ALPACA_SECRET_KEY', '')
    kid = os.getenv(f'ALPACA_KEY_ID_{account_letter}', '')
    sk = os.getenv(f'ALPACA_SECRET_KEY_{account_letter}', '')
    # fallback to generic vars if specific not present
    return kid or os.getenv('ALPACA_KEY_ID', ''), sk or os.getenv('ALPACA_SECRET_KEY', '')


@dataclass
class Settings:
    # ALPACA account selector (A/B/C) — falls back to generic keys
    account: str = os.getenv('ALPACA_ACCOUNT', 'A')
    key_id: str = ''
    secret_key: str = ''
    base_url: str = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    data_feed: str = os.getenv('DATA_FEED', '')
    starting_money: float = float(os.getenv('STARTING_MONEY', '100000'))
    model_id: str = ''

    @property
    def market_data_base(self):
        """Return the market data API base URL - always use live data API for market data"""
        return 'https://data.alpaca.markets'

    def __post_init__(self):
        kid, sk = _select_account(self.account)
        self.key_id = kid
        self.secret_key = sk
        # pick model id for this account (env ALPACA_MODEL_A/B/C)
        self.model_id = os.getenv(f'ALPACA_MODEL_{self.account}', os.getenv('ALPACA_MODEL', ''))


SETTINGS = Settings()


def get_settings_for(account_letter: str | None):
    """Return a Settings instance for a specific account letter (A/B/C).

    If account_letter is None, return the global SETTINGS.
    """
    if not account_letter:
        return SETTINGS
    s = Settings()
    s.account = (account_letter or '').upper()
    # re-run post_init wiring
    s.__post_init__()
    return s
