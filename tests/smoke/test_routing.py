import os
from engine.live import DEFAULT_STRATEGY_ACCOUNT, get_account_for_strategy
from utils.config import get_settings_for


def test_default_mapping():
    assert DEFAULT_STRATEGY_ACCOUNT['EMA'] == 'A'
    assert DEFAULT_STRATEGY_ACCOUNT['XGB'] == 'B'
    assert get_account_for_strategy('EMA') == 'A'


def test_env_override(monkeypatch):
    monkeypatch.setenv('STRATEGY_ACCOUNT_XGB', 'C')
    try:
        assert get_account_for_strategy('XGB') == 'C'
    finally:
        monkeypatch.delenv('STRATEGY_ACCOUNT_XGB', raising=False)


def test_get_settings_for_account():
    sA = get_settings_for('A')
    assert hasattr(sA, 'key_id')
    sB = get_settings_for('B')
    assert hasattr(sB, 'key_id')
