from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import yaml
from pathlib import Path

@dataclass
class Session:
    start_utc: datetime
    end_utc: datetime
    is_open: bool

def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))

def load_cfg(path: str = 'config/market.yml'):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))

def today_market_session(now_utc: datetime | None = None, cfg=None) -> Session:
    cfg = cfg or load_cfg()
    tz_mkt = ZoneInfo(cfg['timezone_market'])
    now_utc = now_utc or datetime.now(timezone.utc)

    d_mkt = now_utc.astimezone(tz_mkt).date()
    # Regular session in market TZ
    start = datetime.combine(d_mkt, _parse_hhmm(cfg['regular_hours']['start']), tzinfo=tz_mkt)
    end   = datetime.combine(d_mkt, _parse_hhmm(cfg['regular_hours']['end']),   tzinfo=tz_mkt)
    start_utc, end_utc = start.astimezone(timezone.utc), end.astimezone(timezone.utc)

    is_open = start_utc <= now_utc <= end_utc
    return Session(start_utc=start_utc, end_utc=end_utc, is_open=is_open)

def next_session_open_close(now_utc: datetime | None = None, cfg=None) -> Session:
    s = today_market_session(now_utc, cfg)
    if s.end_utc <= (now_utc or datetime.now(timezone.utc)):
        nxt = (now_utc or datetime.now(timezone.utc)) + timedelta(days=1)
        anchor = nxt.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        return today_market_session(anchor, cfg)
    return s
