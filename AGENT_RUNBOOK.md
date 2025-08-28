# AGENT RUNBOOK — JuusoTrader (one-step checklist for contributors)

IMPORTANT MEMO: Always activate the repository virtual environment before running any commands. If you are not in `.venv`, stop and activate it now. This runbook assumes the venv is active.

Purpose: provide an exact, single-action-at-a-time runbook for developers and coding agents. Follow each numbered step in order; stop and fix errors before moving on.

Status: aligned with `README.md` — keep both in sync when you change configs or root commands.

## CURRENT STATUS — Session 2025-08-27 ✅

**COMPLETED TASKS:**
- [x] Repository structure validated and working
- [x] Python 3.12 venv created and activated (`.venv`)
- [x] All requirements installed (`alpaca-py`, `requests`, `xgboost`, `torch`, etc.)
- [x] Multi-account support implemented in `.env` (A/B/C accounts with keys)
- [x] Per-account model mapping configured (`ALPACA_MODEL_A/B/C`)
- [x] `utils/config.py` updated with `get_settings_for(account)` function
- [x] `engine/broker_alpaca.py` updated to accept account parameter and lazy import
- [x] `engine/live.py` updated with per-account routing and `DEFAULT_STRATEGY_ACCOUNT` mapping
- [x] `engine/datafeed.py` updated with CSV fallback for offline testing
- [x] Backtest pipeline working (`scripts/backtest_quick.py` → `storage/backtest_report.json`)
- [x] Retrain pipeline working (`scripts/retrain.py` → `models/retrain_report.json`)
- [x] Smoke tests added and passing (`tests/smoke/test_routing.py`)
- [x] Import diagnostics script created (`scripts/_check_imports.py`)
- [x] VS Code settings fixed (`.vscode/settings.json`)
- [x] `pytest.ini` cleaned (removed unknown `python_paths` option)
- [x] All imports validated and working in venv

**IMMEDIATE NEXT TASKS:**
- [ ] Implement per-account model serialization (`models/{account}_{strategy}.model`)
- [ ] Create per-account logging directories (`storage/logs/{account}/`)
- [x] Create per-account logging directories (`storage/logs/{account}/`)
- [ ] Add diagnostics harness (`scripts/diagnostics.py`)
- [ ] Test live paper trading with actual Alpaca accounts

**ENVIRONMENT STATUS:**
- Python: 3.12.x in `.venv` ✅
- Dependencies: all installed ✅
- Accounts: A/B/C configured in `.env` ✅
- Model mapping: EMA→A, XGB/TCN→B, HNS/ENSEMBLE→C ✅

## Setup Checklist (single-step actions)

1) Verify repository root contains the folders `engine/`, `strategies/`, `utils/`, `config/`, `ui/`, `storage/`.

2) Create and activate a Python 3.12 venv in the project root:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Run a one-line sanity check to confirm Python version:
```powershell
python -V
```
Expect `Python 3.12.x`.

4) Upgrade packaging tools:
```powershell
python -m pip install --upgrade pip wheel setuptools
```

5) (Optional GPU) Install PyTorch wheel matching your CUDA. If unsure, skip and install requirements in step 6. Example (cu128 nightly):
```powershell
pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio
```

6) Install repo Python requirements:
```powershell
pip install -r requirements.txt
```

7) Create `.env` from example and edit it to include Alpaca paper keys:
```powershell
copy .env.example .env
# edit .env to add ALPACA_KEY_ID, ALPACA_SECRET_KEY, ALPACA_BASE_URL
```

8) Edit `config/universe.yml` to enable a small set (2–5) of tickers for initial testing.

9) Run bootstrap to prepare folders and DB tables:
```powershell
python scripts/bootstrap.py
```

10) Run the GPU check script (only if you installed torch):
```powershell
python scripts/check_gpu.py
```

11) Start the bot in paper mode (single-step):
```powershell
python engine/live.py --paper
```

12) Confirm logs are being written to `storage/logs/` and `storage/db.sqlite` exists.

13) If errors occur, stop and fix them. Common single fixes:
- Missing keys: open `.env` and correct them.
- CUDA unavailable: reactivate `.venv` and reinstall a compatible torch.
- Import errors: ensure `pip install -r requirements.txt` completed without exceptions.

14) After basic run is green, run a retrain (single-step):
```powershell
python scripts/retrain.py
```

15) Periodic maintenance tasks (each is a single-step action):
- Reset paper account via Alpaca UI (one click) or use API to cancel orders and close positions.
- Reduce universe by editing `config/universe.yml` then restart `engine/live.py`.
- Update dependencies: `pip install -r requirements.txt`.

16) Model → Account isolation policy

This project enforces a 1:1 mapping between Alpaca paper accounts and strategies so that PnL, equity curves and drawdowns are isolated per model. The default mapping is:

- Account A: EMA (`EMA`)
- Account B: XGBoost / TCN (`XGB`, `TCN`)
- Account C: Pattern / Ensemble (`HNS`, `ENSEMBLE`)

To override which Alpaca account a strategy uses, set `STRATEGY_ACCOUNT_{STRATEGY_ID}` in your `.env` (for example `STRATEGY_ACCOUNT_XGB=B`). The engine will create one Broker per account and will not mix keys between accounts.

Next tasks for a coding agent (priority order)

**HIGH PRIORITY:**
- A) Implement per-account model serialization:
  - Modify `scripts/retrain.py` to accept `--account A|B|C` flag
  - Save models as `models/{account}_{strategy}.model`
  - Update strategy classes to load from account-specific files
  
- B) Create per-account logging structure:
  - Ensure logs write to `storage/logs/{account}/`
  - Create `storage/metrics/{account}.json` for P&L tracking

**MEDIUM PRIORITY:**
- C) Add diagnostics harness (`scripts/diagnostics.py`):
  - Run compile, import, smoke tests in sequence
  - Write results to `storage/logs/diagnostics/`
  
- D) Test live paper trading:
  - Validate actual Alpaca API connections work
  - Confirm per-account isolation in practice

**LOW PRIORITY:**
- E) Add dashboard P&L card (`ui/dashboard_card.py`)
- F) Add unit tests (`tests/test_timewin.py`)
- G) Add mock live engine test (`tests/smoke/test_live_mock.py`)

**QUICK COMMANDS TO RESUME WORK:**
```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Run diagnostics
python scripts\_check_imports.py

# Run smoke tests
pytest -q tests/smoke/test_routing.py

# Quick backtest
python scripts/backtest_quick.py

# Start next priority task
# (implement per-account model serialization)
```

## Offline replay & accelerated demos (new)

Use the offline engine to run fast replays against cached sample bars in `storage/sample_bars`. This is useful in the 4 hours before market open to validate models and safety rules.

- Run a short demo (exits after cycles):
```powershell
.venv\Scripts\python.exe engine\live_offline.py --offline --symbols SPY --cycles 3 --sleep 3
```

- Seed a starting position so sell signals can execute in a demo:
```powershell
.venv\Scripts\python.exe engine\live_offline.py --offline --symbols SPY --cycles 1 --seed SPY:100
```

- Accelerated replay (fast, advance N bars per cycle):
```powershell
.venv\Scripts\python.exe engine\live_offline.py --offline --symbols SPY,QQQ --cycles 200 --accel --step 5 --seed SPY:500
```

- When using `--accel` the runner advances internal pointers over the historical bars and sleeps for ~0.01s between cycles so the replay completes quickly and does not block your terminal for long.

Logs and artifacts produced by offline replays:
- Trades: appended to `storage/intraday_backtest_trades.csv` and `storage/logs/trades_{strategy}.csv`
- Analysis: `scripts/analyze_trades.py` writes `storage/trade_summary_per_strategy.csv` and `storage/trade_summary_per_symbol.csv`
- Fee sensitivity: `scripts/simulate_trade_costs.py` writes `storage/trade_summary_with_fees_{bps}.csv`

## Safety defaults and quick readiness checklist

Safety defaults are in `utils/safety.py` and currently set to the Relaxed profile:
- MAX_POS_PCT = 0.03 (3% of capital per symbol)
- MAX_TOTAL_EXPOSURE_PCT = 0.3 (30% total exposure)
- DAILY_LOSS_PCT = 0.10 (10% daily stop)
- MAX_TRADES_PER_SYMBOL_DAY = 5

Readiness quick-check (run these before switching to paper live):
1) Confirm model artifacts exist for each strategy (check `models/` and the reports in `models/*.json`).
2) Run accelerated offline replay with `--accel` across target symbols and review `storage/trade_summary_per_symbol.csv`.
3) Run `scripts/simulate_trade_costs.py` and ensure results survive realistic fees (0.05%–0.2%).
4) Use a small pilot allocation (0.25%–1% per trade) and a 1% stop-loss for the first trading day.
5) Monitor logs in `storage/logs/` and build an alert for DAILY_LOSS_PCT breach.

## Model verification & repair (quick steps)

1. Run the model loader checker to verify `models/*.json` point to loadable binaries:

```powershell
.venv\Scripts\python.exe scripts\check_models_loadable.py
```

2. If any reports fail, attempt automatic repair (the script will probe candidate files and update the `model_file` fields):

```powershell
.venv\Scripts\python.exe scripts\repair_model_reports.py
```

3. Re-run the checker and confirm all reports are OK. If a model still fails to load, inspect the file header (e.g., PyTorch zip, XGBoost native) and either re-serialize the model in a supported format or add a loader fallback.

4. After models verify OK, run an accelerated offline replay across your target universe and then re-run `scripts/analyze_trades.py` + `scripts/simulate_trade_costs.py`.

## Final pre-live checklist (one-line ready checks)

- Confirm `scripts/check_models_loadable.py` returns OK for all `models/*.json`.
- Run `engine/live_offline.py --offline --accel --cycles 200 --step 1` and validate trades survive realistic fees.
- Set environment variables for conservative limits: `MAX_POS_PCT=0.01`, `DAILY_LOSS_PCT=0.05` for pilot.


## Committing changes and pushing
After making code or docs changes, commit and push from the repo root (one-line):
```powershell
git add . ; git commit -m "chore: docs/update runbook + offline replay flags" ; git push
```

---
Revision: 2025-08-28

---
Revision: 2025-08-27
# or
