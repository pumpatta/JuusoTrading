@echo off
REM JuusoTrader Auto-Restart Service
REM This will keep the live trading engine running automatically

echo 🚀 Starting JuusoTrader Auto-Restart Service
echo 📊 Trading across 113 instruments with multi-account support
echo.
echo Press Ctrl+C to stop the service
echo.

python auto_restart.py

echo.
echo 👋 Auto-restart service stopped
pause
