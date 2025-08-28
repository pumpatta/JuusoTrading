@echo off
echo.
echo ===================================
echo 🚀 JuusoTrader - Paper Trading
echo ===================================
echo.

REM Siirry projektin hakemistoon
cd /d "C:\Users\Juuso\source\repos\pumpatta\JuusoTrader"

echo 📈 Käynnistetään trading engine taustalle...
start "JuusoTrader Live Engine" /MIN ".venv\Scripts\python.exe" engine/live.py --paper

echo ⏰ Odotetaan 10 sekuntia että engine käynnistyy...
timeout /t 10 /nobreak >nul

echo.
echo 📊 Käynnistetään dashboard...
echo 🌐 Dashboard aukeaa automaattisesti: http://localhost:8501
echo 💡 Account A: EMA Strategy (30%%)
echo 💡 Account B: XGB ML Strategy (30%%)  
echo 💡 Account C: Enhanced ML + News (40%%)
echo.
echo ⏹️  Pysäytä trading: Ctrl+C
echo ===================================
echo.

".venv\Scripts\python.exe" launch_dashboard.py
