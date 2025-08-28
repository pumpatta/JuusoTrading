# JuusoTrader PowerShell käynnistysskripti
Write-Host ""
Write-Host "===================================" -ForegroundColor Cyan
Write-Host "🚀 JuusoTrader - Paper Trading" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Siirry projektin hakemistoon
Set-Location "C:\Users\Juuso\source\repos\pumpatta\JuusoTrader"

Write-Host "📈 Käynnistetään trading engine taustalle..." -ForegroundColor Yellow
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "engine/live.py --paper" -WindowStyle Minimized

Write-Host "⏰ Odotetaan 10 sekuntia että engine käynnistyy..." -ForegroundColor Blue
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "📊 Käynnistetään dashboard..." -ForegroundColor Yellow
Write-Host "🌐 Dashboard aukeaa automaattisesti: http://localhost:8501" -ForegroundColor Green
Write-Host "💡 Account A: EMA Strategy (30%)" -ForegroundColor White
Write-Host "💡 Account B: XGB ML Strategy (30%)" -ForegroundColor White  
Write-Host "💡 Account C: Enhanced ML + News (40%)" -ForegroundColor White
Write-Host ""
Write-Host "⏹️  Pysäytä trading: Ctrl+C" -ForegroundColor Red
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

& ".\.venv\Scripts\python.exe" launch_dashboard.py
