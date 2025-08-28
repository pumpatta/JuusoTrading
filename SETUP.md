# 🔧 JuusoTrader Setup Instructions

## ⚠️ Ennen ensimmäistä käyttöä

### 1. Kloonaa repository
```bash
git clone https://github.com/yourusername/JuusoTrader.git
cd JuusoTrader
```

### 2. Luo Python virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. Asenna riippuvuudet
```bash
pip install -r requirements.txt
```

### 4. Konfiguroi API avaimet (TÄRKEÄ!)

#### Alpaca API (Paper Trading)
1. Luo tili: https://alpaca.markets/
2. Hanki Paper Trading API avaimet
3. Kopioi `config/production_deployment_template.json` → `config/production_deployment.json`
4. Lisää API avaimesi:
```json
{
  "alpaca": {
    "api_key": "YOUR_ACTUAL_API_KEY",
    "secret_key": "YOUR_ACTUAL_SECRET_KEY"
  }
}
```

#### Execution konfiguraatio
- `config/execution.yml` - Slippage ja spread asetukset
- Oletukset toimivat paper tradingissa

### 5. Testaa asennus
```bash
python scripts/bootstrap.py
```

## 🚀 Käynnistys

### Windows
```cmd
start_trading.bat
```

### PowerShell
```powershell
.\start_trading.ps1
```

### Manuaalinen käynnistys
```bash
# 1. Trading engine taustalle
python engine/live.py --paper &

# 2. Dashboard
python launch_dashboard.py
```

## 📊 Dashboard

- **URL**: http://localhost:8501
- **Päivittyy**: Automaattisesti 30s välein
- **Sisältää**: Portfolio tracking, benchmarks, trade history

## 💡 Trading Strategiat

| Account | Strategia | Pääoma | Teknologia |
|---------|-----------|--------|-------------|
| A | EMA Trend | 30% | Exponential Moving Average |
| B | XGB ML | 30% | XGBoost Machine Learning |
| C | Enhanced ML | 40% | FinBERT + News Sentiment + Patterns |

## 🔒 Turvallisuus

### ✅ Mitä ON TURVALLISTA commitoida:
- Koodi (`.py` tiedostot)
- Template konfiguraatiot
- Dokumentaatio
- Requirements

### ❌ Mitä EI SAA commitoida:
- API avaimet
- Kaupankäyntilokitiedostot
- ML mallit (proprietary)
- Henkilökohtaiset tulokset
- Sample market data

### 🛡️ Suojaukset
- `.gitignore` suojaa arkaluonteiset tiedot
- Template tiedostot API avaimille
- Virtual environment ei mukana repossa

## 🐛 Ongelmanratkaisu

### "No module named 'engine'"
```bash
# Varmista että olet projektin juurihakemistossa
cd JuusoTrader
python engine/live.py --paper
```

### "SIP data subscription" virhe
- ✅ Automaattisesti korjattu
- Järjestelmä siirtyy sample data tilaan
- Ei vaadi maksullista Alpaca SIP tilausta

### Dashboard tyhjä
- Odota 2-3 minuuttia että trading engine käynnistyy
- Tarkista `storage/logs/` hakemisto trade historiale
- Päivitä selain (F5)

### Virtual environment ongelmat
```bash
# Poista vanha ja luo uusi
rmdir /s .venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 📈 Live Trading (Tulevaisuudessa)

**TÄRKEÄ**: Tämä on PAPER TRADING järjestelmä. Live trading vaatii:
- Alpaca Live API avaimet (ei Paper)
- Lisää risk management
- Backtesting ja validointi
- Säännöllinen seuranta

**ÄLÄ käytä oikeaa rahaa ilman kattavaa testausta!**
