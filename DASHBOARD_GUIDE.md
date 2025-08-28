# JuusoTrader Dashboard - Käyttöohje

## 📊 Dashboard Ominaisuudet

### Pääominaisuudet:
- **Reaaliaikainen portfolion seuranta** - Näet portfoliosi arvon kehityksen
- **Vertailu markkinaindekseihin** (Vertailukohteet/Verrokki):
  - NASDAQ (^IXIC)
  - S&P 500 (^GSPC)  
  - SPY ETF
  - QQQ ETF
- **Kaupankäyntihistoria** - Kaikki tehdyt kaupat strategioittain
- **Suorituskykymittarit** - P&L, tuotto-%, Sharpe ratio
- **Strategiakohtainen analyysi** - Jokaisen strategian erikseen

### Dashboard Näkymät:

1. **Päämetriikka-kortit**:
   - 💰 Portfolion nykyarvo
   - 📊 Kokonais P&L (voitto/tappio)
   - 📈 Viimeisin päivätuotto
   - 🔄 Tehtyjen kauppojen määrä

2. **Suorituskykykuvaaja**:
   - Portfolion arvon kehitys normalisoituna (100 = aloitusarvo)
   - Vertailu valittuihin markkinaindekseihin
   - Päivittäiset tuotot erillisessä paneelissa

3. **Strategioiden tilanne**:
   - Tili A: EMA Trend (klassinen)
   - Tili B: XGBoost ML (koneoppiminen)
   - Tili C: Ei aktiivisia strategioita

4. **Kauppahistoria**:
   - Viimeisimmät 20 kauppaa
   - Lataa täydellinen historia CSV-tiedostona
   - Suodatus strategioittain

## 🚀 Käynnistysohjeet

### 1. Käynnistä Dashboard:
```bash
python launch_dashboard.py
```

### 2. Käynnistä Paper Trading (toisessa terminaalissa):
```bash
python engine/live.py --paper
```

### 3. Avaa Dashboard:
- Automaattisesti: http://localhost:8501
- Tai klikkaa terminaalissa näkyvää linkkiä

## ⚙️ Dashboard Asetukset

### Sivupaneelin asetukset:
- **Automaattinen päivitys**: Päivittää dataa 30 sekunnin välein
- **Vertailuindeksit**: Valitse mitkä indeksit näytetään
- **Aikaväli**: 30-365 päivää

### Päivitys:
- 🔄 Manuaalinen päivitys -nappi
- Automaattinen päivitys (oletuksena päällä)
- Cache tyhjentyy automaattisesti

## 📈 Tilitilanne

### ✅ Valmiit Tilit:
- **Tili A (Klassinen)**: EMA Trend strategia - 35% pääomasta
- **Tili B (ML)**: XGBoost strategia - 35% pääomasta

### ⚠️ Ei-aktiiviset:
- **Tili C**: Ei strategioita käytössä
- **TCN Neural Network**: Käytettävissä mutta ei aktiivinen

## 🔧 Tekniset Tiedot

### Riippuvuudet:
- `streamlit` - Dashboard framework
- `plotly` - Interaktiiviset kuvaajat  
- `yfinance` - Markkinadata vertailuindeksejä varten
- `pandas` - Datan käsittely
- `numpy` - Numeeriset laskut

### Tiedostot:
- `storage/logs/trades_*.csv` - Kaupankäyntilokit
- `config/strategies.yml` - Strategioiden konfiguraatio
- `ui/trading_dashboard.py` - Dashboard-koodi
- `launch_dashboard.py` - Käynnistysskripti

## 📊 Datan Tulkinta

### Normalisoidut Arvot:
- Kaikki arvot alkavat 100:sta helpompaa vertailua varten
- Portfolio ja indeksit samalla asteikolla
- Prosentuaalinen muutos näkyy suoraan

### Vertailuindeksit (Verrokki):
- **NASDAQ**: Teknologia-painotteinen
- **S&P 500**: Laaja markkina-indeksi
- **SPY/QQQ**: ETF:t joilla voit verrata strategioidesi suorituskykyä

### Mittarit:
- **P&L**: Profit & Loss (voitto/tappio euroina)
- **Tuotto-%**: Prosentuaalinen tuotto aloituspääomasta
- **Päivätuotto**: Edellisen päivän muutos
- **Sharpe Ratio**: Riskikorjattu tuotto (laskettuna automaattisesti)

## 🎯 Käyttötips

1. **Pidä dashboard auki** kaupankäynnin aikana seurantaa varten
2. **Vertaile indekseihin** nähdäksesi onko strategiasi parempi kuin markkinat
3. **Seuraa kauppojen määrää** - liikaa kauppoja voi syödä voittoja
4. **Tarkkaile P&L trendiä** - onko kehitys positiivinen pitkällä aikavälillä
5. **Lataa data CSV:nä** syvempää analyysiä varten

## ⚡ Pikakomennot

```bash
# Käynnistä koko systeemi:
python launch_dashboard.py      # Terminal 1: Dashboard
python engine/live.py --paper   # Terminal 2: Paper trading

# Generoi testidata:
python scripts/generate_sample_trades.py

# Pysäytä:
Ctrl+C molemmissa terminaaleissa
```

## Päivitys (2025-08-28)

- Dashboard integrates with the offline accelerated replay (`engine/live_offline.py`) for quick demo runs when markets are closed. Run the offline engine in another terminal and refresh the dashboard to view simulated trades.
- Model-check utilities (`scripts/check_models_loadable.py`, `scripts/repair_model_reports.py`) are available to verify that deployed model artifacts are loadable before starting paper-live.

---
**Valmis käyttöön! Dashboard näyttää reaaliaikaisesti portfoliosi suorituskyvyn vs. markkinaindeksit.** 🎉
