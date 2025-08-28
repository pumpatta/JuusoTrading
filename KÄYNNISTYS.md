# 🚀 JuusoTrader - Käynnistysohje

## Nopea käynnistys

### Windows (Command Prompt)
```cmd
start_trading.bat
```

### Windows (PowerShell)
```powershell
.\start_trading.ps1
```

## Mitä tapahtuu

1. **Trading Engine käynnistyy taustalle** 📈
   - Lataa XGB ML-mallit
   - Käynnistää Account C uutisanalyysin (FinBERT)
   - Alkaa generoimaan kauppasignaaleja

2. **Dashboard aukeaa selaimessa** 📊
   - URL: http://localhost:8501
   - Näyttää reaaliaikaisen portfolion tilan
   - Vertailee S&P500 ja NASDAQ indekseihin

## Trading Accountit

| Account | Strategia | Pääoma | Kuvaus |
|---------|-----------|--------|---------|
| **A** | EMA Trend | 30% | Klassinen trendin seuranta |
| **B** | XGB ML | 30% | Koneoppiminen (XGBoost) |
| **C** | Enhanced ML | 40% | ML + uutisanalyysi + pattern recognition |

## Ominaisuudet

### Account C - Enhanced ML
- ✅ **Live uutisanalyysi** FinBERT:llä 
- ✅ **Sentiment scoring** 50+ artikkeli/kierros
- ✅ **Pattern recognition** (Head & Shoulders)
- ✅ **Ensemble consensus** (EMA + XGB)
- ✅ **Jatkuva oppiminen** tuloksista

### Paper Trading
- ✅ Toimii ilman oikeaa rahaa
- ✅ Käyttää sample dataa kun markkinat kiinni
- ✅ SIP data error -korjaus
- ✅ Automaattinen fallback sample dataan

## Dashboard Näkymät

1. **📊 Portfolio Performance** - Kokonaissuorituskyky ajan kuluessa
2. **💰 Account Breakdown** - Account A/B/C erikseen  
3. **📈 Benchmark Comparison** - Vertailu SPY/QQQ/NASDAQ
4. **📋 Trade History** - Kaupankäyntihistoria strategioittain
5. **⚙️ System Status** - Live engine tila ja signaalit

## Pysäyttäminen

Paina **Ctrl+C** dashboard terminaalissa pysäyttääksesi sekä dashboardin että trading enginen.

## Ongelmanratkaisu

### Dashboard näyttää tyhjältä
- Odota 1-2 minuuttia että trading engine generoi ensimmäiset signaalit
- Tarkista että `storage/logs/` hakemistossa on kauppalogeja
- Päivitä dashboard (F5)

### "SIP data subscription" virhe
- ✅ **Korjattu!** Järjestelmä siirtyy automaattisesti sample data tilaan
- Toimii normaalisti ilman maksullista Alpaca SIP tilausta

### Live engine jumissa
- Ensimmäinen käynnistys kestää 2-3 minuuttia (FinBERT malli latautuu)
- Uutisanalyysi hakee 50+ artikkelia joka kierros
- Normal käyttäytyminen markkinoiden ollessa kiinni

## Kehittäjäinfoa

```bash
# Suorita vain trading engine
python engine/live.py --paper

# Suorita vain dashboard  
python launch_dashboard.py

# Offline testing
python engine/live_offline.py --offline

# Strategies config
config/strategies.yml
```

## Päivitys (2025-08-28)

- Lisätty offline-kiihtyvä replay-tila (`engine/live_offline.py`) joka mahdollistaa nopean demon ja non-blocking testauksen sample-datalla.
- Model-tarkistus- ja korjaustyökalut löytyvät `scripts/check_models_loadable.py` ja `scripts/repair_model_reports.py`.
