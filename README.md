# 🎯 NSE Scanner Pro v3.0 — Regime-Aware Trading Platform

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Secrets (Streamlit Cloud → Settings → Secrets)
Paste ONLY these lines (**no backticks**):
```
BREEZE_API_KEY = "your_key"
BREEZE_API_SECRET = "your_secret"
BREEZE_SESSION_TOKEN = "daily_token"
TELEGRAM_BOT_TOKEN = "optional"
TELEGRAM_CHAT_ID = "optional"
```

## v3.0 Changes

| Feature | What Changed |
|---------|-------------|
| 🧠 **Market Regime Engine** | 4 regimes: Expansion, Accumulation, Distribution, Panic. Blocks unsuitable strategies automatically. |
| 🚫 **Intraday Proxies Killed** | ORB, VWAP, Lunch Low require Breeze. No more fake signals from daily data. |
| 💪 **RS Filter** | Long signals require RS > 70 (top 30% stocks vs Nifty). Configurable. |
| 🗺️ **Sector Filter** | Buys in weak sectors get confidence penalty. Strong sectors get bonus. |
| 🎯 **Daily Focus Panel** | Time-aware panel showing exactly what to do NOW based on time of day. |
| ⏱️ **Data Staleness** | Visual warning when scan data is > 15 minutes old. |
| 📱 **Mobile Fix** | Auto-collapse sidebar, responsive CSS, smaller fonts on mobile. |
| 📈 **Charts** | Candlestick with EMA/Volume/RSI + trade levels overlay. |
| 🔄 **Weekly Alignment** | Multi-timeframe confirmation (4-point weekly check) on every signal. |
| 📓 **Journal** | Full trade lifecycle with equity curve and strategy-level analytics. |

## Regime Behavior

| Regime | Position | Ideal Strategies | Blocked |
|--------|---:|---|---|
| 🟢 EXPANSION | 100% | VCP, 52WH, ORB, ATH | — |
| 🟡 ACCUMULATION | 60% | VCP, EMA21, VWAP | — |
| 🟠 DISTRIBUTION | 35% | Short, Mean-reversion | VCP, 52WH, ATH |
| 🔴 PANIC | 15% | Shorts only | All longs |
