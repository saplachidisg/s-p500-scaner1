[README.md](https://github.com/user-attachments/files/21955821/README.md)
# 📈 S&P 500 Weekly Stock Scanner (Optimized)

Optimized Streamlit app that:
- ✅ Scans **S&P 500** stocks for probability of **>5% move in 5 trading days**
- ✅ Uses **technical indicators, FinBERT sentiment**
- ✅ Ensemble ML (XGBoost + RandomForest + Logistic Regression)
- ✅ Walk-forward backtesting
- ✅ Performance log with stop-loss (-3%) & target (+5%)
- ✅ Equity curve tracking
- ✅ Works on iPhone via Streamlit Cloud

## 🚀 Optimizations
- Lookback reduced to **2 years**
- Multiprocessing (parallel scan)
- Cached Yahoo Finance data (24h)
- Cached FinBERT sentiment (1h)
- Progress bar in UI
- Clear cache button

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push repo to GitHub  
2. Deploy `app.py`  
3. Open public URL → Add to iPhone Home Screen → ✅ native-like app
