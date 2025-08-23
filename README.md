# 📈 S&P 500 Weekly Stock Scanner

Streamlit app that:
- ✅ Scans **S&P 500** stocks for probability of **>5% move in 5 trading days**
- ✅ Uses **technical indicators, fundamentals, FinBERT sentiment**
- ✅ Ensemble ML (XGBoost + RandomForest + Logistic Regression)
- ✅ Walk-forward backtesting
- ✅ Performance log with stop-loss (-3%) & target (+5%)
- ✅ Equity curve tracking
- ✅ Works perfectly on iPhone via Streamlit Cloud

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy to Streamlit Cloud
1. Push repo to GitHub  
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)  
3. Select `app.py`  
4. Open public URL → iPhone Safari → Add to Home Screen → ✅ native-like app
