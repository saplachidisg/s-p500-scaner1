import yfinance as yf
from transformers import pipeline
import numpy as np
import streamlit as st

# Φορτώνουμε το FinBERT μία φορά global
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

finbert_model = load_finbert()

# Cached FinBERT sentiment (1 ώρα)
@st.cache_data(ttl=3600)
def get_sentiment_news(ticker: str) -> float:
    try:
        headlines = yf.Ticker(ticker).news
        if not headlines:
            return 0.0

        texts = [h.get("title", "") for h in headlines[:5] if "title" in h]
        if not texts:
            return 0.0

        results = finbert_model(texts)
        scores = [1 if r['label']=="positive" else -1 if r['label']=="negative" else 0 for r in results]
        return float(np.mean(scores))

    except Exception:
        return 0.0
