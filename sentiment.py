import yfinance as yf
from transformers import pipeline
import numpy as np
import streamlit as st

# Cached FinBERT sentiment (refresh every 1 hour)
@st.cache_data(ttl=3600)
def get_sentiment_news(ticker: str) -> float:
    try:
        headlines = yf.Ticker(ticker).news
        if not headlines:
            return 0.0

        # Πάρε μόνο όσα έχουν 'title'
        texts = [h.get("title", "") for h in headlines[:5] if "title" in h]

        if not texts:
            return 0.0

        sentiment_model = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
        results = sentiment_model(texts)
        scores = [1 if r['label'] == "positive" else -1 if r['label'] == "negative" else 0 for r in results]
        return float(np.mean(scores))

    except Exception as e:
        # Αν κάτι πάει στραβά → γύρνα ουδέτερο sentiment
        return 0.0
