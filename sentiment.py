import yfinance as yf
from transformers import pipeline
import numpy as np

sentiment_model = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

def get_sentiment_news(ticker: str) -> float:
    try:
        headlines = yf.Ticker(ticker).news
        if not headlines:
            return 0.0
        texts = [h['title'] for h in headlines[:5]]
        results = sentiment_model(texts)
        scores = [1 if r['label']=="positive" else -1 if r['label']=="negative" else 0 for r in results]
        return float(np.mean(scores))
    except Exception as e:
        print(f"⚠️ Sentiment error {ticker}: {e}")
        return 0.0
