import os
import pandas as pd
import datetime as dt
import yfinance as yf
import streamlit as st
from features import build_features
from model import train_ensemble, predict_probabilities

LOG_FILE = "predictions_log.xlsx"

# --- Cached Yahoo Finance data (24 ώρες) ---
@st.cache_data(ttl=86400)
def get_history_batch(tickers, start, end):
    return yf.download(tickers, start=start, end=end, interval="1d", group_by="ticker", progress=False)

def scan_sp500(limit=100):
    tickers = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]["Symbol"].tolist()[:limit]

    end = dt.date.today()
    start = end - dt.timedelta(days=365 * 2)

    # Κατεβάζουμε ΟΛΑ τα tickers με ένα call
    big_data = get_history_batch(tickers, start, end)

    all_probs = []
    progress = st.progress(0)

    for i, ticker in enumerate(tickers):
        try:
            # Πάρε τα δεδομένα για το συγκεκριμένο ticker
            df = big_data[ticker].dropna()
            if df.empty:
                continue

            feats, labels = build_features(ticker, start, end)
            if feats is None or labels.sum() < 5:
                continue

            X_train, y_train = feats.iloc[:-20], labels.iloc[:-20]
            X_test = feats.iloc[-1:].copy()

            model = train_ensemble(X_train, y_train)
            prob = predict_probabilities(model, X_test)[0]

            last_price = df["Close"].iloc[-1]
            all_probs.append((ticker, prob, last_price))

        except Exception as e:
            print(f"⚠️ Error {ticker}: {e}")
            continue

        progress.progress((i + 1) / len(tickers))

    if not all_probs:
        return pd.DataFrame()

    df = pd.DataFrame(all_probs, columns=["Ticker", "Prob_5perc", "EntryPrice"])
    df["Date"] = pd.to_datetime(end)

    if os.path.exists(LOG_FILE):
        old = pd.read_excel(LOG_FILE)
        df = pd.concat([old, df], ignore_index=True)

    df.to_excel(LOG_FILE, index=False)
    return df.sort_values("Prob_5perc", ascending=False)
