import yfinance as yf
import pandas as pd
import numpy as np
from sentiment import get_sentiment_news

RSI_PERIOD, SMA_FAST, SMA_SLOW, ATR_PERIOD = 14, 20, 50, 14
TARGET_RETURN, LOOKAHEAD_DAYS = 0.05, 5

def rsi(series, period=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), (-delta).clip(lower=0)
    rs = up.rolling(period).mean() / down.rolling(period).mean()
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    hl = (df['High'] - df['Low']).abs()
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def build_features(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    if df.empty: return None, None
    
    df["SMA20"], df["SMA50"] = df["Close"].rolling(SMA_FAST).mean(), df["Close"].rolling(SMA_SLOW).mean()
    df["RSI"], df["ATR"] = rsi(df["Close"]), atr(df)
    df = df.dropna()
    
    future_returns = df["Close"].pct_change(LOOKAHEAD_DAYS).shift(-LOOKAHEAD_DAYS)
    df["Label"] = (future_returns > TARGET_RETURN).astype(int)
    
    sent = get_sentiment_news(ticker)
    df["Sentiment"] = sent
    
    features = df[["RSI","SMA20","SMA50","ATR"]].copy()
    features["Sentiment"] = sent
    
    return features, df["Label"]
