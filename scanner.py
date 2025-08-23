import os
import pandas as pd
import datetime as dt
import yfinance as yf
from features import build_features
from model import train_ensemble, predict_probabilities

LOG_FILE = "predictions_log.xlsx"

def scan_sp500(limit=100):
    tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].tolist()
    end = dt.date.today()
    start = end - dt.timedelta(days=365*5)

    all_probs = []
    for t in tickers[:limit]:
        try:
            feats, labels = build_features(t, start, end)
            if feats is None or labels.sum() < 5:
                continue
            X_train, y_train = feats.iloc[:-20], labels.iloc[:-20]
            X_test = feats.iloc[-1:].copy()

            model = train_ensemble(X_train, y_train)
            prob = predict_probabilities(model, X_test)[0]

            last_price = yf.Ticker(t).history(period="1d")["Close"].iloc[-1]
            all_probs.append((t, prob, last_price))
        except Exception as e:
            print(f"⚠️ {t}: {e}")
            continue

    if not all_probs: return pd.DataFrame()

    df = pd.DataFrame(all_probs, columns=["Ticker","Prob_5perc","EntryPrice"])
    df["Date"] = dt.date.today()

    if os.path.exists(LOG_FILE):
        old = pd.read_excel(LOG_FILE)
        df = pd.concat([old, df], ignore_index=True)
    df.to_excel(LOG_FILE, index=False)

    return df.sort_values("Prob_5perc", ascending=False)

def walk_forward_backtest(ticker, start_year=2010, end_year=2025, train_window=3, test_window=1):
    results = []
    for year in range(start_year, end_year - (train_window + test_window)):
        train_start = f"{year}-01-01"
        train_end   = f"{year+train_window}-01-01"
        test_end    = f"{year+train_window+test_window}-01-01"

        feats, labels = build_features(ticker, train_start, test_end)
        if feats is None or len(feats) < 100:
            continue
        
        train_mask = (feats.index < train_end)
        test_mask  = (feats.index >= train_end) & (feats.index < test_end)

        X_train, y_train = feats[train_mask], labels[train_mask]
        X_test, y_test   = feats[test_mask], labels[test_mask]

        if y_train.sum() < 5 or len(y_test) < 10:
            continue

        model = train_ensemble(X_train, y_train)
        probs = predict_probabilities(model, X_test)
        preds = (probs > 0.5).astype(int)

        hit_rate = (preds == y_test).mean()
        precision = ((preds & y_test).sum() / max(preds.sum(),1))
        results.append((year, hit_rate, precision))
    
    return pd.DataFrame(results, columns=["Year","HitRate","Precision"])

def evaluate_predictions(max_days=10):
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    df = pd.read_excel(LOG_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    results = []

    for _, row in df.iterrows():
        days_passed = (dt.date.today() - row["Date"].date()).days
        if days_passed < max_days:
            continue

        try:
            hist = yf.download(
                row["Ticker"],
                start=row["Date"],
                end=row["Date"] + pd.Timedelta(days=max_days+5),
                progress=False
            )
            if hist.empty:
                continue

            entry_price = row["EntryPrice"]
            lows, highs, closes = hist["Low"].values, hist["High"].values, hist["Close"].values

            stop_loss_price, target_price = entry_price * 0.97, entry_price * 1.05
            outcome, ret = "Hold", (closes[-1] - entry_price) / entry_price

            for low, high, close in zip(lows, highs, closes):
                if low <= stop_loss_price:
                    outcome, ret = "Stopped", -0.03
                    break
                elif high >= target_price:
                    outcome, ret = "Hit", (close - entry_price) / entry_price
                    break

            hit = 1 if outcome == "Hit" else 0
            results.append((row["Date"].date(), row["Ticker"], entry_price, closes[-1], ret, outcome, hit))
        except:
            continue

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results, columns=["Date","Ticker","EntryPrice","FuturePrice","Return","Outcome","Hit"])
