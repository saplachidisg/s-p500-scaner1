import os
import pandas as pd
import datetime as dt
import yfinance as yf
import streamlit as st
from features import build_features
from model import train_ensemble, predict_probabilities

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# -------------------------------------
# Ρυθμίσεις
# -------------------------------------
LOG_FILE = "predictions_log.xlsx"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# --- Google Drive ---
def get_drive_service():
    """Σύνδεση με Google Drive API"""
    flow = InstalledAppFlow.from_client_config(
        {
            "installed": {
                "client_id": st.secrets["GOOGLE_CLIENT_ID"],
                "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        SCOPES,
    )
    creds = flow.run_local_server(port=0)
    return build("drive", "v3", credentials=creds)


def upload_to_drive(file_path, folder_id=None):
    """Ανεβάζει το αρχείο στο Google Drive και επιστρέφει link"""
    service = get_drive_service()
    file_metadata = {"name": os.path.basename(file_path)}
    if folder_id:
        file_metadata["parents"] = [folder_id]

    media = MediaFileUpload(
        file_path,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    file = service.files().create(
        body=file_metadata, media_body=media, fields="id"
    ).execute()

    file_id = file.get("id")
    link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    return link


# --- Cached Yahoo Finance batch download (24 ώρες) ---
@st.cache_data(ttl=86400)
def get_history_batch(tickers, start, end):
    return yf.download(
        tickers, start=start, end=end, interval="1d", group_by="ticker", progress=False
    )


# --- Weekly Scan ---
def scan_sp500(limit=100):
    tickers = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]["Symbol"].tolist()[:limit]

    end = dt.date.today()
    start = end - dt.timedelta(days=365 * 2)

    big_data = get_history_batch(tickers, start, end)

    all_probs = []
    progress = st.progress(0)

    for i, ticker in enumerate(tickers):
        try:
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
        old["Date"] = pd.to_datetime(old["Date"], errors="coerce")
        df = pd.concat([old, df], ignore_index=True)

    df.to_excel(LOG_FILE, index=False)

    # --- Ανεβάζουμε στο Google Drive ---
    try:
        link = upload_to_drive(LOG_FILE)
        st.success(f"📂 Το Excel ανέβηκε στο Google Drive: [Άνοιξέ το εδώ]({link})")
    except Exception as e:
        st.error(f"⚠️ Αποτυχία upload στο Google Drive: {e}")

    return df.sort_values("Prob_5perc", ascending=False)


# --- Walk-Forward Backtest ---
def walk_forward_backtest(
    ticker, start_year=2010, end_year=2025, train_window=3, test_window=1
):
    results = []
    for year in range(start_year, end_year - (train_window + test_window)):
        train_start = f"{year}-01-01"
        train_end = f"{year+train_window}-01-01"
        test_end = f"{year+train_window+test_window}-01-01"

        feats, labels = build_features(ticker, train_start, test_end)
        if feats is None or len(feats) < 100:
            continue

        train_mask = feats.index < train_end
        test_mask = (feats.index >= train_end) & (feats.index < test_end)

        X_train, y_train = feats[train_mask], labels[train_mask]
        X_test, y_test = feats[test_mask], labels[test_mask]

        if y_train.sum() < 5 or len(y_test) < 10:
            continue

        model = train_ensemble(X_train, y_train)
        probs = predict_probabilities(model, X_test)
        preds = (probs > 0.5).astype(int)

        hit_rate = (preds == y_test).mean()
        precision = ((preds & y_test).sum() / max(preds.sum(), 1))
        results.append((year, hit_rate, precision))

    return pd.DataFrame(results, columns=["Year", "HitRate", "Precision"])


# --- Evaluate Predictions ---
def evaluate_predictions(max_days=10):
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    df = pd.read_excel(LOG_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    results = []

    for _, row in df.iterrows():
        days_passed = (dt.date.today() - row["Date"].date()).days
        if days_passed < max_days:
            continue

        try:
            hist = yf.download(
                row["Ticker"],
                start=row["Date"],
                end=row["Date"] + pd.Timedelta(days=max_days + 5),
                progress=False,
            )
            if hist.empty:
                continue

            entry_price = row["EntryPrice"]
            lows, highs, closes = (
                hist["Low"].values,
                hist["High"].values,
                hist["Close"].values,
            )

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
            results.append(
                (
                    row["Date"].date(),
                    row["Ticker"],
                    entry_price,
                    closes[-1],
                    ret,
                    outcome,
                    hit,
                )
            )
        except Exception as e:
            print(f"⚠️ Eval error {row['Ticker']}: {e}")
            continue

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(
        results,
        columns=[
            "Date",
            "Ticker",
            "EntryPrice",
            "FuturePrice",
            "Return",
            "Outcome",
            "Hit",
        ],
    )
