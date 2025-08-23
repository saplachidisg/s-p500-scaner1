import streamlit as st
from scanner import scan_sp500, walk_forward_backtest, evaluate_predictions

st.set_page_config(page_title="S&P 500 Scanner", layout="wide")
st.title("📈 S&P 500 Weekly Stock Scanner")

tab1, tab2, tab3 = st.tabs(["Weekly Scan", "Backtest", "Performance Log"])

with tab1:
    st.header("Weekly Scan")
    n_tickers = st.slider("Number of tickers to scan", 20, 500, 100, step=20)
    top_n = st.slider("Show Top N results", 5, 50, 10, step=5)
    if st.button("🚀 Run Scan"):
        df = scan_sp500(limit=n_tickers)
        if df.empty:
            st.error("⚠️ No results.")
        else:
            df = df.head(top_n)
            st.success(f"✅ Top {top_n} picks")
            st.dataframe(df)

with tab2:
    st.header("Walk-Forward Backtest")
    ticker = st.text_input("Enter ticker", "AAPL")
    if st.button("Run Backtest"):
        df = walk_forward_backtest(ticker)
        if df.empty:
            st.error("⚠️ Not enough data.")
        else:
            st.success("✅ Backtest complete")
            st.dataframe(df)
            st.metric("Mean Hit Rate", f"{df['HitRate'].mean():.2%}")
            st.metric("Mean Precision", f"{df['Precision'].mean():.2%}")

with tab3:
    st.header("Performance Evaluation of Past Picks (Stop -3%, Target +5%)")
    max_days = st.slider("Max days to keep trade open", 5, 15, 10)
    if st.button("📊 Evaluate Past Predictions"):
        perf = evaluate_predictions(max_days=max_days)
        if perf.empty:
            st.warning("⏳ Not enough old predictions yet.")
        else:
            st.success("✅ Performance Results")
            st.dataframe(perf)
            st.metric("Hit Rate", f"{perf['Hit'].mean():.2%}")
            st.metric("Average Return", f"{perf['Return'].mean():.2%}")
            perf["Equity"] = (1 + perf["Return"]).cumprod()
            st.line_chart(perf.set_index("Date")["Equity"])
