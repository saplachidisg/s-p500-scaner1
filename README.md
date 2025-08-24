# 📊 S&P 500 Stock Scanner with ML + Google Drive Upload

Αυτή η εφαρμογή σαρώνει τον S&P 500, εκπαιδεύει ML μοντέλο (ensemble) με τεχνικούς δείκτες + sentiment (FinBERT) και προβλέπει μετοχές που έχουν πιθανότητα ≥5% ανόδου μέσα σε 5 ημέρες.  

Φτιαγμένο με **Streamlit**, ώστε να τρέχει εύκολα στο Cloud ή τοπικά.  
Τα αποτελέσματα αποθηκεύονται αυτόματα σε **Google Drive**.

---

## 🚀 Χαρακτηριστικά
- Σκανάρει τον S&P 500 (Yahoo Finance data).
- Χρησιμοποιεί **FinBERT sentiment analysis** από ειδήσεις.
- ML Ensemble (XGBoost + Logistic Regression).
- Backtesting (walk-forward).
- Stop-loss -3% και take-profit +5%.
- 📂 Αυτόματο **upload Excel στο Google Drive** με τα αποτελέσματα.

---

## ⚙️ Εγκατάσταση

### 1. Clone το repo
```bash
git clone https://github.com/<username>/s-p500-scanner.git
cd s-p500-scanner
