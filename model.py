import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def train_ensemble(X, y):
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    logreg = LogisticRegression(max_iter=1000)

    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("lr", logreg)],
        voting="soft"
    )
    ensemble.fit(X, y)
    return ensemble

def predict_probabilities(model, X_new: pd.DataFrame):
    return model.predict_proba(X_new)[:,1]
