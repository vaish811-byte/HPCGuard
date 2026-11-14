"""Train a clean-only HPCGuard Isolation Forest detector."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from preprocessing import RAW_FEATURES

RANDOM_STATE = 42
N_ESTIMATORS = 300
CALIBRATION_PERCENTILE = 95

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/trace.csv")
    parser.add_argument("--output", default="models/hpcguard_final.pkl")
    parser.add_argument("--results", default="results/calibration_summary.json")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    X = df[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features contain missing, non-numeric, or infinite values.")
    if len(X) < 20:
        raise ValueError("At least 20 clean rows are required.")

    X_train, X_cal = train_test_split(X, test_size=0.20, random_state=RANDOM_STATE)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("isolation_forest", IsolationForest(
            n_estimators=N_ESTIMATORS, max_samples="auto",
            contamination="auto", random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    model.fit(X_train)
    scores = -model.decision_function(X_cal)
    threshold = float(np.percentile(scores, CALIBRATION_PERCENTILE))
    alerts = int((scores > threshold).sum())

    artifact = {
        "project_name": "HPCGuard", "model_name": "IsolationForest",
        "model": model, "features": RAW_FEATURES, "threshold": threshold,
        "score_definition": "negative decision_function; higher is more anomalous",
        "calibration_percentile": CALIBRATION_PERCENTILE,
        "training_rows": len(X_train), "calibration_rows": len(X_cal),
        "n_estimators": N_ESTIMATORS, "random_state": RANDOM_STATE
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    summary = {
        "rows_total": len(X), "training_rows": len(X_train),
        "calibration_rows": len(X_cal), "features": RAW_FEATURES,
        "threshold": threshold, "calibration_alerts": alerts,
        "calibration_alert_rate_percent": 100 * alerts / len(X_cal),
        "note": "Calibration alert rate is not backdoor detection accuracy."
    }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Saved detector:", args.output)

if __name__ == "__main__":
    main()
