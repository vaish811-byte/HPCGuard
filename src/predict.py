"""Predict normal/anomalous labels for new HPC traces."""
import argparse
import joblib
import numpy as np
import pandas as pd

def predict(input_csv, model_path, output_csv):
    artifact = joblib.load(model_path)
    features, threshold, model = artifact["features"], artifact["threshold"], artifact["model"]
    df = pd.read_csv(input_csv)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    X = df[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input contains missing, non-numeric, or infinite values.")
    scores = -model.decision_function(X)
    predictions = (scores > threshold).astype(int)
    result = df.copy()
    result["anomaly_score"] = scores
    result["prediction"] = predictions
    result["label"] = np.where(predictions == 1, "Anomalous", "Normal")
    result.to_csv(output_csv, index=False)
    print(result[["anomaly_score", "prediction", "label"]].head())
    print("Saved:", output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", default="models/hpcguard_final.pkl")
    parser.add_argument("--output", default="results/predictions.csv")
    a = parser.parse_args()
    predict(a.csv, a.model, a.output)
