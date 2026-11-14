# Trained detector artifact

The trained detector is committed in this directory as `hpcguard_final.pkl`.

It was exported from the Colab notebook using `joblib.dump`. The artifact contains:
- the fitted preprocessing + Isolation Forest pipeline;
- the canonical feature order: `cache-references`, `cycles`, `LLC-loads`;
- the clean-calibration anomaly threshold;
- the calibration percentile and random-state metadata.

Load it with `joblib.load("models/hpcguard_final.pkl")`. The prediction script expects the keys `model`, `features`, and `threshold`.

To retrain from the permitted clean CSV, run from the repository root:

```bash
pip install -r requirements.txt
python src/train.py --csv data/trace.csv
```

This regenerates the artifact and writes `results/calibration_summary.json`.

**Important:** The calibration alert rate is not backdoor detection accuracy. Evaluate on the organizers' labelled validation set before reporting Accuracy, TPR, FPR, F1, or AUROC.
