# HPCGuard
**Clean-only anomaly detection using hardware performance counters (HPCs).**

## Problem
The training data contains benign inference traces only. HPCGuard learns normal execution patterns and flags statistically unusual traces for investigation. An anomaly is not proof of a backdoor.

## Dataset
Expected CSV: `data/trace.csv`. Required columns:
- `cache-references`
- `cycles`
- `LLC-loads`

The supplied dataset was inspected in Colab: 800 rows, 3 numeric columns, no missing values, no duplicate rows, and no constant columns. The CSV is not included here; add it only if competition rules permit redistribution. Never commit private validation data.

## Method and finalized implementation parameters
1. Validate the three HPC counters.
2. Split clean traces into 80% training and 20% calibration with `random_state=42`.
3. Fit StandardScaler on training rows only.
4. Fit Isolation Forest with `n_estimators=300`, `max_samples="auto"`, `contamination="auto"`, `random_state=42`, and `n_jobs=-1`.
5. Define anomaly score as `-decision_function(X)`; higher means more anomalous.
6. Set the threshold to the 95th percentile of clean calibration scores.
7. Save the pipeline, feature order, threshold, and metadata as `models/hpcguard_final.pkl`.

The Colab exploration also evaluated engineered ratios, PCA, and One-Class SVM; the recorded clean-calibration observations are in `results/model_comparison.csv`. The committed notebook currently reproduces the raw-feature Isolation Forest baseline only; it does not rerun all comparison experiments.

## Reproduce
```bash
pip install -r requirements.txt
python src/train.py --csv data/trace.csv
```
This creates `models/hpcguard_final.pkl` and `results/calibration_summary.json`.

## Predict
```bash
python src/predict.py --csv data/new_traces.csv --model models/hpcguard_final.pkl --output results/predictions.csv
```
Prediction: `0 = Normal`, `1 = Anomalous`.

## Clean-only experiment observations
The Colab experiments used 640 training and 160 calibration observations. Each model was calibrated at the 95th percentile and produced 8 calibration alerts (5%). The raw-feature Isolation Forest is retained as a simple provisional baseline, not because superior backdoor detection has been proven.

The ratio features are `LLC-loads/cache-references`, `cache-references/cycles`, and `LLC-loads/cycles`. They are derived from the original counters, not independent measurements.

## Limitations
- No backdoor-labelled training traces or private validation labels were available.
- A 5% clean calibration alert rate is not accuracy, TPR, F1, AUROC, or the private-validation FPR.
- Scheduling, CPU load, workload and environment variation can produce unusual HPC traces.
- A backdoor inference may not produce a detectable change in these three counters.
- Final performance claims require evaluation on the organizers' labelled validation set.

## Structure
```text
HPCGuard/
├── data/README.md
├── notebooks/HPCGuard_Anomaly_Detection.ipynb
├── models/                 # generated .pkl artifact after training
├── src/train.py
├── src/predict.py
├── src/preprocessing.py
├── results/model_comparison.csv
├── report/HPCGuard_Report.pdf
└── requirements.txt
```
