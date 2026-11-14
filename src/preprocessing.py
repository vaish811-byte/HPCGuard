"""Input validation and feature construction for HPCGuard."""
import numpy as np
import pandas as pd

RAW_FEATURES = ["cache-references", "cycles", "LLC-loads"]
ENGINEERED_FEATURES = RAW_FEATURES + [
    "LLC_cache_ratio", "cache_cycles_ratio", "LLC_cycles_ratio"
]

def validate_raw_features(data):
    """Return numeric HPC counters in canonical feature order."""
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    missing = [c for c in RAW_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required HPC columns: {missing}")
    frame = frame[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("HPC data contains missing, non-numeric, or infinite values.")
    return frame

def engineer_features(data):
    """Create the three derived ratios explored in the experiments."""
    frame = validate_raw_features(data)
    if (frame["cache-references"] == 0).any() or (frame["cycles"] == 0).any():
        raise ValueError("cache-references and cycles must be non-zero.")
    out = frame.copy()
    out["LLC_cache_ratio"] = out["LLC-loads"] / out["cache-references"]
    out["cache_cycles_ratio"] = out["cache-references"] / out["cycles"]
    out["LLC_cycles_ratio"] = out["LLC-loads"] / out["cycles"]
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Feature engineering produced non-finite values.")
    return out[ENGINEERED_FEATURES]
