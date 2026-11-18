"""Feature engineering shared by the modelling notebook and the Streamlit app.

The saved model contains only scikit-learn / LightGBM objects. All custom logic lives
here and is applied *before* the model, so the artifact never needs this module to load.
"""
import numpy as np
import pandas as pd

TIER_ORDER = ["Economy", "Standard", "Express"]
TIER_TO_INT = {t: i for i, t in enumerate(TIER_ORDER)}

RAW_COLUMNS = ["weight_kg", "distance_km", "warehouse_load", "carrier_tier", "dispatch_date"]
TARGET = "transit_days"

FEATURES = [
    "log_weight",
    "log_distance",
    "warehouse_load",
    "tier_ord",
    "dispatch_dow",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model-ready feature table (columns = FEATURES) for raw shipment rows."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    unknown = set(df["carrier_tier"].astype(str)) - set(TIER_ORDER)
    if unknown:
        raise ValueError(f"unknown carrier_tier values: {sorted(unknown)}")

    dates = pd.to_datetime(df["dispatch_date"])
    out = pd.DataFrame(index=df.index)
    out["log_weight"] = np.log1p(df["weight_kg"].astype(float))
    out["log_distance"] = np.log1p(df["distance_km"].astype(float))
    out["warehouse_load"] = df["warehouse_load"].astype(float)
    out["tier_ord"] = df["carrier_tier"].astype(str).map(TIER_TO_INT).astype(float)
    out["dispatch_dow"] = dates.dt.dayofweek.astype(float)
    return out[FEATURES]


def delay_risk(estimate_days, upper_days, promised_days):
    """Delay-risk label against a promised delivery time.

    High   - the estimate itself is later than the promise
    Medium - the estimate is on time but the upper bound is later than the promise
    Low    - even the upper bound is within the promise
    """
    est, up = np.asarray(estimate_days), np.asarray(upper_days)
    out = np.where(est > promised_days, "High", np.where(up > promised_days, "Medium", "Low"))
    return out.item() if out.ndim == 0 else out
