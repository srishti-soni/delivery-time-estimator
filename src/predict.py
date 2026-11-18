"""Inference helpers: load the saved bundle and turn raw shipment inputs into an estimate and an interval.

The bundle is a plain dict written by notebooks/03_modeling.ipynb with joblib:

    models      {"median", "q0.05", "q0.1", "q0.9", "q0.95"}  LightGBM regressors on log1p(days)
    conformal   {"0.8": q, "0.9": q}                          conformal widening, in log1p(days)
    ranges      training min/max of each numeric input
    metrics     test-set metrics recorded at training time
    ...

Only the notebook and this module know how the bundle is built, so the Streamlit app stays thin.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:  # package import (tests, app) or plain import (Colab, where only the two files are uploaded)
    from .features import TIER_ORDER, add_features, delay_risk
except ImportError:  # pragma: no cover
    from features import TIER_ORDER, add_features, delay_risk

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "transit_model.joblib"
LEVELS = (0.8, 0.9)


def load_model(path=DEFAULT_MODEL_PATH) -> dict:
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}. Run notebooks/03_modeling.ipynb first.")
    return joblib.load(path)


def _bounds(level: float) -> tuple[str, str]:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level}")
    alpha = round((1 - level) / 2, 4)
    return f"q{alpha:g}", f"q{1 - alpha:g}"


def predict_frame(bundle: dict, df: pd.DataFrame, level: float = 0.9) -> pd.DataFrame:
    """Estimate and interval (in days) for every row of `df`.

    `df` needs: weight_kg, distance_km, warehouse_load, carrier_tier, dispatch_date.
    Returns estimate_days, lower_days, upper_days with lower <= estimate <= upper.
    """
    lo_name, hi_name = _bounds(level)
    X = add_features(df)
    m = bundle["models"]
    q = bundle["conformal"][str(level)]

    est = m["median"].predict(X)
    lo = m[lo_name].predict(X) - q
    hi = m[hi_name].predict(X) + q
    # independent quantile models can cross; keep the interval around the estimate
    lo, hi = np.minimum(lo, est), np.maximum(hi, est)

    out = pd.DataFrame(
        {
            "estimate_days": np.expm1(est),
            "lower_days": np.maximum(np.expm1(lo), 0.0),
            "upper_days": np.expm1(hi),
        },
        index=df.index,
    )
    return out


def validate_inputs(bundle: dict, weight_kg, distance_km, carrier_tier, warehouse_load) -> list[str]:
    """Raise ValueError for impossible inputs; return warnings for inputs outside the training range."""
    if carrier_tier not in TIER_ORDER:
        raise ValueError(f"carrier_tier must be one of {TIER_ORDER}, got {carrier_tier!r}")
    if not weight_kg > 0:
        raise ValueError("weight_kg must be greater than 0")
    if not distance_km > 0:
        raise ValueError("distance_km must be greater than 0")
    if not warehouse_load >= 1:
        raise ValueError("warehouse_load must be at least 1")

    warnings = []
    r = bundle["ranges"]
    for name, value in (("weight_kg", weight_kg), ("distance_km", distance_km), ("warehouse_load", warehouse_load)):
        lo, hi = r[name]
        if value < lo or value > hi:
            warnings.append(
                f"{name} = {value:g} is outside the range seen in training ({lo:g} to {hi:g}). "
                "The estimate is an extrapolation and less reliable."
            )
    return warnings


def predict_one(
    bundle: dict,
    weight_kg: float,
    distance_km: float,
    carrier_tier: str,
    warehouse_load: float,
    dispatch_date,
    level: float = 0.9,
    promised_days: float | None = None,
) -> dict:
    """Single-shipment prediction used by the Streamlit app."""
    warnings = validate_inputs(bundle, weight_kg, distance_km, carrier_tier, warehouse_load)
    row = pd.DataFrame(
        [
            {
                "weight_kg": weight_kg,
                "distance_km": distance_km,
                "carrier_tier": carrier_tier,
                "warehouse_load": warehouse_load,
                "dispatch_date": pd.Timestamp(dispatch_date).normalize(),
            }
        ]
    )
    p = predict_frame(bundle, row, level).iloc[0]
    result = {
        "estimate_days": float(p["estimate_days"]),
        "lower_days": float(p["lower_days"]),
        "upper_days": float(p["upper_days"]),
        "level": level,
        "warnings": warnings,
    }
    for k in ("estimate", "lower", "upper"):
        result[f"{k}_hours"] = result[f"{k}_days"] * 24
    if promised_days is not None:
        result["delay_risk"] = delay_risk(result["estimate_days"], result["upper_days"], promised_days)
    return result
