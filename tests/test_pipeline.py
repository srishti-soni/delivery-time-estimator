"""Tests for src/features.py, src/predict.py and the saved model bundle.

Run from the project root:  pytest -q
The model tests need models/transit_model.joblib (produced by notebooks/03_modeling.ipynb).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import predict  # noqa: E402
from src.features import FEATURES, add_features, delay_risk  # noqa: E402

MODEL_PATH = ROOT / "models" / "transit_model.joblib"
needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="model file not built yet")


def _rows(**over):
    base = {"weight_kg": 1.0, "distance_km": 500.0, "warehouse_load": 3, "carrier_tier": "Standard",
            "dispatch_date": "2018-06-04"}  # a Monday
    base.update(over)
    return pd.DataFrame([base])


# ------------------------------------------------------------------ features
def test_add_features_columns_and_values():
    out = add_features(_rows())
    assert list(out.columns) == FEATURES
    assert out.loc[0, "log_distance"] == pytest.approx(np.log1p(500.0))
    assert out.loc[0, "log_weight"] == pytest.approx(np.log1p(1.0))
    assert out.loc[0, "tier_ord"] == 1.0          # Standard
    assert out.loc[0, "dispatch_dow"] == 0.0      # Monday


@pytest.mark.parametrize("tier,expected", [("Economy", 0.0), ("Standard", 1.0), ("Express", 2.0)])
def test_tier_encoding(tier, expected):
    assert add_features(_rows(carrier_tier=tier)).loc[0, "tier_ord"] == expected


def test_add_features_rejects_unknown_tier():
    with pytest.raises(ValueError, match="unknown carrier_tier"):
        add_features(_rows(carrier_tier="Platinum"))


def test_add_features_rejects_missing_column():
    with pytest.raises(ValueError, match="missing columns"):
        add_features(_rows().drop(columns="warehouse_load"))


@pytest.mark.parametrize(
    "est,up,promise,expected",
    [(12, 20, 10, "High"), (8, 14, 10, "Medium"), (5, 9, 10, "Low"), (10, 10, 10, "Low")],
)
def test_delay_risk(est, up, promise, expected):
    assert delay_risk(est, up, promise) == expected


def test_delay_risk_vectorised():
    out = delay_risk(np.array([12, 5]), np.array([20, 9]), 10)
    assert list(out) == ["High", "Low"]


# ------------------------------------------------------------------ model
@pytest.fixture(scope="module")
def bundle():
    return predict.load_model(MODEL_PATH)


@needs_model
def test_bundle_has_expected_parts(bundle):
    assert {"models", "conformal", "ranges", "metrics", "features"} <= set(bundle)
    assert bundle["features"] == FEATURES
    assert {"median", "q0.05", "q0.1", "q0.9", "q0.95"} <= set(bundle["models"])
    assert set(bundle["conformal"]) == {"0.8", "0.9"}
    assert all(v >= 0 for v in bundle["conformal"].values())


@needs_model
@pytest.mark.parametrize("level", [0.8, 0.9])
def test_interval_is_ordered_and_positive(bundle, level):
    df = pd.DataFrame({
        "weight_kg": np.linspace(0.05, 40, 200), "distance_km": np.linspace(5, 3000, 200),
        "warehouse_load": np.tile([1, 3, 10, 30], 50), "carrier_tier": np.tile(["Economy", "Standard", "Express", "Standard"], 50),
        "dispatch_date": "2018-07-10",
    })
    out = predict.predict_frame(bundle, df, level)
    assert (out["lower_days"] >= 0).all()
    assert (out["lower_days"] <= out["estimate_days"]).all()
    assert (out["estimate_days"] <= out["upper_days"]).all()
    assert np.isfinite(out.values).all()


@needs_model
def test_90_interval_is_wider_than_80(bundle):
    df = _rows(distance_km=800.0)
    w90 = predict.predict_frame(bundle, df, 0.9).eval("upper_days - lower_days").iloc[0]
    w80 = predict.predict_frame(bundle, df, 0.8).eval("upper_days - lower_days").iloc[0]
    assert w90 > w80


@needs_model
def test_longer_distance_gives_longer_estimate(bundle):
    est = [predict.predict_one(bundle, 1.0, d, "Standard", 3, "2018-06-04")["estimate_days"] for d in (50, 500, 2000)]
    assert est[0] < est[1] < est[2]


@needs_model
def test_predict_one_shape_and_hours(bundle):
    r = predict.predict_one(bundle, 2.0, 700, "Economy", 5, "2018-06-04", level=0.9, promised_days=10)
    assert r["estimate_hours"] == pytest.approx(r["estimate_days"] * 24)
    assert r["delay_risk"] in {"Low", "Medium", "High"}
    assert r["lower_days"] <= r["estimate_days"] <= r["upper_days"]
    assert r["warnings"] == []


@needs_model
def test_predict_one_matches_batch(bundle):
    one = predict.predict_one(bundle, 1.5, 640, "Express", 4, "2018-06-04", level=0.9)
    batch = predict.predict_frame(bundle, _rows(weight_kg=1.5, distance_km=640, carrier_tier="Express", warehouse_load=4), 0.9)
    assert one["estimate_days"] == pytest.approx(batch.loc[0, "estimate_days"])
    assert one["upper_days"] == pytest.approx(batch.loc[0, "upper_days"])


@needs_model
@pytest.mark.parametrize(
    "kwargs",
    [dict(weight_kg=0), dict(weight_kg=-1), dict(distance_km=0), dict(warehouse_load=0), dict(carrier_tier="Gold")],
)
def test_invalid_inputs_raise(bundle, kwargs):
    args = dict(weight_kg=1.0, distance_km=500, carrier_tier="Standard", warehouse_load=3)
    args.update(kwargs)
    with pytest.raises(ValueError):
        predict.predict_one(bundle, dispatch_date="2018-06-04", **args)


@needs_model
def test_out_of_range_input_warns_but_predicts(bundle):
    r = predict.predict_one(bundle, 1.0, 9000, "Standard", 3, "2018-06-04")
    assert any("distance_km" in w for w in r["warnings"])
    assert np.isfinite(r["estimate_days"])


@needs_model
def test_unsupported_level_raises(bundle):
    with pytest.raises(ValueError, match="level"):
        predict.predict_frame(bundle, _rows(), level=0.95)


@needs_model
def test_missing_model_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="03_modeling"):
        predict.load_model(tmp_path / "nope.joblib")


@needs_model
def test_saved_model_beats_baseline_on_recorded_test_metrics(bundle):
    m = bundle["metrics"]
    assert m["test"]["MAE"] < m["baseline_test"]["MAE"]
