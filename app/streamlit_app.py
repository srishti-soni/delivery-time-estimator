"""Streamlit dispatch tool: live transit-time estimate with a confidence interval.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import TIER_ORDER  # noqa: E402
from src.predict import DEFAULT_MODEL_PATH, load_model, predict_frame, predict_one  # noqa: E402

st.set_page_config(page_title="Delivery Time Estimator", layout="wide")


@st.cache_resource
def get_bundle():
    return load_model(DEFAULT_MODEL_PATH)


try:
    bundle = get_bundle()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

ranges = bundle["ranges"]

import lightgbm, sklearn  # noqa: E402

_trained = bundle.get("versions", {})
_now = {"lightgbm": lightgbm.__version__, "scikit-learn": sklearn.__version__}
_diff = {k: (_trained.get(k), v) for k, v in _now.items() if _trained.get(k) and _trained[k] != v}
if _diff:
    st.warning(
        "Library versions differ from the ones used for training "
        + ", ".join(f"{k}: trained {a}, running {b}" for k, (a, b) in _diff.items())
        + ". Predictions should be checked before relying on them."
    )

st.title("Delivery Time and Transit Delay Estimator")
st.caption("Estimates carrier-pickup-to-delivery time for an order, with a confidence interval.")

# ---------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Shipment")
    weight = st.number_input("Package weight (kg)", min_value=0.01, max_value=500.0, value=1.0, step=0.1)
    distance = st.number_input("Distance, origin to destination (km)", min_value=1.0, max_value=6000.0, value=500.0, step=10.0)
    tier = st.selectbox("Carrier tier", TIER_ORDER, index=1)
    load = st.slider("Warehouse workload (orders the seller dispatches that day)", 1, 100, 3)
    dispatch_day = st.date_input("Dispatch date", value=date.today())
    dispatch_time = st.time_input("Dispatch time", value=time(9, 0))
    st.header("Estimate settings")
    level = st.radio("Interval level", [0.9, 0.8], format_func=lambda x: f"{int(x * 100)}%", horizontal=True)
    promised = st.number_input("Promised delivery time (days)", min_value=1.0, max_value=90.0, value=10.0, step=1.0)

# ---------------------------------------------------------------- prediction
try:
    res = predict_one(bundle, weight, distance, tier, load, dispatch_day, level, promised)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

dispatch_dt = datetime.combine(dispatch_day, dispatch_time)
arrive = lambda days: dispatch_dt + timedelta(days=days)

for w in res["warnings"]:
    st.warning(w)

c1, c2, c3 = st.columns(3)
c1.metric("Estimated transit", f"{res['estimate_days']:.1f} days", f"{res['estimate_hours']:.0f} hours", delta_color="off")
c2.metric(f"{int(level * 100)}% interval", f"{res['lower_days']:.1f} to {res['upper_days']:.1f} days")
c3.metric("Delay risk vs promise", res["delay_risk"], f"promise: {promised:g} days", delta_color="off")

st.subheader("Expected arrival")
a1, a2, a3 = st.columns(3)
a1.metric("Earliest plausible", arrive(res["lower_days"]).strftime("%d %b %Y, %H:%M"))
a2.metric("Most likely", arrive(res["estimate_days"]).strftime("%d %b %Y, %H:%M"))
a3.metric("Latest plausible", arrive(res["upper_days"]).strftime("%d %b %Y, %H:%M"))

fig = go.Figure()
fig.add_trace(go.Scatter(x=[res["lower_days"], res["upper_days"]], y=["transit", "transit"], mode="lines",
                         line=dict(width=14, color="#9ecae1"), name=f"{int(level * 100)}% interval"))
fig.add_trace(go.Scatter(x=[res["estimate_days"]], y=["transit"], mode="markers",
                         marker=dict(size=16, color="#08519c"), name="estimate"))
fig.add_vline(x=promised, line_dash="dash", line_color="firebrick", annotation_text="promise")
fig.update_layout(height=190, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="days in transit",
                  yaxis=dict(showticklabels=False), legend=dict(orientation="h"))
st.plotly_chart(fig, width="stretch")

st.info(
    "Read the interval as a plausible range, not a promise. On the test period a 90% interval covered about "
    f"{bundle['metrics']['coverage_test']['0.9'] * 100:.0f}% of deliveries."
)

# ---------------------------------------------------------------- what-if
st.subheader("What if the distance or the workload changes?")
tab1, tab2 = st.tabs(["Distance", "Warehouse workload"])


def sweep(column, values):
    base = {"weight_kg": weight, "distance_km": distance, "warehouse_load": load, "carrier_tier": tier,
            "dispatch_date": pd.Timestamp(dispatch_day)}
    rows = pd.DataFrame([{**base, column: v} for v in values])
    out = predict_frame(bundle, rows, level)
    out[column] = list(values)
    return out


def band_chart(out, column, label):
    f = go.Figure()
    f.add_trace(go.Scatter(x=out[column], y=out["upper_days"], line=dict(width=0), showlegend=False))
    f.add_trace(go.Scatter(x=out[column], y=out["lower_days"], fill="tonexty", line=dict(width=0),
                           fillcolor="rgba(158,202,225,0.5)", name=f"{int(level * 100)}% interval"))
    f.add_trace(go.Scatter(x=out[column], y=out["estimate_days"], line=dict(color="#08519c"), name="estimate"))
    f.update_layout(height=320, xaxis_title=label, yaxis_title="days in transit", margin=dict(l=10, r=10, t=10, b=10))
    return f


with tab1:
    dists = [int(x) for x in range(25, 3001, 25)]
    st.plotly_chart(band_chart(sweep("distance_km", dists), "distance_km", "distance (km)"), width="stretch")
with tab2:
    loads = list(range(1, 41))
    st.plotly_chart(band_chart(sweep("warehouse_load", loads), "warehouse_load", "warehouse workload (orders per day)"),
                    width="stretch")
    st.caption("Workload has only a small effect in the training data, so this curve is nearly flat.")

# ---------------------------------------------------------------- batch
with st.expander("Batch mode: upload a CSV"):
    st.write("Columns required: `weight_kg`, `distance_km`, `carrier_tier`, `warehouse_load`, `dispatch_date`.")
    template = pd.DataFrame([{"weight_kg": 1.0, "distance_km": 500, "carrier_tier": "Standard",
                              "warehouse_load": 3, "dispatch_date": str(date.today())}])
    st.download_button("Download a template", template.to_csv(index=False), "template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            data = pd.read_csv(up)
            required = ["weight_kg", "distance_km", "carrier_tier", "warehouse_load", "dispatch_date"]
            missing = [c for c in required if c not in data.columns]
            if missing:
                raise ValueError(f"missing columns: {missing}")
            if not ((data["weight_kg"] > 0).all() and (data["distance_km"] > 0).all() and (data["warehouse_load"] >= 1).all()):
                raise ValueError("weight_kg and distance_km must be above 0, and warehouse_load at least 1")
            result = pd.concat([data, predict_frame(bundle, data, level).round(2)], axis=1)
            st.dataframe(result)
            st.download_button("Download estimates", result.to_csv(index=False), "estimates.csv", "text/csv")
        except Exception as exc:  # shown to the user, not swallowed
            st.error(f"Could not process the file: {exc}")

# ---------------------------------------------------------------- about
with st.expander("About this model"):
    m = bundle["metrics"]
    st.markdown(
        f"""
- **Trained on:** Olist Brazilian e-commerce orders, dispatched {bundle['fit_period'][0]} to {bundle['fit_period'][1]}.
- **Test period error:** about {m['test']['MAE']:.1f} days on average (a naive guess is off by {m['baseline_test']['MAE']:.1f}).
- **Interval coverage on the test period:** 80% interval {m['coverage_test']['0.8'] * 100:.0f}%, 90% interval {m['coverage_test']['0.9'] * 100:.0f}%.
- **Main driver:** distance. Weight, workload and carrier tier have small effects in this data.
- **Carrier tier is a proxy** built from freight cost per kg per km, not a real carrier label.
- **Warehouse workload** is the number of orders the same seller handed to the carrier on the same day.
- Estimates for Brazil only. The model is least reliable for long routes.
"""
    )
