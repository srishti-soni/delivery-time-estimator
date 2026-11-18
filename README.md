# E-Commerce Order Delivery Time & Transit Delay Estimator

Estimates how many days an order will spend in transit (carrier pickup to customer delivery) from four inputs, and gives a
confidence interval around the estimate. A Streamlit app turns it into a dispatch tool.

| Input | Meaning |
|---|---|
| `weight_kg` | package weight |
| `distance_km` | straight-line seller-to-customer distance |
| `carrier_tier` | Economy / Standard / Express (a **proxy**, see limitations) |
| `warehouse_load` | orders the same seller handed to the carrier on the same day |

**Data:** [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), 94,466 shipments dispatched October 2016 to September 2018.

## Results (all errors in days)

| | Validation block | Test block (most recent, faster period) |
|---|---|---|
| Final model MAE | 4.86 | 3.03 |
| Baseline MAE (always predict the training median) | 6.33 | 3.77 |
| Improvement over baseline | 23% | 20% |
| R2 | 0.22 | 0.04 |

Interval coverage on the test block: **90% interval covers 89.8%**, **80% interval covers 77.1%**.
Average 90% interval width is about 18 days.

Read these honestly:
- The model beats a naive guess by about a fifth, but the absolute fit is modest. **Distance does almost all the work** (about 75% of the SHAP importance); weight, workload and tier add very little.
- The test block is a faster period than the training data (mean 5.6 days against 9.9), so its absolute MAE looks better than validation. R2 on the test block is close to zero.
- Intervals are wide. The tool gives a plausible range, not a precise date.
- Coverage is uneven: about 97% for routes under 100 km, but about 82% beyond 1,500 km and about 85% for the Economy tier.
- Validation was used to choose models, features and hyper-parameters. The conformal rule "only widen" was adopted after a first test result (see notebook 03, section 7), so the test numbers are not a perfectly untouched estimate.

## Project layout

```
delivery-time-estimator/
├── notebooks/
│   ├── 01_data_preparation.ipynb   raw Olist files -> data/processed/shipments*.csv
│   ├── 02_eda.ipynb                exploration, settles modelling decisions
│   └── 03_modeling.ipynb           split, models, intervals, SHAP, export of the model
├── src/
│   ├── features.py                 feature engineering and delay-risk rule (used by notebook and app)
│   └── predict.py                  loads the model file, returns estimate + interval
├── app/streamlit_app.py            dispatch tool
├── tests/test_pipeline.py          27 tests
├── models/transit_model.joblib     the trained model bundle
├── data/raw/                       Olist CSVs (git-ignored)
├── data/processed/                 shipments.csv, shipments_full.csv (git-ignored, rebuilt by notebook 01)
├── data-preparation.md             design of the data preparation step
└── requirements.txt
```

## How it works

1. **Data preparation (notebook 01).** One row per order-and-seller shipment. Target = carrier pickup to customer delivery. Distance = haversine between seller and customer zip-code centres. Weight = summed product weights. Orders with more than one seller are dropped. Rows with transit time of 0 or below, or above 60 days, are filtered out.
2. **EDA (notebook 02).** The target is strongly right-skewed (so the model is trained on `log1p(days)`), distance dominates, the target drifts over time (so the split is chronological), and the carrier tier turns out to be a proxy for the route.
3. **Modelling (notebook 03).** Chronological split: 60% train, 15% validation, 10% calibration, 15% test (the most recent). Ridge, Random Forest, sklearn Gradient Boosting and LightGBM are compared on validation; LightGBM (L1 objective) wins by a small margin (4.86 days against 4.88 to 4.98 for the others). A small grid is tuned on validation only; the spread across settings is 0.02 days.
4. **Intervals.** LightGBM quantile models give lower and upper bounds. A conformal correction (never allowed to shrink the interval) is measured on the calibration block. On this data the correction came out as 0, so in practice the intervals are the raw quantile intervals.
5. **Delay risk.** The dispatcher enters a promised delivery time: *High* if the estimate is later than the promise, *Medium* if only the upper bound is later, *Low* otherwise. On the test block with a 10-day promise, shipments labelled Low were late 1% of the time, Medium 5%, High 32%.
6. **Model bundle.** One `joblib` file with the models, conformal values, input ranges, metrics and library versions. The app loads it and contains no modelling code.

## Run it

### Locally

```bash
conda create -n xbnb python=3.10
conda activate xbnb
conda install -c conda-forge scipy      # avoids a scipy binary problem seen on macOS with the pip build
pip install -r requirements.txt

pytest -q                               # 27 tests, needs models/transit_model.joblib
streamlit run app/streamlit_app.py
```

To rebuild everything from scratch: download the data (needs a Kaggle API token in the `KAGGLE_API_TOKEN` environment variable, notebook 01 does this), then run notebooks 01, 02 and 03 in order.

### In Google Colab

1. Upload the notebook you want to run. In Colab, open the key icon in the sidebar, add a secret named `KAGGLE_API_TOKEN` and switch on notebook access (needed by notebook 01 only).
2. Notebook 01 downloads the dataset and then offers the two CSVs for download.
3. Notebooks 02 and 03 ask you to upload any file they need when it is missing: `shipments_full.csv` (02), `shipments.csv`, `features.py` and `predict.py` (03). Notebook 03 offers `transit_model.joblib` for download at the end.
4. Put the downloaded model in `models/` on your machine. The app warns if its LightGBM or scikit-learn version differs from the one used for training.

I could only run the notebooks locally. The Colab-specific cells (secrets, upload prompts, downloads) have not been run in Colab.

## The Streamlit app

Inputs: weight, distance, carrier tier, warehouse workload, dispatch date and time, interval level (80% or 90%), promised delivery time.
Outputs: estimate in days and hours, the interval, an earliest / most likely / latest arrival date, a delay-risk label, an interval chart, what-if curves for distance and workload, a batch mode for CSV files, and an "About" section with the metrics.
Inputs outside the training range produce a warning, not an error.

## Limitations

- **Brazil only, 2016 to 2018.** The target drifts over time, so errors on a different period can be larger.
- **Carrier tier is a proxy.** Olist has no carrier field. The tier is built from freight price per kg per km, so it mostly encodes the route. At a fixed distance the three tiers have almost the same transit time. Do not read it as a real service level.
- **Warehouse workload is derived**, as the seller's order count on the dispatch day. 39% of shipments have a load of 1, and its effect on transit time is small.
- **Distance is straight-line**, not road distance.
- **Weight barely matters** in this data (weak correlation, small effect in the model).
- **Wide intervals and uneven coverage:** the 80% interval is a little under its target and coverage drops for long routes and the Economy tier.
- **Limited fit:** R2 is 0.22 on validation and 0.04 on the test block.
- **Small caveat on dispatch time:** the dataset has the dispatch date but the model uses only the day of the week, so the time of day entered in the app affects only the displayed arrival time.
