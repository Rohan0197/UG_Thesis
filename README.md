# Retail Intelligence Platform

A full-stack retail analytics system combining demand forecasting (M5 Walmart dataset)
and customer intelligence (UCI Online Retail dataset) into a self-contained HTML dashboard.

## Datasets

| Dataset | Source | Used for |
|---------|--------|----------|
| **M5 Walmart** | Kaggle M5 competition | Demand forecasting · 10 US stores · 2011–2016 |
| **UCI Online Retail** | UCI ML Repository | Customer intelligence · UK e-commerce · 2009–2011 |

These are **independent datasets** combined into one platform. Revenue figures
from M5 are in USD; UCI customer CLV figures are in GBP.

## Project Structure

```
Project/
├── src/                          # all pipeline + dashboard code
│   ├── data_builder.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── customer_intelligence.py
│   ├── generate_dashboard.py
│   ├── dashboard_fixed.py
│   └── dashboard_template.html
├── raw_data/                     # empty in the repo — see "Data" below
│   └── m5/
├── processed_data/                # empty in the repo — populated by the pipeline
├── models/                        # empty in the repo — populated by model.py
├── retail_dashboard.html          # generated deliverable (already built, viewable as-is)
├── requirements.txt
└── README.md
```

## Data

Raw and processed data (~1.5 GB total, including two >100 MB CSVs) are kept
**outside this repository** in a sibling folder: `../Project_Data/`
(`raw_data/`, `processed_data/`, `models/`). This keeps the git repo small;
none of it is hand-authored — it's either source data or pipeline output.

To run the pipeline, copy the contents back in first:

```
raw_data/
├── m5/                          # Walmart M5 competition data
│   ├── sales_train_evaluation.csv
│   ├── sell_prices.csv
│   └── calendar.csv
├── online_retail_II.xlsx        # UCI Online Retail 2009-2010
└── Online Retail.xlsx           # UCI Online Retail 2010-2011
```

Pipeline (run in order, from the repo root):
  Step 1: src/data_builder.py           → processed_data/{orders, line_items, products, stores}.csv
  Step 2: src/feature_engineering.py    → processed_data/{weekly_features, inventory_base}.csv
  Step 3: src/model.py                  → processed_data/{forecasts, feature_importance, model_metrics}.csv
                                      models/lgbm_demand.pkl
  Step 4: src/customer_intelligence.py  → processed_data/{clv_scores, customer_summary, kmeans_diagnostics}.csv
  Step 5: src/generate_dashboard.py     → retail_dashboard.html  (open in any browser)

Dashboard (alternative — requires Streamlit):
  streamlit run src/dashboard_fixed.py

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy raw_data/, processed_data/, models/ in from ../Project_Data (see "Data" above)

# 3. Run the pipeline (each step takes several minutes), from the repo root
python src/data_builder.py
python src/feature_engineering.py
python src/model.py
python src/customer_intelligence.py

# 4a. Generate the standalone HTML dashboard (recommended)
python src/generate_dashboard.py
#     → Opens retail_dashboard.html in any browser, no server needed

# 4b. OR launch the interactive Streamlit dashboard
streamlit run src/dashboard_fixed.py
```

## Dashboard

`retail_dashboard.html` is a fully self-contained file — no server, no Python needed at
view time. Share it by copying a single file. It contains five pages:

| Page | Content |
|------|---------|
| **Executive Summary** | Total revenue, forecast accuracy, CLV, churn risk, inventory alerts |
| **Demand Forecast** | 12-week forward predictions by store × category with prediction intervals |
| **Inventory Intelligence** | Stock coverage analysis, critical reorder list, category/store breakdown |
| **Customer Intelligence** | RFM segments, CLV vs churn scatter, segment revenue |
| **Model Validation** | Walk-forward results, SHAP feature importance, methodology rationale |

## Model Performance

| Split      | R²     | MAE   | RMSE  | MAPE  | 80% interval coverage |
|------------|--------|-------|-------|-------|----------------------|
| Validation | 0.8947 | 8.98  | 19.97 | 44.4% | ~80% (quantile)      |
| Test       | 0.9110 | 9.14  | 18.16 | 37.8% | ~80% (quantile)      |

MAPE is computed on non-zero weeks only. High MAPE is expected given the large number
of low-volume product-store pairs with sporadic demand.
**R² = 0.911 on the held-out 2016 test set is the primary quality indicator.**

## Key Design Decisions

- **LightGBM with MAE loss** — robust to outliers common in retail demand
- **Walk-forward CV** — no data leakage; mirrors real deployment
- **Quantile regression (α=0.1, 0.9)** — statistically valid 80% prediction intervals
- **Top 200 products** — covers ~80%+ of revenue, keeps feature matrix tractable
- **BG/NBD + Gamma-Gamma CLV** — probabilistic model vs naive linear projection
- **Logistic Regression churn** — calibrated 0–1 probability vs recency heuristic
- **price_vs_median_ratio** — relative price elasticity (unit_price / 8-week rolling median)
- **inventory_base.csv** — lightweight file read by dashboard instead of 125 MB weekly_features.csv
- **Reproducible encodings** — category/dept label maps use `sorted()` for determinism

## Notes on Inventory Data

Stock levels in the Inventory tab are **estimated** from recent sales velocity
(8-week rolling average × simulated weeks-on-hand drawn from a uniform distribution).
The platform does not have access to a real warehouse management system.
For production use, replace the `inventory_base.csv` stock simulation in
`feature_engineering.py` with live WMS data.

## Optional: Hyperparameter Tuning

```python
# In model.py, set:
TUNE = True
N_TRIALS = 50  # increase for a more thorough search
```

Results are saved to `processed_data/best_params.json` and automatically
loaded on subsequent runs.

## Improvements over v1

### feature_engineering.py
- **FIX — is_q4 bug**: was always 0.0 (SHAP = 0) due to month being computed as
  float/NaN for zero-filled rows. Now derived from `week.dt.month.astype(int)` with
  a runtime assertion to catch regressions.
- **NEW — price_vs_median_ratio**: `unit_price / 8-week rolling median price`.
  Captures relative price elasticity that the raw `price_change` feature misses.

### model.py
- **NEW — Quantile prediction intervals**: Three LightGBM models trained — point (L1),
  lower (α=0.10), upper (α=0.90) — producing proper 80% prediction intervals.
  `forecasts.csv` includes `forecast_lower` and `forecast_upper`.
- **NEW — Optuna hyperparameter tuning**: 50-trial Optuna search (opt-in via `TUNE=True`).
- **FIX — is_q4** now has non-zero SHAP importance after the feature_engineering fix.
- **NEW — 80% interval coverage** reported in `model_metrics.csv`.

### customer_intelligence.py
- **NEW — BG/NBD CLV model** (via `lifetimes` library): replaces linear projection with
  the probabilistic BG/NBD + Gamma-Gamma model. Falls back to enhanced linear projection
  if `lifetimes` is not installed.
- **NEW — Logistic Regression churn model**: replaces recency heuristic with a calibrated
  logistic regression (5-fold CV). `churn_risk_score` is a true probability (0–1).
- **NEW — K-Means validation**: elbow + silhouette scores for k=2..8 saved to
  `kmeans_diagnostics.csv`. Justifies k=4.
- **NEW — clv_model_type column** in output CSVs records which method was used.

### generate_dashboard.py + dashboard_template.html (new in v2)
- Replaces the Streamlit-only approach with a **standalone HTML dashboard**.
- All pipeline data injected as JSON at build time; no server needed at view time.
- Five-page sidebar navigation, Chart.js charts, real data wired to every KPI and chart.
- Inventory simulation uses a realistic stock distribution (1–8 weeks on-hand) so
  Critical/Warning/Safe categories reflect genuine spread across SKUs.
- Zero-demand items (avg_weekly_qty < 1 unit/week) excluded from inventory analysis.

### dashboard_fixed.py (Streamlit — improved over original dashboard.py)
- Dataset labels throughout (M5 Walmart vs UCI Online Retail).
- Inventory disclaimer (estimated from sales velocity, not live WMS).
- MAPE explanation ribbon (why 37–44% is acceptable).
- Human-readable product labels (`FOODS_3_586` → `Foods · Item 586`).
- Forecast chart uses `forecast_lower`/`forecast_upper` quantile bounds.
- CLV model type displayed (BG/NBD or linear).
- K-Means silhouette chart to justify k=4.
- `st.tabs()` navigation + sidebar filters.
