# Tesla EV Deliveries — ML Pipeline (2015–2025)

This notebook builds a full machine learning pipeline on Tesla's global delivery and production data. The goal was to predict average vehicle price and forecast future deliveries, while keeping the runtime under a minute on Kaggle.

---

## Dataset

Tesla EV Deliveries & Production Data (2015–2025) by nalisha on Kaggle. It covers estimated deliveries, production units, average price, battery capacity, range, CO2 savings, and charging station counts across different models and regions.

Dataset path on Kaggle:
```
/kaggle/input/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025/tesla_deliveries_dataset_2015_2025.csv
```

---

## What this pipeline does

**Preprocessing** — removes duplicates, fixes data types, and caps outliers using IQR.

**EDA** — two charts: total deliveries by year and deliveries by region.

**Feature Engineering** — builds 20+ features including price-to-range ratios, lag features (up to 3 months), 3-month rolling averages, cyclical month encoding, and label-encoded categoricals.

**Regression** — trains Ridge, Random Forest, Gradient Boosting, XGBoost, and LightGBM to predict `Avg_Price_USD`. All models reported with R², MAE, RMSE, and MAPE.

**Hyperparameter Tuning** — uses `RandomizedSearchCV` with 3-fold CV and 10 iterations to tune XGBoost and LightGBM. Results are blended 50/50 for the final ensemble.

**Diagnostics** — actual vs predicted plots, residual analysis, and cross-validation scores.

**Time Series Forecasting** — Holt-Winters, SARIMA, and an XGBoost lag model on monthly global deliveries, with a 24-month future projection.

---

## Results

Best model consistently hits R² above 0.75. Full leaderboard saved to `results_summary.csv`.

---

## How to run

Paste the entire script into **one single Kaggle cell** and run. No setup needed — all libraries are pre-installed in the Kaggle Python image.

For local use:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm statsmodels
python tesla_ml_pipeline.py
```

---

## Output files

- `eda_plots.png` — deliveries by year and region
- `model_comparison.png` — feature importances and R² comparison
- `diagnostics.png` — residual plots and CV scores
- `ts_forecast.png` — forecast vs actual + future projection
- `results_summary.csv` — full model leaderboard
