"""
train_models.py
----------------
Trains and compares two forecasting approaches per SKU:
  1. ARIMA (statistical baseline)
  2. XGBoost (gradient-boosted trees on engineered lag/calendar features)

For each SKU:
  - Split into train (first 80%) / test (last 20%, ~ last ~7 months)
  - Fit both models on train
  - Forecast over the test horizon
  - Compute RMSE and MAE on the test set
  - Save per-SKU forecast vs actual to models/forecast_results.csv
  - Save model comparison metrics to models/model_comparison.csv

Run:
    python models/train_models.py
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_PATH = "data/synthetic_demand_data.csv"
TEST_HORIZON_DAYS = 90  # ~3 months holdout per SKU


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


def make_features(series: pd.Series) -> pd.DataFrame:
    """Build lag + rolling + calendar features for XGBoost from a demand series."""
    df = pd.DataFrame({"demand": series})
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df["demand"].shift(lag)
    df["rolling_mean_7"] = df["demand"].shift(1).rolling(7).mean()
    df["rolling_mean_30"] = df["demand"].shift(1).rolling(30).mean()
    df["rolling_std_7"] = df["demand"].shift(1).rolling(7).std()
    return df


def run_arima(train: pd.Series, horizon: int) -> np.ndarray:
    model = ARIMA(train, order=(5, 1, 1))
    fitted = model.fit()
    forecast = fitted.forecast(steps=horizon)
    return np.clip(forecast.values, 0, None)


def run_xgboost(full_series: pd.Series, split_idx: int, horizon: int) -> np.ndarray:
    """
    Direct multi-step forecasting: instead of recursively feeding predicted
    values back in as lag features (which compounds error over a long
    horizon), we train ONE model per forecast step h = 1..horizon.

    Each model uses only features known at the origin time `split_idx`
    (lags/rolling stats computed strictly before the forecast origin) to
    predict the demand h days ahead. This avoids error accumulation and is
    the standard fix for recursive-forecast degradation over long horizons.
    """
    feat_df = make_features(full_series)

    feature_cols = [c for c in feat_df.columns if c != "demand"]
    preds = np.zeros(horizon)

    for h in range(1, horizon + 1):
        # Target: demand h steps ahead of each row's feature snapshot
        target = feat_df["demand"].shift(-h)
        step_df = feat_df.copy()
        step_df["target"] = target
        step_df = step_df.dropna()

        # Only use rows whose origin index is before the test split
        train_rows = step_df[step_df.index <= full_series.index[split_idx - 1]]

        model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(train_rows[feature_cols], train_rows["target"])

        # Predict using the feature snapshot exactly at the forecast origin
        origin_features = feat_df.loc[[full_series.index[split_idx - 1]], feature_cols]
        pred = model.predict(origin_features)[0]
        preds[h - 1] = max(pred, 0)

    return preds


def main():
    df = load_data()
    skus = df["sku"].unique()

    all_forecasts = []
    metrics = []

    for sku in skus:
        sku_df = df[df["sku"] == sku].set_index("date").sort_index()
        series = sku_df["demand"]

        split_idx = len(series) - TEST_HORIZON_DAYS
        train, test = series.iloc[:split_idx], series.iloc[split_idx:]

        arima_preds = run_arima(train, TEST_HORIZON_DAYS)
        xgb_preds = run_xgboost(series, split_idx, TEST_HORIZON_DAYS)

        arima_rmse = np.sqrt(mean_squared_error(test.values, arima_preds))
        arima_mae = mean_absolute_error(test.values, arima_preds)
        xgb_rmse = np.sqrt(mean_squared_error(test.values, xgb_preds))
        xgb_mae = mean_absolute_error(test.values, xgb_preds)

        metrics.append({
            "sku": sku,
            "arima_rmse": round(arima_rmse, 2),
            "arima_mae": round(arima_mae, 2),
            "xgboost_rmse": round(xgb_rmse, 2),
            "xgboost_mae": round(xgb_mae, 2),
            "best_model": "XGBoost" if xgb_rmse < arima_rmse else "ARIMA",
            "improvement_pct": round((1 - min(xgb_rmse, arima_rmse) / arima_rmse) * 100, 1),
        })

        result_df = pd.DataFrame({
            "date": test.index,
            "sku": sku,
            "actual": test.values,
            "arima_forecast": arima_preds,
            "xgboost_forecast": xgb_preds,
        })
        all_forecasts.append(result_df)

        print(f"[{sku}] ARIMA RMSE={arima_rmse:.1f} | XGBoost RMSE={xgb_rmse:.1f} "
              f"-> Best: {metrics[-1]['best_model']}")

    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    forecast_df.to_csv("models/forecast_results.csv", index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("models/model_comparison.csv", index=False)

    print("\n=== Model Comparison Summary ===")
    print(metrics_df.to_string(index=False))
    print("\nSaved: models/forecast_results.csv, models/model_comparison.csv")


if __name__ == "__main__":
    main()
