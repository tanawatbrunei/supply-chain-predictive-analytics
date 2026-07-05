"""
inventory_threshold.py
-----------------------
Step 3: Translate demand forecasts into actionable inventory policy —
Safety Stock (SS) and Reorder Point (ROP) per SKU.

This is the step that makes the forecasting work in Step 1-2 useful for a
procurement/planning team: a forecast alone doesn't prevent stockouts,
but SS + ROP tell the buyer WHEN to place a PO and HOW MUCH buffer to hold.

Method
------
Standard inventory theory under demand uncertainty during lead time:

    Safety Stock (SS)   = Z * sigma_dLT
    Reorder Point (ROP) = (avg_daily_demand * lead_time_days) + SS

Where:
    Z            = service-level factor (e.g. Z=1.65 for 95% service level,
                   Z=2.33 for 99%). Higher service level = more buffer =
                   higher holding cost, so this is a business trade-off,
                   not a purely technical one.
    sigma_dLT    = combined uncertainty over the lead-time window, from TWO
                   sources:
                     1) natural day-to-day demand volatility (demand_std),
                        scaled by sqrt(lead_time_days) since variances of
                        independent days add up over the lead-time window
                     2) forecast model error (RMSE from Step 2) — a model
                        that forecasts poorly needs a bigger safety buffer
                        to compensate for its own uncertainty
    avg_daily_demand = recent average daily demand (last 90 days), so the
                   reorder point reflects current, not historical-average,
                   demand level

Output: models/inventory_thresholds.csv
Columns: sku, avg_daily_demand, demand_std, lead_time_days, forecast_rmse,
         service_level, z_score, safety_stock, reorder_point, unit_cost,
         safety_stock_value

Run:
    python models/inventory_threshold.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

DEMAND_PATH = "data/synthetic_demand_data.csv"
MODEL_COMPARISON_PATH = "models/model_comparison.csv"
RECENT_WINDOW_DAYS = 90

# Business decision: target service level (probability of NOT stocking out
# during the lead-time window). 95% is a common default for non-critical
# components; critical/long-lead-time parts often justify 97-99%.
SERVICE_LEVEL = 0.95


def compute_z_score(service_level: float) -> float:
    return norm.ppf(service_level)


def main():
    demand_df = pd.read_csv(DEMAND_PATH, parse_dates=["date"])
    metrics_df = pd.read_csv(MODEL_COMPARISON_PATH)

    z = compute_z_score(SERVICE_LEVEL)
    results = []

    for sku in demand_df["sku"].unique():
        sku_df = demand_df[demand_df["sku"] == sku].sort_values("date")
        recent = sku_df.tail(RECENT_WINDOW_DAYS)

        avg_daily_demand = recent["demand"].mean()
        demand_std = recent["demand"].std()
        lead_time_days = int(sku_df["lead_time_days"].iloc[0])
        unit_cost = sku_df["unit_cost"].iloc[0]

        # Forecast error (RMSE) from the best model for this SKU, per day of
        # the forecast horizon on average -> treated as an additional daily
        # uncertainty component
        row = metrics_df[metrics_df["sku"] == sku].iloc[0]
        best_rmse = min(row["arima_rmse"], row["xgboost_rmse"])

        # Combine natural demand volatility and forecast-model uncertainty
        # (added in quadrature, i.e. combined as independent variance
        # sources), then scaled by sqrt(lead_time) since lead time spans
        # multiple independent demand days
        combined_daily_std = np.sqrt(demand_std**2 + best_rmse**2)
        sigma_dLT = combined_daily_std * np.sqrt(lead_time_days)

        safety_stock = z * sigma_dLT
        reorder_point = (avg_daily_demand * lead_time_days) + safety_stock

        results.append({
            "sku": sku,
            "avg_daily_demand": round(avg_daily_demand, 1),
            "demand_std": round(demand_std, 1),
            "lead_time_days": lead_time_days,
            "forecast_rmse": round(best_rmse, 1),
            "service_level": SERVICE_LEVEL,
            "z_score": round(z, 2),
            "safety_stock": round(safety_stock),
            "reorder_point": round(reorder_point),
            "unit_cost": unit_cost,
            "safety_stock_value_usd": round(safety_stock * unit_cost, 2),
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv("models/inventory_thresholds.csv", index=False)

    print(f"Service level target: {SERVICE_LEVEL*100:.0f}% (Z={z:.2f})\n")
    print(result_df.to_string(index=False))
    print(f"\nTotal safety stock value across all SKUs: "
          f"${result_df['safety_stock_value_usd'].sum():,.2f}")
    print("\nSaved: models/inventory_thresholds.csv")


if __name__ == "__main__":
    main()
