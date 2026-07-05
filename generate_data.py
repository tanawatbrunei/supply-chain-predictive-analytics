"""
generate_data.py (v2 — grounded in real market data)
-----------------------------------------------------
Builds daily SKU-level demand data using a REAL macroeconomic driver:

    FRED series IPB53122S - "Industrial Production: Durable Goods Materials:
    Semiconductors, Printed Circuit Boards, and Other" (Board of Governors
    of the Federal Reserve System, monthly, Index 2017=100)
    Source: https://fred.stlouisfed.org/series/IPB53122S

Why this approach instead of pure synthetic data:
    Company-level SKU demand is confidential and not publicly available for
    any real firm. However, component-level demand for PCB/semiconductor
    parts is well known to move with broader semiconductor industry output
    (this is a standard "leading indicator" / exogenous-variable technique
    used in real S&OP and demand planning). Anchoring the simulation to a
    real, citable government index means:
      - The macro trend, the 2021-2022 chip shortage dip/rebound, and the
        2024-2026 recovery are REAL, not invented.
      - Each SKU's demand = real macro trend (scaled/shaped per component
        type) + SKU-specific idiosyncratic noise and calendar effects.

Output: data/synthetic_demand_data.csv
Columns: date, sku, demand, lead_time_days, unit_cost
"""

import numpy as np
import pandas as pd

np.random.seed(42)

FRED_RAW_PATH = "data/fred_semiconductor_ip_index_raw.csv"

# Each SKU reacts to the macro semiconductor cycle differently:
#   sensitivity: how strongly the SKU's demand tracks the macro index swings
#   base_scale:  converts the index level into a realistic unit-demand level
#   noise:       idiosyncratic day-to-day demand noise (company/SKU-specific)
SKU_PROFILES = {
    "PCB-A100-CTRLBOARD": {"sensitivity": 1.0, "base_scale": 3.2, "noise": 30, "lead_time": 45, "cost": 12.50},
    "PCBA-B220-PWRMOD":   {"sensitivity": 0.6, "base_scale": 1.8, "noise": 22, "lead_time": 60, "cost": 28.75},
    "CONN-C310-HARNESS":  {"sensitivity": 1.3, "base_scale": 6.5, "noise": 55, "lead_time": 30, "cost": 3.20},
    "SENSOR-D410-TEMP":   {"sensitivity": 0.8, "base_scale": 1.6, "noise": 18, "lead_time": 75, "cost": 8.90},
    "IC-E510-MCU":        {"sensitivity": 1.5, "base_scale": 4.0, "noise": 40, "lead_time": 90, "cost": 15.40},
}


def load_fred_index() -> pd.Series:
    df = pd.read_csv(FRED_RAW_PATH, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df["value"]


def to_daily_index(monthly_index: pd.Series) -> pd.Series:
    """Upsample the monthly FRED index to daily frequency via linear
    interpolation, so it can drive a daily demand simulation."""
    daily_range = pd.date_range(monthly_index.index.min(), monthly_index.index.max(), freq="D")
    daily = monthly_index.reindex(daily_range).interpolate(method="linear")
    return daily


def generate_sku_series(daily_index: pd.Series, profile: dict) -> np.ndarray:
    dates = daily_index.index
    n = len(dates)

    # Macro-driven base demand: real FRED index scaled to unit-demand level
    macro_component = daily_index.values * profile["base_scale"] * profile["sensitivity"]

    # Weekly seasonality (factories run lighter on weekends)
    weekday = dates.dayofweek.values
    weekly_season = np.where(weekday >= 5, -macro_component * 0.30, 0)

    # SKU-specific idiosyncratic noise (company-level variation not captured
    # by the macro index alone)
    noise = np.random.normal(0, profile["noise"], n)

    demand = macro_component + weekly_season + noise
    demand = np.clip(demand, a_min=0, a_max=None)
    return np.round(demand).astype(int)


def main():
    monthly_index = load_fred_index()
    daily_index = to_daily_index(monthly_index)

    frames = []
    for sku, profile in SKU_PROFILES.items():
        demand = generate_sku_series(daily_index, profile)
        df = pd.DataFrame({
            "date": daily_index.index,
            "sku": sku,
            "demand": demand,
            "lead_time_days": profile["lead_time"],
            "unit_cost": profile["cost"],
        })
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    out_path = "data/synthetic_demand_data.csv"
    full_df.to_csv(out_path, index=False)

    print(f"Generated {len(full_df):,} rows across {len(SKU_PROFILES)} SKUs -> {out_path}")
    print(f"Date range: {full_df['date'].min().date()} to {full_df['date'].max().date()}")
    print("\nDemand summary by SKU:")
    print(full_df.groupby("sku")["demand"].describe()[["mean", "std", "min", "max"]])

    # Sanity check: the 2021 chip-shortage dip is a real feature of the FRED
    # index at MONTHLY resolution (annual averages smooth it out), e.g. the
    # index fell from 126.2 (May 2021) to 118.5 (Jul 2021) before recovering.
    print("\nSanity check - real FRED index around the 2021 chip shortage (monthly):")
    print(monthly_index.loc["2021-04-01":"2021-10-01"])


if __name__ == "__main__":
    main()
