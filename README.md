# Supply Chain Predictive Analytics & Inventory Optimization

**Forecasting electronic component demand and translating it into actionable inventory policy — grounded in real semiconductor industry data.**

---

## Business Problem

Procurement teams sourcing electronic components (PCB/PCBA, connectors, sensors, ICs) face a recurring trade-off:

- **Order too little / too late** → production line stoppage, expedite fees, damaged supplier relationships
- **Order too much / too early** → cash tied up in inventory, obsolescence risk, higher holding costs

This is especially acute for components with **long lead times** (60-90 days) sourced from overseas suppliers, where demand can shift meaningfully between the moment a PO is placed and the moment parts arrive — the exact environment of automotive/electronics manufacturing sourcing (PCB, PCBA, connectors, ICs).

This project builds a full pipeline — from demand forecasting to a concrete reorder policy — to answer three questions a procurement/planning function actually needs answered:

1. **How much of each component will we need, and when?**
2. **How much forecast error should we plan around?**
3. **Given that uncertainty, when should we reorder, and how much safety buffer do we need?**

---

## Data

Company-level SKU demand is confidential at every real manufacturer, so this project grounds its simulation in a **real, public, citable economic indicator** rather than inventing numbers from scratch:

> **FRED series `IPB53122S`** — *Industrial Production: Durable Goods Materials: Semiconductors, Printed Circuit Boards, and Other* (Board of Governors of the Federal Reserve System, monthly index, 2017=100)
> Source: https://fred.stlouisfed.org/series/IPB53122S

This index is a well-established **leading indicator** for component-level demand in electronics manufacturing — the same logic used in real S&OP/demand-planning functions that track semiconductor industry output as an exogenous driver.

**Why not a ready-made Kaggle dataset?** Most public retail forecasting datasets (e.g. Walmart/M5) don't reflect the demand dynamics of B2B electronic component procurement — no long supplier lead times, no chip-shortage-style supply shocks. Anchoring to a real semiconductor production index keeps the case study inside the correct domain.

**Simulation approach:**
- The real FRED index (2015 - 2026) is interpolated from monthly to daily resolution
- 5 representative SKUs (PCB control board, power module, connector harness, temperature sensor, MCU) are simulated as the macro index scaled by a SKU-specific *sensitivity* + idiosyncratic daily noise + weekly seasonality (factories run lighter on weekends)
- This means the **2021 chip-shortage dip and the 2023-2026 recovery are real events**, not fabricated — verified directly against the raw FRED values (see `data/fred_semiconductor_ip_index_raw.csv`)

| File | Description |
|---|---|
| `data/fred_semiconductor_ip_index_raw.csv` | Raw real monthly index from FRED |
| `data/generate_data.py` | Builds daily SKU-level demand from the real index |
| `data/synthetic_demand_data.csv` | Output: 20,695 rows, 5 SKUs, 2015-2026 |

---

## Step 1-2: Demand Forecasting

**Goal:** predict daily demand per SKU over a 90-day horizon (matching typical procurement lead times) and quantify forecast accuracy.

### Feature engineering
For the machine-learning model, the raw time series is converted into a feature table: lag values (1/7/14/30 days), rolling mean/std (7/30-day windows), and calendar features (day-of-week, month, day-of-year).

### Models compared
| Model | Type | Notes |
|---|---|---|
| **ARIMA(5,1,1)** | Classical statistical baseline | Simple, interpretable, no external features |
| **XGBoost** | Gradient-boosted trees | Uses lag/rolling/calendar features |

### A real finding worth highlighting: recursive vs. direct forecasting
The first XGBoost implementation used **recursive forecasting** (predicting one day, then feeding that prediction back in as a lag feature for the next day, repeated 90 times). Result: **ARIMA outperformed XGBoost on every SKU**, because prediction errors compounded over the long horizon.

**Fix:** switched to **direct multi-step forecasting** — training one specialized model per forecast step (1 model for "1 day ahead," a separate model for "2 days ahead," ... up to 90), each predicting straight from features known at the forecast origin, with no dependency on the model's own earlier predictions. This is the standard remedy for recursive-forecast error accumulation over long horizons.

**Result after the fix:** XGBoost outperformed ARIMA on all 5 SKUs, cutting RMSE by 48-75%.

| SKU | ARIMA RMSE | XGBoost RMSE | Improvement |
|---|---|---|---|
| PCB-A100-CTRLBOARD | 84.2 | 26.1 | 69.0% |
| PCBA-B220-PWRMOD | 38.2 | 19.9 | 48.0% |
| CONN-C310-HARNESS | 201.3 | 52.3 | 74.0% |
| SENSOR-D410-TEMP | 36.0 | 16.0 | 55.5% |
| IC-E510-MCU | 148.7 | 36.9 | 75.2% |

*(RMSE penalizes large errors more heavily than small ones — appropriate here since a single large miss, e.g. a stockout on a 90-day-lead-time MCU, is far costlier than several small ones.)*

| File | Description |
|---|---|
| `models/train_models.py` | Trains and compares ARIMA vs. XGBoost per SKU |
| `models/forecast_results.csv` | Daily actual vs. forecast values for the test period |
| `models/model_comparison.csv` | RMSE/MAE comparison table |

---

## Step 3: Inventory Threshold — Safety Stock & Reorder Point

**Goal:** turn a forecast into a decision — *when to place the PO, and how much buffer to hold.*

### Method
```
Safety Stock (SS)   = Z × σ_dLT
Reorder Point (ROP) = (avg_daily_demand × lead_time_days) + SS
```

- **Z** — service-level factor (95% target → Z = 1.64). Higher service level = more buffer = higher holding cost; a business trade-off, not a purely technical one.
- **σ_dLT** — combined uncertainty over the lead-time window, from *two* sources added in quadrature: (1) natural day-to-day demand volatility, and (2) the forecasting model's own RMSE from Step 2. A model with higher forecast error needs a bigger safety buffer to compensate — safety stock sized on demand volatility alone would understate real risk.
- **avg_daily_demand** — trailing 90-day average, so the reorder point reflects current demand, not the full 2015-2026 history.

### Results (95% service level target)

| SKU | Reorder Point (units) | Safety Stock (units) | Safety Stock Value |
|---|---|---|---|
| CONN-C310-HARNESS | 47,318 | 2,131 | $6,819 |
| IC-E510-MCU | 99,577 | 2,656 | $40,902 |
| PCB-A100-CTRLBOARD | 26,837 | 1,046 | $13,078 |
| PCBA-B220-PWRMOD | 12,140 | 543 | $15,621 |
| SENSOR-D410-TEMP | 17,879 | 582 | $5,176 |
| **Total** | | | **$81,597** |

**Business insight:** IC-E510-MCU carries the highest safety stock value — driven by its 90-day lead time combined with a relatively high unit cost. This is a concrete, quantified starting point for a supplier negotiation on lead-time reduction: every day shaved off the MCU's lead time reduces the working capital locked up in safety stock.

| File | Description |
|---|---|
| `models/inventory_threshold.py` | Computes Safety Stock and Reorder Point per SKU |
| `models/inventory_thresholds.csv` | Final output table |

---

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost statsmodels scipy

python data/generate_data.py          # Step 1: build demand data from real FRED index
python models/train_models.py         # Step 2: train & compare ARIMA vs. XGBoost
python models/inventory_threshold.py  # Step 3: compute safety stock & reorder point
```

## Tech Stack
`Python` · `pandas` · `NumPy` · `XGBoost` · `statsmodels (ARIMA)` · `scikit-learn` · `SciPy`

## Author
**Tanawat Hanpa** — Procurement Engineer specializing in global sourcing, supplier management, and data-driven procurement analytics.
[LinkedIn](https://linkedin.com/in/tanawat-hanpa) · [GitHub](https://github.com/tanawatbrunei)
