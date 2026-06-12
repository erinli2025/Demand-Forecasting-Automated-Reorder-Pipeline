# Demand Forecasting & Automated Reorder Pipeline

End-to-end supply chain pipeline: from raw order data to weekly demand forecasts and automated purchase order suggestions.

Built as a portfolio project to demonstrate the core skills of a Commercial Planning Data Scientist — forecasting demand, quantifying uncertainty, and connecting model output to operational decisions.

---

## Business context

Retail and e-commerce warehouses face a continuous planning problem: order too early and capital is tied up in slow-moving stock; order too late and you face stockouts. This project simulates the core of an automated replenishment system:

1. **Forecast** weekly demand per SKU
2. **Calculate** a reorder point based on lead time and safety stock
3. **Output** a daily purchase suggestion list — the same artefact a planning team would review each morning

---

## Dataset

**DataCo Supply Chain Dataset** — 180,519 order lines across 118 SKUs, January 2015 to January 2018. 

Link: https://data.mendeley.com/datasets/8gx2fvg2k6/5

The raw data contains order-level rows. The first step in both notebooks is aggregating these into a weekly time series per SKU — reducing 180k rows to ~1,300 rows of clean weekly demand signals.

Only SKUs with at least 130 weeks of continuous history are used (9 SKUs), ensuring models have enough data to learn seasonal patterns reliably.

---

## Project structure

```
demand_forecast_v1_prophet.ipynb    # Baseline: Prophet time series model
demand_forecast_v2_xgboost.ipynb    # Improved: XGBoost with engineered features
README.md
```

---

## Notebook 1 — Prophet baseline

**Approach:** Facebook Prophet treats demand forecasting as a decomposable time series problem — trend + yearly seasonality + noise. No feature engineering required; Prophet learns patterns directly from the date and value columns.

**Key design choices:**
- Multiplicative seasonality mode — demand spikes scale with the overall level of the series, which fits retail data better than additive
- Predictions clipped at zero — demand cannot be negative
- 95% prediction intervals used downstream to size safety stock

**Reorder logic:**

```
reorder_point  = demand_during_leadtime + safety_stock
safety_stock   = z × σ_weekly × √(lead_time_weeks)
```

Where σ is estimated from the width of Prophet's prediction interval — a principled way to derive uncertainty without a separate model.

**Results (8-week held-out test set):**

| Metric | Value |
|---|---|
| Median MAE | 38.9 units / week |
| Median MAPE | 229.9% |

The high MAPE is partly a data artefact: the dataset ends with an abrupt demand drop in the final week (a truncation effect, not a real demand change). Both models are penalised for not predicting this anomaly. MAE is a more reliable metric here.

---

## Notebook 2 — XGBoost with business features

**Approach:** Reframe the time series problem as supervised learning. XGBoost cannot reason about time directly, so time patterns are encoded as explicit features. This allows plugging in business signals — like promotional discounts — alongside historical patterns.

**Features engineered:**

| Feature | What it captures |
|---|---|
| `lag_1` … `lag_4` | Last 4 weeks of actual demand |
| `rolling_mean_4` | Short-term demand level |
| `rolling_mean_12` | Long-term demand baseline |
| `rolling_std_4` | Recent demand volatility |
| `month`, `quarter` | Seasonal patterns |
| `avg_discount` | Weekly average discount rate — promotion effect |

All rolling features use `shift(1)` to prevent data leakage: at prediction time, we only know what happened *before* the current week.

**Global model:** A single XGBoost model trained on all 9 SKUs combined. This allows the model to learn shared demand patterns across products and improves performance on SKUs with lower individual volume — a design choice motivated by production deployments at scale.

**Results vs Prophet baseline:**

| Metric | Prophet | XGBoost |
|---|---|---|
| Median MAE | 38.9 units / week | 34.5 units / week |
| Median MAPE | 229.9% | 211.4% |
| SKUs improved | — | 6 / 9 |

**Feature importance finding:** `avg_discount` ranked above calendar features (month, quarter), confirming that promotional activity has a stronger measurable effect on weekly demand than seasonal timing alone. This supports the business case for including promotional planning calendars in future model versions.

**Where XGBoost underperformed:** One SKU (Field & Stream Sportsman 16 Gun Fire Safe) saw MAPE increase from 480% to 554%. Low-volume, irregular-demand SKUs are a known weakness of global models — a per-SKU fallback model would be appropriate in production.

---

## Reorder output

Both notebooks produce the same output format — a daily purchase suggestion list:

```
============================================================
  AUTOMATED REORDER SUGGESTIONS — 2024-01-15
============================================================
  Under Armour Girls' Toddler Spine Surge...    order:  1256 units  (stock: 401, ROP: 508)
============================================================
  1 of 9 SKUs flagged for reorder
============================================================
```

In a production system, this output would be written to a purchasing database with `status = "pending_review"` and surfaced to the planning team each morning — not sent directly to suppliers.

---

## What I would do next

- **More SKUs:** Relax the history threshold and use cross-SKU embeddings (product category, price tier) to handle cold-start for new products
- **Probabilistic XGBoost:** Add quantile regression to produce prediction intervals, removing the dependency on residual-based safety stock estimation
- **Lead time variability:** Current model assumes fixed lead time; incorporating supplier variability would improve safety stock accuracy
- **Monitoring layer:** Alert when actual demand deviates from forecast by more than 2σ for three consecutive weeks — an early warning that the model needs retraining

---

## Skills demonstrated

- Time series aggregation from transactional data (pandas)
- Demand forecasting: Prophet and XGBoost
- Feature engineering: lag features, rolling statistics, calendar encoding
- Train/test evaluation with MAE and MAPE
- Safety stock calculation using forecast uncertainty
- End-to-end pipeline from raw data to operational output
- Model comparison and honest error analysis

---

## How to run

```bash
pip install pandas numpy matplotlib prophet xgboost scikit-learn

# Place DataCoSupplyChainDataset.csv in the same directory, then:
jupyter notebook demand_forecast_v1_prophet.ipynb
jupyter notebook demand_forecast_v2_xgboost.ipynb
```
