# Walmart Weekly Revenue Forecasting
### $452,868 Error Per Store Per Week → $38,661

[![Live](https://img.shields.io/badge/Live-Streamlit-red)](https://lnkd.in/dFTq7rts)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)]()

**Live Dashboard:** https://lnkd.in/dFTq7rts

---

## The Real Problem

**Not exploration. One specific question with
one specific answer that enables specific decisions.**

**Question answered:**
Can ML predict weekly revenue per Walmart store
accurately enough to replace Monday morning
guessing with data-driven inventory and staffing
decisions — with a quantified confidence range
per store?

**Why this matters:**
- 45 stores × $414,207 weekly error improvement
  = $18.6M potential annual forecasting value
- Average weekly error before: $452,868 per store
- Average weekly error after: $38,661 per store
- Reduction: 91.5% improvement in dollar accuracy

---

## Who This Is Built For

**Every page of the dashboard serves one stakeholder
decision — not general exploration.**

| Stakeholder | Decision | Weekly $ at stake |
|-------------|---------|-------------------|
| Store operations manager | Staff scheduling 7 days ahead | $50K-$200K labour |
| Supply chain team | Inventory order quantities Monday | $100K-$500K per order |
| Finance FP&A analyst | Budget vs actual variance | 45 reports weekly |
| Regional director | Store intervention identification | Monthly allocation |
| CFO | Board revenue forecast input | Quarterly planning |

---

## Results

| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| MAPE | 88.54% | 4.41% | 95% better |
| MAE | $452,868 | $38,661 | $414K saved |
| RMSE | $526,000 | $55,603 | $470K better |
| R² | -0.7812 | 0.9810 | +1.76 |
| WMAE | — | $39,104 | Kaggle official |
| Stores improved | 0 of 45 | 45 of 45 | 100% |
| Within 10% | — | 92.8% | 570 of 615 |
| Alerts triggered | — | 0 of 21 months | Stable |

---

## Step 1 — Business Problem

- Question: Can we predict weekly store revenue
  with MAPE below 5% using public data only?
- Revenue range: $263K to $2.55M — 10x spread
- 5 statistical tests confirmed ML is right approach
- Dataset: completed Kaggle competition — prize
  money signals real commercial value
- Success criteria defined before any code:
  MAPE below 5%, R² above 0.95, 90% within 10%

---

## Step 2 — Data and EDA

- 6,435 rows — 45 stores — Feb 2010 to Oct 2012
- Zero nulls — zero duplicates — confirmed

**5 tests before any modelling:**

| Test | Result | Decision |
|------|--------|----------|
| Holiday T-test | p=0.003 | Feature confirmed valid |
| Store ANOVA | p=0.000 | Store ID essential |
| Fuel_Price | p=0.447 | Keep — non-linear value |
| Normality | Non-normal | MAPE over RMSE |
| YoY trend | p=0.430 | No trend feature |

**Finding that changed the analysis:**
- Christmas week = BELOW average sales
- December 28th = post-Christmas decline
- Every public notebook treated it as a peak
- Fixed: separate is_christmas feature built
  to capture decline not peak
- Result: prevented over-prediction during
  highest-stakes retail weeks of the year

**Outlier decision:**
- 34 IQR-flagged rows kept — all real holiday peaks
- Removing them: model never learns holiday peaks
- Holiday MAPE post-FE: 3.2% — decision validated

---

## Step 3 — Feature Engineering

**8 raw → 33 features | 95% of improvement here**

| Category | Count | Top feature | Business purpose |
|----------|-------|-------------|-----------------|
| Raw | 6 | Store, Holiday_Flag | Baseline |
| Time | 4 | month, quarter | Seasonal context |
| Cyclic | 4 | week_sin, week_cos | Week 52 adjacent week 1 |
| Lag | 4 | lag_52 | #1 SHAP — $800K impact |
| Rolling | 3 | rolling_std_4 | #2 SHAP — volatility |
| Store baseline | 3 | store_avg, store_cv | Store identity |
| Holiday interactions | 3 | holiday_x_lag1 | Holiday vs normal |
| Holiday types | 4 | is_thanksgiving | Each holiday distinct |
| Pre/post holiday | 2 | pre_holiday | Surrounding weeks |
| Calendar | 1 | is_month_end | Month-end pattern |

**lag_52 impact:**
- Before: MAPE 88.54% R² -0.78
- After lag_52 added: MAPE 4.35% R² 0.98
- One feature = 95% of total improvement

**Human decisions:**
- Cyclic encoding: week 52 and week 1 are 1 week
  apart not 51 — sin/cos corrects this
- Keep Fuel_Price despite p=0.447: SHAP confirmed
  6th most important feature
- Keep outliers: holiday MAPE 3.2% validates this

---

## Step 4 — Split

- Chronological 70/15/15
- Train: 2,866 — Val: 614 — Test: 615
- Test set never touched until final evaluation
- Random split rejected: leaks future into training
  — inflates MAPE by 2 to 4 percentage points
- Every public notebook uses random split — invalid

---

## Step 5 — Model Selection

| Model | MAPE | Decision |
|-------|------|---------|
| Decision Tree | 11.32% | Out |
| Random Forest | 9.17% | Out — won initially |
| XGBoost | 9.94% | Out |
| LightGBM | Best | Selected |

- Random Forest won initial comparison at 9.17%
- LightGBM chosen: lag features benefit more
  from gradient boosting — validated post FE

---

## Step 6 — Evaluation

**Metrics with stakeholder meaning:**

| Metric | Value | Stakeholder hears |
|--------|-------|-----------------|
| MAPE 4.41% | Avg % error | $44,100 on $1M store |
| MAE $38,661 | Avg dollar | Order buffer amount |
| RMSE $55,603 | Large error | Worst case weekly |
| R² 0.9810 | 98.1% explained | All patterns captured |
| WMAE $39,104 | Holiday weighted | Competition aligned |

**Decisions per problem store:**

| Store | MAPE | Action required |
|-------|------|----------------|
| 39 | 5.81% | +6% inventory buffer |
| 42 | 5.23% | +6% inventory buffer |
| 43 | 4.40% | -4% order reduction |
| 44 | 4.98% | $15K threshold not % |

**Benchmark — top 15 to 20%:**

| Notebook | RMSE | Split | Valid? |
|----------|------|-------|--------|
| CatBoost | $48,856 | Random | No |
| LightGBM public | $48,934 | Random | No |
| GradientBoosting | $51,701 | Random | No |
| XGBoost | $53,832 | Random | No |
| **This project** | **$55,603** | **Chronological** | **Yes** |

- No leaderboard — closed competition
- All 5 public notebooks use invalid random split
- Estimated top 15 to 20% using honest evaluation
- Only approach with MAPE + WMAE + deployment
  + SHAP + monitoring combined

---

## Step 7 — Deployment

| Dashboard page | Stakeholder | Decision |
|---------------|-------------|---------|
| Dashboard | Manager + director | Weekly KPI review |
| Error Analysis | Supply chain | Store buffer identification |
| Error Heatmap | Regional director | Seasonal patterns |
| Store Deep Dive | Store manager | Individual review |
| Monitoring | Data team | Model health |
| Live Predictor | Operations | Monday ordering |

---

## Step 8 — Monitoring

- 0 of 21 months triggered 8% MAPE alert
- Fuel_Price drift: 9.5% — near 10% threshold
- 4 retraining triggers defined and documented

---

## Business Impact

| Before | After | Improvement |
|--------|-------|-------------|
| $452,868 weekly error | $38,661 | -$414,207 |
| 0 stores reliable | 41 of 45 | 91% coverage |
| No confidence range | Per-store quantified | Actionable |
| Manual variance input | Automated weekly | Time saved |
| 294.79% worst MAPE | 4.98% | -289.81pp |

## Recommendations

- Deploy 41 stores immediately
- Stores 39 and 42: add 6% inventory buffer
- Store 44: use $15K absolute threshold
- Collect promotion data: closes remaining gap
- Retrain if Fuel_Price exceeds 10% drift

---

## Human Decision Log

| Decision | Standard | Mine | Result |
|----------|----------|------|--------|
| Outliers | Remove | Keep | Holiday MAPE 3.2% |
| Metric | RMSE | MAPE | Fair across 10x range |
| Split | Random | Chronological | No leakage |
| Fuel_Price | Remove | Keep | SHAP #6 feature |
| Christmas | Peak | Decline | No over-prediction |
| Encoding | Raw | Cyclic | Week 52 and 1 adjacent |

---

## Tech Stack
Python | LightGBM | SHAP | Streamlit | Plotly |
Scipy | Scikit-learn | Pandas | NumPy

## Files
| File | Description |
|------|-------------|
| app.py | 6-page dashboard |
| walmart_model.pkl | Trained model |
| walmart_features.pkl | 33 features |
| test_results.csv | 615 predictions |

```bash
pip install -r requirements.txt
streamlit run app.py
```
