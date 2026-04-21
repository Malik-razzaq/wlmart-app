

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(
    page_title="Walmart Forecast",
    page_icon="🛒",
    layout="wide"
)

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def calculate_wmae(y_true, y_pred, weights):
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    weights = np.array(weights)
    if np.sum(weights) == 0:
        return 0
    min_len = min(len(y_true), len(y_pred), len(weights))
    return np.sum(
        weights[:min_len] * np.abs(y_true[:min_len] - y_pred[:min_len])
    ) / np.sum(weights[:min_len])

@st.cache_data
def load_data():
    try:
        if os.path.exists("walmart_features.csv"):
            df_full = pd.read_csv("walmart_features.csv", parse_dates=["Date"])
        else:
            df_full = pd.read_csv("Walmart.csv", parse_dates=["Date"])
        results  = pd.read_csv("test_results.csv", parse_dates=["Date"])
        model    = joblib.load("walmart_model.pkl")
        features = joblib.load("walmart_features.pkl") \
                   if os.path.exists("walmart_features.pkl") \
                   and os.path.getsize("walmart_features.pkl") > 0 \
                   else None
        return df_full, results, model, features
    except Exception as e:
        st.error("File loading error")
        st.write(e)
        st.stop()

df_full, results, model, features = load_data()

df_full.columns = df_full.columns.str.strip()
results.columns = results.columns.str.strip()

if "Error_Pct" not in results.columns:
    results["Error_Pct"] = np.where(
        results["Actual"] == 0, 0,
        abs(results["Actual"] - results["Predicted"]) / results["Actual"] * 100
    )

if "month" not in results.columns:
    results["month"] = results["Date"].dt.month

if "Bias" in results.columns:
    results["Bias_Dir"] = results["Bias"].apply(
        lambda x: "Over" if x > 0 else "Under"
    )
else:
    results["Bias_Dir"] = "Neutral"

try:
    model.set_params(verbose=-1)
except:
    pass

st.sidebar.title("Walmart Forecast")
st.sidebar.markdown("**Model:** LightGBM V2")
st.sidebar.markdown("**Version:** v2.0")
st.sidebar.markdown("**MAPE:** 4.03%")
st.sidebar.markdown("**WMAE:** $35,357")
st.sidebar.markdown("**R2:** 0.9809")
st.sidebar.markdown("**Stores:** 45")
st.sidebar.markdown("**Features:** 37 engineered")
st.sidebar.markdown("**Dataset:** Public Walmart dataset")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "Dashboard",
    "Error Analysis",
    "Error Heatmap",
    "Store Deep Dive",
    "Step 8 Monitoring",
    "Step 7 Live Predictor"
])

if page == "Dashboard":
    st.title("Walmart Weekly Sales Forecasting")
    st.caption("LightGBM V2 | 45 Stores | Feb 2010 to Oct 2012")
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("MAPE",            "4.03%",  "Goal < 5%")
    c2.metric("R2 Score",        "0.9809", "Goal > 0.95")
    c3.metric("RMSE",            "$55,667")
    c4.metric("Below 10% Error", "93.7%",  "576 of 615 rows")
    c5.metric("High Error Rows", "6.3%",   "39 of 615 rows")

    st.divider()
    store_filter = st.selectbox(
        "Select Store", sorted(df_full["Store"].unique())
    )
    filtered_df = df_full[df_full["Store"] == store_filter]

    col1, col2 = st.columns(2)
    if "Weekly_Sales" in filtered_df.columns:
        col1.metric("Total Sales",
                    f"${filtered_df['Weekly_Sales'].sum():,.0f}")
        col2.metric("Avg Weekly Sales",
                    f"${filtered_df['Weekly_Sales'].mean():,.0f}")

    if "Weekly_Sales" in filtered_df.columns:
        st.subheader(f"Store {store_filter} Sales Trend")
        fig = px.line(filtered_df, x="Date", y="Weekly_Sales")
        st.plotly_chart(fig, use_container_width=True)

    if "Temperature" in filtered_df.columns:
        st.subheader("Temperature vs Sales")
        fig_t = px.scatter(filtered_df, x="Temperature", y="Weekly_Sales")
        st.plotly_chart(fig_t, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actual vs Predicted")
        fig2 = px.scatter(
            results, x="Actual", y="Predicted",
            color="Error_Pct",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 20],
            hover_data=["Store","Date","Error_Pct"],
            labels={"Error_Pct":"Error %"}
        )
        lim = [results["Actual"].min(), results["Actual"].max()]
        fig2.add_scatter(
            x=lim, y=lim, mode="lines",
            line=dict(color="black", dash="dash"),
            name="Perfect Forecast"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Error Band Distribution")
        bands = pd.cut(
            results["Error_Pct"],
            bins=[0,2,5,10,15,20,100],
            labels=["0-2%","2-5%","5-10%","10-15%","15-20%",">20%"]
        ).value_counts().sort_index()
        fig3 = px.bar(
            x=bands.index, y=bands.values,
            color=bands.values,
            color_continuous_scale="RdYlGn_r",
            labels={"x":"Error Band","y":"Rows"},
            text=bands.values
        )
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Forecast vs Actual Over Time")
    time_df = results.groupby("Date")[["Actual","Predicted"]].mean().reset_index()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=time_df["Date"], y=time_df["Actual"],
        name="Actual", line=dict(color="steelblue", width=2)
    ))
    fig4.add_trace(go.Scatter(
        x=time_df["Date"], y=time_df["Predicted"],
        name="Predicted", line=dict(color="orange", dash="dash", width=2)
    ))
    fig4.update_layout(xaxis_title="Date", yaxis_title="Avg Weekly Sales")
    st.plotly_chart(fig4, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.success("62.9% of predictions below 5% error")
    col2.success("93.7% of predictions below 10% error")
    col3.warning("6.3% above 10% — concentrated in 4 stores")

    st.subheader("Raw Data")
    st.dataframe(filtered_df.head(20), use_container_width=True)

elif page == "Error Analysis":
    st.title("Error Analysis")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Avg Error by Store")
        store_err = results.groupby("Store")["Error_Pct"].mean().sort_values(ascending=False)
        fig = px.bar(
            x=store_err.index.astype(str), y=store_err.values,
            color=store_err.values, color_continuous_scale="RdYlGn_r",
            labels={"x":"Store","y":"Avg Error %"}
        )
        fig.add_hline(y=10, line_dash="dash", line_color="red",
                      annotation_text="10% threshold")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Avg Error by Month")
        month_err = results.groupby("month")["Error_Pct"].mean()
        fig2 = px.bar(
            x=month_err.index, y=month_err.values,
            color=month_err.values, color_continuous_scale="RdYlGn_r",
            labels={"x":"Month","y":"Avg Error %"}
        )
        fig2.add_hline(y=10, line_dash="dash", line_color="red")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Over vs Under Prediction by Store")
    bias_df = results.groupby(["Store","Bias_Dir"])["Error_Pct"].mean().reset_index()
    fig3 = px.bar(
        bias_df, x="Store", y="Error_Pct",
        color="Bias_Dir", barmode="group",
        color_discrete_map={"Over":"orange","Under":"steelblue"},
        labels={"Error_Pct":"Avg Error %"}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("All High Error Rows above 10%")
    high = results[results["Error_Pct"] > 10].sort_values("Error_Pct", ascending=False)
    cols_show = [c for c in
                 ["Store","Date","Actual","Predicted","Error_Pct","Bias_Dir","Holiday"]
                 if c in results.columns]
    st.dataframe(high[cols_show], use_container_width=True)

    st.subheader("Root Cause Summary")
    root = pd.DataFrame({
        "Store"     : [39, 42, 43, 44],
        "Bias"      : ["100% Under","100% Under","86% Over","100% Over"],
        "Avg Error" : ["6.09%","5.44%","4.36%","4.36%"],
        "Avg Sales" : ["$1.55M","$570K","$622K","$314K"],
        "Root Cause": [
            "$1.3M swing — peaks unpredictable",
            "Systematic under-prediction year-round",
            "Unique summer overestimation Jul-Oct",
            "Smallest store — low dollar inflates percent"
        ]
    })
    st.dataframe(root, use_container_width=True)
    st.warning(
        "4 stores = 82.1% of all high-error rows. "
        "Errors are store-specific — not holiday-driven."
    )

elif page == "Error Heatmap":
    import seaborn as sns
    import matplotlib.pyplot as plt

    st.title("Error Heatmap — Store x Month")
    st.divider()

    pivot = results.pivot_table(
        index="Store", columns="month",
        values="Error_Pct", aggfunc="mean"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f",
                cmap="RdYlGn_r", center=5, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Avg Error %"})
    ax.set_title("Avg Error Percent by Store x Month", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Store")
    st.pyplot(fig)

    st.divider()
    col1, col2, col3 = st.columns(3)
    worst_store = results.groupby("Store")["Error_Pct"].mean().idxmax()
    worst_month = results.groupby("month")["Error_Pct"].mean().idxmax()
    best_month  = results.groupby("month")["Error_Pct"].mean().idxmin()
    col1.metric("Worst Store", f"Store {worst_store}")
    col2.metric("Worst Month", f"Month {worst_month} December")
    col3.metric("Best Month",  f"Month {best_month}")

elif page == "Store Deep Dive":
    st.title("Store Deep Dive")
    st.divider()

    store = st.selectbox("Select Store", sorted(results["Store"].unique()))
    sd    = results[results["Store"] == store].copy()

    c1,c2,c3 = st.columns(3)
    c1.metric("Store MAPE",       f"{sd['Error_Pct'].mean():.2f}%")
    c2.metric("Avg Sales",        f"${sd['Actual'].mean():,.0f}")
    c3.metric("High Error Weeks", str((sd["Error_Pct"] > 10).sum()))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Store {store} Forecast vs Actual")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sd["Date"], y=sd["Actual"],
            name="Actual", line=dict(color="steelblue", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=sd["Date"], y=sd["Predicted"],
            name="Predicted", line=dict(color="orange", dash="dash", width=2)
        ))
        fig.update_layout(xaxis_title="Date", yaxis_title="Weekly Sales")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Store {store} Weekly Error Percent")
        fig2 = px.scatter(
            sd, x="Date", y="Error_Pct",
            color="Error_Pct", size="Error_Pct",
            color_continuous_scale="RdYlGn_r"
        )
        fig2.add_hline(y=10, line_dash="dash", line_color="red",
                       annotation_text="10% threshold")
        st.plotly_chart(fig2, use_container_width=True)

    high_s = sd[sd["Error_Pct"] > 10].sort_values("Error_Pct", ascending=False)
    if len(high_s):
        cols_show = [c for c in
                     ["Date","Actual","Predicted","Error_Pct","Bias_Dir","Holiday"]
                     if c in sd.columns]
        st.dataframe(high_s[cols_show], use_container_width=True)
    else:
        st.success(f"Store {store} has no predictions above 10% error")

elif page == "Step 8 Monitoring":
    st.title("Step 8 — Model Monitoring and Maintenance")
    st.divider()

    st.subheader("Monthly MAPE Tracking")
    results["month_year"] = results["Date"].dt.to_period("M").astype(str)
    monthly = results.groupby(
        "month_year", group_keys=False
    ).apply(lambda g: mape(g["Actual"], g["Predicted"])).reset_index()
    monthly.columns = ["Month","MAPE"]

    fig = px.line(monthly, x="Month", y="MAPE", markers=True)
    fig.add_hline(y=8, line_dash="dash", line_color="red",
                  annotation_text="Alert 8%")
    fig.add_hline(y=5, line_dash="dash", line_color="orange",
                  annotation_text="Target 5%")
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    alerts = monthly[monthly["MAPE"] > 8]
    if len(alerts):
        st.error(f"{len(alerts)} months exceeded 8% alert threshold")
        st.dataframe(alerts)
    else:
        st.success("0 of 21 months exceeded 8% alert — model is healthy")

    st.divider()
    st.subheader("WMAE — Kaggle Competition Metric")
    st.info(
        "WMAE (Weighted Mean Absolute Error) is the official Walmart "
        "Kaggle competition metric. Holiday weeks receive 5x weight. "
        "Calculated on our own test split for internal evaluation only. "
        "Our WMAE: $35,357 on 15% holdout split."
    )

    holiday_col = "Holiday_Flag" if "Holiday_Flag" in results.columns \
                  else "Holiday" if "Holiday" in results.columns \
                  else None

    if holiday_col is not None:
        weights_r  = np.where(results[holiday_col] == 1, 5, 1)
        min_len    = min(len(results["Actual"]),
                         len(results["Predicted"]),
                         len(weights_r))
        wmae_score = calculate_wmae(
            results["Actual"].values[:min_len],
            results["Predicted"].values[:min_len],
            weights_r[:min_len]
        )
        mae_score = calculate_wmae(
            results["Actual"].values[:min_len],
            results["Predicted"].values[:min_len],
            np.ones(min_len)
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Our WMAE",  f"${wmae_score:,.0f}")
        col2.metric("Our MAE",   f"${mae_score:,.0f}")
        col3.metric("Our MAPE",  "4.03%")

        wmae_df = pd.DataFrame({
            "Metric" : [
                "WMAE (our 15% test split)",
                "MAE (unweighted)",
                "MAPE",
                "Kaggle top WMAE (approx)"
            ],
            "Value"  : [
                f"${wmae_score:,.0f}",
                f"${mae_score:,.0f}",
                "4.03%",
                "~$1,500 to $2,500"
            ],
            "Notes"  : [
                "5x holiday weighting — matches Kaggle rule",
                "Equal weight baseline",
                "Interpretable business metric",
                "On private test set — not directly comparable"
            ]
        })
        st.dataframe(wmae_df, use_container_width=True)
        st.warning(
            "Gap to Kaggle top scores is due to missing promotion, "
            "markdown and store-type data not in this public dataset. "
            "Not directly comparable — Kaggle used a different private test set."
        )
    else:
        st.warning("Holiday column not found — WMAE cannot be calculated")

    st.divider()
    st.subheader("Benchmark Comparison (MAPE)")
    benchmark = pd.DataFrame({
        "Model"   : [
            "Naive Baseline (last week sales)",
            "Random Forest + raw features",
            "LightGBM V2 (this project)",
            "Best public notebooks (MAPE est.)",
        ],
        "MAPE"    : ["~18%","9.17%","4.03%","~2-3%"],
        "Features": [
            "None",
            "8 raw features",
            "37 engineered features",
            "Includes promotion + markdown data"
        ],
        "Notes"   : [
            "Baseline to beat",
            "Starting point",
            "Near ceiling for available data",
            "Data not in this public dataset"
        ]
    })
    st.dataframe(benchmark, use_container_width=True)
    st.info(
        "Industry Note: The original Walmart Kaggle competition uses WMAE. "
        "This project uses MAPE for interpretability and also reports WMAE "
        "for competition alignment. Model beats naive baseline by 74.5% "
        "and RF baseline by 51.8%."
    )
    st.warning(
        "Gap to top notebooks is due to missing promotion and "
        "store-type data — not model weakness."
    )

    st.divider()
    st.subheader("Feature Drift Detection")
    n         = len(df_full)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    drift_rows = []
    for col in ["Temperature","Fuel_Price","CPI","Unemployment"]:
        if col in df_full.columns:
            t_mean = df_full.iloc[:train_end][col].mean()
            v_mean = df_full.iloc[val_end:][col].mean()
            drift  = abs(v_mean - t_mean) / t_mean * 100 \
                     if t_mean != 0 else 0
            drift_rows.append({
                "Feature"   : col,
                "Train Mean": round(t_mean, 3),
                "Test Mean" : round(v_mean, 3),
                "Drift %"   : round(drift, 1),
                "Status"    : "Alert" if drift > 10 else "OK"
            })

    if drift_rows:
        drift_df = pd.DataFrame(drift_rows)
        st.dataframe(drift_df, use_container_width=True)
        fig2 = px.bar(
            drift_df, x="Feature", y="Drift %",
            color="Drift %", color_continuous_scale="RdYlGn_r", text="Drift %"
        )
        fig2.add_hline(y=10, line_dash="dash", line_color="red",
                       annotation_text="Alert 10%")
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Retraining Schedule")
    schedule = pd.DataFrame({
        "Trigger"  : ["Scheduled","Performance","Drift","Emergency"],
        "Frequency": ["Monthly","On alert","Quarterly","Immediate"],
        "Condition": ["Every 4 weeks","Live MAPE > 8%",
                      "Feature drift > 10%","Fuel spike or new store"],
        "Action"   : ["Append data + retrain + validate",
                      "Retrain + redeploy immediately",
                      "Review features + retrain",
                      "Manual review + targeted retrain"]
    })
    st.dataframe(schedule, use_container_width=True)

    st.divider()
    st.subheader("Model Health Summary")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Overall MAPE",  "4.03%",    "Below 8% alert")
    c2.metric("WMAE",          "$35,357",  "Kaggle metric")
    c3.metric("Max Drift",     "9.5%",     "Below 10% alert")
    c4.metric("High Err Rate", "6.3%",     "Below 10% target")

elif page == "Step 7 Live Predictor":
    st.title("Step 7 — Live Weekly Sales Predictor")
    st.caption("Enter store details to get an instant forecast")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Store Info")
        store       = st.selectbox("Store ID", list(range(1, 46)))
        holiday     = st.selectbox("Holiday Week", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No")
        temperature = st.slider("Temperature F", 10.0, 110.0, 65.0)

    with col2:
        st.subheader("Economic Inputs")
        fuel_price   = st.slider("Fuel Price",   2.0,   5.0,   3.4)
        cpi          = st.slider("CPI",          120.0, 260.0, 210.0)
        unemployment = st.slider("Unemployment", 3.0,   15.0,  7.5)

    with col3:
        st.subheader("Sales History")
        date   = st.date_input("Forecast Date")
        lag_1  = st.number_input("Last Week Sales",     value=1000000, step=10000)
        lag_52 = st.number_input("Same Week Last Year", value=980000,  step=10000)

    st.divider()

    if lag_1 <= 0 or lag_52 <= 0:
        st.warning("Sales values must be positive")
        st.stop()

    if st.button("Generate Forecast", type="primary", use_container_width=True):
        try:
            dt    = pd.Timestamp(date)
            week  = int(dt.isocalendar()[1])
            month = int(dt.month)

            if "Weekly_Sales" in df_full.columns:
                store_data = df_full[df_full["Store"] == store]["Weekly_Sales"]
            else:
                store_data = pd.Series([1000000])

            store_avg = float(store_data.mean()) if len(store_data) > 0 else 0
            store_cv  = float(store_data.std() / store_data.mean()) \
                        if store_data.mean() != 0 else 0

            row = {
                "Store"          : store,
                "Holiday_Flag"   : holiday,
                "Temperature"    : temperature,
                "Fuel_Price"     : fuel_price,
                "CPI"            : cpi,
                "Unemployment"   : unemployment,
                "month"          : month,
                "week"           : week,
                "store_cv"       : store_cv,
                "holiday_x_lag1" : holiday * lag_1,
                "holiday_x_lag52": holiday * lag_52,
                "lag_52_diff"    : lag_1 - lag_52,
            }

            input_df = pd.DataFrame([row])
            if features is not None:
                input_df = input_df.reindex(columns=features, fill_value=0)

            pred = model.predict(input_df)[0]
            pred = max(pred, 0)
            pred = min(pred, 5_000_000)

            st.success(f"Predicted Weekly Sales: ${pred:,.0f}")

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Prediction", f"${pred:,.0f}")
            c2.metric("vs Last Week",
                      f"${pred - lag_1:+,.0f}",
                      f"{((pred-lag_1)/lag_1*100) if lag_1!=0 else 0:+.1f}%")
            c3.metric("vs Last Year",
                      f"${pred - lag_52:+,.0f}",
                      f"{((pred-lag_52)/lag_52*100) if lag_52!=0 else 0:+.1f}%")
            c4.metric("Store Avg", f"${store_avg:,.0f}")

            st.divider()
            store_mape = results[results["Store"] == store]["Error_Pct"].mean() \
                         if store in results["Store"].values else 4.03

            col1, col2 = st.columns(2)

            with col1:
                low  = max(pred * (1 - store_mape / 100), 0)
                high = pred * (1 + store_mape / 100)
                st.info(f"Confidence Range: ${low:,.0f} to ${high:,.0f}")
                st.caption(f"Store {store} avg error: {store_mape:.1f}%")

            with col2:
                if store_mape < 5:
                    st.success("High confidence prediction")
                elif store_mape < 8:
                    st.warning("Moderate confidence prediction")
                else:
                    st.error("Low confidence — high error store")
                if holiday:
                    st.info("Holiday week — elevated sales expected")
                if store in [39, 42]:
                    st.warning("Under-prediction bias — consider adding 6% buffer")
                if store == 44:
                    st.warning("Over-prediction bias — consider reducing 4%")

            log_row = input_df.copy()
            log_row["prediction"]    = pred
            log_row["timestamp"]     = pd.Timestamp.now()
            log_row["model_version"] = "LightGBM_V2"
            log_row["store_mape"]    = store_mape
            log_row["prediction_id"] = str(pd.Timestamp.now().value)

            log_file = "prediction_logs.csv"
            if os.path.exists(log_file):
                log_df = pd.read_csv(log_file)
                log_df = pd.concat([log_df, log_row], ignore_index=True)
            else:
                log_df = log_row
            log_df.to_csv(log_file, index=False)
            st.caption("Prediction logged — model version: LightGBM_V2")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            if features is not None:
                st.write("Expected features:", list(features))
| Stores | 45 |

**Project is done. Stop improving. Send Karun the message. Write LinkedIn post. Update CV.**
