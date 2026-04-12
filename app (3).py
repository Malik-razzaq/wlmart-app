import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(
    page_title="Walmart Forecast",
    page_icon="🛒",
    layout="wide"
)

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

@st.cache_data
def load_data():
    results  = pd.read_csv("test_results.csv",     parse_dates=["Date"])
    df_full  = pd.read_csv("walmart_features.csv", parse_dates=["Date"])
    model    = joblib.load("walmart_model.pkl")
    features = joblib.load("walmart_features.pkl")
    return results, df_full, model, features

results, df_full, model, features = load_data()

try:
    model.set_params(verbose=-1)
except:
    pass

if "Bias_Dir" not in results.columns:
    results["Bias_Dir"] = results["Bias"].apply(
        lambda x: "Over" if x > 0 else "Under"
    )

st.sidebar.title("Walmart Forecast")
st.sidebar.markdown("**Model:** LightGBM V2")
st.sidebar.markdown("**MAPE:** 4.43%")
st.sidebar.markdown("**R2:** 0.9809")
st.sidebar.markdown("**Stores:** 45")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "Dashboard",
    "Error Analysis",
    "Error Heatmap",
    "Store Deep Dive",
    "Step 8 Monitoring",
    "Step 7 Live Predictor"
])

# ════════════════════════════
# DASHBOARD
# ════════════════════════════
if page == "Dashboard":
    st.title("Walmart Weekly Sales Forecasting")
    st.caption("LightGBM V2 | 45 Stores | Feb 2010 to Oct 2012")
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("MAPE",            "4.43%",  "Goal < 5%")
    c2.metric("R2 Score",        "0.9809", "Goal > 0.95")
    c3.metric("RMSE",            "$55,667")
    c4.metric("Below 10% Error", "93.7%",  "576 of 615 rows")
    c5.metric("High Error Rows", "6.3%",   "39 of 615 rows")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actual vs Predicted")
        fig = px.scatter(
            results, x="Actual", y="Predicted",
            color="Error_Pct",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 20],
            hover_data=["Store","Date","Error_Pct"],
            labels={"Error_Pct":"Error %"}
        )
        lim = [results["Actual"].min(), results["Actual"].max()]
        fig.add_scatter(
            x=lim, y=lim, mode="lines",
            line=dict(color="black", dash="dash"),
            name="Perfect Forecast"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Error Band Distribution")
        bands = pd.cut(
            results["Error_Pct"],
            bins=[0,2,5,10,15,20,100],
            labels=["0-2%","2-5%","5-10%","10-15%","15-20%",">20%"]
        ).value_counts().sort_index()
        fig2 = px.bar(
            x=bands.index, y=bands.values,
            color=bands.values,
            color_continuous_scale="RdYlGn_r",
            labels={"x":"Error Band","y":"Rows"},
            text=bands.values
        )
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Forecast vs Actual Over Time")
    time_df = results.groupby("Date")[["Actual","Predicted"]].mean().reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=time_df["Date"], y=time_df["Actual"],
        name="Actual", line=dict(color="steelblue", width=2)
    ))
    fig3.add_trace(go.Scatter(
        x=time_df["Date"], y=time_df["Predicted"],
        name="Predicted", line=dict(color="orange", dash="dash", width=2)
    ))
    fig3.update_layout(xaxis_title="Date", yaxis_title="Avg Weekly Sales")
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.success("62.9% of predictions below 5% error")
    col2.success("93.7% of predictions below 10% error")
    col3.warning("6.3% above 10% — concentrated in 4 stores")

# ════════════════════════════
# ERROR ANALYSIS
# ════════════════════════════
elif page == "Error Analysis":
    st.title("Error Analysis")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Avg Error by Store")
        store_err = results.groupby("Store")["Error_Pct"].mean().sort_values(ascending=False)
        fig = px.bar(
            x=store_err.index.astype(str), y=store_err.values,
            color=store_err.values,
            color_continuous_scale="RdYlGn_r",
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
            color=month_err.values,
            color_continuous_scale="RdYlGn_r",
            labels={"x":"Month","y":"Avg Error %"}
        )
        fig2.add_hline(y=10, line_dash="dash", line_color="red")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Over vs Under Prediction by Store")
    bias_df = results.groupby(
        ["Store","Bias_Dir"]
    )["Error_Pct"].mean().reset_index()
    fig3 = px.bar(
        bias_df, x="Store", y="Error_Pct",
        color="Bias_Dir", barmode="group",
        color_discrete_map={"Over":"orange","Under":"steelblue"},
        labels={"Error_Pct":"Avg Error %"}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("All High Error Rows above 10%")
    high = results[results["Error_Pct"] > 10].sort_values(
        "Error_Pct", ascending=False
    )
    st.dataframe(
        high[["Store","Date","Actual","Predicted","Error_Pct","Bias_Dir","Holiday"]],
        use_container_width=True
    )

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

# ════════════════════════════
# ERROR HEATMAP
# ════════════════════════════
elif page == "Error Heatmap":
    st.title("Error Heatmap — Store x Month")
    st.divider()

    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = results.pivot_table(
        index="Store", columns="month",
        values="Error_Pct", aggfunc="mean"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        pivot, annot=True, fmt=".1f",
        cmap="RdYlGn_r", center=5,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Avg Error %"}
    )
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

# ════════════════════════════
# STORE DEEP DIVE
# ════════════════════════════
elif page == "Store Deep Dive":
    st.title("Store Deep Dive")
    st.divider()

    store = st.selectbox("Select Store", sorted(results["Store"].unique()))
    sd    = results[results["Store"] == store].copy()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Store MAPE",       f"{sd['Error_Pct'].mean():.2f}%")
    c2.metric("Avg Sales",        f"${sd['Actual'].mean():,.0f}")
    c3.metric("High Error Weeks", str((sd["Error_Pct"] > 10).sum()))
    c4.metric("Dominant Bias",    sd["Bias_Dir"].value_counts().index[0])

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
            color_continuous_scale="RdYlGn_r",
            hover_data=["Actual","Predicted","Bias_Dir"]
        )
        fig2.add_hline(
            y=10, line_dash="dash", line_color="red",
            annotation_text="10% threshold"
        )
        st.plotly_chart(fig2, use_container_width=True)

    high_s = sd[sd["Error_Pct"] > 10].sort_values("Error_Pct", ascending=False)
    if len(high_s):
        st.subheader(f"Store {store} High Error Weeks")
        st.dataframe(
            high_s[["Date","Actual","Predicted","Error_Pct","Bias_Dir","Holiday"]],
            use_container_width=True
        )
    else:
        st.success(f"Store {store} has no predictions above 10% error")

# ════════════════════════════
# STEP 8 MONITORING
# ════════════════════════════
elif page == "Step 8 Monitoring":
    st.title("Step 8 Model Monitoring and Maintenance")
    st.divider()

    st.subheader("Monthly MAPE Tracking")
    results["month_year"] = results["Date"].dt.to_period("M").astype(str)
    monthly = results.groupby("month_year").apply(
        lambda g: mape(g["Actual"], g["Predicted"])
    ).reset_index()
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
    st.subheader("Feature Drift Detection")
    n         = len(df_full)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    drift_rows = []
    for col in ["Temperature","Fuel_Price","CPI","Unemployment"]:
        t_mean = df_full.iloc[:train_end][col].mean()
        v_mean = df_full.iloc[val_end:][col].mean()
        drift  = abs(v_mean - t_mean) / t_mean * 100
        drift_rows.append({
            "Feature"   : col,
            "Train Mean": round(t_mean, 3),
            "Test Mean" : round(v_mean, 3),
            "Drift %"   : round(drift, 1),
            "Status"    : "Alert" if drift > 10 else "OK"
        })

    drift_df = pd.DataFrame(drift_rows)
    st.dataframe(drift_df, use_container_width=True)

    fig2 = px.bar(
        drift_df, x="Feature", y="Drift %",
        color="Drift %",
        color_continuous_scale="RdYlGn_r",
        text="Drift %"
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
        "Condition": [
            "Every 4 weeks",
            "Live MAPE > 8%",
            "Feature drift > 10%",
            "Fuel spike or new store"
        ],
        "Action": [
            "Append data retrain validate",
            "Retrain redeploy",
            "Review features retrain",
            "Manual review"
        ]
    })
    st.dataframe(schedule, use_container_width=True)

    st.divider()
    st.subheader("Model Health Summary")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Overall MAPE",  "4.43%",    "Below 8% alert")
    c2.metric("Worst Month",   "Dec 7.07%","Below 8% alert")
    c3.metric("Max Drift",     "9.5%",     "Below 10% alert")
    c4.metric("High Err Rate", "6.3%",     "Below 10% target")

# ════════════════════════════
# STEP 7 LIVE PREDICTOR
# ════════════════════════════
elif page == "Step 7 Live Predictor":
    st.title("Step 7 Live Weekly Sales Predictor")
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

        dt    = pd.Timestamp(date)
        week  = int(dt.isocalendar()[1])
        month = int(dt.month)

        store_data = df_full[df_full["Store"] == store]["Weekly_Sales"]
        store_avg  = float(store_data.mean())
        store_cv   = float(store_data.std() / store_data.mean())

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

        for f in features:
            if f not in input_df.columns:
                input_df[f] = 0

        input_df = input_df[list(features)]
        pred     = model.predict(input_df)[0]

        st.success(f"Predicted Weekly Sales: ${pred:,.0f}")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Prediction",   f"${pred:,.0f}")
        c2.metric("vs Last Week",
                  f"${pred - lag_1:+,.0f}",
                  f"{(pred - lag_1) / lag_1 * 100:+.1f}%")
        c3.metric("vs Last Year",
                  f"${pred - lag_52:+,.0f}",
                  f"{(pred - lag_52) / lag_52 * 100:+.1f}%")
        c4.metric("Store Avg", f"${store_avg:,.0f}")

        st.divider()

        store_mape = results[results["Store"] == store]["Error_Pct"].mean() \
                     if store in results["Store"].values else 4.43

        col1, col2 = st.columns(2)
        with col1:
            low  = pred * (1 - store_mape / 100)
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
        log_row["prediction"] = pred
        log_row["timestamp"]  = pd.Timestamp.now()

        try:
            log_df = pd.read_csv("prediction_logs.csv")
            log_df = pd.concat([log_df, log_row], ignore_index=True)
        except:
            log_df = log_row

        log_df.to_csv("prediction_logs.csv", index=False)
        st.caption("Prediction logged")