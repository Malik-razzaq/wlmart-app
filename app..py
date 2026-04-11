import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Walmart Forecast", page_icon="🛒", layout="wide")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    results  = pd.read_csv("test_results.csv", parse_dates=["Date"])
    df_full  = pd.read_csv("walmart_features.csv", parse_dates=["Date"])
    model    = joblib.load("walmart_model.pkl")
    features = joblib.load("walmart_features.pkl")
    return results, df_full, model, features

results, df_full, model, features = load_data()

# Silence LightGBM warnings
try:
    model.set_params(verbose=-1)
except:
    pass

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("Walmart Forecast")
st.sidebar.markdown("**Model:** LightGBM V2")
st.sidebar.markdown("**MAPE:** 4.43%")
st.sidebar.markdown("**R2:** 0.9809")
st.sidebar.markdown("**Stores:** 45")

page = st.sidebar.radio("Navigate", [
    "Dashboard",
    "Step 7 Live Predictor"
])

# ----------------------------
# DASHBOARD
# ----------------------------
if page == "Dashboard":

    st.title("Walmart Weekly Sales Forecasting")
    st.divider()

    c1,c2,c3 = st.columns(3)
    c1.metric("MAPE", "4.43%")
    c2.metric("R2 Score", "0.9809")
    c3.metric("Accuracy (<10%)", "93.7%")

    st.subheader("Actual vs Predicted")

    fig = px.scatter(
        results, x="Actual", y="Predicted",
        color="Error_Pct",
        color_continuous_scale="RdYlGn_r"
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# LIVE PREDICTOR
# ----------------------------
elif page == "Step 7 Live Predictor":

    st.title("Live Weekly Sales Predictor")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        store = st.selectbox("Store", list(range(1, 46)))
        holiday = st.selectbox("Holiday", [0,1], format_func=lambda x: "Yes" if x else "No")
        temperature = st.slider("Temperature", 10.0, 110.0, 65.0)

    with col2:
        fuel_price = st.slider("Fuel Price", 2.0, 5.0, 3.4)
        cpi = st.slider("CPI", 120.0, 260.0, 210.0)
        unemployment = st.slider("Unemployment", 3.0, 15.0, 7.5)

    with col3:
        date = st.date_input("Date")
        lag_1 = st.number_input("Last Week Sales", value=1000000)
        lag_52 = st.number_input("Last Year Same Week", value=980000)

    st.divider()

    # ----------------------------
    # VALIDATION
    # ----------------------------
    if lag_1 <= 0 or lag_52 <= 0:
        st.warning("Sales values must be positive")
        st.stop()

    # ----------------------------
    # PREDICTION
    # ----------------------------
    if st.button("Generate Forecast", use_container_width=True):

        dt = pd.Timestamp(date)

        # Base features
        row = {
            "Store": store,
            "Holiday_Flag": holiday,
            "Temperature": temperature,
            "Fuel_Price": fuel_price,
            "CPI": cpi,
            "Unemployment": unemployment,
            "month": dt.month,
            "week": int(dt.isocalendar()[1]),
            "store_cv": float(df_full[df_full["Store"] == store]["Weekly_Sales"].std() /
                              df_full[df_full["Store"] == store]["Weekly_Sales"].mean()),
            "holiday_x_lag1": holiday * lag_1,
            "holiday_x_lag52": holiday * lag_52,
            "lag_52_diff": lag_1 - lag_52
        }

        input_df = pd.DataFrame([row])

        # ----------------------------
        # STRICT FEATURE MATCH
        # ----------------------------
        missing = [f for f in features if f not in input_df.columns]

        if missing:
            st.error(f"Missing features: {missing}")
            st.stop()

        input_df = input_df[features]

        # ----------------------------
        # PREDICT
        # ----------------------------
        pred = model.predict(input_df)[0]

        st.success(f"Predicted Weekly Sales: ${pred:,.0f}")

        # ----------------------------
        # METRICS
        # ----------------------------
        c1,c2,c3 = st.columns(3)

        c1.metric("Prediction", f"${pred:,.0f}")
        c2.metric("vs Last Week", f"{(pred-lag_1):,.0f}")
        c3.metric("vs Last Year", f"{(pred-lag_52):,.0f}")

        # ----------------------------
        # CONFIDENCE
        # ----------------------------
        store_mape = results[results["Store"] == store]["Error_Pct"].mean()

        if store_mape < 5:
            st.success("High confidence prediction")
        elif store_mape < 8:
            st.warning("Moderate confidence")
        else:
            st.error("Low confidence")

        # ----------------------------
        # LOGGING (PRODUCTION FEATURE)
        # ----------------------------
        log_row = input_df.copy()
        log_row["prediction"] = pred
        log_row["timestamp"] = pd.Timestamp.now()

        try:
            log_df = pd.read_csv("prediction_logs.csv")
            log_df = pd.concat([log_df, log_row], ignore_index=True)
        except:
            log_df = log_row

        log_df.to_csv("prediction_logs.csv", index=False)