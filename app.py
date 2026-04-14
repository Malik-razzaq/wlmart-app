import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Walmart Dashboard", layout="wide")

# -------------------------
# LOAD DATA (SAFE VERSION)
# -------------------------
@st.cache_data
def load_data():
    try:
        # Try main file
        if os.path.exists("walmart_features.csv"):
            df_full = pd.read_csv("walmart_features.csv")
        else:
            df_full = pd.read_csv("Walmart.csv")

        results = pd.read_csv("test_results.csv")

        model = joblib.load("walmart_model.pkl")
        features = joblib.load("walmart_features.pkl")

        return df_full, results, model, features

    except Exception as e:
        st.error(f"File loading error: {e}")
        st.stop()

df_full, results, model, features = load_data()

# -------------------------
# CLEAN DATA (VERY IMPORTANT)
# -------------------------
df_full.columns = df_full.columns.str.strip().str.replace(" ", "_")
results.columns = results.columns.str.strip().str.replace(" ", "_")

# SAFE DATE CONVERSION ✅
if "Date" in df_full.columns:
    df_full["Date"] = pd.to_datetime(df_full["Date"], errors="coerce")
    df_full = df_full.dropna(subset=["Date"])

if "Date" in results.columns:
    results["Date"] = pd.to_datetime(results["Date"], errors="coerce")
    results = results.dropna(subset=["Date"])

# ADD MISSING COLUMNS SAFELY
if "Error_Pct" not in results.columns and "Actual" in results.columns:
    results["Error_Pct"] = abs(results["Actual"] - results["Predicted"]) / results["Actual"] * 100

if "Bias" not in results.columns:
    results["Bias"] = results["Predicted"] - results["Actual"]

if "Bias_Dir" not in results.columns:
    results["Bias_Dir"] = results["Bias"].apply(lambda x: "Over" if x > 0 else "Under")

if "month" not in results.columns:
    results["month"] = results["Date"].dt.month

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("🛒 Walmart App")

page = st.sidebar.radio("Navigate", [
    "Dashboard",
    "Error Analysis",
    "Live Predictor"
])

# -------------------------
# DASHBOARD
# -------------------------
if page == "Dashboard":

    st.title("📊 Walmart Sales Dashboard")

    store = st.selectbox("Select Store", sorted(df_full["Store"].unique()))
    filtered_df = df_full[df_full["Store"] == store]

    st.subheader("Sales Trend")
    fig = px.line(filtered_df, x="Date", y="Weekly_Sales")
    st.plotly_chart(fig, use_container_width=True)

    if "Temperature" in filtered_df.columns:
        st.subheader("Temperature vs Sales")
        fig2 = px.scatter(filtered_df, x="Temperature", y="Weekly_Sales")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Actual vs Predicted")

    if "Actual" in results.columns and "Predicted" in results.columns:
        fig3 = px.scatter(
            results,
            x="Actual",
            y="Predicted",
            color="Error_Pct"
        )
        st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# ERROR ANALYSIS
# -------------------------
elif page == "Error Analysis":

    st.title("🔍 Error Analysis")

    store_err = results.groupby("Store")["Error_Pct"].mean().reset_index()

    fig = px.bar(store_err, x="Store", y="Error_Pct")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("High Error Rows")
    high = results[results["Error_Pct"] > 10]
    st.dataframe(high)

# -------------------------
# LIVE PREDICTOR
# -------------------------
elif page == "Live Predictor":

    st.title("🔮 Sales Predictor")

    store = st.number_input("Store", 1, 45, 1)
    temperature = st.slider("Temperature", 0.0, 120.0, 70.0)
    fuel_price = st.slider("Fuel Price", 1.0, 5.0, 3.0)
    cpi = st.slider("CPI", 100.0, 300.0, 200.0)
    unemployment = st.slider("Unemployment", 1.0, 15.0, 7.0)

    if st.button("Predict"):

        row = {
            "Store": store,
            "Temperature": temperature,
            "Fuel_Price": fuel_price,
            "CPI": cpi,
            "Unemployment": unemployment
        }

        input_df = pd.DataFrame([row])

        for f in features:
            if f not in input_df.columns:
                input_df[f] = 0

        input_df = input_df[features]

        pred = model.predict(input_df)[0]

        st.success(f"Predicted Sales: ${pred:,.0f}")
