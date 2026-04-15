import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_percentage_error

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Walmart Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

# =========================
# HELPER
# =========================
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100


# =========================
# LOAD DATA (SAFE VERSION)
# =========================
@st.cache_data
def load_data():
    try:
        base_path = os.path.dirname(__file__)

        def safe_read(file):
            path = os.path.join(base_path, file)
            if os.path.exists(path):
                return pd.read_csv(path)
            else:
                st.error(f"❌ Missing file: {file}")
                st.stop()

        df_full = safe_read("walmart_features.csv")
        results = safe_read("test_results.csv")

        # ---- DATE FIX ----
        if "Date" in df_full.columns:
            df_full["Date"] = pd.to_datetime(df_full["Date"], errors="coerce")

        if "Date" in results.columns:
            results["Date"] = pd.to_datetime(results["Date"], errors="coerce")

        # ---- MODEL ----
        model = None
        if os.path.exists(os.path.join(base_path, "walmart_model.pkl")):
            model = joblib.load(os.path.join(base_path, "walmart_model.pkl"))

        # ---- FEATURES ----
        features = None
        if os.path.exists(os.path.join(base_path, "walmart_features.pkl")):
            features = joblib.load(os.path.join(base_path, "walmart_features.pkl"))

        return df_full, results, model, features

    except Exception as e:
        st.error("❌ File loading error")
        st.write(e)
        st.stop()


df_full, results, model, features = load_data()

# =========================
# CLEAN DATA
# =========================
df_full.columns = df_full.columns.str.strip()
results.columns = results.columns.str.strip()

if "Error_Pct" not in results.columns and "Actual" in results.columns:
    results["Error_Pct"] = (
        abs(results["Actual"] - results["Predicted"]) / results["Actual"] * 100
    )

if "month" not in results.columns and "Date" in results.columns:
    results["month"] = results["Date"].dt.month

if "Bias" in results.columns:
    results["Bias_Dir"] = results["Bias"].apply(
        lambda x: "Over" if x > 0 else "Under"
    )

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🛒 Walmart Forecast")

page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "🔍 Error Analysis",
    "🗺️ Error Heatmap",
    "🏪 Store Deep Dive",
    "📈 Monitoring",
    "🔮 Live Predictor"
])

# =========================
# DASHBOARD
# =========================
if page == "📊 Dashboard":

    st.title("📊 Walmart Sales Dashboard")

    if "Store" not in df_full.columns:
        st.error("Missing 'Store' column")
        st.stop()

    store = st.selectbox("Select Store", sorted(df_full["Store"].unique()))
    df = df_full[df_full["Store"] == store]

    st.subheader("Sales Trend")

    if "Date" in df.columns and "Weekly_Sales" in df.columns:
        fig = px.line(df, x="Date", y="Weekly_Sales")
        st.plotly_chart(fig, use_container_width=True)

    if "Temperature" in df.columns:
        st.subheader("Temperature vs Sales")
        fig2 = px.scatter(df, x="Temperature", y="Weekly_Sales")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Raw Data")
    st.dataframe(df.head(20))


# =========================
# ERROR ANALYSIS
# =========================
elif page == "🔍 Error Analysis":

    st.title("🔍 Error Analysis")

    fig = px.histogram(results, x="Error_Pct", nbins=30)
    st.plotly_chart(fig, use_container_width=True)

    high = results[results["Error_Pct"] > 10]
    st.subheader("High Error Rows (>10%)")
    st.dataframe(high)


# =========================
# HEATMAP
# =========================
elif page == "🗺️ Error Heatmap":

    st.title("🗺️ Error Heatmap")

    import seaborn as sns
    import matplotlib.pyplot as plt

    pivot = results.pivot_table(
        index="Store", columns="month",
        values="Error_Pct", aggfunc="mean"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, cmap="RdYlGn_r", annot=True, fmt=".1f", ax=ax)
    st.pyplot(fig)


# =========================
# STORE DEEP DIVE
# =========================
elif page == "🏪 Store Deep Dive":

    st.title("🏪 Store Deep Dive")

    store = st.selectbox("Select Store", sorted(results["Store"].unique()))
    sd = results[results["Store"] == store]

    st.metric("MAPE", f"{sd['Error_Pct'].mean():.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sd["Date"], y=sd["Actual"], name="Actual"))
    fig.add_trace(go.Scatter(x=sd["Date"], y=sd["Predicted"], name="Predicted"))
    st.plotly_chart(fig, use_container_width=True)


# =========================
# MONITORING
# =========================
elif page == "📈 Monitoring":

    st.title("📈 Model Monitoring")

    results["month_year"] = results["Date"].dt.to_period("M").astype(str)

    monthly = results.groupby("month_year").apply(
        lambda g: mape(g["Actual"], g["Predicted"])
    ).reset_index()

    monthly.columns = ["Month", "MAPE"]

    fig = px.line(monthly, x="Month", y="MAPE", markers=True)
    st.plotly_chart(fig, use_container_width=True)


# =========================
# LIVE PREDICTOR
# =========================
elif page == "🔮 Live Predictor":

    st.title("🔮 Sales Predictor")

    if model is None:
        st.error("Model file missing")
        st.stop()

    store = st.number_input("Store", 1, 45, 1)
    temp = st.slider("Temperature", 10.0, 110.0, 65.0)
    fuel = st.slider("Fuel Price", 2.0, 5.0, 3.0)

    if st.button("Predict"):

        input_df = pd.DataFrame([{
            "Store": store,
            "Temperature": temp,
            "Fuel_Price": fuel
        }])

        if features is not None:
            for f in features:
                if f not in input_df.columns:
                    input_df[f] = 0
            input_df = input_df[features]

        pred = model.predict(input_df)[0]

        st.success(f"Predicted Sales: ${pred:,.0f}")
