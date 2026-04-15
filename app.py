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

# =========================
# Helper
# =========================
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100


# =========================
# LOAD DATA (FIXED)
# =========================
@st.cache_data
def load_data():
    try:
        # Try correct file
        try:
            df_full = pd.read_csv("walmart_features.csv", parse_dates=["Date"])
        except:
            df_full = pd.read_csv("Walmart.csv", parse_dates=["Date"])

        results = pd.read_csv("test_results.csv", parse_dates=["Date"])

        model = joblib.load("walmart_model.pkl")

        # Features optional
        try:
            features = joblib.load("walmart_features.pkl")
        except:
            features = None

        return results, df_full, model, features

    except Exception as e:
        st.error("❌ File loading error")
        st.write(e)
        st.stop()


results, df_full, model, features = load_data()


# =========================
# CLEANING (FIXED)
# =========================
df_full.columns = df_full.columns.str.strip()
results.columns = results.columns.str.strip()

# Fix Error %
if "Error_Pct" not in results.columns:
    results["Error_Pct"] = abs(results["Actual"] - results["Predicted"]) / results["Actual"] * 100

# Fix month
if "month" not in results.columns:
    results["month"] = results["Date"].dt.month

# Fix Bias
if "Bias" in results.columns and "Bias_Dir" not in results.columns:
    results["Bias_Dir"] = results["Bias"].apply(lambda x: "Over" if x > 0 else "Under")
else:
    results["Bias_Dir"] = "Neutral"


# Fix model verbosity
try:
    model.set_params(verbose=-1)
except:
    pass


# =========================
# SIDEBAR
# =========================
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


# =========================
# DASHBOARD
# =========================
if page == "Dashboard":
    st.title("Walmart Weekly Sales Forecasting")
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("MAPE", "4.43%")
    c2.metric("R2", "0.9809")
    c3.metric("RMSE", "$55,667")
    c4.metric("<10% Error", "93.7%")
    c5.metric(">10% Error", "6.3%")

    st.divider()

    fig = px.scatter(results, x="Actual", y="Predicted", color="Error_Pct")
    st.plotly_chart(fig, use_container_width=True)


# =========================
# ERROR ANALYSIS
# =========================
elif page == "Error Analysis":
    st.title("Error Analysis")

    store_err = results.groupby("Store")["Error_Pct"].mean()

    fig = px.bar(store_err, title="Error by Store")
    st.plotly_chart(fig, use_container_width=True)


# =========================
# HEATMAP
# =========================
elif page == "Error Heatmap":
    import seaborn as sns
    import matplotlib.pyplot as plt

    pivot = results.pivot_table(index="Store", columns="month", values="Error_Pct")

    fig, ax = plt.subplots()
    sns.heatmap(pivot, cmap="RdYlGn_r", ax=ax)
    st.pyplot(fig)


# =========================
# STORE DEEP DIVE
# =========================
elif page == "Store Deep Dive":
    store = st.selectbox("Store", sorted(results["Store"].unique()))
    sd = results[results["Store"] == store]

    st.metric("MAPE", f"{sd['Error_Pct'].mean():.2f}%")

    fig = px.line(sd, x="Date", y=["Actual","Predicted"])
    st.plotly_chart(fig, use_container_width=True)


# =========================
# STEP 8 MONITORING
# =========================
elif page == "Step 8 Monitoring":
    st.title("Monitoring")

    results["month_year"] = results["Date"].dt.to_period("M").astype(str)

    monthly = results.groupby("month_year").apply(
        lambda g: mape(g["Actual"], g["Predicted"])
    ).reset_index()

    monthly.columns = ["Month","MAPE"]

    fig = px.line(monthly, x="Month", y="MAPE")
    st.plotly_chart(fig)


# =========================
# STEP 7 LIVE PREDICTOR
# =========================
elif page == "Step 7 Live Predictor":

    st.title("Live Predictor")

    store = st.selectbox("Store", list(range(1,46)))
    holiday = st.selectbox("Holiday", [0,1])
    temp = st.slider("Temperature", 10.0, 110.0, 65.0)

    fuel = st.slider("Fuel", 2.0, 5.0, 3.4)
    cpi = st.slider("CPI", 120.0, 260.0, 210.0)
    unemp = st.slider("Unemployment", 3.0, 15.0, 7.5)

    lag1 = st.number_input("Last Week", value=1000000)
    lag52 = st.number_input("Last Year", value=980000)

    if st.button("Predict"):

        row = {
            "Store":store,
            "Holiday_Flag":holiday,
            "Temperature":temp,
            "Fuel_Price":fuel,
            "CPI":cpi,
            "Unemployment":unemp,
            "month":1,
            "week":1,
            "store_cv":0,
            "holiday_x_lag1":holiday*lag1,
            "holiday_x_lag52":holiday*lag52,
            "lag_52_diff":lag1-lag52
        }

        df = pd.DataFrame([row])

        # Safe feature handling
        if features is not None:
            for f in features:
                if f not in df.columns:
                    df[f] = 0
            df = df[list(features)]

        pred = model.predict(df)[0]

        st.success(f"Prediction: ${pred:,.0f}")
