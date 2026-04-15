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
    page_title="Walmart Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

# =========================
# LOAD DATA (FIXED ONLY HERE)
# =========================
@st.cache_data
def load_data():
    try:
        # Try original file
        try:
            df_full = pd.read_csv("walmart_features.csv", parse_dates=["Date"])
        except:
            # Fallback to your actual file
            df_full = pd.read_csv("Walmart.csv", parse_dates=["Date"])

        results  = pd.read_csv("test_results.csv", parse_dates=["Date"])
        model    = joblib.load("walmart_model.pkl")

        try:
            features = joblib.load("walmart_features.pkl")
        except:
            features = None

        return df_full, results, model, features

    except Exception as e:
        st.error("File loading error")
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
    results["month"] = pd.to_datetime(results["Date"]).dt.month

if "Bias" in results.columns:
    results["Bias_Dir"] = results["Bias"].apply(
        lambda x: "Over" if x > 0 else "Under"
    )

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
    "Step 8 — Monitoring",
    "Step 7 — Live Predictor"
])

# =========================
# DASHBOARD
# =========================
if page == "Dashboard":
    st.title("Walmart Weekly Sales Forecasting")
    st.caption("LightGBM V2 | 45 Stores | Feb 2010 to Oct 2012")
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("MAPE", "4.43%", "Goal < 5%")
    c2.metric("R2 Score", "0.9809", "Goal > 0.95")
    c3.metric("RMSE", "$55,667")
    c4.metric("Below 10% Error", "93.7%")
    c5.metric("High Error Rows", "6.3%")

    store_filter = st.selectbox(
        "Select Store",
        sorted(df_full["Store"].unique())
    )
    filtered_df = df_full[df_full["Store"] == store_filter]

    st.metric("Total Sales", f"${filtered_df['Weekly_Sales'].sum():,.0f}")

    fig = px.line(filtered_df, x="Date", y="Weekly_Sales")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# ERROR ANALYSIS
# =========================
elif page == "Error Analysis":
    st.title("Error Analysis")

    store_err = results.groupby("Store")["Error_Pct"].mean()
    st.bar_chart(store_err)

# =========================
# HEATMAP
# =========================
elif page == "Error Heatmap":
    st.title("Error Heatmap")

    pivot = results.pivot_table(
        index="Store", columns="month",
        values="Error_Pct", aggfunc="mean"
    )

    st.dataframe(pivot)

# =========================
# STORE DEEP DIVE
# =========================
elif page == "Store Deep Dive":
    st.title("Store Deep Dive")

    store = st.selectbox("Store", sorted(results["Store"].unique()))
    sd = results[results["Store"] == store]

    st.metric("MAPE", f"{sd['Error_Pct'].mean():.2f}%")

# =========================
# STEP 8 — MONITORING
# =========================
elif page == "Step 8 — Monitoring":
    st.title("Model Monitoring")

    results["month_year"] = results["Date"].dt.to_period("M").astype(str)

    monthly = results.groupby("month_year").apply(
        lambda g: mape(g["Actual"], g["Predicted"])
    ).reset_index()

    monthly.columns = ["Month", "MAPE"]

    fig = px.line(monthly, x="Month", y="MAPE")
    st.plotly_chart(fig)

# =========================
# STEP 7 — LIVE PREDICTOR
# =========================
elif page == "Step 7 — Live Predictor":
    st.title("Live Predictor")

    store = st.selectbox("Store", list(range(1, 46)))
    temp = st.slider("Temperature", 10.0, 110.0, 65.0)
    fuel = st.slider("Fuel Price", 2.0, 5.0, 3.4)
    cpi = st.slider("CPI", 120.0, 260.0, 210.0)
    unemp = st.slider("Unemployment", 3.0, 15.0, 7.5)
    lag1 = st.number_input("Last Week Sales", value=1000000)
    lag52 = st.number_input("Last Year Sales", value=980000)

    if st.button("Predict"):

        row = pd.DataFrame([{
            "Store": store,
            "Temperature": temp,
            "Fuel_Price": fuel,
            "CPI": cpi,
            "Unemployment": unemp,
            "lag_1": lag1,
            "lag_52": lag52
        }])

        if features is not None:
            for f in features:
                if f not in row.columns:
                    row[f] = 0
            row = row[features]

        pred = model.predict(row)[0]

        st.success(f"Prediction: ${pred:,.0f}")
