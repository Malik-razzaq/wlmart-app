import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Walmart Sales Dashboard", layout="wide")

st.title("📊 Walmart Weekly Sales Dashboard")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    # Load main dataset (FIXED: using your csv)
    df_full = pd.read_csv("Walmart.csv", parse_dates=["Date"])
    
    # Clean column names (IMPORTANT FIX)
    df_full.columns = df_full.columns.str.strip().str.title()

    # Load predictions (optional)
    try:
        results = pd.read_csv("test_results.csv")
    except:
        results = None

    # Load model (optional)
    try:
        with open("walmart_model.pkl", "rb") as f:
            model = pickle.load(f)
    except:
        model = None

    return df_full, results, model

df_full, results, model = load_data()

# ---------------------------
# SIDEBAR FILTER
# ---------------------------
st.sidebar.header("Filters")

store = st.sidebar.selectbox(
    "Select Store",
    sorted(df_full["Store"].unique())
)

filtered_df = df_full[df_full["Store"] == store]

# ---------------------------
# KPI SECTION
# ---------------------------
st.subheader("📌 Key Metrics")

col1, col2 = st.columns(2)

col1.metric("Total Sales", f"{filtered_df['Weekly_Sales'].sum():,.0f}")
col2.metric("Average Sales", f"{filtered_df['Weekly_Sales'].mean():,.0f}")

# ---------------------------
# SALES TREND
# ---------------------------
st.subheader("📈 Weekly Sales Trend")

fig = px.line(
    filtered_df,
    x="Date",
    y="Weekly_Sales",
    title=f"Store {store} Sales Over Time"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# ADDITIONAL INSIGHTS
# ---------------------------
st.subheader("🌡️ Temperature vs Sales")

fig2 = px.scatter(
    filtered_df,
    x="Temperature",
    y="Weekly_Sales",
    title="Impact of Temperature on Sales"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# PREDICTION VS ACTUAL
# ---------------------------
if results is not None and "Predicted" in results.columns:
    st.subheader("📊 Actual vs Predicted")

    fig3 = px.scatter(
        results,
        x="Weekly_Sales",
        y="Predicted",
        title="Actual vs Predicted Sales"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# MODEL INFO
# ---------------------------
st.subheader("🤖 Model Info")

if model is not None:
    st.success("Model loaded successfully ✅")
    st.write(type(model))
else:
    st.warning("Model not loaded")

# ---------------------------
# RAW DATA VIEW
# ---------------------------
st.subheader("📂 Raw Data")

st.dataframe(filtered_df.head())
