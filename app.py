import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# Page config
st.set_page_config(page_title="Walmart Sales Forecast", layout="wide")

st.title("📊 Walmart Weekly Sales Forecast Dashboard")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    # Load dataset (FIXED: using .pkl instead of .csv)
    df_full = pd.read_pickle("walmart_features.pkl")
    
    # Load test results
    results = pd.read_csv("test_results.csv")
    
    # Load trained model
    with open("walmart_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    return df_full, results, model

df_full, results, model = load_data()

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------
st.sidebar.header("Filters")

store = st.sidebar.selectbox("Select Store", sorted(df_full["Store"].unique()))
dept = st.sidebar.selectbox("Select Dept", sorted(df_full["Dept"].unique()))

filtered_df = df_full[(df_full["Store"] == store) & (df_full["Dept"] == dept)]

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
st.subheader("📈 Sales Trend")

fig = px.line(
    filtered_df,
    x="Date",
    y="Weekly_Sales",
    title="Weekly Sales Over Time"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# PREDICTION VS ACTUAL
# ---------------------------
st.subheader("📊 Actual vs Predicted")

if "Predicted" in results.columns:
    fig2 = px.scatter(
        results,
        x="Weekly_Sales",
        y="Predicted",
        title="Actual vs Predicted Sales"
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("⚠️ 'Predicted' column not found in test_results.csv")

# ---------------------------
# MODEL INFO
# ---------------------------
st.subheader("🤖 Model Info")

st.write("Model loaded successfully ✅")
st.write(type(model))
