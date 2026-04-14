import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Walmart Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Walmart Sales Dashboard")

# -------------------------
# SAFE LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    try:
        df_full  = pd.read_csv("walmart_features.csv", parse_dates=["Date"])
        results  = pd.read_csv("test_results.csv", parse_dates=["Date"])
        model    = joblib.load("walmart_model.pkl")

        try:
            features = joblib.load("walmart_features.pkl")
        except:
            features = None

        return df_full, results, model, features

    except Exception as e:
        st.error("❌ File loading error")
        st.write(e)
        st.stop()

df_full, results, model, features = load_data()

# -------------------------
# CLEAN DATA
# -------------------------
df_full.columns = df_full.columns.str.strip()
results.columns = results.columns.str.strip()

# Fix missing columns
required_cols = ["Store", "Weekly_Sales"]
for col in required_cols:
    if col not in df_full.columns:
        st.error(f"❌ Missing column: {col}")
        st.stop()

# Add Error %
if "Error_Pct" not in results.columns and "Actual" in results.columns:
    results["Error_Pct"] = abs(results["Actual"] - results["Predicted"]) / results["Actual"] * 100

# Add month
if "month" not in results.columns and "Date" in results.columns:
    results["month"] = pd.to_datetime(results["Date"]).dt.month

# Bias
if "Bias" in results.columns:
    results["Bias_Dir"] = results["Bias"].apply(lambda x: "Over" if x > 0 else "Under")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", [
    "Dashboard",
    "Error Analysis",
    "Heatmap",
    "Store Deep Dive",
    "Predictor"
])

# -------------------------
# DASHBOARD
# -------------------------
if page == "Dashboard":

    store = st.selectbox("Select Store", sorted(df_full["Store"].unique()))
    filtered_df = df_full[df_full["Store"] == store]

    col1, col2 = st.columns(2)
    col1.metric("Total Sales", f"${filtered_df['Weekly_Sales'].sum():,.0f}")
    col2.metric("Average Sales", f"${filtered_df['Weekly_Sales'].mean():,.0f}")

    st.subheader("Sales Trend")
    fig = px.line(filtered_df, x="Date", y="Weekly_Sales")
    st.plotly_chart(fig, use_container_width=True)

    if "Temperature" in df_full.columns:
        st.subheader("Temperature vs Sales")
        fig2 = px.scatter(filtered_df, x="Temperature", y="Weekly_Sales")
        st.plotly_chart(fig2, use_container_width=True)

    if "Actual" in results.columns:
        st.subheader("Actual vs Predicted")
        fig3 = px.scatter(results, x="Actual", y="Predicted")
        st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# ERROR ANALYSIS
# -------------------------
elif page == "Error Analysis":

    st.subheader("Error by Store")

    store_err = results.groupby("Store")["Error_Pct"].mean()
    fig = px.bar(x=store_err.index, y=store_err.values)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Error by Month")
    month_err = results.groupby("month")["Error_Pct"].mean()
    fig2 = px.bar(x=month_err.index, y=month_err.values)
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# HEATMAP
# -------------------------
elif page == "Heatmap":

    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = results.pivot_table(
        index="Store",
        columns="month",
        values="Error_Pct",
        aggfunc="mean"
    )

    fig, ax = plt.subplots()
    sns.heatmap(pivot, annot=True, cmap="RdYlGn_r")
    st.pyplot(fig)

# -------------------------
# STORE DEEP DIVE
# -------------------------
elif page == "Store Deep Dive":

    store = st.selectbox("Select Store", sorted(results["Store"].unique()))
    sd = results[results["Store"] == store]

    st.metric("Store Error", f"{sd['Error_Pct'].mean():.2f}%")

    fig = px.line(sd, x="Date", y="Actual")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# PREDICTOR
# -------------------------
elif page == "Predictor":

    st.subheader("Make Prediction")

    store = st.selectbox("Store", sorted(df_full["Store"].unique()))
    temp = st.slider("Temperature", 10.0, 110.0, 60.0)

    if st.button("Predict"):

        try:
            input_df = pd.DataFrame([{
                "Store": store,
                "Temperature": temp
            }])

            # If features exist → align
            if features:
                for f in features:
                    if f not in input_df.columns:
                        input_df[f] = 0
                input_df = input_df[features]

            pred = model.predict(input_df)[0]

            st.success(f"Predicted Sales: ${pred:,.0f}")

        except Exception as e:
            st.error("Prediction failed")
            st.write(e)
