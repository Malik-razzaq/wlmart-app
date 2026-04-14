import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.title("Walmart Sales App")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    # Load files safely
    results = pd.read_csv("test_results.csv")
    df_full = pd.read_csv("Walmart.csv", parse_dates=["Date"])

    # FIX: clean column names
    df_full.columns = df_full.columns.str.strip().str.title()

    # Load model
    with open("walmart_model.pkl", "rb") as f:
        model = pickle.load(f)

    return results, df_full, model

results, df_full, model = load_data()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("Filters")

# FIX: only Store (no Dept)
store = st.sidebar.selectbox(
    "Select Store",
    sorted(df_full["Store"].unique())
)

filtered_df = df_full[df_full["Store"] == store]

# -------------------------
# MAIN OUTPUT (same style)
# -------------------------
st.write("Filtered Data", filtered_df.head())

# Plot
fig = px.line(filtered_df, x="Date", y="Weekly_Sales")
st.plotly_chart(fig)

# -------------------------
# OPTIONAL MODEL OUTPUT
# -------------------------
if "Predicted" in results.columns:
    st.write("Predictions Preview")
    st.write(results.head())
