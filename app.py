import streamlit as st
import pandas as pd

st.set_page_config(page_title="Remaining Useful Life (RUL) - CMAPSS", layout="centered")

st.title("🚀 Remaining Useful Life (RUL) Prediction - CMAPSS")
st.markdown("""
Interactive visualization of a predictive model trained on CMAPSS (NASA) engine data.

Select an engine unit to analyze its status and RUL estimation.
""")

# Load prediction data
DATA_PATH = "05_Results/predictions_validation_FD001.csv"
df = pd.read_csv(DATA_PATH)

# Confirm expected columns exist (fixed lowercase!)
if 'unit_number' not in df.columns or 'predicted_RUL' not in df.columns:
    st.error("❌ 'unit_number' or 'predicted_RUL' column not found in the CSV.")
    st.stop()

# Select engine unit
engine_ids = df['unit_number'].unique()
selected_engine = st.selectbox("Select engine unit:", engine_ids)

# Filter dataframe
filtered_df = df[df['unit_number'] == selected_engine]

# Show predictions
st.subheader("📉 Predicted RUL for selected unit:")
st.line_chart(filtered_df[['predicted_RUL']].reset_index(drop=True))
