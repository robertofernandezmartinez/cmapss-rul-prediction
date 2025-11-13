import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Remaining Useful Life (RUL) - CMAPSS", layout="centered")

st.title("🚀 Remaining Useful Life (RUL) Prediction - CMAPSS")
st.markdown("""
Interactive visualization of a predictive model trained on CMAPSS (NASA) engine data.

Select an engine unit to analyze its status and RUL estimation.
""")

# Load data
DATA_PATH = "05_Results/predictions_validation_FD001.csv"
df = pd.read_csv(DATA_PATH)

# Validate expected columns
if 'unit_number' not in df.columns or 'predicted_RUL' not in df.columns:
    st.error("❌ 'unit_number' or 'predicted_RUL' column not found in the CSV.")
    st.stop()

# Engine selection
engine_ids = df['unit_number'].unique()
selected_engine = st.selectbox("Select engine unit:", engine_ids)

# Filter selected engine
filtered_df = df[df['unit_number'] == selected_engine]

# Plot with thresholds
st.subheader("📉 Predicted RUL for selected unit:")
fig, ax = plt.subplots()
ax.plot(filtered_df['predicted_RUL'].values, label='Predicted RUL')
ax.axhline(y=30, color='yellow', linestyle='--', label='Warning Threshold (30)')
ax.axhline(y=20, color='red', linestyle='--', label='Critical Threshold (20)')
ax.set_ylabel("RUL")
ax.set_xlabel("Cycle")
ax.legend()
st.pyplot(fig)

# Maintenance alert
if filtered_df['predicted_RUL'].min() < 20:
    st.warning("⚠️ Maintenance recommended soon: Predicted RUL drops below 20 cycles.")
