import streamlit as st
import pandas as pd

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
required_cols = ['unit_number', 'time_in_cycles', 'predicted_RUL']
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"❌ Missing columns in CSV: {missing}")
    st.stop()

# Engine selection
engine_ids = df['unit_number'].unique()
selected_engine = st.selectbox("Select engine unit:", engine_ids)

# Filter selected engine
filtered_df = df[df['unit_number'] == selected_engine].sort_values("time_in_cycles")

# --- INTERACTIVE PLOT (NOT STATIC) ---
st.subheader("📉 Predicted RUL for selected unit:")

# Build an interactive dataframe for plotting
plot_df = filtered_df[["time_in_cycles", "predicted_RUL"]].rename(
    columns={"time_in_cycles": "Cycle", "predicted_RUL": "Predicted RUL"}
)

# Add threshold helper columns so Streamlit will draw them as flat lines
plot_df["Warning (30)"] = 30
plot_df["Critical (20)"] = 20

st.line_chart(plot_df, x="Cycle", y=["Predicted RUL", "Warning (30)", "Critical (20)"])

# --- Maintenance alert ---
if filtered_df['predicted_RUL'].min() < 20:
    st.warning("⚠️ Maintenance recommended soon: Predicted RUL drops below 20 cycles.")
elif filtered_df['predicted_RUL'].min() < 30:
    st.warning("⚠️ RUL below warning threshold (30). Plan maintenance soon.")
