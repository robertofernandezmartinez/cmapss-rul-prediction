import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import cloudpickle
import os

# Page config
st.set_page_config(page_title="RUL Prediction - CMAPSS", layout="wide")

# --- PATH HANDLER ---
# Get the absolute path of the current directory to avoid "File Not Found" errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "05_Results", "predictions_validation_FD001.csv")
MODEL_PATH = os.path.join(BASE_DIR, "04_Models", "pipe_execution.pickle")

# Title and intro
st.title("🚀 Remaining Useful Life (RUL) Prediction - CMAPSS")
st.markdown("""
Interactive visualization of a predictive model trained on NASA's CMAPSS engine dataset.
""")

# --- LOAD DATA ---
if not os.path.exists(DATA_PATH):
    st.error(f"❌ Data file not found at: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Select engine unit
engine_ids = df["unit_number"].unique()
selected_engine = st.selectbox("Select engine unit:", engine_ids)

# Filter data for selected unit
filtered_df = df[df["unit_number"] == selected_engine]

# Define thresholds
WARNING = 80
CRITICAL = 50

# --- MAINTENANCE THRESHOLDS ---
warning_cycles = filtered_df[filtered_df["predicted_RUL"] < WARNING]["time_in_cycles"]
critical_cycles = filtered_df[filtered_df["predicted_RUL"] < CRITICAL]["time_in_cycles"]

warning_cycle = int(warning_cycles.iloc[0]) if not warning_cycles.empty else None
critical_cycle = int(critical_cycles.iloc[0]) if not critical_cycles.empty else None

st.subheader("📌 Maintenance Thresholds")
col1, col2 = st.columns(2)
with col1:
    if warning_cycle:
        st.warning(f"Warning zone starts at: **Cycle {warning_cycle}**")
    else:
        st.info("Warning zone not reached.")
with col2:
    if critical_cycle:
        st.error(f"Critical zone starts at: **Cycle {critical_cycle}**")
    else:
        st.info("Critical zone not reached.")

# --- VISUALIZATION ---
fig = go.Figure()

# Predicted RUL curve
fig.add_trace(go.Scatter(
    x=filtered_df["time_in_cycles"],
    y=filtered_df["predicted_RUL"],
    mode="lines",
    name="Predicted RUL",
    line=dict(color="#00D4FF", width=3),
    hovertemplate='Cycle: %{x}<br>RUL: %{y:.2f}<extra></extra>'
))

# Threshold lines
fig.add_hline(y=WARNING, line_dash="dash", line_color="orange", annotation_text="Warning")
fig.add_hline(y=CRITICAL, line_dash="dash", line_color="red", annotation_text="Critical")

fig.update_layout(
    height=600,
    title=f"RUL Decay for Engine {selected_engine}",
    xaxis_title="Cycle",
    yaxis_title="Remaining Useful Life",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# --- REAL-TIME MODEL STATUS ---
if st.checkbox("Show Model Info"):
    if os.path.exists(MODEL_PATH):
        st.success(f"✅ Active Model Found: {os.path.basename(MODEL_PATH)}")
    else:
        st.warning("⚠️ Model pickle not found. Using static CSV data only.")