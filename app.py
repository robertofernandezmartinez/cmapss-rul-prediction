import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Remaining Useful Life (RUL) - CMAPSS", layout="wide")

st.title("🚀 Remaining Useful Life (RUL) Prediction - CMAPSS")
st.markdown("""
Interactive visualization of a predictive model trained on CMAPSS (NASA) engine data.

Select an engine unit to analyze its status and predicted Remaining Useful Life (RUL).
""")

# Load prediction data
DATA_PATH = "05_Results/predictions_validation_FD001.csv"
df = pd.read_csv(DATA_PATH)

# Validate column names
expected_cols = ["unit_number", "time_in_cycles", "predicted_RUL"]
if not all(col in df.columns for col in expected_cols):
    st.error(f"❌ CSV file must contain columns: {expected_cols}")
    st.stop()

# Engine selector
engine_ids = df["unit_number"].unique()
selected_engine = st.selectbox("Select engine unit:", engine_ids)

# Filter data
filtered_df = df[df["unit_number"] == selected_engine]

# Thresholds
WARNING = 80
CRITICAL = 50

# Create graph
fig = go.Figure()

# Predicted RUL curve
fig.add_trace(go.Scatter(
    x=filtered_df["time_in_cycles"],
    y=filtered_df["predicted_RUL"],
    mode="lines",
    name="Predicted RUL",
    line=dict(color="lightblue", width=3)
))

# Yellow warning line
fig.add_hline(
    y=WARNING,
    line_dash="dash",
    line_color="yellow",
    annotation_text=f"Warning ({WARNING})",
    annotation_position="top right"
)

# Red critical line
fig.add_hline(
    y=CRITICAL,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Critical ({CRITICAL})",
    annotation_position="bottom right"
)

# Layout adjustments → taller plot, clearer axes
fig.update_layout(
    height=650,  # ← makes the chart taller
    yaxis=dict(title="RUL (Remaining Useful Life)", range=[0, max(filtered_df["predicted_RUL"]) + 20]),
    xaxis=dict(title="Cycle"),
    title=f"Predicted RUL for Engine {selected_engine}",
    template="plotly_dark",
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# Alerts
lowest_rul = filtered_df["predicted_RUL"].min()

if lowest_rul < CRITICAL:
    st.error("🛑 **CRITICAL: RUL below 50 cycles. Immediate maintenance required.**")
elif lowest_rul < WARNING:
    st.warning("⚠️ **Warning: RUL below 80 cycles. Prepare maintenance.**")
else:
    st.success("✅ Engine condition stable. No maintenance required yet.")
