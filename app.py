import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Remaining Useful Life (RUL) - CMAPSS", layout="wide")

# Title and intro
st.title("🚀 Remaining Useful Life (RUL) Prediction - CMAPSS")
st.markdown("""
Interactive visualization of a predictive model trained on NASA's CMAPSS engine dataset.

Select an engine unit to explore its predicted Remaining Useful Life (RUL) and identify when maintenance is recommended.
""")

# Load prediction data
DATA_PATH = "05_Results/predictions_validation_FD001.csv"
df = pd.read_csv(DATA_PATH)

# Check required columns
expected_cols = ["unit_number", "time_in_cycles", "predicted_RUL"]
if not all(col in df.columns for col in expected_cols):
    st.error(f"❌ CSV file must contain columns: {expected_cols}")
    st.stop()

# Select engine unit
engine_ids = df["unit_number"].unique()
selected_engine = st.selectbox("Select engine unit:", engine_ids)

# Filter data for selected unit
filtered_df = df[df["unit_number"] == selected_engine]

# Define thresholds
WARNING = 80
CRITICAL = 50

# ------------------------------
# 📌 Maintenance Threshold Info
# ------------------------------
warning_cycles = filtered_df[filtered_df["predicted_RUL"] < WARNING]["time_in_cycles"]
critical_cycles = filtered_df[filtered_df["predicted_RUL"] < CRITICAL]["time_in_cycles"]

warning_cycle = int(warning_cycles.iloc[0]) if not warning_cycles.empty else None
critical_cycle = int(critical_cycles.iloc[0]) if not critical_cycles.empty else None

st.subheader("📌 Maintenance Thresholds")

if warning_cycle:
    st.markdown(f"🟨 **Warning zone** starts at: **Cycle {warning_cycle}**")
else:
    st.markdown("🟨 **Warning zone** not reached for this engine.")

if critical_cycle:
    st.markdown(f"🟥 **Critical zone** starts at: **Cycle {critical_cycle}**")
else:
    st.markdown("🟥 **Critical zone** not reached for this engine.")

# ------------------------------
# 📈 Plot RUL Over Time
# ------------------------------
fig = go.Figure()

# Predicted RUL curve
fig.add_trace(go.Scatter(
    x=filtered_df["time_in_cycles"],
    y=filtered_df["predicted_RUL"],
    mode="lines",
    name="Predicted RUL",
    line=dict(color="lightblue", width=3)
))

# Add warning and critical lines
fig.add_hline(
    y=WARNING,
    line_dash="dash",
    line_color="yellow",
    annotation_text=f"Warning ({WARNING})",
    annotation_position="top right"
)

fig.add_hline(
    y=CRITICAL,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Critical ({CRITICAL})",
    annotation_position="bottom right"
)

# Format layout
fig.update_layout(
    height=650,
    yaxis=dict(title="RUL (Remaining Useful Life)", range=[0, max(filtered_df["predicted_RUL"]) + 20]),
    xaxis=dict(title="Cycle"),
    title=f"Predicted RUL for Engine {selected_engine}",
    template="plotly_dark",
    showlegend=True
)

# Show plot
st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# 🚨 Status Alert
# ------------------------------
lowest_rul = filtered_df["predicted_RUL"].min()

if lowest_rul < CRITICAL:
    st.error("🛑 **CRITICAL: RUL below 50 cycles. Immediate maintenance required.**")
elif lowest_rul < WARNING:
    st.warning("⚠️ **Warning: RUL below 80 cycles. Prepare maintenance.**")
else:
    st.success("✅ Engine condition stable. No maintenance required yet.")
