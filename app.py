import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib  # Switched from cloudpickle to joblib
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================================
# 1. MODEL DNA (REQUIRED TO LOAD THE .PKL)
# =====================================================================
# We must define the custom class here so Streamlit knows how to unpack the model
class NASAFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.selected_features = [
            'time_in_cycles', 'sensor_11', 'sensor_4', 'sensor_12',
            'sensor_7', 'sensor_15', 'sensor_21', 'sensor_20'
        ]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        if isinstance(df.columns[0], int) or df.columns[0] == 0:
            n_cols = df.shape[1]
            columns = (['unit_number', 'time_in_cycles'] + 
                       [f'op_setting_{i}' for i in range(1, 4)] + 
                       [f'sensor_{i}' for i in range(1, n_cols - 4)])
            df.columns = columns[:n_cols]
        for col in self.selected_features:
            if col not in df.columns: df[col] = 0.0 
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df[self.selected_features].values.astype(np.float64)

# =====================================================================
# 2. CONFIG & PATHS
# =====================================================================
st.set_page_config(page_title="RUL Prediction - CMAPSS", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "05_Results", "predictions_validation_FD001.csv")
# UPDATED: Path matches the new naming convention from Notebook 07
MODEL_PATH = os.path.join(BASE_DIR, "04_Models", "nasa_model.pkl")

st.title("✈️ Remaining Useful Life (RUL) Prediction - CMAPSS")
st.markdown("Interactive visualization of predictive maintenance for jet engines using NASA's dataset.")

# =====================================================================
# 3. DATA LOADING
# =====================================================================
if not os.path.exists(DATA_PATH):
    st.error(f"❌ Prediction results not found at: {DATA_PATH}. Please run the Execution Script first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Engine Selector
engine_ids = df["unit_number"].unique()
selected_engine = st.selectbox("Select engine unit:", engine_ids)
filtered_df = df[df["unit_number"] == selected_engine]

# Threshold Settings
WARNING = 80
CRITICAL = 50

# =====================================================================
# 4. MAINTENANCE ALERTS
# =====================================================================
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

# =====================================================================
# 5. VISUALIZATION
# =====================================================================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=filtered_df["time_in_cycles"],
    y=filtered_df["predicted_RUL"],
    mode="lines",
    name="Predicted RUL",
    line=dict(color="#00D4FF", width=3),
    hovertemplate='Cycle: %{x}<br>RUL: %{y:.2f}<extra></extra>'
))

fig.add_hline(y=WARNING, line_dash="dash", line_color="orange", annotation_text="Warning")
fig.add_hline(y=CRITICAL, line_dash="dash", line_color="red", annotation_text="Critical")

fig.update_layout(
    height=600,
    title=f"RUL Decay Analysis for Engine Unit {selected_engine}",
    xaxis_title="Operational Cycle",
    yaxis_title="Remaining Useful Life (Cycles)",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# 6. MODEL STATUS CHECK
# =====================================================================
if st.checkbox("Show System Info"):
    if os.path.exists(MODEL_PATH):
        try:
            # We use joblib now
            joblib.load(MODEL_PATH)
            st.success(f"✅ Autonomous Pipeline Active: {os.path.basename(MODEL_PATH)}")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
    else:
        st.warning("⚠️ Autonomous model (.pkl) not found. Displaying cached results only.")