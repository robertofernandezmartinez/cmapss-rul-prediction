import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================================
# MODEL DNA (MANDATORY FOR CUSTOM PIPELINES)
# =====================================================================
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
        # Handle raw column names if they are numeric
        if isinstance(df.columns[0], int) or df.columns[0] == 0:
            n_cols = df.shape[1]
            columns = (['unit_number', 'time_in_cycles'] + 
                       [f'op_setting_{i}' for i in range(1, 4)] + 
                       [f'sensor_{i}' for i in range(1, n_cols - 4)])
            df.columns = columns[:n_cols]
            
        for col in self.selected_features:
            if col not in df.columns:
                df[col] = 0.0 
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
        return df[self.selected_features].values.astype(np.float64)

# =====================================================================
# UI CONFIGURATION
# =====================================================================
st.set_page_config(page_title="NASA RUL Prediction", page_icon="✈️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "05_Results", "predictions_validation_FD001.csv")
MODEL_PATH = os.path.join(BASE_DIR, "04_Models", "nasa_model.pkl")

st.title("✈️ Jet Engine Predictive Maintenance")
st.markdown("Predicting **Remaining Useful Life (RUL)** using NASA's CMAPSS dataset.")

# =====================================================================
# DATA LOADING
# =====================================================================
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at {DATA_PATH}. Please check folder names.")
    st.stop()

df = pd.read_csv(DATA_PATH)
engine_ids = df["unit_number"].unique()
selected_engine = st.selectbox("Select Engine Unit:", engine_ids)
filtered_df = df[df["unit_number"] == selected_engine]

# Thresholds
WARNING = 80
CRITICAL = 50

# =====================================================================
# DASHBOARD METRICS
# =====================================================================
warning_cycles = filtered_df[filtered_df["predicted_RUL"] < WARNING]["time_in_cycles"]
critical_cycles = filtered_df[filtered_df["predicted_RUL"] < CRITICAL]["time_in_cycles"]

w_start = int(warning_cycles.iloc[0]) if not warning_cycles.empty else "N/A"
c_start = int(critical_cycles.iloc[0]) if not critical_cycles.empty else "N/A"

col1, col2 = st.columns(2)
with col1:
    st.metric("Warning Threshold (Cycles)", f"Starts at: {w_start}")
with col2:
    st.metric("Critical Threshold (Cycles)", f"Starts at: {c_start}", delta_color="inverse")

# =====================================================================
# VISUALIZATION
# =====================================================================
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered_df["time_in_cycles"],
    y=filtered_df["predicted_RUL"],
    mode="lines",
    line=dict(color="#00D4FF", width=3),
    name="Predicted RUL"
))

fig.add_hline(y=WARNING, line_dash="dash", line_color="orange")
fig.add_hline(y=CRITICAL, line_dash="dash", line_color="red")

fig.update_layout(template="plotly_dark", height=500, xaxis_title="Cycles", yaxis_title="RUL")
st.plotly_chart(fig, use_container_width=True)