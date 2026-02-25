import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# =====================================================================
# 1. PAGE CONFIGURATION (Optimized)
# =====================================================================
st.set_page_config(
    page_title="NASA Jet Engine RUL",
    page_icon="✈️",
    layout="wide"
)

# Using native metrics container (no heavy custom CSS for faster load)
st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# =====================================================================
# 2. ASSET LOADING (With Spinner for UX)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "04_Models", "nasa_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "05_Results", "predictions_validation_FD001.csv")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        return None, None
    m = joblib.load(MODEL_PATH)
    d = pd.read_csv(DATA_PATH)
    return m, d

# Visual feedback during the first load
with st.spinner('Synchronizing with Fleet Data...'):
    pipeline, df_results = load_assets()

if pipeline is None:
    st.error("Assets missing. Please ensure 04_Models/nasa_model.pkl exists.")
    st.stop()

# =====================================================================
# 3. SIDEBAR & LOGIC
# =====================================================================
unit_ids = df_results['unit_number'].unique()
selected_unit = st.sidebar.selectbox("Engine Unit ID:", unit_ids)

engine_data = df_results[df_results['unit_number'] == selected_unit].sort_values('time_in_cycles')
current_cycle = int(engine_data['time_in_cycles'].max())
predicted_rul = float(engine_data['predicted_RUL'].iloc[-1])

# =====================================================================
# 4. DASHBOARD DISPLAY
# =====================================================================
st.title("✈️ Predictive Maintenance Dashboard")
st.info(f"Analysis for Engine #{selected_unit} | Data synced with latest cycle")

m1, m2, m3 = st.columns(3)
m1.metric("Total Cycles", current_cycle)
m2.metric("Remaining Useful Life", f"{predicted_rul:.0f} cycles")
m3.metric("Health Index", "CRITICAL" if predicted_rul < 50 else "STABLE")

# =====================================================================
# 5. CHART (Dynamic Theme Support)
# =====================================================================
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=engine_data['time_in_cycles'], 
    y=engine_data['predicted_RUL'],
    name="RUL Trend",
    line=dict(color='#00d4ff', width=3)
))

# Thresholds
fig.add_hline(y=50, line_color="red", line_dash="dash")

# THE SECRET FOR COLOR ADAPTATION:
# We don't force 'plotly_dark'. We let it be transparent to adapt to the UI
fig.update_layout(
    margin=dict(l=10, r=10, t=30, b=10),
    height=400,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='gray')
)

st.plotly_chart(fig, use_container_width=True)