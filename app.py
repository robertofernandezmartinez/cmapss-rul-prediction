import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# =====================================================================
# 1. PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="NASA Jet Engine RUL Predictor",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS for a professional look (similar to Stockout UI)
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetric"] {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3e4259;
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# =====================================================================
# 2. PATHS & MODEL LOADING (Native Strategy)
# =====================================================================
# We use relative paths so it works both on your Mac and on the Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "04_Models", "nasa_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "05_Results", "predictions_validation_FD001.csv")

@st.cache_resource
def load_prediction_assets():
    """
    Loads the native Sklearn Pipeline and the pre-calculated results.
    No custom classes needed here anymore.
    """
    if not os.path.exists(MODEL_PATH):
        return None, None
    
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
    return model, data

pipeline, df_results = load_prediction_assets()

if pipeline is None or df_results is None:
    st.error(f"Critical Error: Assets not found. Check: {MODEL_PATH}")
    st.stop()

# =====================================================================
# 3. SIDEBAR - ENGINE SELECTION
# =====================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/724/724082.png", width=80)
st.sidebar.title("Fleet Management")
st.sidebar.markdown("Select an engine unit to analyze its Remaining Useful Life (RUL).")

unit_ids = df_results['unit_number'].unique()
selected_unit = st.sidebar.selectbox("Select Engine Unit:", unit_ids)

# Filter data for the selected engine
engine_data = df_results[df_results['unit_number'] == selected_unit].sort_values('time_in_cycles')
current_cycle = int(engine_data['time_in_cycles'].max())
predicted_rul = float(engine_data['predicted_RUL'].iloc[-1])

# =====================================================================
# 4. MAIN DASHBOARD - KEY METRICS
# =====================================================================
st.title("✈️ Jet Engine Maintenance AI")
st.markdown(f"**Unit ID:** {selected_unit} | **System Status:** Active Monitoring")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Current Flight Cycles", value=current_cycle)

with col2:
    # Color logic for RUL
    if predicted_rul < 50:
        status = "🚨 CRITICAL"
    elif predicted_rul < 80:
        status = "⚠️ WARNING"
    else:
        status = "✅ HEALTHY"
    st.metric(label="Predicted RUL (Remaining Cycles)", value=f"{predicted_rul:.0f}", delta=status)

with col3:
    # Estimating percentage of life remaining (assuming 200 cycles avg life)
    life_perc = (predicted_rul / 200) * 100
    st.metric(label="Estimated Life %", value=f"{life_perc:.1f}%")

# =====================================================================
# 5. VISUAL ANALYSIS - RUL DEGRADATION
# =====================================================================
st.subheader("RUL Degradation Over Time")

fig = go.Figure()

# Add RUL line
fig.add_trace(go.Scatter(
    x=engine_data['time_in_cycles'],
    y=engine_data['predicted_RUL'],
    mode='lines',
    name='Predicted RUL',
    line=dict(color='#00D4FF', width=3)
))

# Threshold lines
fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critical Limit")
fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Warning Zone")

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Time in Cycles",
    yaxis_title="Remaining Useful Life",
    margin=dict(l=20, r=20, t=40, b=20),
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# 6. STRATEGIC RECOMMENDATION
# =====================================================================
st.markdown("---")
if predicted_rul < 50:
    st.error(f"**URGENT**: Unit {selected_unit} has reached the critical limit. Schedule immediate engine overhaul.")
elif predicted_rul < 80:
    st.warning(f"**MAINTENANCE ALERT**: Unit {selected_unit} shows signs of significant wear. Plan inspection within the next 20 cycles.")
else:
    st.success(f"**OPTIMAL**: Unit {selected_unit} is performing within safety parameters. No immediate action required.")

st.caption("Predictive Maintenance Hub v1.0 | CMAPSS-RUL Engine")