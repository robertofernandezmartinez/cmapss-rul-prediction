import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Remaining Useful Life (RUL) - CMAPSS", layout="centered")

st.title("🚀 Remaining Useful Life (RUL) Prediction - CMAPSS")
st.markdown("""
Interactive visualization of a predictive model trained on CMAPSS (NASA) engine data.

Select an engine unit to analyze its status and RUL estimation.
""")

# Load prediction data
DATA_PATH = "05_Results/predictions_validation_FD001.csv"
df = pd.read_csv(DATA_PATH)

# Confirm expected columns exist
if 'unit_number' not in df.columns or 'predicted_rul' not in df.columns:
    st.error("❌ 'unit_number' or 'predicted_rul' column not found in the CSV.")
    st.stop()

# Select engine unit
engine_ids = df['unit_number'].unique()
selected_engine = st.selectbox("Select engine unit:", engine_ids)

# Filter dataframe
filtered_df = df[df['unit_number'] == selected_engine].reset_index(drop=True)

# Add maintenance thresholds
rul_warning = 30
rul_critical = 20

# Create the Altair chart
line = alt.Chart(filtered_df.reset_index()).mark_line().encode(
    x=alt.X('index:Q', title='Cycle'),
    y=alt.Y('predicted_rul:Q', title='Predicted RUL')
).properties(
    width=700,
    height=400,
    title="📉 Predicted RUL for selected unit:"
)

# Add horizontal lines at 30 and 20
rule_warning = alt.Chart(pd.DataFrame({'y': [rul_warning]})).mark_rule(color='orange', strokeDash=[6,4]).encode(y='y')
rule_critical = alt.Chart(pd.DataFrame({'y': [rul_critical]})).mark_rule(color='red', strokeDash=[4,2]).encode(y='y')

# Combine charts
st.altair_chart(line + rule_warning + rule_critical, use_container_width=True)

# Optional alert text
min_rul = filtered_df['predicted_rul'].min()
if min_rul < rul_critical:
    st.error("🔴 CRITICAL: Predicted RUL dropped below 20 cycles. Immediate maintenance required.")
elif min_rul < rul_warning:
    st.warning("🟠 Warning: Predicted RUL below 30 cycles. Plan maintenance soon.")
else:
    st.success("✅ RUL above maintenance thresholds.")
