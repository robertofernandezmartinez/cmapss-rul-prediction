import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# Title
st.title("🚀 Remaining Useful Life (RUL) Prediction - CMAPSS")

st.write("Interactive visualization of a predictive model trained on CMAPSS (NASA) engine data.")
st.write("Select an engine unit to analyze its status and RUL estimation.")

@st.cache_resource
def load_model():
    with open("04_Models/pipe_execution.pickle", "rb") as f:
        model = cloudpickle.load(f)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("05_Results/predictions_validation_FD001.csv")
    st.write("📋 DataFrame columns:")
    st.write(df.columns)
    return df

model = load_model()
df = load_data()

# Engine selector
engine_ids = df['unit_number'].unique()
selected_engine = st.selectbox("Select Engine Unit:", engine_ids)

# Filter data
engine_data = df[df['unit_number'] == selected_engine]

# Plot RUL over time
fig = px.line(engine_data, x='time_cycles', y='RUL', title=f"RUL over Time - Engine {selected_engine}")
st.plotly_chart(fig)

# SHAP explanation (optional)
st.subheader("Model Explanation with SHAP")
if st.checkbox("Show SHAP Explanation"):
    explainer = shap.Explainer(model.named_steps['regressor'])  # Assuming pipeline
    shap_values = explainer(engine_data.drop(columns=['RUL', 'unit_number']))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, engine_data.drop(columns=['RUL', 'unit_number']))
    st.pyplot()
