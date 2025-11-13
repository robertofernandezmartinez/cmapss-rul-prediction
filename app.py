import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="CMAPSS RUL Dashboard", layout="wide")
st.title("🚀 Predicción de Remaining Useful Life (RUL) - CMAPSS")
st.markdown("""
Visualización interactiva del modelo predictivo entrenado con los datos CMAPSS (NASA).  
Selecciona una unidad (engine) y analiza su evolución y estado actual.
""")

# --- FUNCIONES DE CARGA ---
@st.cache_resource
def load_model():
    with open("04_Models/pipe_execution.pickle", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("02_Data/02_Validation/validation_FD001.csv")
    return df

# Cargar modelo y datos
model = load_model()
data = load_data()

# --- INTERFAZ ---
unit_ids = data['unit_number'].unique()
selected_unit = st.selectbox("Selecciona una unidad:", unit_ids)

df_unit = data[data['unit_number'] == selected_unit].sort_values('time_in_cycles')

# Predicciones
X = df_unit.drop(columns=['unit_number', 'time_in_cycles'])
rul_pred = model.predict(X)
df_unit['RUL_pred'] = rul_pred

# --- GRÁFICO RUL ---
col1, col2 = st.columns([2, 1])

with col1:
    fig = px.line(df_unit, x='time_in_cycles', y='RUL_pred',
                  title=f"Evolución del RUL - Unidad {selected_unit}",
                  labels={'time_in_cycles': 'Ciclos', 'RUL_pred': 'RUL Predicho'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    current_rul = df_unit['RUL_pred'].iloc[-1]
    st.metric(label="RUL actual (ciclos)", value=int(current_rul))
    threshold = st.slider("Umbral de mantenimiento (ciclos):", 10, 50, 20)
    if current_rul < threshold:
        st.error("⚠️ ¡Mantenimiento urgente recomendado!")
    else:
        st.success("✅ Estado del motor: dentro de parámetros normales")

# --- SHAP opcional ---
st.divider()
st.subheader("🧩 Interpretabilidad del modelo (SHAP)")

if st.checkbox("Mostrar SHAP values (último ciclo)"):
    X_sample = X.iloc[[-1]]
    try:
        explainer = shap.Explainer(model['model'])
        shap_values = explainer(X_sample)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"No fue posible calcular SHAP: {e}")
