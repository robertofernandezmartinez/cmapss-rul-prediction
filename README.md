# ✈️ Remaining Useful Life (RUL) Prediction - CMAPSS Jet Engine Simulated Data

Source: [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

## 📘 Project Overview

- This machine learning project predicts the **Remaining Useful Life (RUL)** of aircraft engines using NASA’s CMAPSS dataset.
- The goal is to estimate **how many operating cycles each engine has left before failure**, based on its sensor readings.
- The workflow follows a complete supervised regression pipeline, utilizing a **Native Scikit-Learn Pipeline** for seamless production deployment.

## 📊 Streamlit Web App – Remaining Useful Life (RUL) Prediction

This project includes an interactive Streamlit dashboard that visualizes the predicted Remaining Useful Life (RUL) for each engine unit in the NASA CMAPSS FD001 dataset.

### 🔗 Try the App Online: [https://cmapss-rul-prediction.streamlit.app/](https://cmapss-rul-prediction.streamlit.app/)

## 🛠️ Tech Stack & Environment
- **Python 3.13+**
- **Libraries**: `Pandas 2.2.3` (Cloud Optimized), `NumPy 1.26.4`, `Scikit-Learn 1.6.1`, `XGBoost 2.1.4`, `Plotly`, and `Streamlit`.
- **Serialization**: `joblib` for native pipeline persistence and autonomous deployment.

## 🎯 This Project Focuses on FD001

The CMAPSS dataset contains four subsets (FD001–FD004) of increasing complexity.  
This project uses **FD001**, providing a controlled environment to develop and validate the RUL prediction pipeline, allowing for:  
- A clearer understanding of degradation patterns.  
- Straightforward feature selection and model comparison.  
- A solid performance baseline before extending to more complex scenarios.  

## ⚙️ Workflow Summary  

1. **Data Preparation & Quality**
   - Cleaned raw CMAPSS data using `sep=r'\s+'` for modern Pandas compatibility.
   - Handled column renaming and unit-specific life-cycle labeling.

2. **Feature Engineering & Selection**
   - Applied normalization and scaling using native `ColumnTransformer`.
   - Selected the **Top 8 Sensors** using Permutation Importance (PI) and correlation analysis.

3. **Production-Ready Modeling (Stockout Strategy)**
   - Abandoned custom classes in favor of a **Native Scikit-Learn Pipeline**.
   - Combined preprocessing (`StandardScaler`) and **XGBoost** into a single `.pkl` artifact.
   - Optimized for **Streamlit Cloud** by maintaining compatibility with Pandas < 3.0.

4. **Retraining & Execution**
   - **Retraining**: Automated script to update the model using native library components.
   - **Execution**: Production inference script that processes sensor data through the serialized pipeline without external dependencies.

## 🧩 Notebooks  

| # | Notebook | Description |
|---|-----------|-------------|
| 1 | `01_setup.ipynb` | Environment setup, library installation, and folder structure. |
| 2 | `02_data_quality.ipynb` | Data cleaning and initial integrity checks. |
| 3 | `03_eda.ipynb` | EDA: distributions, correlations, and degradation insights. |
| 4 | `04_feature_engineering.ipynb` | Scaling, normalization, and sensor transformations. |
| 5 | `05_feature_preselection.ipynb` | Feature selection using MI, RFE, and Permutation Importance. |
| 6 | `06_modelling_regression.ipynb` | Regression model training, tuning, and evaluation. |
| 7 | `07_retraining_execution_scripts.ipynb` | **Production Core**: Native pipeline assembly and retraining/execution logic. |

## 📊 Key Metrics (FD001)

- **MAE: 30.8** - Average prediction error ≈ 30 cycles.
- **RMSE: 44.0** - Moderate error dispersion; the model generalizes well.
- **R²: 0.62** - Explains ~62% of RUL variance.

### 📊 Dashboard Features

- **Engine Selector**: Choose any engine unit from the validation set.
- **Interactive Plotly Chart**: Dynamic theme adaptation for dark/light modes.
- **Maintenance Thresholds**:
    - 🟧 **Warning**: Below 80 cycles.
    - 🟥 **Critical**: Below 50 cycles.

### 📁 Files Used by the Dashboard

- `app.py` — Streamlit application code.
- `/04_Models/nasa_model.pkl` — The native production pipeline.
- `/05_Results/predictions_validation_FD001.csv` — Results for dashboard visualization.
- `requirements.txt` — Environment manifest.