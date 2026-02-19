# 🧭 Remaining Useful Life (RUL) Prediction - CMAPSS Jet Engine Simulated Data

Source: [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

## 📘 Project Overview

- This machine learning project predicts the **Remaining Useful Life (RUL)** of aircraft engines using NASA’s CMAPSS dataset.
- The goal is to estimate **how many operating cycles each engine has left before failure**, based on its sensor readings.
- The workflow follows a complete supervised regression pipeline, from preprocessing to production-ready inference.

## 📊 Streamlit Web App – Remaining Useful Life (RUL) Prediction

This project includes an interactive Streamlit dashboard that visualizes the predicted Remaining Useful Life (RUL) for each engine unit in the NASA CMAPSS FD001 dataset. It is designed for engineers, analysts, or decision‑makers who need a quick way to assess engine health and upcoming maintenance needs.

### 🔗 Try the App Online: [https://cmapss-rul-prediction.streamlit.app/](https://cmapss-rul-prediction.streamlit.app/)

## 🛠️ Tech Stack & Environment
- **Python 3.13+**
- **Libraries**: `Pandas 3.0.1`, `NumPy 1.26.4` (stability-optimized), `Scikit-Learn 1.8.0`, `XGBoost 3.2.0`, `Plotly`, and `Streamlit`.
- **Serialization**: `cloudpickle` for cross-version pipeline compatibility.

## 🎯 This Project Focuses on FD001

The CMAPSS dataset contains four subsets (FD001–FD004) of increasing complexity.  
This project uses **FD001**, which includes a single operating condition and one fault mode. Focusing on FD001 provides a controlled environment to develop and validate the RUL prediction pipeline, allowing for:  
- A clearer understanding of degradation patterns.  
- Straightforward feature selection and model comparison.  
- A solid performance baseline before extending to more complex scenarios (FD002–FD004).  

## ⚙️ Workflow Summary  

1. **Data Preparation & Quality**
   - Cleaned raw CMAPSS data (train/test sets).  
   - Handled Pandas 2.0+ whitespace separators using `sep='\s+'`.
   - Renamed and standardized columns (`unit_number`, `time_in_cycles`, `sensors`).  

2. **Feature Engineering & Selection**
   - Applied normalization, Yeo–Johnson and Quantile transformations, and standardization.  
   - Compared three feature-selection methods: Mutual Information (MI), Recursive Feature Elimination (RFE), and **Permutation Importance (PI)**.  
   - Identified and eliminated highly correlated features to reduce redundancy.  

3. **Modeling**
   - Built regression pipelines using **Scikit-learn** and **XGBoost**.  
   - Trained and tuned models via `RandomizedSearchCV`.  
   - Evaluated with **MAE**, **RMSE**, and **R²** metrics.  
   - Saved the final trained pipeline using `cloudpickle` for production readiness.  

4. **Execution (Inference)**
   - Loaded the production pipeline to predict RUL on validation/new data.  
   - Generated Predicted RUL for each unit and cycle.  
   - Exported results to `/05_Results/predictions_validation_FD001.csv` for dashboarding.

## 🧩 Notebooks  

| # | Notebook | Description |
|---|-----------|-------------|
| 1 | `01_setup.ipynb` | Environment setup, library installation, and folder structure. |
| 2 | `02_data_quality.ipynb` | Data cleaning, renaming, and missing values handling. |
| 3 | `03_eda.ipynb` | EDA: distributions, correlations, and degradation insights. |
| 4 | `04_feature_engineering.ipynb` | Scaling, normalization, and Yeo–Johnson transformations. |
| 5 | `05_feature_selection.ipynb` | Feature selection using MI, RFE, and Permutation Importance. |
| 6 | `06_modeling.ipynb` | Regression model training, tuning, and evaluation. |
| 7 | `07_preparation_of_production_code.ipynb` | Final pipeline assembly and cloudpickle serialization. |
| 8 | `08_retraining_script.ipynb` | Automated retraining script for model updates. |
| 9 | `09_execution_script.ipynb` | Production inference for validation or new sensor data. |

## 📊 Key Metrics (FD001)

- **MAE: 30.8** - Average prediction error ≈ 30 cycles.
- **RMSE: 44.0** - Moderate error dispersion; the model generalizes well.
- **R²: 0.62** - Explains ~62% of RUL variance.

### 📊 Dashboard Features

- **Engine Selector**: Choose any engine unit from the validation set.
- **Interactive Plotly Chart**: Zoom and inspect exact cycle/RUL values.
- **Maintenance Thresholds**:
    - 🟧 **Warning**: Below 80 cycles.
    - 🟥 **Critical**: Below 50 cycles.

### 📁 Files Used by the Dashboard

- `app.py` — Streamlit application code using relative paths for portability.
- `/05_Results/predictions_validation_FD001.csv` — Predictions generated in notebook **09**.
- `requirements.txt` — Environment manifest (Streamlit, Plotly, Pandas, etc.).