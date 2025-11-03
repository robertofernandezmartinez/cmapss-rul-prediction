# 🧭 Remaining Useful Life (RUL) Prediction - CMAPSS Jet Engine Simulated Data

Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

### 📘 Project Overview

- This machine learning project predicts the **Remaining Useful Life (RUL)** of aircraft engines using NASA’s CMAPSS dataset.
- The goal is to estimate **how many operating cycles each engine has left before failure**, based on its sensor readings.
- The workflow follows a complete supervised regression pipeline, from preprocessing to production-ready inference.


### 🎯 This Project Focuses on FD001

The CMAPSS dataset contains four subsets (FD001–FD004) of increasing complexity.  
This project uses **FD001**, which includes a single operating condition and one fault mode.  

Focusing on FD001 provides a controlled environment to develop and validate the RUL prediction pipeline.  
It minimizes variability from different flight regimes or multiple fault types, allowing for:  
- a clearer understanding of degradation patterns,  
- straightforward feature selection and model comparison, and  
- a solid performance baseline before extending to more complex scenarios (FD002–FD004).  


### ⚙️ Workflow Summary  

1. **Data Preparation & Quality**  
   - Cleaned raw CMAPSS data (train/test sets).  
   - Removed duplicates and missing data.  
   - Renamed and standardized columns (`unit_number`, `time_in_cycles`, `sensors`).  

2. **Feature Engineering & Selection**  
   - Applied normalization, Yeo–Johnson and Quantile transformations, and standardization.  
   - Compared three feature-selection methods:  
     - Mutual Information (MI)  
     - Recursive Feature Elimination (RFE)  
     - Permutation Importance (PI)  
   - Chose **Mutual Information** as final method.  
   - Checked correlations and eliminated redundancy.  

3. **Modeling**  
   - Built regression pipelines using scikit-learn and XGBoost.  
   - Trained and tuned models via `RandomizedSearchCV`.  
   - Evaluated with **MAE**, **RMSE**, and **R²** metrics.  
   - Selected the best-performing model and saved it as a production pipeline.
   - Created a retraining code to fit the model again in case conditions change anytime in the future.  

4. **Execution (Inference)**  
   - Loaded the trained pipeline with `cloudpickle`.  
   - Applied it to validation/new data.  
   - Generated **Predicted RUL** for each unit and cycle.  
   - Exported results to `/05_Results/predictions_validation_FD001.csv` just to check.


### 🧩 Notebooks  

| # | Notebook | Description |
|---|-----------|-------------|
| 1 | `01_setup.ipynb` | Environment setup, library installation, and folder structure creation. |
| 2 | `02_data_quality.ipynb` | Data cleaning, renaming, duplicates, and missing values handling. |
| 3 | `03_eda.ipynb` | Exploratory Data Analysis (EDA): variable distributions, correlations, and initial insights. |
| 4 | `04_feature_engineering.ipynb` | Feature transformations: scaling, normalization, Yeo–Johnson, and Quantile transformations. |
| 5 | `05_feature_selection.ipynb` | Feature selection and ranking using MI, RFE, and PI methods. |
| 6 | `06_modeling.ipynb` | Regression model training, tuning, and evaluation using cross-validation. |
| 7 | `07_retraining_code.ipynb` | Final retraining using the optimal model and best hyperparameters. |
| 8 | `08_execution:code.ipynb` | Production inference: predicting Remaining Useful Life (RUL) on validation or new data. |


### 📊 Key Metrics (FD001)

- MAE: 30.8 - Average prediction error ≈ 30 cycles - Not perfect but acceptable for FD001.
- RMSE: 44.0 - Moderate error dispersion - Model generalizes well.
- R²:	0.62	- Explains ~62 % of RUL variance - Strong predictive signal.
