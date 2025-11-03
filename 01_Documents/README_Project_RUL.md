🧭 README — RUL Prediction (CMAPSS)

Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

📘 Project Overview

- This project predicts the **Remaining Useful Life (RUL)** of aircraft engines using NASA’s CMAPSS dataset.
- The goal is to estimate **how many operating cycles each engine has left before failure**, based on its sensor readings.
- The workflow follows a complete supervised regression pipeline — from preprocessing to production-ready inference.


### 🎯 Why This Project Focuses on FD001

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

- MAE: 30.8 - Average prediction error ≈ 30 cycles (acceptable for FD001).
- RMSE: 44.0 - Moderate error dispersion: model generalizes well.
- R²	0.62	Explains ~62 % of RUL variance — strong predictive signal.



### Readme by NASA:

Data Set: FD001
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)

Data Set: FD002
Train trjectories: 260
Test trajectories: 259
Conditions: SIX 
Fault Modes: ONE (HPC Degradation)

Data Set: FD003
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: TWO (HPC Degradation, Fan Degradation)

Data Set: FD004
Train trjectories: 248
Test trajectories: 249
Conditions: SIX 
Fault Modes: TWO (HPC Degradation, Fan Degradation)


**Experimental Scenario**

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine ñ i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:
1)	unit number
2)	time, in cycles
3)	operational setting 1
4)	operational setting 2
5)	operational setting 3
6)	sensor measurement  1
7)	sensor measurement  2
...
26)	sensor measurement  26


Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, ìDamage Propagation Modeling for Aircraft Engine Run-to-Failure Simulationî, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
