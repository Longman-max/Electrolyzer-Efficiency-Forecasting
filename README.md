# HydroGrid-AI

## Overview

This project implements an intelligent energy management system for a hybrid microgrid. The core objective is to optimize grid stability and maximimize renewable energy utilization by forecasting electrical **Load** and **Solar Generation**. These forecasts drive a simulation of a **Hydrogen Energy Storage System**, balancing energy flows between solar production, load demand, batteries, and hydrogen storage (Electrolyzer & Fuel Cell). This is an implementation of the electrolyzer efficiency study with various ML algorithms.

The project explores a wide range of analytical techniques, from classical statistical models to advanced machine learning and deep learning architectures, to identify the most accurate forecasting methods.

## Features

- **Multi-Model Forecasting:** Implementation and comparison of diverse models:
  - **Ensemble Methods:** XGBoost (with Genetic Algorithm optimization), Random Forest, Extra Trees.
  - **Deep Learning:** LSTM (Long Short-Term Memory) networks, LSTM Autoencoders.
  - **Classical/Statistical:** ARIMA, Linear Regression, SVR (Support Vector Regression).
  - **Hybrid Approach:** A Stacking Regressor combining XGBoost, Extra Trees, and LSTM.
- **Hyperparameter Optimization:** Uses Genetic Algorithms (GA) to fine-tune XGBoost parameters for maximal accuracy.
- **Physics-Based Simulation:** Simulates a complete Hydrogen storage cycle:
  - **Electrolyzer:** Converts excess power to Hydrogen (70% efficiency).
  - **Fuel Cell:** Converts Hydrogen to power during deficits (50% efficiency).
  - **Storage:** Tracks Hydrogen levels (kWh) with capacity constraints.
- **Model Predictive Control (MPC):** Includes a controller logic class to demonstrate how forecasts dictate energy dispatch decisions.

## Model Performance Summary

### Load Forecasting (Target: `Next_Load_kW`)

_Sorted by R² (Higher is better)_

| Model             | MAE (kW) | RMSE (kW) | R² Score  |
| ----------------- | -------- | --------- | --------- |
| **GA-XGBoost**    | **7.89** | **11.36** | **0.825** |
| XGBoost           | 8.09     | 11.70     | 0.814     |
| Random Forest     | 8.45     | 12.16     | 0.799     |
| Extra Trees       | 8.54     | 12.38     | 0.792     |
| Hybrid Regressor  | 8.54     | _N/A_     | 0.792     |
| LSTM              | 9.23     | 13.23     | 0.762     |
| SVR               | 10.19    | 14.64     | 0.709     |
| LSTM Autoencoder  | 11.62    | 15.77     | 0.662     |
| Linear Regression | 12.71    | 17.02     | 0.607     |
| ARIMA             | 12.71    | 17.02     | 0.607     |

### Solar Forecasting (Target: `Next_Solar_kW`)

_Sorted by R² (Higher is better)_

| Model             | MAE (kW)  | R² Score  |
| ----------------- | --------- | --------- |
| **XGBoost**       | **10.37** | **0.957** |
| Random Forest     | 10.39     | 0.956     |
| Extra Trees       | 10.51     | 0.953     |
| Hybrid Regressor  | 10.51     | 0.953     |
| LSTM              | 12.65     | 0.950     |
| SVR               | 12.98     | 0.937     |
| Linear Regression | 23.28     | 0.894     |
| ARIMA             | 23.29     | 0.894     |

## Key Observations

1.  **Best Performing Model**:

    - The **Genetic Algorithm Optimized XGBoost (GA-XGBoost)** achieved the highest predictive accuracy for Load forecasting, with an R² of **0.825** and the lowest Mean Absolute Error (7.89 kW).
    - This highlights the value of hyperparameter tuning over default settings or more complex architectures for this specific dataset.

2.  **Solar vs. Load Predictability**:

    - Solar generation proved to be highly predictable, with most models achieving an R² above **0.95**. This is likely due to the strong correlation with temporal features (Time of Day) and clear cyclical patterns.
    - Load forecasting was more challenging (best R² ~0.82), reflecting the more stochastic nature of energy consumption.

3.  **Model Architecture Comparison**:

    - **Tree-Based Ensembles Dominance**: Methods like XGBoost, Random Forest, and Extra Trees consistently outperformed Deep Learning (LSTM) and Classical (Linear/ARIMA) methods.
    - **Deep Learning Performance**: The LSTM and LSTM Autoencoder models, while capable, did not surpass the gradient boosting methods. The Autoencoder specifically had a lower R² (0.66), suggesting that the reconstruction-based feature extraction might not have been optimal for this specific regression task compared to direct supervision.
    - **Baselines**: Linear Regression and ARIMA provided the lowest performance, indicating that the relationships in the data are highly non-linear.

4.  **Hybrid Approaches**:
    - The **Hybrid Regressor** performed well (matching Extra Trees) but did not outperform the single best optimized XGBoost model. This suggests that the simple stacking of base models (XGBoost, Extra Trees, LSTM) didn't capture additional variance beyond what XGBoost alone could find.

## Hyperparameter Optimization Results

The Genetic Algorithm optimization process for XGBoost settled on the following optimal hyperparameters after 3 generations:

- `n_estimators`: **150**
- `learning_rate`: **0.1**
- `max_depth`: **3**

These parameters suggest that a moderate number of robust, shallow trees (depth 3) generalized better than deeper, potentially overfitting trees.

## Appendix: Detailed Model Outputs

### A. Genetic Algorithm Optimization Log

_Source: Cell 38 Output_

```text
Training GA-Optimized XGBoost...
Starting GA Optimization over 3 generations...
Generation 1: Best MSE = 145.8098
Generation 2: Best MSE = 145.8098
Generation 3: Best MSE = 145.8098
Best Parameters Found: {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 3}
GA-XGBoost - Load MAE: 7.8913, R2: 0.8247
```

### B. LSTM Autoencoder Training Summary

_Source: Cell 36 Output_

```text
Epoch 1/20 ... loss: 0.8781
...
Epoch 20/20 ... loss: 0.3895
Autoencoder - Load MAE: 11.6188, R2: 0.6623
```

### C. ARIMA Model Results

_Source: Cell 26 Output_

```text
Training ARIMA for Load...
ARIMA Load - MAE: 12.7127, R2: 0.6065

Training ARIMA for Solar...
ARIMA Solar - MAE: 23.2896, R2: 0.8936
```

### D. Visual Output Descriptions

_The notebook generated several key visualizations which should be included in the final report:_

1.  **Hydrogen Storage vs. Solar Generation (Scatter Plots)**:

    - Displayed for each model (Random Forest, XGBoost, etc.).
    - Shows the correlation between Solar Energy generated and energy stored in Hydrogen.
    - Includes "Actual Fit" vs "Predicted Fit" lines to visually assess model bias.

2.  **Learning Curve (XGBoost)**:

    - Plots the Training Score vs. Cross-Validation Score as the training set size increases.
    - Diagnostics for overfitting (gap between lines) or underfitting (low scores).

3.  **Model Comparison Bar Chart**:

    - Visual comparison of MSE and RMSE across all models.
    - highlights GA-XGBoost as the lowest error bar.

4.  **MPC Simulation Plot**:
    - Visualizes the control actions (hydrogen production vs consumption) over a simulated horizon.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Longman-max/HydroGrid-AI.git
    cd HydroGrid-AI
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    _Note: The project requires `tensorflow`, `xgboost`, `mlxtend`, `scikit-learn`, `pandas`, `numpy`, and `matplotlib`._

## Usage

### Running the Analysis

You can run the analysis using the Jupyter Notebook or the Python script.

- **Jupyter Notebook:**
  Open `model.ipynb` to view the step-by-step data processing, exploratory data analysis (EDA), model training, and visualization.

  ```bash
  jupyter notebook model.ipynb
  ```

- **Python Script:**
  Run the standalone script to execute the training and evaluation pipeline.
  ```bash
  python model.py
  ```

### Input Data

The model relies on `original_load.csv`, which must contain the following columns (processed effectively within the script):

- `Date`: Timestamp.
- `Solar_kW`: Solar power generation.
- `Gen_kW`: Other generator power.
- `Load_kW`: Power consumption/demand.
- `Excess_kW`: Excess renewable energy.
- `Battery_SOC`: Battery State of Charge.

## Project Structure

- `model.ipynb`: The primary research notebook containing code, visualizations, and markdown explanations.
- `model.py`: A script version of the model pipeline for batch execution.
- `model_observations.md`: Technical report summarizing model performance metrics and insights.
- `original_load.csv`: Historical energy data used for training and testing.
- `requirements.txt`: Python package dependencies.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
