import streamlit as st

st.set_page_config(page_title="Time Series Info", page_icon="💡")

st.sidebar.success("Select a page above.")

st.write('Forecasts of data on the left in the "📈 Prediction" page')

st.markdown(
"""
# **Project: Time Series Forecasting and Outlier Detection**


### 🔍 Overview
This project focuses on forecasting sales and identifying anomalies in time series data using various statistical and machine learning methods. The primary tool for interaction is a Streamlit web application.

### 📦 Dataset Description

The [dataset](https://www.kaggle.com/datasets/samuelcortinhas/time-series-practice-dataset) used is from Kaggle. It contains simulated sales data across a 10-year period (2010–2019).

**Features:**

- date
- store_id (7 unique stores)
- product_id (10 unique products)
- number_sold (target variable)
- Train period: 2010–2018 and Test period: 2019 
- There are no missing values in the dataset.

### 🎯 Goals

1. Forecast future values of number_sold for various data granularities (total, store, product).
2. Compare model performance using forecasting metrics.
3. Detect and visualize anomalies in forecast vs. actual values.
4. Build an interactive Streamlit dashboard to explore results.

### ⚙️ Model Summary

- Prophet (Meta): Decomposition-based, easy to implement, automatically detects seasonality/trends, and supports uncertainty intervals.
- SARIMA: Traditional statistical model. Requires manual tuning and works poorly with unstable seasonality or missing data.
- Linear Regression: Simple and interpretable, but needs manual feature engineering (lags, time encoding), and can't handle nonlinearity or autocorrelation.
- LightGBM: A powerful ML model. Captures nonlinearities and handles categorical features, but lacks native time-awareness and needs careful feature engineering.
- LSTM: Neural network built for sequential data. Captures long-term dependencies and nonlinear patterns, but is complex to train and less interpretable.

### 📏 Evaluation Metrics

For comparison these models I used *MAPE* (Mean Absolute Percentage Error) as the main evaluation metric:  
- It’s scale-independent
- It’s interpretable (shows average % error)
- It was also recommended by the dataset author
""")

st.latex(r"""
\text{MAPE} = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right| \cdot 100\%
""")
st.latex(r"""
\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} \left| y_t - \hat{y}_t \right|
""")
st.latex(r"""
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2 }
""")

st.markdown(
"""


### 🧪 Models Compared
To evaluate forecasting quality, we tested several models (for the total level):

| Model                | MAPE    | MAE     | RMSE   | 
|----------------------|---------|---------|--------|
| Prophet              | 0.14%   | 76      | 94     | 
| SARIMA               | 0.22%   | 121     | 147    |
| Logistic Regression  | 0.11%   | 61      | 76     | 
| LightGBM             | 0.20%   | 113     | 145    | 
| LSTM                 | 0.15%   | 85      | 104    |

Despite the fact that logistic regression has shown the best results, but for simplicity, let's leave the prophet for now.

### 🚨 Anomaly Detection Methods

Time series may contain unexpected patterns, spikes, or drops. We tested three anomaly detection strategies:

1. Prophet Confidence Interval (default)

- Flags anomalies when actual values fall outside Prophet’s predicted intervals.
- Based on bootstrapped residuals and uncertainty in trend/seasonality components.
- Simple, interpretable 

2. STL + Z-Score

- Decomposes time series into trend/seasonality/residuals.
- Flags points where the residual exceeds a Z-threshold (e.g., |Z| > 2).
- Good for local outliers; flexible and method-agnostic.

3. Isolation Forest

- Tree-based unsupervised anomaly detector.
- No need for forecast or intervals; works directly on observed/predicted residuals.
- Handles complex distributions, but less interpretable.
- Using multiple methods helps balance sensitivity and robustness across different anomaly types.

⚠️ On our dataset, Prophet tends to be quite sensitive and flags many anomalies — which may reflect both its uncertainty estimates and potential model overfitting.

➡️ Using multiple methods helps balance sensitivity and robustness across different anomaly types and datasets. The choice of method should consider the nature of your data, the expected anomalies, and your tolerance for false positives.


### 💻 Streamlit App

- An interactive dashboard was built with Streamlit for:
- Selecting data granularity (total/store/product)
- Choosing forecast horizon and anomaly method
- Visualizing predictions, intervals, and detected outliers
- Comparing performance metrics

⚠️ Note: Although logistic regression had the best metrics, we selected Prophet as the primary model for deployment due to simplicity and confidence interval support.

""")