import streamlit as st
import pandas as pd
import joblib
from utils.plot_utils import plot_forecast_generic
from prophet import Prophet
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import os
import json


st.set_page_config(page_title="Forecast", page_icon="📈")

st.markdown("# Sales monitoring")
st.write(

    """
    This page visualizes time series forecasts and highlights anomalies using different detection methods.

    Use the sidebar to adjust the model, granularity, forecast horizon, and anomaly detection strategy.
    """
)

horizon_map_days = {
        "1 day": 1,
        "7 days": 7,
        "1 month": 30,
        "3 months": 90,
        "6 months": 180,
        "1 year": 365
    }

horizon_map_months = {
        "1 day": 1,
        "7 days": 1,
        "1 month": 1,
        "3 months": 3,
        "6 months": 6,
        "1 year": 12
    }

# sidebar filters 
st.sidebar.header("Configuration")
data_level = st.sidebar.selectbox("Select dataset", ["total", "store", "product"])
granularity = st.sidebar.radio("Granularity", ["daily", "monthly"])
model_type = st.sidebar.selectbox("Model", ["Prophet"])

horizon_options = ["1 day", "7 days", "1 month", "3 months", "6 months", "1 year"]
horizon_label = st.sidebar.selectbox("Forecast horizon", horizon_options)

if granularity == "daily":
        forecast_horizon = horizon_map_days[horizon_label]
        forecast_freq = "D"
else:
        forecast_horizon = horizon_map_months[horizon_label]
        forecast_freq = "MS"

# Select anomaly detection strategy:
# - Prophet interval: flags points outside predicted confidence interval
# - STL + Z-score: uses residual decomposition and standard score thresholding
# - Isolation Forest: unsupervised model that isolates outliers using decision trees
anomaly_method = st.sidebar.selectbox(
        "Anomaly detection method",
        ["Prophet interval", "STL + Z-score", "Isolation Forest"]
    )
show_anomalies = st.sidebar.checkbox("Show anomalies", True)
show_interval = st.sidebar.checkbox("Show confidence interval", True)
start_date = st.sidebar.date_input("Start date (optional)", value='2018-10-01')

is_started_from_root = os.path.exists(".git")
prefix = "time_series" if is_started_from_root else ""

@st.cache_data
def load_data(prefix:str = ""):
    if len(prefix) != 0 and not prefix.endswith('/'):
        prefix += '/'

    data = {}
    for level in ["total", "store", "product"]:
        for freq in ["daily", "monthly"]:
            train_path = f"{prefix}data/df_{freq}_{level}_train.csv"
            test_path = f"{prefix}data/df_{freq}_{level}_test.csv"
            try:
                train = pd.read_csv(train_path, parse_dates=["Date"])
                test = pd.read_csv(test_path, parse_dates=["Date"])
                data[f"{freq}_{level}"] = (train, test)
            except:
                break
    return data

data_dict = load_data(prefix)
data_key = f"{granularity}_{data_level}"

if data_key not in data_dict:
    st.error(f"No data available for {data_key}")
    st.stop()

train_df, test_df = data_dict[data_key]

entity_col = None
if data_level == "store":
    entity_col = "store"
elif data_level == "product":
    entity_col = "product"

entity_id = None
if entity_col:
    unique_entities = sorted(train_df[entity_col].unique())
    entity_id = st.selectbox(f"Select {entity_col}", unique_entities)

@st.cache_data
def load_forecasts(prefix:str = ""):
    if len(prefix) != 0 and not prefix.endswith('/'):
        prefix += '/'
    forecasts = {}
    for model in ["prophet"]:
        for freq in ["daily", "monthly"]:
            for level in ["total", "store", "product"]:
                key = f"{model}_{freq}_{level}"
                path = f"{prefix}forecasts/{key}.pkl"
                try:
                    forecasts[key] = joblib.load(path)
                except:
                    pass
    return forecasts

forecast_all = load_forecasts(prefix)
forecast_key = f"{model_type.lower()}_{granularity}_{data_level}"

if forecast_key not in forecast_all:
    st.error(f"No forecast available for {forecast_key}")
    st.stop()

forecast_dict = forecast_all[forecast_key]

if data_level == "total":
    train_filtered = train_df.rename(columns={"Date": "ds", "number_sold": "y"})
    test_filtered = test_df.rename(columns={"Date": "ds"}).set_index("ds")
    forecast_df = forecast_dict
else:
    train_filtered = train_df[train_df[entity_col] == entity_id].copy().rename(columns={"Date": "ds", "number_sold": "y"})
    test_filtered = test_df[test_df[entity_col] == entity_id].copy().rename(columns={"Date": "ds"}).set_index("ds")
    forecast_df = forecast_dict.get(entity_id)

if forecast_df is None:
    st.warning(f"No forecast for {entity_col} {entity_id}")
    st.stop()

forecast_df = forecast_df.copy()
if 'ds' in forecast_df.columns:
    forecast_df = forecast_df.set_index('ds')
forecast_df.index = pd.to_datetime(forecast_df.index)
if "yhat" in forecast_df.columns:
    forecast_df = forecast_df.rename(columns={"yhat": "y_pred"})

# Finds common dates between forecast_df and test_filtered.  
# forecast_df can contain forecasts for dates that are not in test_filtered, and vice versa.
common_idx = forecast_df.index.intersection(test_filtered.index)
forecast_df = forecast_df.loc[common_idx]
forecast_df = forecast_df.iloc[:forecast_horizon]
forecast_df = forecast_df.dropna(subset=['y_pred'])

fig = plot_forecast_generic(
        y_train=train_filtered.set_index("ds")["y"],
        y_test=test_filtered["number_sold"],
        forecast_df=forecast_df,
        model_name=f"{model_type} — {data_level.title()}{f' {entity_id}' if entity_id else ''}",
        show_anomalies=show_anomalies,
        interval_confidence=show_interval,
        start_date=pd.to_datetime(start_date) if start_date else None,
        anomaly_method=anomaly_method
    )

st.plotly_chart(fig, use_container_width=True)
