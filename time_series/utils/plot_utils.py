import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest


def plot_forecast_generic(y_train,
                          y_test,
                          forecast_df, 
                          model_name="Model", 
                          show_anomalies=True,
                          start_date=None, 
                          interval_confidence=True,
                          anomaly_method="Prophet interval"):
    """
    Universal forecast visualizer for time series predictions with optional anomaly detection.

    Parameters:
    - y_train : Historical training data (indexed by datetime).
    - y_test : Actual test data to compare predictions against (indexed by datetime).
    - forecast_df : DataFrame with model predictions. 
        Expected columns: 'y_pred' or 'yhat', optionally 'yhat_upper' and 'yhat_lower'.
    - model_name : Title to display on the plot.
    - show_anomalies : If True, mark detected anomalies on the plot.
    - start_date : Optional filter to only show data after this date.
    - interval_confidence : If True, plot confidence interval from forecast.
    - anomaly_method (str): Method used for anomaly detection.
        Options:
        - "Prophet interval (default)": Marks actual values outside of [lower, upper] bounds.
        - "STL + Z-score": Uses residuals from STL decomposition and flags points with |Z| > 3.
        - "Isolation Forest": Applies a tree-based unsupervised model on forecast errors.

    """
    if 'ds' in forecast_df.columns:
        forecast_df = forecast_df.set_index('ds')
    if 'yhat' in forecast_df.columns:
        forecast_df.rename(columns={'yhat': 'y_pred'}, inplace=True)
    if 'yhat_upper' in forecast_df.columns:
        forecast_df.rename(columns={'yhat_upper': 'upper'}, inplace=True)
    if 'yhat_lower' in forecast_df.columns:
        forecast_df.rename(columns={'yhat_lower': 'lower'}, inplace=True)

    y_train.index = pd.to_datetime(y_train.index)
    y_test.index = pd.to_datetime(y_test.index)
    forecast_df.index = pd.to_datetime(forecast_df.index)

    forecast_df = forecast_df.reindex(y_test.index)
    forecast_df = forecast_df.dropna(subset=['y_pred'])
    y_pred = forecast_df['y_pred']

    mae = mean_absolute_error(y_test.loc[y_pred.index], y_pred)
    mape = mean_absolute_percentage_error(y_test.loc[y_pred.index], y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_test.loc[y_pred.index], y_pred))

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        y_train = y_train.loc[y_train.index >= start_date]
        y_test = y_test.loc[y_test.index >= start_date]
        forecast_df = forecast_df.loc[forecast_df.index >= start_date]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_train.index, y=y_train.values,
        mode='lines', name='Train', line=dict(color='gray', dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=y_test.index, y=y_test.values,
        mode='lines', name='Actual (Test)', line=dict(color='gray')
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['y_pred'],
        mode='lines', name='Prediction', line=dict(color='red', width=2)
    ))

    if interval_confidence and 'upper' in forecast_df.columns and 'lower' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df['upper'],
            mode='lines', name='Upper Bound', line=dict(dash='dot', color='green'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df['lower'],
            mode='lines', name='Lower Bound', line=dict(dash='dot', color='green'),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            showlegend=True
        ))

    if show_anomalies and not y_test.empty and not forecast_df.empty:
        if anomaly_method == "Prophet interval (default)":
            anomalies = (y_test.loc[y_pred.index] < forecast_df["lower"]) | (y_test.loc[y_pred.index] > forecast_df["upper"])
        elif anomaly_method == "STL + Z-score":
            resid = y_test.loc[y_pred.index] - y_pred
            stl = STL(resid, period=7)
            z_scores = (stl.fit().resid - resid.mean()) / resid.std()
            anomalies = abs(z_scores) > 2
        elif anomaly_method == "Isolation Forest":
            iso = IsolationForest(contamination=0.05)
            is_outlier = iso.fit_predict((y_test.loc[y_pred.index] - y_pred).values.reshape(-1, 1))
            anomalies = is_outlier == -1
        else:
            anomalies = pd.Series(False, index=y_test.index)

        if anomalies.any():
            anomaly_points = y_test.loc[y_pred.index][anomalies]
            errors = anomaly_points - forecast_df.loc[anomaly_points.index]['y_pred']

            fig.add_trace(go.Scatter(
                x=anomaly_points.index,
                y=anomaly_points.values,
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8, symbol='x'),
                customdata=[f"{error:.0f}" for error in errors],
                hovertemplate=(
                    "Date: %{x}<br>"
                    "Actual: %{y}<br>"
                    "Error: Δ=%{customdata}<extra></extra>"
                )
            ))

    fig.update_layout(
        title=f"{model_name} — MAPE: {mape:.2f}%, MAE: {mae:.0f}, RMSE: {rmse:.0f}",
        xaxis_title="Date",
        yaxis_title="Sales",
        template='plotly_white',
        height=500
    )

    return fig
