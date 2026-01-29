#!/usr/bin/env python3
"""Interactive ARIMA Stock Forecaster - Production-Grade Streamlit App"""

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ARIMA Stock Forecaster", page_icon="📈", layout="wide")

st.title("📈 Interactive ARIMA Stock Forecaster")
st.markdown("""
Demonstrates advanced time series modeling on **3 years of real Google stock data**.

**Features:** Multiple forecasting methods • Parameter tuning • 30-day validation • Performance metrics
""")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('stock_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date').reset_index(drop=True)
    except FileNotFoundError:
        return None

df = load_data()
if df is None:
    st.error("⚠️ Data not found! Run: `python3 get_data.py`")
    st.stop()

train_size = len(df) - 30
train_df, test_df = df.iloc[:train_size].copy(), df.iloc[train_size:].copy()
train_prices, test_prices = train_df['Close'].values, test_df['Close'].values
test_dates = test_df['Date'].values

# Sidebar Controls
st.sidebar.header("⚙️ Forecasting Methods")
col1, col2 = st.sidebar.columns(2)

with col1:
    use_arima = st.checkbox("ARIMA", value=True, help="AutoRegressive Integrated MA")
    use_naive = st.checkbox("Naive", value=True, help="Last value")
    use_drift = st.checkbox("Drift", value=False, help="Linear drift")
    use_ar = st.checkbox("AR", value=False, help="Autoregressive")

with col2:
    use_ma = st.checkbox("Moving Avg", value=False)
    use_ses = st.checkbox("Exp. Smooth", value=False)
    use_linear = st.checkbox("Linear Trend", value=False)

if use_arima or use_ar:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 ARIMA Parameters")
    c1, c2, c3 = st.sidebar.columns(3)
    p = c1.slider("p", 0, 10, 5, help="AR lags")
    d = c2.slider("d", 0, 2, 1, help="Differencing")
    q = c3.slider("q", 0, 10, 0, help="MA lags")
else:
    p, d, q = 5, 1, 0

if use_ar:
    ar_lags = st.sidebar.slider("AR Lags", 1, 15, 5)
if use_ma:
    ma_window = st.sidebar.slider("MA Window", 5, 60, 20)
if use_ses:
    alpha = st.sidebar.slider("α (smoothing)", 0.01, 0.99, 0.3, 0.01)

st.sidebar.markdown("---")
show_actuals = st.sidebar.toggle("👁️ Show Actual Values", False)
show_conf = st.sidebar.toggle("📊 Show 95% CI", True)
show_table = st.sidebar.toggle("📋 Show Table", True)

# Forecasting Functions
def forecast_arima(data, steps, p, d, q):
    try:
        model = ARIMA(data, order=(p,d,q)).fit()
        fc = model.get_forecast(steps=steps)
        return fc.predicted_mean.values, fc.conf_int().values
    except:
        return None, None

def forecast_ar(data, steps, lags):
    try:
        model = AutoReg(data, lags=lags).fit()
        fc = model.get_forecast(steps=steps)
        return fc.predicted_mean.values, fc.conf_int().values
    except:
        return None, None

def forecast_naive(data, steps):
    return np.full(steps, data[-1]), None

def forecast_drift(data, steps):
    drift = (data[-1] - data[0]) / (len(data) - 1)
    return np.array([data[-1] + (i+1)*drift for i in range(steps)]), None

def forecast_ma(data, steps, window):
    return np.full(steps, np.mean(data[-window:])), None

def forecast_ses(data, steps, a):
    level = data[0]
    for x in data:
        level = a * x + (1-a) * level
    return np.full(steps, level), None

def forecast_linear(data, steps):
    n = len(data)
    coeffs = np.polyfit(np.arange(n), data, 1)
    return np.array([coeffs[0]*(n+i) + coeffs[1] for i in range(steps)]), None

# Generate Forecasts
forecasts, conf_ints = {}, {}

if use_arima:
    f, c = forecast_arima(train_prices, 30, p, d, q)
    if f is not None:
        forecasts['ARIMA'], conf_ints['ARIMA'] = f, c

if use_ar:
    f, c = forecast_ar(train_prices, 30, ar_lags if 'ar_lags' in locals() else 5)
    if f is not None:
        forecasts['AR'], conf_ints['AR'] = f, c

if use_naive:
    forecasts['Naive'], _ = forecast_naive(train_prices, 30)
if use_drift:
    forecasts['Drift'], _ = forecast_drift(train_prices, 30)
if use_ma:
    forecasts['MA'], _ = forecast_ma(train_prices, 30, ma_window if 'ma_window' in locals() else 20)
if use_ses:
    forecasts['SES'], _ = forecast_ses(train_prices, 30, alpha if 'alpha' in locals() else 0.3)
if use_linear:
    forecasts['Linear'], _ = forecast_linear(train_prices, 30)

# Calculate Metrics
metrics = {}
for name, fc in forecasts.items():
    rmse = np.sqrt(mean_squared_error(test_prices, fc))
    mae = mean_absolute_error(test_prices, fc)
    mape = np.mean(np.abs((test_prices - fc) / test_prices)) * 100
    metrics[name] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# Display Metrics
st.subheader("📊 Performance Metrics")

if metrics:
    best = min(metrics, key=lambda x: metrics[x]['RMSE'])
    m1, m2, m3 = st.columns(3)
    m1.metric("🏆 Best (RMSE)", best, f"${metrics[best]['RMSE']:.2f}")
    m2.metric("Training Days", f"{len(train_df):,}", f"{len(test_df)} test")
    m3.metric("Date Range", train_df['Date'].min().strftime('%b %Y'), f"to {test_df['Date'].max().strftime('%b %Y')}")
    
    st.markdown("---")
    st.subheader("📈 Model Comparison")
    df_metrics = pd.DataFrame(metrics).T.sort_values('RMSE')
    df_display = df_metrics.copy()
    df_display['RMSE'] = df_display['RMSE'].apply(lambda x: f"${x:.2f}")
    df_display['MAE'] = df_display['MAE'].apply(lambda x: f"${x:.2f}")
    df_display['MAPE'] = df_display['MAPE'].apply(lambda x: f"{x:.2f}%")
    st.dataframe(df_display, use_container_width=True)

# Visualization
st.markdown("---")
st.subheader("📉 Forecast Chart")

fig = go.Figure()
colors = {'ARIMA': '#667eea', 'AR': '#764ba2', 'Naive': '#FF6B6B', 'Drift': '#FFA07A', 
          'MA': '#45B7D1', 'SES': '#4ECDC4', 'Linear': '#95E1D3', 'Historical': '#2C3E50'}

fig.add_trace(go.Scatter(x=train_df['Date'], y=train_df['Close'], mode='lines',
                         name='Historical', line=dict(color=colors['Historical'], width=2)))

for name, fc in forecasts.items():
    color = colors.get(name, '#999')
    
    if show_conf and name in conf_ints and conf_ints[name] is not None:
        ci = conf_ints[name]
        fig.add_trace(go.Scatter(x=test_dates, y=ci[:,1], line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=test_dates, y=ci[:,0], line=dict(width=0), fill='tonexty',
                                 fillcolor=color+'20', showlegend=False, hoverinfo='skip'))
    
    fig.add_trace(go.Scatter(x=test_dates, y=fc, mode='lines+markers', name=name,
                             line=dict(color=color, width=2.5, dash='dash'), marker=dict(size=4)))

if show_actuals:
    fig.add_trace(go.Scatter(x=test_dates, y=test_prices, mode='lines+markers', name='Actual',
                             line=dict(color='#2ECC71', width=3), marker=dict(size=6)))

fig.add_vline(x=train_df['Date'].iloc[-1].isoformat(), line_dash="dot", line_color="#999",
              annotation_text="Train/Test Split", annotation_position="top right")

fig.update_layout(title="Google Stock: Forecast Comparison", xaxis_title="Date",
                  yaxis_title="Price (USD)", height=700, template="plotly_white",
                  hovermode="x unified", legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"))

zoom_start = train_df['Date'].iloc[-90]
fig.update_xaxes(range=[zoom_start, test_df['Date'].iloc[-1]])

st.plotly_chart(fig, use_container_width=True)

# Forecast Table
if show_table and forecasts:
    st.markdown("---")
    st.subheader("📊 30-Day Forecast Values")
    table = {'Date': test_dates}
    for name, fc in forecasts.items():
        table[name] = [f"${v:.2f}" for v in fc]
    if show_actuals:
        table['Actual'] = [f"${v:.2f}" for v in test_prices]
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("**📚 Portfolio Project** | Streamlit + Statsmodels + Plotly | 3 years GOOGL data • 30-day validation")
