# 📈 Interactive ARIMA Stock Forecaster

Production-grade time series forecasting app built with Streamlit.

## 🚀 Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Download data
python3 get_data.py

# Run app
streamlit run app.py
```

## ✨ Features

- **7 Forecasting Methods**: ARIMA, AR, Naive, Drift, MA, Exponential Smoothing, Linear Trend
- **Interactive Controls**: Real-time parameter tuning (p, d, q)
- **Performance Metrics**: RMSE, MAE, MAPE comparison
- **Visual Comparison**: Plotly charts with 95% confidence intervals
- **Holdout Validation**: 30-day test set

## 📊 Data

- Source: Yahoo Finance (yfinance)
- Symbol: GOOGL
- Duration: 3 years daily prices
- Split: Training (3yr - 30d), Test (30d)

## 🛠️ Tech Stack

- Python 3.8+
- Streamlit (web framework)
- Statsmodels (ARIMA)
- Plotly (visualization)
- Pandas, NumPy, Scikit-learn

## 🌐 Deployment

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

## 📝 License

MIT - Free for portfolio use
