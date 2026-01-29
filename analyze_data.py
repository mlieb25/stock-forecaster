#!/usr/bin/env python3
"""
Data Analysis Script
Analyzes the Google stock data and generates insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data():
    """Load the stock data."""
    df = pd.read_csv('googl_us_d.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def basic_stats(df):
    """Display basic statistics."""
    print("=" * 60)
    print("GOOGLE STOCK DATA ANALYSIS")
    print("=" * 60)
    print()
    print(f"Dataset Overview:")
    print(f"  Total Records: {len(df):,}")
    print(f"  Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Time Span: {(df['Date'].max() - df['Date'].min()).days} days")
    print()
    
    print("Price Statistics (Close):")
    print(f"  Current Price: ${df['Close'].iloc[-1]:.2f}")
    print(f"  All-Time High: ${df['Close'].max():.2f} ({df.loc[df['Close'].idxmax(), 'Date'].strftime('%Y-%m-%d')})")
    print(f"  All-Time Low: ${df['Close'].min():.2f} ({df.loc[df['Close'].idxmin(), 'Date'].strftime('%Y-%m-%d')})")
    print(f"  Mean Price: ${df['Close'].mean():.2f}")
    print(f"  Median Price: ${df['Close'].median():.2f}")
    print(f"  Std Dev: ${df['Close'].std():.2f}")
    print()
    
    # Calculate returns
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    print("Daily Returns Statistics:")
    print(f"  Mean Daily Return: {df['Daily_Return'].mean():.4f}%")
    print(f"  Volatility (Std Dev): {df['Daily_Return'].std():.4f}%")
    print(f"  Best Day: +{df['Daily_Return'].max():.2f}% ({df.loc[df['Daily_Return'].idxmax(), 'Date'].strftime('%Y-%m-%d')})")
    print(f"  Worst Day: {df['Daily_Return'].min():.2f}% ({df.loc[df['Daily_Return'].idxmin(), 'Date'].strftime('%Y-%m-%d')})")
    print()
    
    # Volume statistics
    print("Volume Statistics:")
    print(f"  Mean Volume: {df['Volume'].mean():,.0f}")
    print(f"  Median Volume: {df['Volume'].median():,.0f}")
    print(f"  Max Volume: {df['Volume'].max():,.0f} ({df.loc[df['Volume'].idxmax(), 'Date'].strftime('%Y-%m-%d')})")
    print()
    
    # Price change over time
    first_price = df['Close'].iloc[0]
    last_price = df['Close'].iloc[-1]
    total_return = ((last_price - first_price) / first_price) * 100
    
    print("Overall Performance:")
    print(f"  Starting Price: ${first_price:.2f}")
    print(f"  Ending Price: ${last_price:.2f}")
    print(f"  Total Return: {total_return:+.2f}%")
    print()

def check_missing_data(df):
    """Check for missing data."""
    print("Missing Data Check:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✅ No missing values detected")
    else:
        print("  ⚠️ Missing values found:")
        for col, count in missing[missing > 0].items():
            print(f"    - {col}: {count} missing values")
    print()

def trend_analysis(df):
    """Analyze trends by year."""
    df['Year'] = df['Date'].dt.year
    
    print("Yearly Performance:")
    yearly_stats = df.groupby('Year').agg({
        'Close': ['first', 'last', 'min', 'max'],
        'Volume': 'mean'
    })
    
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]
        first = year_data['Close'].iloc[0]
        last = year_data['Close'].iloc[-1]
        year_return = ((last - first) / first) * 100
        
        print(f"  {year}:")
        print(f"    Start: ${first:.2f} | End: ${last:.2f}")
        print(f"    Return: {year_return:+.2f}%")
        print(f"    High: ${year_data['Close'].max():.2f} | Low: ${year_data['Close'].min():.2f}")
        print(f"    Avg Volume: {year_data['Volume'].mean():,.0f}")
        print()

def check_stationarity(df):
    """Basic stationarity check."""
    from scipy import stats
    
    print("Stationarity Check (Visual Inspection):")
    
    # Split data into 3 periods
    n = len(df) // 3
    period1 = df['Close'].iloc[:n]
    period2 = df['Close'].iloc[n:2*n]
    period3 = df['Close'].iloc[2*n:]
    
    print(f"  Period 1 (Early): Mean=${period1.mean():.2f}, Std=${period1.std():.2f}")
    print(f"  Period 2 (Mid):   Mean=${period2.mean():.2f}, Std=${period2.std():.2f}")
    print(f"  Period 3 (Late):  Mean=${period3.mean():.2f}, Std=${period3.std():.2f}")
    print()
    print("  Note: Significant differences in mean/std suggest non-stationarity.")
    print("        Consider differencing (d=1 in ARIMA) if means vary substantially.")
    print()

def autocorrelation_check(df):
    """Check autocorrelation at different lags."""
    print("Autocorrelation Analysis (Sample Lags):")
    
    from pandas.plotting import autocorrelation_plot
    
    lags = [1, 5, 10, 20, 30]
    for lag in lags:
        corr = df['Close'].autocorr(lag=lag)
        print(f"  Lag {lag:2d}: {corr:.4f}")
    
    print()
    print("  High autocorrelation (>0.8) at early lags suggests AR component.")
    print("  Gradual decay suggests trend/non-stationarity.")
    print()

def recommend_arima_params(df):
    """Provide ARIMA parameter recommendations."""
    print("=" * 60)
    print("ARIMA PARAMETER RECOMMENDATIONS")
    print("=" * 60)
    print()
    
    # Calculate first difference
    df['Diff'] = df['Close'].diff()
    
    # Check if differencing helps
    original_std = df['Close'].std()
    diff_std = df['Diff'].dropna().std()
    
    print("Differencing Analysis:")
    print(f"  Original Std Dev: ${original_std:.2f}")
    print(f"  After Differencing: ${diff_std:.2f}")
    
    if diff_std < original_std * 0.7:
        print("  → Recommendation: Use d=1 (differencing helps stabilize variance)")
        d_rec = 1
    else:
        print("  → Recommendation: Try d=0 first (data may be stationary)")
        d_rec = 0
    print()
    
    # Autocorrelation suggests p (AR order)
    lag1_corr = df['Close'].autocorr(lag=1)
    if lag1_corr > 0.9:
        print("  Strong lag-1 autocorrelation detected.")
        print("  → Recommendation: Try p=3 to p=7 (higher AR order)")
        p_rec = "3-7"
    elif lag1_corr > 0.7:
        print("  Moderate lag-1 autocorrelation detected.")
        print("  → Recommendation: Try p=2 to p=5")
        p_rec = "2-5"
    else:
        print("  Low lag-1 autocorrelation.")
        print("  → Recommendation: Try p=1 to p=3")
        p_rec = "1-3"
    print()
    
    print("  MA component (q): Start with q=0 or q=1")
    print("  → Try q=0 first, then experiment with q=1 or q=2")
    print()
    
    print("Suggested Starting Points:")
    print(f"  • Conservative: ARIMA({p_rec.split('-')[0]}, {d_rec}, 0)")
    print(f"  • Moderate: ARIMA(5, {d_rec}, 0)")
    print(f"  • Exploratory: ARIMA({p_rec.split('-')[1]}, {d_rec}, 1)")
    print()

def main():
    """Main analysis function."""
    # Load data
    df = load_data()
    
    # Run all analyses
    basic_stats(df)
    check_missing_data(df)
    trend_analysis(df)
    check_stationarity(df)
    autocorrelation_check(df)
    recommend_arima_params(df)
    
    print("=" * 60)
    print("✅ Analysis Complete!")
    print("=" * 60)
    print()
    print("Next Steps:")
    print("  1. Run the Streamlit app: streamlit run app.py")
    print("  2. Try the recommended ARIMA parameters")
    print("  3. Compare multiple forecasting methods")
    print("  4. Toggle 'Show Actual Values' to validate performance")
    print()

if __name__ == "__main__":
    main()
