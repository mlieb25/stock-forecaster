#!/usr/bin/env python3
"""Data Download Script - Downloads 3 years of Google stock data"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker="GOOGL", years=3):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    print("=" * 60)
    print(f"Downloading {ticker} Stock Data")
    print("=" * 60)
    print(f"Period: {start_date.date()} to {end_date.date()}\n")
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df[["Close"]].reset_index()
        df.columns = ["Date", "Close"]
        df.to_csv("stock_data.csv", index=False)
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Data saved to stock_data.csv")
        print("=" * 60)
        print(f"Records: {len(df):,}")
        print(f"Range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Price: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        print("\nNext: streamlit run app.py")
        return True
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    fetch_stock_data()
