#!/usr/bin/env python3
"""
Setup Verification Script
Checks if all required files and dependencies are in place.
"""

import sys
import os
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: Found")
        return True
    else:
        print(f"❌ {description}: Missing")
        return False

def check_dependencies():
    """Check if required Python packages are installed."""
    print("\nChecking Python Dependencies:")
    print("="  * 60)
    
    required = [
        ('streamlit', 'Streamlit'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('statsmodels', 'Statsmodels'),
        ('plotly', 'Plotly'),
        ('sklearn', 'Scikit-learn')
    ]
    
    missing = []
    for module, name in required:
        try:
            __import__(module)
            print(f"✅ {name}: Installed")
        except ImportError:
            print(f"❌ {name}: Not installed")
            missing.append(name)
    
    if missing:
        print("\n⚠️  Missing dependencies detected!")
        print("\nTo install, run:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True

def check_data():
    """Check and display data info."""
    print("\nChecking Data:")
    print("="  * 60)
    
    if not Path('googl_us_d.csv').exists():
        print("❌ Data file not found!")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv('googl_us_d.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"✅ Data file: googl_us_d.csv")
        print(f"   Records: {len(df):,}")
        print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
        print(f"   Latest price: ${df['Close'].iloc[-1]:.2f}")
        
        # Check for last 30 days (test set)
        if len(df) >= 30:
            print(f"   Test set: Last 30 days available for validation ✅")
        else:
            print(f"   ⚠️  Warning: Less than 30 days of data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading  {e}")
        return False

def main():
    """Main verification function."""
    print("\n" + "=" * 60)
    print("🔍 STOCK FORECASTER - SETUP VERIFICATION")
    print("=" * 60)
    
    # Check files
    print("\nChecking Project Files:")
    print("="  * 60)
    
    files_ok = all([
        check_file('app.py', 'Main application'),
        check_file('requirements.txt', 'Dependencies file'),
        check_file('README.md', 'Documentation'),
        check_file('analyze_data.py', 'Data analysis script'),
        check_file('.streamlit/config.toml', 'Streamlit config'),
        check_file('googl_us_d.csv', 'Stock data')
    ])
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check data
    data_ok = check_data()
    
    # Summary
    print("\n" + "=" * 60)
    if files_ok and deps_ok and data_ok:
        print("✅ ALL CHECKS PASSED!")
        print("=" * 60)
        print("\n🚀 You're ready to go!\n")
        print("Next steps:")
        print("  1. Analyze the ")
        print("     python analyze_data.py")
        print("\n  2. Run the app:")
        print("     streamlit run app.py")
        print("\n  3. Open browser:")
        print("     http://localhost:8501")
        print("\n  4. (Optional) Initialize Git:")
        print("     git init")
        print("     git add .")
        print("     git commit -m 'Initial commit'")
        print()
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("=" * 60)
        print("\nPlease fix the issues above before running the app.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
