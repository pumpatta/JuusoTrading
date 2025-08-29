#!/usr/bin/env python3
"""
Simple dashboard test to debug issues
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import os

st.title("🔧 JuusoTrader - Debug Dashboard")
st.markdown("---")

# Test basic functionality
st.write("✅ Streamlit is working")

# Check if data exists
st.subheader("📁 Data Status")

log_dir = Path('storage/logs')
if log_dir.exists():
    st.success("✅ Storage/logs directory exists")
    
    log_files = list(log_dir.glob('*.csv'))
    if log_files:
        st.success(f"✅ Found {len(log_files)} log files")
        for file in log_files:
            st.write(f"- {file.name}")
            
            # Try to read the file
            try:
                df = pd.read_csv(file)
                st.write(f"  📊 {len(df)} rows loaded")
                st.write(f"  📋 Columns: {list(df.columns)}")
                
                if not df.empty:
                    st.dataframe(df.head(3))
                    
            except Exception as e:
                st.error(f"  ❌ Error reading {file.name}: {e}")
    else:
        st.warning("⚠️ No CSV files found in storage/logs")
else:
    st.error("❌ Storage/logs directory does not exist")

# Test import
st.subheader("📦 Import Test")
try:
    import yfinance as yf
    st.success("✅ yfinance imported successfully")
except Exception as e:
    st.error(f"❌ yfinance import failed: {e}")

try:
    import plotly.graph_objects as go
    st.success("✅ plotly imported successfully")
except Exception as e:
    st.error(f"❌ plotly import failed: {e}")

# Current working directory
st.subheader("📂 Working Directory")
st.write(f"Current directory: {os.getcwd()}")
st.write(f"Files in current directory:")
for item in os.listdir('.'):
    st.write(f"- {item}")
