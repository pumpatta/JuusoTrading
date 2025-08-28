#!/usr/bin/env python3
"""
Launch script for JuusoTrader Dashboard
"""

import subprocess
import sys
from pathlib import Path
import os

def main():
    # Set working directory to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Set PYTHONPATH
    os.environ['PYTHONPATH'] = str(project_root)
    
    print("🚀 Käynnistetään JuusoTrader Dashboard...")
    print(f"📁 Työhakemisto: {project_root}")
    print("🌐 Dashboard käynnistyy selaimessa...")
    print("⏹️  Pysäytä Ctrl+C")
    print("-" * 50)
    
    try:
        # Run streamlit
        cmd = [
            sys.executable, 
            "-m", "streamlit", 
            "run", 
            "ui/trading_dashboard.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n📊 Dashboard pysäytetty!")
    except Exception as e:
        print(f"❌ Virhe dashboardin käynnistyksessä: {e}")
        print("Varmista että kaikki riippuvuudet on asennettu:")
        print("pip install streamlit plotly yfinance")

if __name__ == "__main__":
    main()
