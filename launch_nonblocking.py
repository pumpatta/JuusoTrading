#!/usr/bin/env python3
"""
JuusoTrader - Non-blocking launcher
Starts both live engine and dashboard without blocking terminal
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("🚀 JuusoTrader - Non-blocking Launcher")
    print("=" * 50)
    
    # Python executable path
    python_exe = project_dir / ".venv" / "Scripts" / "python.exe"
    
    if not python_exe.exists():
        print("❌ Virtual environment not found!")
        print("Run: python -m venv .venv && .venv\\Scripts\\activate && pip install -r requirements.txt")
        return 1
    
    try:
        # 1. Start live engine in background
        print("📈 Starting live trading engine...")
        live_process = subprocess.Popen(
            [str(python_exe), "engine/live.py", "--paper"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        )
        
        print(f"✅ Live engine started (PID: {live_process.pid})")
        
        # Wait a bit for engine to initialize
        print("⏰ Waiting 5 seconds for engine initialization...")
        time.sleep(5)
        
        # 2. Start dashboard in background  
        print("📊 Starting dashboard...")
        dashboard_process = subprocess.Popen(
            [str(python_exe), "launch_dashboard.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            text=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        )
        
        print(f"✅ Dashboard started (PID: {dashboard_process.pid})")
        print()
        print("🌐 Dashboard URL: http://localhost:8501")
        print("💡 Both processes running in separate windows")
        print()
        print("📋 Trading Accounts:")
        print("   Account A: EMA Strategy (30%)")
        print("   Account B: XGB ML Strategy (30%)")  
        print("   Account C: Enhanced ML + News (40%)")
        print()
        print("🛑 To stop trading:")
        print("   - Close the console windows, or")
        print("   - Run: taskkill /F /IM python.exe")
        print()
        print("✅ JuusoTrader launched successfully!")
        
        # Save process IDs for later cleanup
        with open("trading_processes.txt", "w") as f:
            f.write(f"live_engine={live_process.pid}\n")
            f.write(f"dashboard={dashboard_process.pid}\n")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error starting JuusoTrader: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
