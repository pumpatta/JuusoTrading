#!/usr/bin/env python3
"""
JuusoTrader - Stop Script
Cleanly stops all trading processes
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🛑 JuusoTrader - Stop Script")
    print("=" * 40)
    
    try:
        # Method 1: Kill by process name
        print("🔍 Stopping Python processes...")
        result = subprocess.run(
            ["taskkill", "/F", "/IM", "python.exe"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Python processes stopped")
        else:
            print("⚠️ No Python processes found to stop")
        
        # Method 2: Kill by specific PIDs if file exists
        pid_file = Path("trading_processes.txt")
        if pid_file.exists():
            print("🔍 Checking saved process IDs...")
            with open(pid_file, "r") as f:
                for line in f:
                    if "=" in line:
                        process_name, pid = line.strip().split("=")
                        try:
                            subprocess.run(
                                ["taskkill", "/F", "/PID", pid],
                                capture_output=True
                            )
                            print(f"✅ Stopped {process_name} (PID: {pid})")
                        except:
                            pass
            
            # Clean up PID file
            pid_file.unlink()
            print("🗑️ Cleaned up process file")
        
        # Method 3: Kill Streamlit specifically
        print("🔍 Stopping Streamlit processes...")
        subprocess.run(
            ["taskkill", "/F", "/FI", "WINDOWTITLE eq Streamlit*"],
            capture_output=True
        )
        
        print()
        print("✅ JuusoTrader stopped successfully!")
        print("💡 All trading processes terminated")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error stopping JuusoTrader: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
