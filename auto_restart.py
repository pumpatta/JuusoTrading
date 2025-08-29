#!/usr/bin/env python3
"""
Auto-restart script for the live trading engine
This script will automatically restart the engine if it crashes or gets terminated
"""

import subprocess
import time
import sys
import os
from datetime import datetime

def log_message(message):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def main():
    """Main auto-restart loop"""
    restart_count = 0
    max_restarts = 10  # Maximum restarts per hour
    restart_window = 3600  # 1 hour in seconds
    restart_times = []

    log_message("🚀 Starting JuusoTrader Auto-Restart Service")
    log_message("📊 Trading across 113 instruments with multi-account support")

    while True:
        try:
            # Clean up old restart times
            current_time = time.time()
            restart_times = [t for t in restart_times if current_time - t < restart_window]

            # Check if we've exceeded max restarts
            if len(restart_times) >= max_restarts:
                log_message(f"⚠️  Too many restarts ({max_restarts}) in {restart_window/3600:.1f} hours")
                log_message("⏸️  Pausing for 1 hour before continuing...")
                time.sleep(3600)
                restart_times = []
                continue

            log_message(f"🔄 Starting live engine (attempt {restart_count + 1})")
            restart_times.append(current_time)
            restart_count += 1

            # Start the live engine
            process = subprocess.Popen([
                sys.executable, "engine/live.py", "--paper"
            ], cwd=os.path.dirname(os.path.abspath(__file__)))

            # Wait for the process to complete
            return_code = process.wait()

            if return_code == 0:
                log_message("✅ Engine exited cleanly")
                break
            else:
                log_message(f"❌ Engine exited with code {return_code}")
                log_message("🔄 Restarting in 30 seconds...")
                time.sleep(30)

        except KeyboardInterrupt:
            log_message("🛑 Received keyboard interrupt")
            log_message("👋 Shutting down auto-restart service")
            break
        except Exception as e:
            log_message(f"💥 Auto-restart error: {e}")
            log_message("🔄 Restarting in 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    main()
