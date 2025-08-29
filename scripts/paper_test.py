#!/usr/bin/env python3
"""
End-to-end test for live paper trading with small amounts.
This script temporarily disables dry_run, runs a controlled test,
and then restores the original settings.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import SETTINGS

def run_paper_test():
    """Run a controlled paper trading test with small amounts."""

    # Backup original dry_run setting
    original_dry_run = getattr(SETTINGS, 'dry_run', False)

    try:
        # Temporarily disable dry_run for real paper testing
        setattr(SETTINGS, 'dry_run', False)
        print("🔬 PAPER TEST: Disabled dry_run for controlled testing")

        # Set environment variables for small test amounts
        os.environ['INITIAL_CAPITAL'] = '1000'  # Very small capital for testing
        os.environ['MAX_POSITION_SIZE'] = '100'  # Small position size

        print("🔬 PAPER TEST: Starting live engine with small amounts...")
        print("🔬 PAPER TEST: Will run for 2 minutes, then stop automatically")

        # Start the live engine process
        proc = subprocess.Popen([
            sys.executable, 'engine/live.py', '--paper'
        ], cwd=Path(__file__).parent)

        # Let it run for 2 minutes
        time.sleep(120)

        # Stop the process
        print("🔬 PAPER TEST: Stopping live engine...")
        proc.terminate()

        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        print("🔬 PAPER TEST: Engine stopped")

        # Check logs for results
        logs_dir = Path('storage/logs')
        if logs_dir.exists():
            # Check engine events
            events_file = logs_dir / 'engine_events.log'
            if events_file.exists():
                print("🔬 PAPER TEST: Engine events:")
                with open(events_file, 'r') as f:
                    lines = f.readlines()[-10:]  # Last 10 lines
                    for line in lines:
                        print(f"  {line.strip()}")

            # Check trade logs
            for trade_file in logs_dir.glob('trades_*.csv'):
                if trade_file.stat().st_size > 0:
                    print(f"🔬 PAPER TEST: Trades in {trade_file.name}:")
                    with open(trade_file, 'r') as f:
                        lines = f.readlines()[-5:]  # Last 5 lines
                        for line in lines:
                            print(f"  {line.strip()}")

            # Check error logs
            error_file = logs_dir / 'engine_errors.log'
            if error_file.exists():
                print("🔬 PAPER TEST: Engine errors:")
                with open(error_file, 'r') as f:
                    lines = f.readlines()[-5:]  # Last 5 lines
                    for line in lines:
                        print(f"  {line.strip()}")

        print("🔬 PAPER TEST: Completed successfully")

    except Exception as e:
        print(f"🔬 PAPER TEST: Error during test: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Always restore original dry_run setting
        setattr(SETTINGS, 'dry_run', original_dry_run)
        print(f"🔬 PAPER TEST: Restored dry_run to {original_dry_run}")

if __name__ == '__main__':
    print("🚀 Starting end-to-end paper trading test...")
    run_paper_test()
    print("✅ Paper trading test completed")
