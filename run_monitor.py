import subprocess
import sys
from datetime import datetime

def run_monitor():
    while True:
        try:
            print(f"\nStarting monitor at {datetime.now()}")
            # Run the monitor script
            process = subprocess.Popen([sys.executable, 'website_monitor.py'])
            process.wait()
        except Exception as e:
            print(f"Monitor crashed: {str(e)}")
            print("Restarting...")

if __name__ == "__main__":
    run_monitor() 