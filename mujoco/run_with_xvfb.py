#!/usr/bin/env python3
"""
Wrapper script to run MuJoCo simulation with XVFB
"""

import os
import subprocess
import sys

def main():
    # Set display environment variable
    os.environ['DISPLAY'] = ':99'
    
    # Start Xvfb in the background
    xvfb_process = subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'])
    
    try:
        # Wait a moment for Xvfb to start
        import time
        time.sleep(2)
        
        # Run the main simulation
        from headless_cdpr_xvfb import main as simulation_main
        simulation_main()
        
    finally:
        # Clean up Xvfb
        xvfb_process.terminate()
        xvfb_process.wait()

if __name__ == "__main__":
    main()