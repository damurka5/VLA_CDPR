#!/usr/bin/env python3
"""
Simple test script for headless CDPR simulation
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# from headless_cdpr import main as run_headless_simulation
from headless_cdpr import main as run_headless_simulation

if __name__ == "__main__":
    print("Starting Headless CDPR Simulation...")
    run_headless_simulation()