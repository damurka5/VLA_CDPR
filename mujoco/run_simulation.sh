#!/bin/bash
# run_simulation.sh

# Start virtual display
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# Run your simulation
python headless_cdpr_xvfb.py

# Kill Xvfb when done
pkill -f Xvfb