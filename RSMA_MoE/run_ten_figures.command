#!/bin/bash
# Double-click in Finder (or run ./run_ten_figures.command) to run all ten figures.
cd "$(dirname "$0")"
python3 run_ten_figures.py "$@"
echo
echo "Finished. Press Enter to close."
read
