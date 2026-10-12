#!/bin/bash
# Double-click (or: bash run_on_mac.command). Creates a local venv, runs tests,
# then the full experiment (500 reps; ~5-15 min depending on cores).
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
python verify.py
python run_experiments.py --reps 500 --out results
echo
echo "Finished. Figures (PDF+PNG) and LaTeX tables are in the 'results' folder."
read -r -p 'Press Enter to close...' _
