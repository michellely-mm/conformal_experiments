@echo off
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --quiet -r requirements.txt
python verify.py
python run_experiments.py --reps 500 --out results
echo Finished. See the results folder.
pause
