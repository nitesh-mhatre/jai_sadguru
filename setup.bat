@echo off
echo Setting up Jai Sadguru - NIFTY F&O Expert
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
echo.
echo Done! Run with:
echo   venv\Scripts\activate
echo   python main.py
