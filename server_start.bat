@echo off
REM Activate the virtual environment first
call "D:\PreProcessor\A8---Contract-Atlas\venv\Scripts\activate.bat"

set FLASK_ENV=production
set URL_PREFIX=/preprocessor
echo Starting Contract Atlas in PRODUCTION mode...
python run.py

