@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Starting Servers...
start cmd /k "uvicorn api:app --host 0.0.0.0 --port 8000 --reload"
start cmd /k "python app.py"

echo Application started!
echo Frontend: http://localhost:5000
echo Backend: http://localhost:8000/docs
pause
