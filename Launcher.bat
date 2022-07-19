@echo off & title %~nx0 & color 02

echo ---- DoE Launcher ----


echo Starting Frontend
start "" http://localhost:8080

echo Starting Server
cd DoE/Server/
python main.py
pause