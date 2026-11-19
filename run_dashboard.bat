@echo off
title AI Smart Traffic Management System 2030
echo =========================================================
echo   AI SMART TRAFFIC MANAGEMENT SYSTEM - 2030 EDITION
echo   Final Year Project - Computer Vision ^& Deep-MARL
echo =========================================================
echo.

if exist .venv\Scripts\streamlit.exe (
    echo Starting Streamlit Control Room on http://localhost:5000 ...
    .venv\Scripts\streamlit.exe run app.py --server.port 5000
) else (
    echo Starting with default python environment...
    streamlit run app.py --server.port 5000
)
pause
