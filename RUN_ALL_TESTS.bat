@echo off
title NTRO Test Suite (35 Tests) -- SIH26147
color 0E
cls
echo ==============================================================================
echo   Running All 35 Unit & Integration Tests...
echo ==============================================================================
echo.
cd /d "%~dp0"
python -m unittest discover tests
echo.
echo ==============================================================================
echo Tests complete. Press any key to close.
echo ==============================================================================
pause >nul
