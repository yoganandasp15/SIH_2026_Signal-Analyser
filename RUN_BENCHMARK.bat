@echo off
title NTRO Signal Benchmark (25 Signals) -- SIH26147
color 0A
cls
echo ==============================================================================
echo   NTRO 25-Signal Automated SigIDWiki Defense Benchmark
echo ==============================================================================
echo.
cd /d "%~dp0"
python scripts\benchmark_13_signals.py
echo.
echo ==============================================================================
echo Benchmark complete. Press any key to close.
echo ==============================================================================
pause >nul
