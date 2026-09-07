@echo off
title NTRO Autonomous Signal Intelligence Workstation (SIH26147)
color 0B
cls

echo ==============================================================================
echo   NTRO Autonomous Signal Intelligence Workstation -- SIH26147
echo   National Technical Research Organisation (NTRO) / MoD
echo ==============================================================================
echo.

cd /d "%~dp0"

:: 0. Guard against running without extracting all project files
if exist "%~dp0app.py" goto :FILES_OK
echo ==============================================================================
echo   [ERROR] Project files not found in this folder!
echo ==============================================================================
echo.
echo You appear to be running START_DASHBOARD.bat directly inside the ZIP preview
echo or without extracting all files first.
echo.
echo Please follow these simple steps:
echo   1. Right-click the downloaded ntro_signal_analyzer.zip file
echo   2. Click "Extract All..." and extract to your Desktop or Documents
echo   3. Open the newly extracted folder
echo   4. Double-click START_DASHBOARD.bat again
echo.
echo ==============================================================================
pause
exit /b 1

:FILES_OK
set "PYTHON_CMD="

:: 1. Check if python is already working in PATH
python -c "import sys; sys.exit(0)" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=python"
    goto :PYTHON_READY
)

:: Check py launcher
py -3 -c "import sys; sys.exit(0)" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=py -3"
    goto :PYTHON_READY
)

:: Check standard system installation paths
if exist "C:\Python314\python.exe" (
    set "PATH=C:\Python314;C:\Python314\Scripts;%PATH%"
    set "PYTHON_CMD=python"
    goto :PYTHON_READY
)
if exist "C:\Python313\python.exe" (
    set "PATH=C:\Python313;C:\Python313\Scripts;%PATH%"
    set "PYTHON_CMD=python"
    goto :PYTHON_READY
)
if exist "C:\Python312\python.exe" (
    set "PATH=C:\Python312;C:\Python312\Scripts;%PATH%"
    set "PYTHON_CMD=python"
    goto :PYTHON_READY
)
if exist "C:\Python311\python.exe" (
    set "PATH=C:\Python311;C:\Python311\Scripts;%PATH%"
    set "PYTHON_CMD=python"
    goto :PYTHON_READY
)
if exist "C:\Python310\python.exe" (
    set "PATH=C:\Python310;C:\Python310\Scripts;%PATH%"
    set "PYTHON_CMD=python"
    goto :PYTHON_READY
)

:: Check user LocalAppData Python installations
for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do (
    if exist "%%D\python.exe" (
        set "PATH=%%D;%%D\Scripts;%PATH%"
        set "PYTHON_CMD=python"
        goto :PYTHON_READY
    )
)

:: 2. Automatic Silent Background Python Installation
echo [*] Python was not detected on this machine.
echo [*] Starting automatic zero-touch download and installation of Python 3.12...
echo [*] No manual steps required. Please wait while Python installs...
echo.

set "INSTALLER=%TEMP%\python-3.12.8-amd64.exe"

if exist "%SystemRoot%\System32\curl.exe" goto :DOWNLOAD_WITH_CURL

echo [*] Downloading Python installer via PowerShell...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe', '%INSTALLER%')"
goto :AFTER_DOWNLOAD

:DOWNLOAD_WITH_CURL
echo [*] Downloading Python installer via curl...
"%SystemRoot%\System32\curl.exe" -L -f -s -o "%INSTALLER%" https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe

:AFTER_DOWNLOAD
if exist "%INSTALLER%" goto :RUN_INSTALLER

echo.
echo ==============================================================================
echo   [ERROR] Could not download Python installer automatically.
echo   Please check your internet connection or install Python manually:
echo   https://www.python.org/downloads/
echo ==============================================================================
pause
exit /b 1

:RUN_INSTALLER
echo [*] Installing Python silently into user profile and configuring PATH...
start /wait "" "%INSTALLER%" /passive InstallAllUsers=0 PrependPath=1 Include_test=0 Include_pip=1 SimpleInstall=1

:: Scan newly created LocalAppData directory
for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do (
    if exist "%%D\python.exe" (
        set "PATH=%%D;%%D\Scripts;%PATH%"
        set "PYTHON_CMD=python"
    )
)

python -c "import sys; sys.exit(0)" >nul 2>&1
if %ERRORLEVEL% equ 0 goto :INSTALL_SUCCESS

echo.
echo ==============================================================================
echo   [ERROR] Python installation finished but could not be initialized.
echo   Please run the downloaded installer manually from:
echo   %INSTALLER%
echo   Make sure to check the box: "Add python.exe to PATH"
echo ==============================================================================
pause
exit /b 1

:INSTALL_SUCCESS
echo [*] Python 3.12 installed and configured successfully!
echo.

:PYTHON_READY
if "%PYTHON_CMD%"=="" set "PYTHON_CMD=python"

echo [*] Python environment ready:
%PYTHON_CMD% --version
echo.

:: 3. Verify and auto-install Python packages
echo [*] Verifying required DSP libraries...
%PYTHON_CMD% -c "import streamlit, scipy, numpy, plotly" >nul 2>&1
if %ERRORLEVEL% equ 0 goto :LIBRARIES_READY

echo [*] Installing required libraries from requirements.txt...
echo [*] This will take a moment on initial setup...
%PYTHON_CMD% -m pip install -r requirements.txt
if %ERRORLEVEL% equ 0 goto :LIBRARIES_READY

echo.
echo ==============================================================================
echo   [ERROR] Failed to install required Python libraries via pip.
echo   Please check your internet connection and run:
echo   %PYTHON_CMD% -m pip install -r requirements.txt
echo ==============================================================================
pause
exit /b 1

:LIBRARIES_READY
echo [*] All required DSP and visualization libraries are verified.
echo.

:: 4. Resolve Local WiFi IP address
set "WIFI_IP=127.0.0.1"
%PYTHON_CMD% -c "import socket; print(socket.gethostbyname(socket.gethostname()))" > "%TEMP%\_ntro_ip.txt" 2>nul
if not exist "%TEMP%\_ntro_ip.txt" goto :IP_RESOLVED
set /p WIFI_IP=<"%TEMP%\_ntro_ip.txt"
del "%TEMP%\_ntro_ip.txt" >nul 2>&1

:IP_RESOLVED
:: 5. Port collision handling
set "PORT=8501"
%PYTHON_CMD% -c "import socket; s=socket.socket(); s.connect(('127.0.0.1', 8501)); s.close()" >nul 2>&1
if not %ERRORLEVEL% equ 0 goto :PORT_OK
echo [!] Port 8501 is occupied by an existing process.
echo [!] Automatically switching to Port 8502...
set "PORT=8502"

:PORT_OK
echo ==============================================================================
echo   NTRO Autonomous Signal Intelligence Workstation is Running!
echo.
echo   LOCAL ACCESS on this laptop:
echo   -^> http://localhost:%PORT%
echo   -^> http://127.0.0.1:%PORT%
echo.
echo   NETWORK ACCESS for teammates on the same WiFi:
echo   -^> http://%WIFI_IP%:%PORT%
echo.
echo   KEEP THIS WINDOW OPEN to maintain the workstation server.
echo   Press Ctrl+C in this window when you want to stop the server.
echo ==============================================================================
echo.

:: Automatically open browser after 2 seconds
start "" /b cmd /c "ping 127.0.0.1 -n 3 >nul & start http://localhost:%PORT%"

:: Run Streamlit server bound to 0.0.0.0 for full local network access
%PYTHON_CMD% -m streamlit run app.py --server.port %PORT% --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false

echo.
echo ==============================================================================
echo   Workstation server has stopped.
echo ==============================================================================
pause
