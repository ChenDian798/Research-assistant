@echo off
cd /d "%~dp0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

set "LOG_DIR=%~dp0logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "OUT_LOG=%~2"
set "ERR_LOG=%~3"
if "%OUT_LOG%"=="" set "OUT_LOG=%LOG_DIR%\web_backend_%PORT%.out.log"
if "%ERR_LOG%"=="" set "ERR_LOG=%LOG_DIR%\web_backend_%PORT%.err.log"

echo Research Agent Web will write stdout to:
echo   "%OUT_LOG%"
echo Research Agent Web will write stderr to:
echo   "%ERR_LOG%"
echo.

set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
set "CODEX_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "PYTHON_CMD="

if exist "%PROJECT_PYTHON%" (
  set "PYTHON_CMD=%PROJECT_PYTHON%"
) else if exist "%CODEX_PYTHON%" (
  set "PYTHON_CMD=%CODEX_PYTHON%"
) else (
  set "PYTHON_CMD=python"
)

echo Using Python:
echo   "%PYTHON_CMD%"
echo.

"%PYTHON_CMD%" -c "import dotenv" 1>nul 2>nul
if errorlevel 1 (
  echo Missing Python dependency: python-dotenv
  echo Install dependencies with:
  echo   "%PYTHON_CMD%" -m pip install -r requirements.txt
  echo.
  echo Missing Python dependency: python-dotenv>>"%ERR_LOG%"
  echo Install dependencies with: "%PYTHON_CMD%" -m pip install -r requirements.txt>>"%ERR_LOG%"
  exit /b 1
)

"%PYTHON_CMD%" web_app.py %PORT% 1>>"%OUT_LOG%" 2>>"%ERR_LOG%"
