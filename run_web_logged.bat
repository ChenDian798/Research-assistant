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

set "CODEX_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%CODEX_PYTHON%" (
  "%CODEX_PYTHON%" web_app.py %PORT% 1>>"%OUT_LOG%" 2>>"%ERR_LOG%"
) else (
  python web_app.py %PORT% 1>>"%OUT_LOG%" 2>>"%ERR_LOG%"
)
