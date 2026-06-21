@echo off
cd /d "%~dp0"
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"
set "URL=http://127.0.0.1:%PORT%"
set "LOG_DIR=%~dp0logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing '%URL%/health' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %ERRORLEVEL% EQU 0 (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process '%URL%'"
  exit /b 0
)

for /f %%I in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"
set "OUT_LOG=%LOG_DIR%\web_backend_%PORT%_%STAMP%.out.log"
set "ERR_LOG=%LOG_DIR%\web_backend_%PORT%_%STAMP%.err.log"

start "Research Agent Server %PORT%" cmd /k call "%~dp0run_web_logged.bat" %PORT% "%OUT_LOG%" "%ERR_LOG%"
timeout /t 2 /nobreak >nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process '%URL%'"