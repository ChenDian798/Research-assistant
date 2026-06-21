@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PORTS=%*"
if "%PORTS%"=="" set "PORTS=8000 8001 8002"

set "LOG_DIR=%~dp0logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%I in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"

for %%P in (%PORTS%) do call :START_ONE %%P

echo.
echo Started or opened ports: %PORTS%
echo Backend logs are written under "%LOG_DIR%" with the port in each filename.
echo Search audit logs are written as logs\search_audit_portPORT_*.json.
echo Annotation entries are appended to the annotation markdown file with the port in each section.
exit /b 0

:START_ONE
set "PORT=%~1"
set "URL=http://127.0.0.1:%PORT%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing '%URL%/health' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %ERRORLEVEL% EQU 0 (
  echo Port %PORT% is already running. Opening %URL%
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process '%URL%'"
  exit /b 0
)

set "OUT_LOG=%LOG_DIR%\web_backend_%PORT%_%STAMP%.out.log"
set "ERR_LOG=%LOG_DIR%\web_backend_%PORT%_%STAMP%.err.log"
echo Starting port %PORT%
start "Research Agent Server %PORT%" cmd /k call "%~dp0run_web_logged.bat" %PORT% "%OUT_LOG%" "%ERR_LOG%"
timeout /t 2 /nobreak >nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process '%URL%'"
exit /b 0