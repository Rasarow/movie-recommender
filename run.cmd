@echo off
setlocal
set "taskPython=%~dp0.venv\Scripts\python.exe"
if not exist "%taskPython%" (
    echo Run setup.py with Python 3.12 first.
    exit /b 1
)
pushd "%~dp0"
if errorlevel 1 exit /b 1
if "%~1"=="" (
    "%taskPython%" main.py --help
) else (
    "%taskPython%" %*
)
set "taskExitCode=%ERRORLEVEL%"
popd
exit /b %taskExitCode%
