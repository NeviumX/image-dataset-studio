@echo off
setlocal
pushd "%~dp0" || goto :failed
set "UV_CACHE_DIR=%~dp0.cache\uv"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found. Install uv and add it to PATH.
    goto :failed
)

if not exist ".venv-cpu\Scripts\image-dataset-studio.exe" call "%~dp0install.bat"
if errorlevel 1 goto :failed
if not exist ".venv-cuda\Scripts\image-dataset-studio.exe" call "%~dp0install.bat"
if errorlevel 1 goto :failed

set "ACCELERATOR=cpu"
".venv-cuda\Scripts\python.exe" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if not errorlevel 1 set "ACCELERATOR=cuda"

set "UV_PROJECT_ENVIRONMENT=.venv-%ACCELERATOR%"
uv sync --locked --extra %ACCELERATOR%
if errorlevel 1 goto :failed

start "" "%~dp0.venv-%ACCELERATOR%\Scripts\image-dataset-studio.exe" %*
if errorlevel 1 goto :failed

popd
exit /b 0

:failed
echo Could not start Image Dataset Studio.
pause
popd
exit /b 1
