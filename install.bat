@echo off
setlocal
pushd "%~dp0" || goto :failed
set "UV_CACHE_DIR=%~dp0.cache\uv"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found. Install uv and add it to PATH.
    goto :failed
)

echo Installing CPU environment...
set "UV_PROJECT_ENVIRONMENT=.venv-cpu"
uv sync --locked --extra cpu
if errorlevel 1 goto :failed
".venv-cpu\Scripts\python.exe" -c "import einops, torch, onnxruntime as ort; assert not torch.version.cuda; assert 'CPUExecutionProvider' in ort.get_available_providers(); print('CPU:', torch.__version__, ort.__version__, 'einops:', einops.__version__)"
if errorlevel 1 goto :failed
".venv-cpu\Scripts\python.exe" -c "from image_dataset_studio.app import self_check; self_check()"
if errorlevel 1 goto :failed

echo Installing CUDA environment...
set "UV_PROJECT_ENVIRONMENT=.venv-cuda"
uv sync --locked --extra cuda
if errorlevel 1 goto :failed
".venv-cuda\Scripts\python.exe" -c "import einops, torch, onnxruntime as ort; assert torch.version.cuda, 'CUDA build of PyTorch is missing'; assert 'CPUExecutionProvider' in ort.get_available_providers(); assert 'CUDAExecutionProvider' in ort.get_available_providers(); print('CUDA:', torch.__version__, ort.__version__, 'einops:', einops.__version__, 'GPU available:', torch.cuda.is_available())"
if errorlevel 1 goto :failed
".venv-cuda\Scripts\python.exe" -c "from image_dataset_studio.app import self_check; self_check()"
if errorlevel 1 goto :failed

echo CPU and CUDA environments are ready. Use start.bat to launch the app.
popd
exit /b 0

:failed
echo Environment setup failed. Check the message above.
pause
popd
exit /b 1
