@echo off
setlocal
rem Activate py310 through conda so TensorFlow can find the CUDA DLLs.
set "AWE_CONDA=C:\ProgramData\anaconda3\Scripts\conda.exe"
set "AWE_ENV=C:\Users\domin\.conda\envs\py310"
if not exist "%AWE_CONDA%" (
  echo ERROR: Conda not found at %AWE_CONDA%
  exit /b 1
)
rem Fail before processing if this machine's expected GPU is unavailable.
call "%AWE_CONDA%" run --prefix "%AWE_ENV%" --no-capture-output python -c "import sys,tensorflow as tf; print('Python:',sys.executable); g=tf.config.list_physical_devices('GPU'); print('GPUs:',g); sys.exit(0 if g else 1)"
if errorlevel 1 (
  echo ERROR: py310 GPU preflight failed. No pipeline was started.
  exit /b 1
)
call "%AWE_CONDA%" run --prefix "%AWE_ENV%" --no-capture-output python -u "%~dp0run_awe_pipeline_batch.py" %*
exit /b %errorlevel%
