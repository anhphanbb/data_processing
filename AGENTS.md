# Running the AWE pipeline

- Use the activated Conda environment `C:\Users\domin\.conda\envs\py310` for production runs on this machine.
- Launch batches with `run_pipeline_py310.cmd` and explicit year/month/start-step arguments. It uses `conda run` and checks TensorFlow GPU visibility before processing.
- Calling `py310\python.exe` directly does not activate the environment and can leave CUDA DLLs unavailable. Do not use that approach for production launches.
- For single-orbit runs, use `C:\ProgramData\anaconda3\Scripts\conda.exe run --prefix C:\Users\domin\.conda\envs\py310 --no-capture-output python run_awe_pipeline_orbit.py ...` after verifying GPU visibility.
- Before resuming a run, check process identity and completion receipts. Preserve verified completed products and select only unfinished orbits; never launch overlapping copies of the same work.
- Synthetic tests must use an isolated Numba cache and importable module names. Do not write test-module aliases into production caches.
