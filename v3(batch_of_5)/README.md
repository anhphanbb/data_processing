# AWE pipeline with bounded orbit batches

Run **run_awe_pipeline_batch.py** for the new batch version. It processes five
orbits through all six stages before starting the next batch, using up to 20
file workers at a stage. The original sequential runner remains available as
`run_awe_pipeline_orbit.py` and is also a required helper for the batch runner.

Extract the whole folder somewhere such as `D:\Github\data_processing_batch`.
Keep both runners and the four scientific processing scripts together.

## Settings

Edit the settings at the top of **run_awe_pipeline_batch.py**:

```python
YEAR = 2023
MONTH = 11
SOC_ROOT = Path(r'E:\soc')
ORBITS_AT_A_TIME = 5
WORKERS = 20
SAVE_L0D = False
SAVE_L1F = False
SAVE_L1R = False
CONTINUE_ON_ERROR = True
OVERWRITE = False
```

`WORKERS` is the total file-worker limit for the batch, not the number per orbit.
Five complete orbits supply 20 filter files to filtering, regridding and
coadding. The final stage has fewer tasks in some phases and may use fewer than
20 workers. Smaller batches also have fewer available tasks.

The batch runner has its own path/interpreter settings. Copy any custom paths
from your previous runner into this file. Its settings override the imported
sequential helper's settings when building child commands.

The two existing ML scripts and their models are still required:

- `D:\Github\solar_panels\4_and_5_mlspb.py`
- `D:\Github\cloud_restructured\4_and_5_mlcloud.py`

They remain in their original repositories; they are not included in this ZIP.
Relative model paths resolve from each ML script's own directory. Set their
paths in `SCRIPTS`, and set `PYTHON_EXE`/`ML_PYTHON_EXE` if using separate Conda
environments. Use the same scientific dependencies as your working pipeline.

## Run

In Spyder, open `run_awe_pipeline_batch.py`, edit its settings and run it.
Or, from a prompt in the extracted folder:

```bat
python run_awe_pipeline_batch.py --year 2023 --month 11
```

Explicit batch size and workers:

```bat
python run_awe_pipeline_batch.py --orbits-at-a-time 5 --workers 20
```

`--batch-size` is an alias for `--orbits-at-a-time`.

Other options:

```bat
python run_awe_pipeline_batch.py --limit 5
python run_awe_pipeline_batch.py --orbits 2 3 4 5 6
python run_awe_pipeline_batch.py --batch-size 1 --workers 4
python run_awe_pipeline_batch.py --batch-size 3 --workers 12
python run_awe_pipeline_batch.py --save-l0d --save-l1f --save-l1r
python run_awe_pipeline_batch.py --start-step 2
python run_awe_pipeline_batch.py --no-continue-on-error
python run_awe_pipeline_batch.py --work-root F:\awe_temp
python run_awe_pipeline_batch.py --dry-run
```

`--limit` counts pending orbit attempts including failures, excluding matching
completed orbits that are skipped. The last batch can be smaller than the
configured size. `--start-step 2` uses existing L0D q20 but still needs all four
original L0C filters. Input discovery is not recursive. Only one version and
timestamp per orbit is accepted; duplicate or incomplete L0C groups stop global
preflight. Dry run prints the batch commands without changing files.

## What happens in each batch

1. Run MLSPB for the batch's q20 files, loading its model once.
2. Filter / flat-field up to 20 filter files in parallel.
3. Regrid up to 20 files in parallel and create one master grid per orbit.
4. Run MLCloud for the batch's q20 files, loading its model once.
5. Coadd up to 20 files in parallel.
6. Produce L2A, L3A and L3C, then save completed orbits to the normal folders.

There is one ML process at a time. The batch version does not launch competing
TensorFlow processes for multiple orbits on the GPU. Failed-stage retries may
load a model again. No numerical or metadata changes were made to the four
scientific scripts; they are byte-for-byte copies of the supplied versions.

## Storage

| Product | Default behavior |
| --- | --- |
| Original L0C | Kept unchanged |
| L0D | Temporary; deleted after this orbit's four L1F outputs pass validation |
| L1F | Temporary; deleted after this orbit's four L1R outputs and master pass validation |
| L1R NetCDF | Temporary; deleted after this orbit's four L1A outputs pass validation |
| Master grid | Always saved under `l1r/YYYY/MM/MasterNNNNN.mgf` |
| L1C and L1A | Saved |
| L2A and L3A | Saved at 2 km and 10 km |
| L3C | Saved at 2 km |

The switches control retention; intermediate files must still be written while
the batch is being processed. Temporary files go under `SOC_ROOT/_orbit_work`
unless overridden. L0C is initially copied into the workspace to protect the
originals, then that temporary copy is removed after flat-fielding succeeds.

Temporary demand is approximately five orbits rather than one with the default
batch size, plus files retained from failed attempts. Final publication copies
one orbit's retained products at a time before deleting its staged copy; allow
space for that overlap. File publication is atomic per file, not an orbit-wide
transaction.

Already saved L0D/L1F/L1R from earlier runs are never automatically deleted.
Existing L0D used with `--start-step 2` is never overwritten or deleted,
regardless of the retention flag. L0C, L1C, L1A, L2A, L3A, L3C and masters still
accumulate; this is bounded temporary storage, not constant total storage.

## Failure handling and logs

`CONTINUE_ON_ERROR = True` is the default. If a batch stage reports an error,
the runner retries that stage once per affected orbit in separate workspaces.
This matters because a failed L2A task can otherwise prevent L3A/L3C from being
attempted for other good orbits in the same batch. A missing or unreadable
output after a successful child exit also triggers an isolated retry.

An orbit whose retry fails is removed from the active batch. Its remaining
files are moved under that batch's `failed/<orbit>` folder. Other successful
orbits continue to the next stage, and later batches still run. The retry starts
at the failed stage, not from L0C. Disk/filesystem errors that prevent safe
isolation or recording progress can still stop the run.

Use `--no-continue-on-error` to stop after a detected failure without the
isolated retries. A running batch child finishes/returns before its failures
are handled. Ctrl+C interrupts the whole run. If the run is interrupted, the
remaining workspace is retained. A later invocation starts failed orbits again
from the selected start step in a fresh workspace; it does not automatically
resume an older failed workspace.

Logs are under:

`SOC_ROOT/pipeline_logs/YYYY/MM/<run timestamp>/`

- `summary.json`: each orbit's status, error and relevant paths.
- `batch_NNNN/step_N.log`: shared stage output for the batch.
- `batch_NNNN/batch_status.json`: batch stage timings and elapsed time.
- `batch_NNNN/orbits/NNNNN/status.json`: orbit status and stage validation results.
- `batch_NNNN/orbits/NNNNN/validate_N.*`: output checks.
- `batch_NNNN/orbits/NNNNN/retry_step_N/`: isolated retry logs when needed.
- `batch_NNNN/reports/`: available shared stage timing/error CSVs.

Batch timings are shared wall times, not independent per-orbit processing
times; do not sum the same batch time once for each orbit. The final console
summary lists completed, failed/aborted and skipped counts. A completed run
with failures returns exit code 1. Failed workspaces can accumulate; inspect
and remove them yourself when no longer needed.

## Reruns and existing outputs

Matching completion records let this runner skip successful orbits when their
input metadata, script bytes, output metadata and relevant settings still match.
Changing batch size alone does not invalidate them. Changing workers currently
does, because the shared fingerprint includes that setting. External model and
calibration files and environment versions are not fingerprinted: explicitly
rerun after changing them.

**The new batch runner does not reuse the old sequential runner's completion
records.** Existing outputs without a matching batch record are reported as a
failed/conflicting orbit and left untouched. Select unprocessed orbits, or
intentionally reprocess using:

```bat
python run_awe_pipeline_batch.py --orbits 1 2 3 4 5 --overwrite
```

`--overwrite` reruns and replaces matching output filenames. Differently named
older outputs remain. If publishing stopped midway, use `--overwrite` on a
rerun to finish replacing that orbit's partial set of published products.

## Compatibility and testing

Both runners accept `awe_l0c_...` or `awe_l0d_...` inside L0D. If MLCloud keeps an
L1R filename inside L1C, only that generated output is renamed for coadding.
The step 4 check accepts `Processed_MLCloud` or `MLCloud`; raw cloud saving does
not need to be enabled. Validation opens each NetCDF, checks required variables
are nonempty, reads endpoints, and checks master-grid sizes. It is a structural
check, not an exhaustive scientific validation.

All 36 automated tests passed: the existing 25 tests plus 11 batch tests for
five-orbit scheduling, 20-worker command construction, smaller final batches,
cleanup and retention, failure isolation, retries, interruptions, L0D reuse,
publication failure and read-only dry runs. Batch tests use small stand-in files
and mocked scientific stages. A real multi-orbit run has not been tested here;
actual speedup and peak memory/disk use need measuring on your machine.

The `test_*.py` files are optional development checks. They do not run as part
of processing, and you do not need to run them. The scientific scripts, both
runners, and your external ML scripts/models are the files needed for use.

## Spyder import and overwrite settings

The batch runner loads `run_awe_pipeline_orbit.py` directly from its own folder,
even when Spyder's working directory is elsewhere. The helper file must still
be present beside the batch runner. Extract the whole ZIP; do not copy only
`run_awe_pipeline_batch.py`.

Set `OVERWRITE = True` near the top of the batch runner to intentionally
reprocess and replace matching output files, then run normally in Spyder.
The CLI `--overwrite` enables it and `--no-overwrite` disables it regardless
of the setting. With Spyder's runfile magic, use:

```python
%runfile D:/Github/data_processing/run_awe_pipeline_batch.py --args="--overwrite"
```
