# Folder layout

## SSH input staging (isolated development branch)

The batch runner can download only the remote `v02` inputs needed by its active
batches, verify SHA-256, and remove raw inputs after successful output publication.
See [SSH_INPUTS.md](SSH_INPUTS.md) for January 2024 commands, SSH key/agent setup,
disk requirements, and interrupted-transfer recovery. Use an activated `py310`
environment; `run_pipeline_py310.cmd` includes a GPU preflight (adjust its local
Conda paths on another computer). These changes are isolated from any running
pipeline in the original checkout.

The current pipeline scripts remain at the repository root. Historical versions
are preserved under `archive/`:

- `archive/v1(steps_in_sequence)/`
- `archive/v2/`
- `archive/v3(batch_of_5)/`
- `archive/v4(before_filling_gaps)/`
- `archive/v5(handling_duplicates)/`
- `archive/v6(bad_frames_removal)/`

Archived scripts, tests, documentation and reports retain their original contents;
their historical settings apply only to those snapshots. Generated logs, temporary
products and model weights stay local.

Both runners default to November 2023 with server publishing and L1R retention
enabled. L1R (including master grids), L1A, L3A and L3C are published under
`Y:\soc\<level>\version02p1\2023\11` (L3A retains its resolution subfolders).
Raw L0C inputs still come from `Y:\soc\l0c\version02\2023\11`; filenames retain
their `v02` suffix. The standalone copy script also includes L1R and defaults to
`version02p1` for November 2023.

MLSPB-flagged pixels are retained in L1F, L1R and L1C Radiance. The compressed
`MLSPB_Pixel_Mask` variable follows Radiance dimensions: 1 excludes a sample,
0 leaves it eligible (normal geometry, missing-data and brightness checks still apply).
Filtering and normalization of clean pixels still exclude flagged input. Added
frames retain separately processed values inside flagged boxes; the trim overlap
is copied exactly. Intrinsically missing pixels remain missing.
Regridding marks a target pixel if any of its four source corners is flagged.
MLCloud masks its inference copy, and coadding excludes flagged samples before
averaging. L1A consumes the mask rather than copying a center-frame mask onto an
average. Older NaN-masked products remain supported. Regenerate from L1F onward
to obtain retained pixels; existing output files are not updated automatically.

The active runners use the ML scripts and config.txt beside them. Model weights
remain local in models/; models/manifest.json records the expected hashes.
Glare annotations default to csv/all_months.csv; combine_monthly_csv.py rebuilds it.

# AWE pipeline: selective retries and three-batch scheduling

Replace both runners together: `run_awe_pipeline_batch.py` and
`run_awe_pipeline_orbit.py`. The other five Python scripts in this package are
unchanged copies of your supplied versions. Keep their canonical filenames.
Step 0, MLSPB and MLCloud scripts/models remain external at the paths in SCRIPTS.

Review YEAR, MONTH, SOC_ROOT, GLARE_CSV and save flags before running. The supplied
YEAR=2023, MONTH=11, SAVE_L0C=True and OVERWRITE=True are preserved.
OVERWRITE=True replaces matching saved outputs. This is an opt-in run, not a
measured speedup: benchmark the same inputs before choosing permanent settings.

## Defaults and per-stage workers

```python
OVERLAP_BATCHES = True
MAX_PROCESSING_BATCHES = 3
ORBITS_AT_A_TIME = 10
PREFETCH_STEP0 = True
WORKERS = 20
STEP_WORKERS = {2: None, 3: None, 5: None, 6: None}
```

None inherits WORKERS / --workers. For example, set STEP_WORKERS[2] = 12 to
use 12 file workers for filtering. No unmeasured worker reduction is enabled
by default. The command-line equivalent is:

```text
python run_awe_pipeline_batch.py --workers 20 --step-2-workers 12 --step-3-workers 8
```

Use --step-5-workers and --step-6-workers for coadding and final products.
Per-stage worker settings are included in completion fingerprints.

There is still only one CPU stage, one ML stage, one setup task and one
preparation task at a time. Three batches may be processing, plus at most one
preparing/prepared batch. This needs more temporary storage than two batches.
The ML processes still load their models per batch; persistent models and
compression changes are not part of this revision.

To compare with the previous admission limit:

```text
python run_awe_pipeline_batch.py --max-processing-batches 2
```

--no-overlap-batches retains sequential batch execution with optional Step 0
prefetch. Oldest ready batches retain priority. Preparation can still use CPU
and disk while a CPU stage runs. The scheduler's CPU slot is not hardware CPU
utilization. Workspaces remain isolated per batch.

## Selective retries

A failed batch subprocess no longer automatically retries every orbit.
The runner checks each orbit's complete required output set. Valid fresh
outputs continue; missing or invalid outputs trigger one isolated retry of
that orbit's stage. Step 6 checks every configured L2A/L3A resolution and L3C.
A successful exit with invalid outputs also triggers a retry.

Before each batch or retry attempt, any existing stage outputs for that orbit
are moved into `previous_attempts/step_N/batch/` or `previous_attempts/step_N/retry/`
inside the batch workspace. This includes partial files and alternate ML output
names. Stage output directories therefore start empty for that orbit: stale
products cannot satisfy validation. Step 6 retains its input master grid.
The files are moved, not copied, and are preserved for diagnosis if the batch
fails or is interrupted. They are removed with the workspace when the entire
batch completes. Partial retry outputs still require space until then.

The existing validation checks required variables, nonempty arrays, readable
endpoints and master-grid structure. It does not perform a full scientific
quality audit of every pixel. Scientific screening and formulas are unchanged.
Fail-fast and cancellation behavior are preserved.

## Timing

`scheduler_timing.json` still records total wall time, stage resource waits and
task durations. Each `batch_status.json` stage now records:

- batch_process_seconds: initial input check and subprocess duration.
- validation_seconds: initial per-orbit output validation and MLCloud renaming.
- retry_seconds: isolated retry work, including its validation and failure handling.
- retry_count and validation_failures.
- retained_after_batch_error: orbits accepted without repeating the stage.
- archived_previous_files: files isolated before the initial attempt.
- workers: resolved CPU stage worker count (null for ML stages).
- stage_total_seconds: stage execution through orbit cleanup, before report
  archiving and the final status write.

These durations do not separate model startup from inference. Other overhead
means the timing components need not sum exactly to the stage total.
A completed scheduler task can contain individual orbit failures; summary.json
remains authoritative. Do not sum overlapping task durations as wall time.

## Validation performed for this revision

Python compilation and eight simulated tests passed: mixed batch success/failure,
stale-output isolation, partial Step 6 output, corrupt output despite exit zero,
persistent retry failure, cancellation, fail-fast behavior, and worker settings
including CLI parsing. Run the bundled checks with:

```text
python tests/test_runner_update.py
```

The tests simulate processing and file validation; they do not run TensorFlow,
real NetCDF processing, or the Windows data paths. Full data validation and timing
must be done on your machine. No scientific processing script was modified.

---

# AWE boxes mode: preserve trim, then extend

This revision replaces the previous longest-clean-interval flat-field method.
It first runs the ORIGINAL trim algorithm on the original selected trim interval.
It captures that exact flat field, copies the processed trim Radiance into the
longer output, and computes only the extra frames using the captured flat field.

## What is preserved

For the same L0C inputs, duplicate handling, selection mode and numerical engine,
L1F Radiance within the trim interval is copied exactly, with one explicit
exception: MLSPB-flagged detector boxes are set to NaN. All other values, including
existing NaNs, remain the same. No filtering, normalization or quantization is
repeated on those copied values. Coordinates and timestamps come from the same
original frames. Synthetic tests compare all four filters with exact equality,
not a numeric tolerance.

The calibration is the original arithmetic: original image filtering, original
zero handling, original number-of-frames denominator, accumulation order and
central normalization. Masked-frame processing and valid-sample averaging are
NOT substituted into that calculation. The flat field is frozen during extension.

`--selection-mode cxx` (default) reproduces historical trim boundaries, including
any MLSPB-positive boundary or internal frames that the original trim selection
retains. These also contribute to the original flat field, intentionally: dropping
them would change the baseline. If you choose `--selection-mode clean`, use it for
BOTH trim and boxes comparisons. That option trims flagged ends, not internal
flagged frames. This revision preserves a baseline; it does not assert that the
legacy calibration is scientifically optimal.

## Extra frames

Only frames outside the original trim interval use the mask-aware bright/dark
filter and normalization. Flagged boxes are excluded from their calculations.
The original flat field is used unchanged. The valid central 20x20 region sets
the scale; if it has no usable pixels, the valid footprint supplies the fallback.
Unsupported pixels remain NaN. No temporal interpolation or gap filling is added.

## Settings

Keep `MLSPB_MODE = 'boxes'` in the runner; use `'trim'` for a baseline run.
`MIN_CLEAN_FRAMES = 100` / `--min-clean-frames` is retained as a compatibility
setting. It now means a minimum count of fully unflagged frames INSIDE the original
trim interval. It does not select a different interval, split at gaps, or change
the flat-field denominator. The minimum is an operational guard, not a calibrated
scientific threshold.

## Files to replace

Replace these scientific scripts together:
- filter_flatfield_to_1lf.py
- regrid_l1f_to_l1r.py
- coadd_to_l1a.py
- generate_l2a_l3a_l3c.py
- 4_and_5_mlcloud.py (at the path used by step 4, normally
  D:\Github\cloud_restructured\4_and_5_mlcloud.py)

Only flat-field processing changes numerically in this revision. The other four
scripts now also recognize `MLSPB_Mask_Mode = boxes_trim_reference` and retain
the previously added missing-data handling. They also recognize the older
`boxes_clean_reference` products. Both runners are included unchanged; keep your
own month/path settings and models. Existing step-0 glare and MLSPB models remain
external. Re-run from step 2 with matching L0C/L0D, or run the full pipeline.
Use separate output trees for comparisons; longer filenames/timestamps may leave
old differently named products alongside new ones if you reuse a folder.

## Diagnostics and limits

`Flat_Field_Normalization(time)` codes:
- 0: no usable support in an added frame
- 1: central normalization in an added frame
- 2: footprint fallback in an added frame
- 3: copied original trim result, then explicit box mask applied

The reference interval, Epoch endpoints and policy are saved as global attributes.
Existing q20-only conditional quality/gap/duplicate reports remain in place.
A temporary trim L1F is written during processing and removed afterward. Allow
additional temporary disk space for one trimmed file per active worker.

The exact-preservation guarantee applies at L1F for unmasked pixels, relative to
the trim calculation in this supplied script. L3C need not be identical: extending
the track changes master-grid construction, coadding windows and swath overlaps.
This change deliberately does not force two different orbit extents onto the same
master grid. Compare same-Epoch detector pixels at L1F first when diagnosing bias.

Validation: numerical checks plus synthetic four-filter processing through L3A
and L3C, including exact trim-overlap equality, duplicate removal and preserved
gaps. Cloud inference uses a mock, not the trained model. Real orbit validation
and performance measurements remain to be done on your data.

---

## Existing batch runner guide

The earlier runner guide follows for storage and orchestration details. The
processing and validation descriptions above supersede earlier compatibility
claims below.

# AWE pipeline with bounded orbit batches

Run **run_awe_pipeline_batch.py** for the new batch version. It processes five
orbits through copy/glare removal and all six later stages before starting the next batch, using up to 20
file workers at a stage. The original sequential runner remains available as
`run_awe_pipeline_orbit.py` and is also a required helper for the batch runner.

Extract the whole folder somewhere such as `D:\Github\data_processing_batch`.
Keep both runners, `0_copy_and_remove_glare.py`, and the four scientific processing scripts together.

## Settings

Edit the settings at the top of **run_awe_pipeline_batch.py**:

```python
YEAR = 2023
MONTH = 11
SOC_ROOT = Path(r'E:\soc')
ORBITS_AT_A_TIME = 10
WORKERS = 20
START_STEP = 0
L0C_SOURCE_ROOT = Path(r'Y:\soc\l0c\version02')
GLARE_CSV = None
SAVE_L0C = False
SAVE_L0D = False
SAVE_L1F = False
SAVE_L1R = True
CONTINUE_ON_ERROR = True
OVERWRITE = True
```

`START_STEP = 0` reads raw files from `L0C_SOURCE_ROOT/YYYY/MM`. It writes
trimmed, glare-removed copies directly into the temporary workspace before MLSPB.
Set `SAVE_L0C = True` (or `--save-l0c`) to publish those copies to
`SOC_ROOT/l0c/YYYY/MM` after the orbit completes. With the default `False`,
they are deleted after flat-fielding passes validation. Raw source files are
always preserved. Existing saved L0C files are not deleted when saving is off.

`GLARE_CSV = None` selects `csv/IndividualMonths/glare_intervals_Nov2023.csv`
relative to the runners for November 2023, with the month/year adjusted to
`YEAR`/`MONTH`. Set an explicit path or use `--glare-csv` for another CSV.
The CSV must contain `Orbit #`, `glare_initial`, and `glare_final`.
The first and last 10 frames are trimmed, then glare intervals are applied as
zero-based, inclusive indices in that trimmed sequence. This preserves the
supplied glare-removal algorithm. Step 0 requires numpy, pandas, and netCDF4.

Use `START_STEP = 1` / `--start-step 1` for existing processed L0C under
`SOC_ROOT/l0c/YYYY/MM`, avoiding a second trim. Step 2 also uses existing L0D.
For either starting point, existing L0C files are only copied into temporary
storage and never republished, regardless of `SAVE_L0C`.

Both runners support the same source, CSV, and retention options:

```bat
python run_awe_pipeline_batch.py --year 2023 --month 12 --save-l0c
python run_awe_pipeline_batch.py --l0c-source-root Y:\soc\l0c\version02 --glare-csv D:\Github\data_processing\csv\IndividualMonths\glare_intervals_Dec2023.csv --year 2023 --month 12
python run_awe_pipeline_batch.py --start-step 1
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

Step 0 prepares each selected orbit, with up to four filter workers. At most one additional batch is preparing or waiting beyond the processing limit. Exact source paths are passed to the copy
script, so it does not process the rest of the source month. Each orbit has its
own `step_0.log` and validation record. A preparation failure is isolated and
logged; it is not automatically retried. Successful orbits then enter:

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
| Raw source L0C | Kept unchanged |
| Processed L0C | Temporary; save with `SAVE_L0C = True` or `--save-l0c` |
| L0D | Temporary; deleted after this orbit's four L1F outputs pass validation |
| L1F | Temporary; deleted after this orbit's four L1R outputs and master pass validation |
| L1R NetCDF | Temporary; deleted after this orbit's four L1A outputs pass validation |
| Master grid | Always saved under `l1r/YYYY/MM/MasterNNNNN.mgf` |
| L1C and L1A | Saved |
| L2A and L3A | Saved at 2 km and 10 km |
| L3C | Saved at 2 km |

The switches control retention; intermediate files must still be written while
the batch is being processed. Temporary files go under `SOC_ROOT/_orbit_work`
unless overridden. Step 0 writes processed L0C directly into the workspace; steps 1/2 copy existing
L0C there. Unsaved temporary L0C is removed after flat-fielding succeeds.

Temporary demand is at most three five-orbit batches (two processing, one preparing/ready)
with the default overlap settings, plus files retained from failed attempts. Final publication copies
one orbit's retained products at a time before deleting its staged copy; allow
space for that overlap. File publication is atomic per file, not an orbit-wide
transaction.

Already saved L0C/L0D/L1F/L1R from earlier runs are never automatically deleted.
Existing L0D used with `--start-step 2` is never overwritten or deleted,
regardless of the retention flag. Retained L0C, L1C, L1A, L2A, L3A, L3C and masters still
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
The step-0 CSV path and contents are included in the completion fingerprint.
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

All 44 orchestration tests passed, including new step-0 coverage for saved and
discarded L0C, source preservation, preparation failure isolation, and read-only
dry runs. Tests use small stand-in files and mocked scientific stages. A real
AWE/NetCDF processing run has not been tested here; speed and peak memory/disk
use need measuring on your machine.

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
