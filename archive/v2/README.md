# AWE pipeline — one orbit at a time

This is a separate version based on the five supplied working scripts. Extract
this folder anywhere, for example `D:\Github\data_processing_orbit`.
Run **run_awe_pipeline_orbit.py** to get the storage-saving behavior.

The four processing scripts are byte-for-byte copies of the supplied versions.
Their scientific calculations, coordinates, metadata and pixel timestamps have
not been edited. The new runner controls orbit selection, paths, file retention,
validation and restart records. It uses four file workers by default.

## Setup

Edit the settings at the top of `run_awe_pipeline_orbit.py`:

```python
YEAR = 2023
MONTH = 11
SOC_ROOT = Path(r'E:\soc')
SAVE_L0D = False
SAVE_L1F = False
SAVE_L1R = False
START_STEP = 1
CONTINUE_ON_ERROR = True
WORKERS = 4
```

The two existing ML scripts are still required:

- `D:\Github\solar_panels\4_and_5_mlspb.py`
- `D:\Github\cloud_restructured\4_and_5_mlcloud.py`

They were not included in the attachments, so they are not included in this
archive. Set their paths in `SCRIPTS` if necessary. The runner uses the same ML
bridge as the original runner: it overrides `INPUT_FOLDER`, `OUTPUT_FOLDER`,
`CSV_OUTPUT_FOLDER`, and `OVERWRITE`, then calls `main()`. Relative model paths
are resolved from each ML script's own folder. It assumes the supplied working
runner's ML interface is still valid.

Use your existing processing environment. `PYTHON_EXE` and `ML_PYTHON_EXE` can
point to different Anaconda environments. NetCDF validation uses `PYTHON_EXE`,
which must have netCDF4. Existing scientific dependencies are still required.

## Run

In Spyder, open the new runner, edit its settings, and run it. In a prompt:

```bat
python run_awe_pipeline_orbit.py --year 2023 --month 11
```

It runs all six steps for one orbit before starting the next orbit:

1. MLSPB: L0C → L0D
2. Filter / flat-field: L0C + L0D → L1F
3. Regrid: L1F → L1R + master grid
4. MLCloud: L1R q20 → L1C q20
5. Coadd: L1R p12/p14/bkg + L1C q20 → L1A
6. Temperature / swaths: L1A + master grid → L2A / L3A / L3C

For a first actual run, choose one orbit that is present in your input folder:

```bat
python run_awe_pipeline_orbit.py --orbits 142
```

Other options:

```bat
python run_awe_pipeline_orbit.py --limit 10
python run_awe_pipeline_orbit.py --save-l0d
python run_awe_pipeline_orbit.py --save-l1f
python run_awe_pipeline_orbit.py --save-l1r
python run_awe_pipeline_orbit.py --save-l1f --save-l1r
python run_awe_pipeline_orbit.py --start-step 2
python run_awe_pipeline_orbit.py --work-root F:\awe_temp
python run_awe_pipeline_orbit.py --dry-run
```

`--start-step 2` uses existing matching L0D q20 files and runs steps 2–6.
The corresponding four L0C files are still required. Existing L0D is not rewritten
in this mode. `--limit` counts pending orbit attempts (including failures), excluding
completed orbits that are skipped. Dry run lists commands without loading ML or
scientific dependencies and does not change any files.

Only one version and timestamp per orbit is accepted in each month's L0C folder;
all four filters must be present. Duplicate or incomplete groups stop preflight.
Input discovery is not recursive.

## Storage behavior

| Product | Default behavior |
| --- | --- |
| Original L0C | Kept unchanged in the original folder |
| L0D | Temporary; removed after all four L1F outputs pass validation. Keep with `--save-l0d`. |
| L1F | Temporary; removed after all four L1R outputs and master pass validation |
| L1R NetCDF | Temporary; removed after all four L1A outputs pass validation |
| Master grid | Always saved as `l1r/YYYY/MM/MasterNNNNN.mgf` |
| L1C | Saved |
| L1A | Saved |
| L2A | Saved at 2 km and 10 km |
| L3A | Saved at 2 km and 10 km |
| L3C | Saved at 2 km |

The options control **retention**, not whether L0D/L1F/L1R are written at all.
The existing stages still need these files on disk while processing the current
orbit. Keeping the file interfaces avoids changing the scientific pipeline.

Temporary files go under `SOC_ROOT/_orbit_work` by default, with a unique folder
for each attempted orbit. Four L0C files are initially copied there to isolate
ML processing from the originals; that temporary L0C copy is removed after
flat-fielding succeeds. The temporary workload is bounded to one active orbit,
not a whole month's L1F/L1R. Final products are copied to their normal month
folders only after the orbit finishes successfully. A short-lived extra copy of
retained products is needed during publication, including when the temporary
folder is on another drive. Publication uses an atomic replacement per file,
not a transaction across the whole orbit.

This reduces intermediate storage but does not make total storage constant:
L0C, L1C, L1A, L2A, L3A, L3C and master grids still accumulate; L0D also
accumulates if `SAVE_L0D = True`. It does not
download or delete L0C. It also does not delete pre-existing L0D/L1F/L1R files from
older runs; the retention switches apply to files generated in this run.

## Failures and reruns

Every stage must exit successfully and produce all expected orbit/filter files.
The runner opens each NetCDF output, verifies required variables are nonempty,
and reads their endpoints. It checks the master header and full byte count.
These are structural checks, not a full scientific comparison or an exhaustive
check of every compressed data chunk.

By default, a processing, validation or saving failure is logged and the runner
continues to the next orbit. It keeps the remaining temporary files and prints
their location. These retained files can accumulate when several orbits fail.

Set `CONTINUE_ON_ERROR = False`, or run with `--no-continue-on-error`, to stop on
the first failure. `--continue-on-error` explicitly enables continuation.
Ctrl+C always stops the whole run.

The terminal prints completed/failed/skipped counts and failed orbit numbers at
the end. A run-level `summary.json` in the timestamped log folder records each
orbit's status, error, log path and temporary path when created; it is updated
after each orbit. A run that continues but has any failures returns exit code 1,
so automation can still detect an incomplete run.

Global preflight errors (such as missing scripts, incomplete/ambiguous L0C
input groups, or missing L0D inputs for start step 2) still stop before the orbit
loop. The continuation option applies to errors within that loop. Files already
removed after a successful downstream stage are not restored. Logs, validation
specifications, available stage timing CSVs and status are stored under:

`SOC_ROOT/pipeline_logs/YYYY/MM/<run timestamp>/<orbit>/`

Successful orbits have a completion record under
`SOC_ROOT/pipeline_orbit_state/YYYY/MM/`. On another run, they are skipped when
the input paths/sizes/modification times, script contents, settings and recorded
output file sizes/modification times match. Outputs are not re-read in full to
skip an orbit. External ML model files, environments and calibration files are
not fingerprinted: use `--overwrite` after changing them.

A failed orbit restarts from the selected start step in a new temporary folder;
it does not resume halfway through the failed orbit. Inspect or remove that
failed attempt's temporary folder yourself when it is no longer needed. The
runner does not remove older failed workspaces automatically.

Existing outputs without a matching completion record mark that orbit as failed
before processing; continuation follows the setting above. To intentionally rerun them:

```bat
python run_awe_pipeline_orbit.py --orbits 142 --overwrite
```

This reruns the selected orbit and replaces matching output filenames. If a new
run produces a different filename (for example, a different start timestamp),
the old differently named output remains and must be managed separately. For
comparison runs, using a separate SOC root with a copy of the relevant L0C is
preferable. If saving failed midway, rerun with `--overwrite` to finish replacing
the partial set of published files.

## Performance and validation

One orbit at a time is suitable for this pipeline because its calculations are
orbit-local. The four filters can still run in parallel. It may be slower than
processing many orbits together: each ML script reloads its model per orbit,
and subprocess startup repeats. Measure actual elapsed time with one or a few
orbits on your machine before scaling up.

Validation performed for this package:

- Syntax compilation of all supplied and new Python files.
- Exact byte comparison of the four processing copies with the attachments.
- Twenty-one orchestration tests covering cleanup, all retention combinations,
  failure before cleanup, failed validation, existing L0D reuse, duplicate or
  missing filters, missing final products, completion records, saving failure,
  continuing after errors, stopping when disabled, Ctrl+C, failed-attempt limits, both ML naming
  conventions, duplicate rejection, existing L0D lookup, L0D retention,
  cleanup after flat-fielding, and preservation after flat-fielding failures.
- Dry-run command construction using a temporary four-filter input group.

The orchestration tests use tiny stand-in files and mocked scientific stages.
A real orbit was not processed here: the L0C data, ML scripts/models and the
full processing environment were not supplied. Numerical equivalence has not
been measured for this runner.

To rerun the orchestration tests:

```bat
python -m unittest discover -s . -p "test_*.py" -v
```

## ML output filename compatibility

MLSPB may save `awe_l0c_q20_...nc` inside the L0D folder. The runner accepts
either that name or `awe_l0d_q20_...nc`, including with `--start-step 2`, and
still requires the MLSPB variable during validation. It rejects duplicates
rather than choosing between two files for the same orbit.

If MLCloud similarly keeps an `awe_l1r_q20_...nc` name inside the temporary
L1C folder, the runner renames that generated output to `awe_l1c_q20_...nc`
before coadding, because the coadding script requires an L1C filename. The
original L1R file and the NetCDF contents are unchanged.

## Optional L0D retention

`SAVE_L0D = False` is the default. MLSPB writes L0D inside the current orbit's
temporary folder. Once flat-fielding succeeds and all four L1F files pass
validation, that temporary L0D folder is removed. If flat-fielding fails, L0D
remains available in the failed workspace.

Set `SAVE_L0D = True` or pass `--save-l0d` to keep generated L0D in the normal
`l0d/YYYY/MM` output folder after the orbit completes. Use `--no-save-l0d` to
override a True setting on the command line.

With `--start-step 2`, existing L0D is copied into the workspace for processing.
The original is never deleted or overwritten, regardless of `SAVE_L0D`. Older
L0D files from previous runs are not automatically removed.
