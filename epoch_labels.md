# Epoch-based cloud labels

Use NetCDF (`.nc`) as the master label file. It stores exact numeric epoch values,
their source metadata, compressed binary cloud flags, and label provenance in one
file. CSV is useful for inspecting or exchanging a flat table; both are supported.

Run with the original **v01 q20 L1R file used during labeling**, not regenerated
data. The converter reads only Epoch and the Radiance shape, not the images.

```powershell
& C:\Users\domin\.conda\envs\py310\python.exe .\convert_cloud_labels_to_epoch.py --l1r "E:\soc\l1r\awe_l1r_q20_2023336T0000_00002_v01.nc" --labels .\csv\mlcloud\cloud_intervals_cleaned_filtered_10272025.csv --output .\epoch_labels\orbit00002.nc
```

Replace the example L1R path with the actual original file. Change output to
`orbit00002.csv` for CSV. Existing output files are never overwritten.
Dependencies: Python 3.10+, numpy, netCDF4.

## Interpretation

- Input bounds are zero-based and inclusive; an interval 585–595 labels all 11
  original frames. Overlapping intervals are combined.
- Every original epoch and every box represented by the CSV columns gets a flag:
  `1 = cloud`, `0 = no cloud`. Unlisted frames are clear within those boxes.
  Other boxes are not inferred to be clear.
- Both `(x,y)start` / `(x,y)end` columns and the cleaned table's
  `Start (Top)` / `End (Top)` etc. are accepted. Top/Middle/Bottom correspond to
  `(7,2)` / `(7,1)` / `(7,0)`. Box coordinates are zero-based.
- NetCDF contains scalar `orbit`, `epoch(epoch)`, `box(box)` coordinate strings,
  and `cloud(epoch, box)`. CSV contains `orbit,epoch,box,cloud` plus source epoch
  units/calendar and filename. NetCDF additionally stores the label CSV hash,
  source paths and conversion policies.
- Epoch values are preserved without rounding, UTC conversion or interpolation.
  The repository's AWE time convention can differ from a literal reading of the
  epoch units, so the script deliberately does not invent calendar timestamps.
- Blank rows are ignored. Invalid bounds, missing endpoints, duplicate epochs,
  and matching rows marked Ground Lights fail explicitly. Use the filtered
  binary cloud labels, or resolve non-cloud annotations first.
- An orbit with an explicit blank label row is clear. An orbit entirely absent
  from the CSV fails by default, since absence can also mean unreviewed. Supply
  `--absent-orbit-is-clear` only when you know the absent orbit was reviewed.

## Using the labels with a later data version

Join by `(orbit, epoch, box)` using the same numeric epoch convention. Removed
frames simply have no corresponding new frame. Added epochs without an original
label are **unknown**, not no-cloud. Do not expand old cloud intervals between
their endpoint epochs: doing so would silently assign labels to added frames.
If epoch values themselves changed between versions, exact joining will not
work; that requires a separately validated time-matching policy.

The single-file converter handles one orbit per invocation. A v01 filename and
in-range intervals cannot alone prove that the file matches the original labeling data.

## Convert all labeled orbits

From `D:\Github\data_processing`, run:

```powershell
& C:\Users\domin\.conda\envs\py310\python.exe .\batch_convert_cloud_labels_to_epoch.py `
  --l1r-root "Y:\soc\l1r" `
  --labels .\csv\mlcloud\cloud_intervals_cleaned_filtered_10272025.csv `
  --output-dir .\epoch_labels\v01 `
  --format nc
```

The batch script reads all distinct nonblank `Orbit #` entries, recursively scans
the year/month folders, and writes one NetCDF per orbit with a unique matching
q20 v01 file. Only listed orbits are processed, including explicit blank interval
rows. A timestamped CSV report records each orbit's source, output, counts and
status. Missing files, multiple matches, existing outputs and invalid labels are
reported without stopping other orbits; the command exits nonzero if any orbit
was not converted. No existing output is overwritten.

Use `--format csv` for flat CSV outputs. Use `--filter bkg` only if the original
intervals were labeled against bkg frames (or you independently verified that its
Epoch sequence is identical to the original labeling source). The default is
`--filter q20`, matching the local cloud-labeling viewer. There is no automatic
fallback to another filter, since its frame sequence might differ. If an orbit
has multiple matching files, narrow `--l1r-root` to the appropriate year/month or
dataset; the script will not guess which one is correct.

Run this from a Windows session where `Y:` is mounted. If it is unavailable in
that session, use the share's actual UNC path instead.

Tests: `python -m unittest discover -s tests -p test_epoch_labels.py`.
