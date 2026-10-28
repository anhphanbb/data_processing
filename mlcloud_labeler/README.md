# MLCloud Review — local Windows tool

Double-click `launch.cmd`, then keep its console open while reviewing. It opens
http://127.0.0.1:8765 in your browser. Close the console or press Ctrl+C to stop.
It uses your existing `py310` environment (numpy, netCDF4, Pillow); no installation
or internet connection is required. If port 8765 is occupied, run
`python server.py --port 8766` from this directory.

## Workflow

1. Enter a folder containing L1R or L1C q20 NetCDF files and click **Load folder**.
   Subfolders are included, up to 1,000 files. Select an orbit and **Open orbit**.
   The first folder is the previous orbit 02 pipeline validation output; subsequent sessions remember your last successful folder. Paste a Windows path (for example `E:\soc\l1r\2023\11`) into **Data folder**, click **Load folder**, select an orbit, and click **Open orbit**.
2. Choose either the cleaned expert CSV or the filtered training-label CSV.
   Click **Open orbit** to apply a changed selection. Expert labels remain read-only.
3. Scrub or play frames. Left/right advances one frame; Shift+left/right advances
   ten. Up/down jumps backward/forward 20 frames. Mouse wheel over the image steps frames. Space toggles playback when not typing. **V** applies the original viewer’s 0.4th/99.7th-percentile auto contrast; **B/C** expand/shrink the range. Missing data is black. Images use lower-left origin, matching the
   original Python viewer. The 15×3 grid is shown; labels apply to x=7, y=0/1/2.
4. Click an expert timeline bar to jump to its start. Use **Start review from
   expert labels** to copy them into an editable review, or add new intervals.
   **I** and **O** mark the start/end frame when not typing in a form.
   Bounds are zero-based and inclusive. Review intervals can be edited, deleted,
   and undone. Cloud, ground-light, and uncertain labels remain distinct.
5. **Save draft** persists intervals and reviewer information. **Validate review**
   additionally uses the default local reviewer and requires confirmation of source/frame alignment, and
   confirmation that you reviewed all three rows throughout the orbit.
6. **Export training CSV** downloads the current validated cloud-only review using
   the existing `(7,2)start`/`end`, `(7,1)`, `(7,0)` training columns. Resolve
   uncertain or ground-light labels first: this tool does not silently export them
   as cloud or teach the current trainer how to exclude them. Empty reviews export
   headers only; the existing trainer does not consume negative-only orbit rows.

## Data integrity and scope

- No NetCDF file or original expert CSV is modified. No files are uploaded.
- Expert CSVs lack date and source-file identity. Orbit-number matches are only
  candidates; regenerated/trimmed data can have different frame indexing. The
  default orbit 02 file is a viewer demonstration, not proof of label alignment.
- Reviews are keyed by source-file path, size, modification time, and label source.
  `reviews/` contains current JSON reviews and `reviews/history/` every revision.
  Copies include the label CSV SHA-256, file identity, reviewer, and local timestamp.
  Back up this folder. Browser edits are not autosaved.
- The local server binds only to 127.0.0.1 and rejects foreign origins. It is not
  a multi-user hosted service. It reads 16-frame float32 blocks, reuses a read-only NetCDF handle, downloads all compressed frames to browser storage and keeps up to 192 decoded frames (about 69 MB) nearby. Contrast is applied locally without another request. HDF5 access is serialized; a jump outside the cache still depends on disk/network latency.
- This first version supports expert comparison and manual review, not model
  inference, retraining, automatic frame alignment, or collaborative accounts.
- The cleaned table's Top/Middle/Bottom columns map to (7,2)/(7,1)/(7,0), matching
  the existing center-column training format. Rows with Ground Lights text retain
  that class. Invalid label rows are reported rather than silently repaired.

## Checks

`python test_server.py` exercises real HTTP requests, NetCDF frame orientation,
expert loading, review persistence, revision conflicts, validation, CSV export,
out-of-bounds rejection, class handling, and origin restrictions.

## HTML and startup

`index.html` is the interface, not a standalone viewer: do not double-click it.
Use `launch.cmd` to run the local Python service and open your browser. Nothing
is uploaded. A browser-only version would require a NetCDF/HDF5 WebAssembly
reader and browser file selection; this implementation uses the existing Python
scientific environment for reliable local-file access.

The compact viewer follows `D:\Github\cloud_restructured\1_view_images.py`:
lower-origin square grayscale image, center boundaries at x=139/159,
all 45 box numbers, optional grid, percentile contrast, and ISS position.

Performance check on orbit 00002 (local disk): 16 sequential PNG requests took
0.201 s; one 16-frame raw block took 0.008 s. This is a warm local HTTP/data-path
measurement, not a guaranteed end-to-end playback or remote-drive speedup.

## Whole-orbit temporary download

Opening an orbit starts a complete download of its radiance frames into this
browser's IndexedDB storage, using the existing web app's compressed-chunk
loader. The bar above the viewer reports progress and **Ready** when complete.
After **Ready**, stepping and random seeks read from the local browser cache,
not from the source NetCDF or network drive. Only nearby decoded chunks stay
in memory (12 × 16 frames, about 69 MB); all compressed chunks stay on browser
storage. Downloading may take time on first use, especially for a network drive.
Interrupted downloads resume when the same unchanged orbit is reopened.

**Clear temporary download** removes only the currently opened orbit's cached
chunks and pauses viewing; **Download orbit / retry** starts it again. Original
NetCDFs and saved label reviews are unaffected. The cache is per browser and
may be evicted by the browser under storage pressure. The initial download
preserves full float32 radiance and missing pixels (lossless gzip).
