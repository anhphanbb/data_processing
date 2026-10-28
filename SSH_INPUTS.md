# Process remote L0C inputs one batch at a time

This feature is developed on `codex/ssh-batch-inputs`. It does not change a running
pipeline in another checkout. Do not switch that running checkout to this branch.
Use a separate clone/worktree to test it. Models are local assets and must be
provided in that checkout's `models/` folder.

## Requirements

- Activated `py310`, model files, and the normal science dependencies.
- OpenSSH `ssh` and `sftp` on PATH on the processing computer.
- `python3` on the remote server for listing, file metadata and SHA-256.
- SSH key/agent authentication for unattended use. No passwords are stored or
  accepted by the pipeline. A passphrase-protected key can be loaded into an agent.
- A verified server host key in the processing user's `known_hosts`.

Test authentication from the same account/environment that will run the pipeline:

```powershell
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes anh@awex2.physics.usu.edu "python3 --version"
```

If this fails, establish SSH access interactively first and arrange key/agent
authentication. Verify any new host fingerprint with the server administrator;
do not disable host-key verification. `--ssh-key C:\path\to\key` and
`--ssh-port 22` are available when needed.

## January 2024

Activate the environment. The portable command below uses the active Python;
the included `run_pipeline_py310.cmd` also activates and checks the GPU, but its
Conda paths must be adjusted for a different computer.

```powershell
conda activate py310
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python run_awe_pipeline_batch.py --year 2024 --month 1 --start-step 0 --ssh-host anh@awex2.physics.usu.edu --ssh-root /awedata2/temp_v2/tmp_202401 --soc-root E:\soc --work-root E:\_orbit_work --no-save-to-server --save-l1r --orbits 621 --dry-run
```

The dry run reads the remote directory inventory but downloads no science files.
Remove `--dry-run` to test orbit 621. For the month, remove `--orbits 621` and add
`--no-overwrite` to skip outputs with matching completion receipts from the same
code/settings. Keep the same paths and settings between the test and month run.
All four filters must exist and share a timestamp. Only `v02` files belonging to
the requested year/month are selected; `v22` is ignored. Incomplete/duplicate
selected orbits are reported before processing starts.

The example writes outputs locally. For server publication, replace
`--no-save-to-server` with `--save-to-server --server-root Y:\soc --version-folder version02p1`
once that destination is accessible. This destination affects completion receipts,
so use the same destination for your test and full run.

## Transfers, disk use and recovery

Raw inputs are staged under `WORK_ROOT/ssh_inputs/<source-id>/<orbit>/` (override
the parent with `--ssh-cache-root`). Step 0 reads these files and writes its normal
trimmed/glare-filtered products into the batch workspace. Remote files are read only.
The existing scheduler prepares one batch at a time alongside processing; its
default allows three processing batches plus one preparing/prepared batch. Allow
space for their raw inputs and intermediate outputs, not just one orbit.

Downloads use SFTP `reget` into `.partial` files. Remote size, timestamp and SHA-256
are recorded. Full local size and SHA-256 must match before a download is published
to step 0. Unchanged verified cache files are reused. Interrupted partial transfers
resume when the command is rerun; changed source metadata or corrupt downloads are
discarded and downloaded again. Remote hashing reads each selected file on the
server but does not transfer its contents to the client.

Raw files are removed only after the orbit's output publication and completion
receipt succeed. Failed-orbit raw inputs and processing intermediates are retained
for investigation/retry, so repeated failures can increase disk usage. No transfer
cache is removed simply because a stage failed. `--no-overwrite` is recommended
when resuming the same code/settings so verified completed orbits are skipped.
There is no automatic retry loop for authentication/transfer failures.

Do not run the same orbit twice concurrently against a shared cache/output folder.
A `.download.lock` prevents concurrent transfers for an orbit. After an abrupt
process kill, verify that its process tree has stopped before manually removing
that orbit's stale lock. Keep partial files and metadata if you want transfer resume.

Offline validation: `python -m unittest discover -s tests -p test_ssh_batch_inputs.py`.
These tests exercise staging and runner integration with simulated transports;
a real SSH/SFTP orbit transfer still needs to be validated on the target computer.
