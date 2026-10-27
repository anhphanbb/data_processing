"""Copy completed AWE L1R, L1A, L3A and L3C monthly products to a mapped server.

Edit SETTINGS and run in Spyder, or use:
    python copy_awe_products_to_server.py --year 2023 --month 11
    python copy_awe_products_to_server.py --dry-run
    python copy_awe_products_to_server.py --overwrite --verify
    python copy_awe_products_to_server.py --resolutions 2

Requires only Python's standard library. Run after processing has finished.
All matching local NetCDF files in the selected month are candidates; this
script does not use pipeline run records to select a particular run.
Existing filenames are skipped unless overwrite is enabled. No files are
removed from the source or destination. Filenames (including v02) are preserved;
VERSION_FOLDER controls only the server directory name.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time

# ------------------------- SETTINGS -------------------------
SOURCE_ROOT = Path(r"E:\soc")
SERVER_ROOT = Path(r"Y:\soc")  # A UNC path can also be used.
VERSION_FOLDER = "version02p1"
YEAR = 2023
MONTH = 11
L3A_RESOLUTIONS = (2,10)  # Use (2,) to copy only L3A 2x2.
OVERWRITE = False         # True replaces existing files of the same name.
DRY_RUN = False           # True lists actions without writing anything.
VERIFY_SHA256 = False    # True also rereads and compares file contents.
# ------------------------------------------------------------


def folder_pairs(source_root, server_root, version, year, month, resolutions):
    ym = Path(f"{year:04d}") / f"{month:02d}"
    yield "l1r", source_root / "l1r" / ym, server_root / "l1r" / version / ym
    yield "l1a", source_root / "l1a" / ym, server_root / "l1a" / version / ym
    for resolution in dict.fromkeys(resolutions):
        sub = f"{resolution}x{resolution}"
        yield "l3a", source_root / "l3a" / ym / sub, server_root / "l3a" / version / ym / sub
    yield "l3c", source_root / "l3c" / ym, server_root / "l3c" / version / ym


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def copy_one(source, destination, overwrite=False, verify=False):
    """Stage beside the destination; publish only after transfer checks pass."""
    if destination.exists() and not overwrite:
        return "skipped"
    before = source.stat()
    if before.st_size == 0:
        raise ValueError(f"Empty source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial",
                                dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        # Some mapped server shares reject copy2's chmod/copystat calls
        # (WinError 50). Transfer bytes only; do not copy filesystem metadata.
        # NetCDF variables and attributes are part of these bytes and remain intact.
        with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
        if temporary.stat().st_size != before.st_size:
            raise OSError("Copied file size does not match the source")
        if verify and sha256(source) != sha256(temporary):
            raise OSError("SHA-256 verification failed")
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise OSError("Source changed during copying; rerun after processing finishes")
        if overwrite:
            os.replace(temporary, destination)
        elif os.name == "nt":
            # On Windows rename fails if another process created the target.
            try:
                os.rename(temporary, destination)
            except FileExistsError:
                return "skipped"
        else:
            # Atomic no-clobber publication for POSIX filesystems.
            try:
                os.link(temporary, destination)
            except FileExistsError:
                return "skipped"
        return "copied"
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    p.add_argument("--server-root", type=Path, default=SERVER_ROOT)
    p.add_argument("--version-folder", default=VERSION_FOLDER)
    p.add_argument("--year", type=int, default=YEAR)
    p.add_argument("--month", type=int, choices=range(1, 13), default=MONTH)
    p.add_argument("--resolutions", type=int, nargs="+", choices=(2, 10), default=L3A_RESOLUTIONS)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE)
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=DRY_RUN)
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=VERIFY_SHA256)
    args = p.parse_args(argv)
    if not 1 <= args.month <= 12 or not 1 <= args.year <= 9999:
        p.error("Invalid year or month")
    if not args.version_folder or args.version_folder in (".", "..") or any(
            c in args.version_folder for c in '/\\:'):
        p.error("--version-folder must be a single directory name")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Require the mapped share to exist; never fabricate an unavailable root.
    for label, root in (("Source", args.source_root), ("Server", args.server_root)):
        if not root.is_dir():
            print(f"ERROR: {label} root is unavailable: {root}", flush=True)
            return 1
    if args.source_root.resolve() == args.server_root.resolve():
        print("ERROR: Source and server roots must differ.", flush=True)
        return 1
    started = time.monotonic()
    counts = Counter()
    print(f"AWE copy: {args.year:04d}/{args.month:02d}; overwrite={args.overwrite}; "
          f"dry_run={args.dry_run}; SHA256={args.verify}", flush=True)
    for level, source_dir, target_dir in folder_pairs(
            args.source_root, args.server_root, args.version_folder,
            args.year, args.month, args.resolutions):
        print(f"\n{source_dir}\n  -> {target_dir}", flush=True)
        try:
            if not source_dir.is_dir():
                raise FileNotFoundError(f"Source folder missing: {source_dir}")
            files = sorted(p for p in source_dir.glob(f"awe_{level}_*.nc") if p.is_file())
            if not files:
                print("  WARNING: No matching NetCDF files.", flush=True)
                counts["empty_folders"] += 1
                continue
        except OSError as exc:
            print(f"  ERROR: {exc}", flush=True)
            counts["failed"] += 1
            continue
        for index, source in enumerate(files, 1):
            destination = target_dir / source.name
            try:
                if args.dry_run:
                    status = "skipped" if destination.exists() and not args.overwrite else "would_copy"
                else:
                    status = copy_one(source, destination, args.overwrite, args.verify)
                counts[status] += 1
                print(f"  [{index}/{len(files)}] {status.upper()}: {source.name}", flush=True)
            except (OSError, ValueError) as exc:
                counts["failed"] += 1
                print(f"  [{index}/{len(files)}] FAILED: {source.name}: {exc}", flush=True)
    print("\nSummary: " + ", ".join(f"{key}={counts[key]}" for key in
          ("copied", "skipped", "would_copy", "failed", "empty_folders")) +
          f"; elapsed={time.monotonic() - started:.1f}s", flush=True)
    return 1 if counts["failed"] or counts["empty_folders"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
