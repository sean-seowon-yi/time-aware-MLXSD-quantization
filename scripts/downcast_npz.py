"""
Rewrite .npz files in-place, downcasting float32 arrays to float16.
Usage: python scripts/downcast_npz.py <directory>

Disk-safe strategy: close file handle, unlink original, then write smaller
version. The space is freed before the write begins. These are regenerable
cache files so losing a file on interruption is acceptable.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np


def convert_file(path: str) -> tuple[int, int, bool]:
    """Returns (old_bytes, new_bytes, changed)."""
    old_size = os.path.getsize(path)

    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    # File handle closed — unlink will actually free blocks on macOS.

    if not any(v.dtype == np.float32 for v in arrays.values()):
        return old_size, old_size, False

    converted = {
        k: v.astype(np.float16) if v.dtype == np.float32 else v
        for k, v in arrays.items()
    }

    os.unlink(path)
    np.savez_compressed(path, **converted)
    return old_size, os.path.getsize(path), True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.directory, "**/*.npz"), recursive=True))
    if not files:
        print("No .npz files found.")
        return

    print(f"Found {len(files)} .npz files")
    total_old = total_new = 0
    start = time.time()

    for i, path in enumerate(files, 1):
        try:
            old, new, changed = convert_file(path)
        except Exception as e:
            print(f"\nERROR {path}: {e}", file=sys.stderr)
            continue

        total_old += old
        total_new += new
        elapsed = time.time() - start
        rate = i / elapsed
        eta = (len(files) - i) / rate if rate > 0 else 0
        status = f"{'→f16' if changed else 'skip'}"
        print(
            f"\r[{i}/{len(files)}] {status}  "
            f"freed {(total_old - total_new) / 1e9:.1f} GB  "
            f"ETA {eta/60:.1f}m",
            end="", flush=True,
        )

    print()
    print(f"\nDone in {(time.time()-start)/60:.1f} min")
    print(
        f"Before: {total_old/1e9:.1f} GB  "
        f"After: {total_new/1e9:.1f} GB  "
        f"Saved: {(total_old-total_new)/1e9:.1f} GB "
        f"({100*(total_old-total_new)/max(total_old,1):.0f}%)"
    )


if __name__ == "__main__":
    main()
