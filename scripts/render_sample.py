"""Render the 8 standard-view PNGs for every part in a corpus from its STLs.

Unlike ``cadquarry export --formats render`` (which re-executes CadQuery for
every part and removes the STL afterwards unless ``stl`` is also requested),
this renders directly off the STL files already on disk, in parallel, and
never touches them. Used to populate the committed sample's ``renders/`` dir.

Usage:
    python scripts/render_sample.py <dataset_dir> [--workers N]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Make ``cadquarry`` importable in this process and in forked workers when the
# script is run directly as ``python scripts/render_sample.py`` (sys.path[0] is
# then the scripts/ dir, not the repo root).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _render_one(stl_path_str: str, renders_dir_str: str) -> tuple[str, bool, str]:
    from cadquarry.export import export_renders

    stl_path = Path(stl_path_str)
    pid = stl_path.stem
    try:
        export_renders(stl_path, Path(renders_dir_str) / pid)
        return pid, True, ""
    except Exception as exc:  # pragma: no cover - reported to caller
        return pid, False, str(exc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="Corpus directory (contains geometry/*.stl)")
    ap.add_argument("--workers", type=int, default=0, help="Processes (0 = all cores)")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    geo_dir = dataset / "geometry"
    renders_dir = dataset / "renders"
    stls = sorted(geo_dir.glob("*.stl"))
    if not stls:
        print(f"error: no STL files in {geo_dir}", file=sys.stderr)
        return 1

    import os

    workers = args.workers or os.cpu_count() or 1
    renders_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(stls)} parts × 8 views into {renders_dir}/ "
          f"({workers} workers)…")

    t0 = time.time()
    done = 0
    failed: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_render_one, str(p), str(renders_dir)) for p in stls]
        for fut in as_completed(futs):
            pid, ok, err = fut.result()
            done += 1
            if not ok:
                failed.append((pid, err))
            if done % 100 == 0 or done == len(stls):
                print(f"  {done}/{len(stls)}  ({time.time() - t0:.0f}s)")

    dt = time.time() - t0
    print(f"\nDone. {len(stls) - len(failed)}/{len(stls)} parts rendered "
          f"in {dt:.1f}s ({len(stls) * 8} PNGs).")
    if failed:
        print(f"  {len(failed)} failed:")
        for pid, err in failed[:20]:
            print(f"    {pid}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
