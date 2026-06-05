"""Render the 8 standard-view PNGs for every part in a corpus from its STLs.

Unlike ``cadquarry export --formats render`` (which re-executes CadQuery for
every part and removes the STL afterwards unless ``stl`` is also requested),
this renders directly off the STL files already on disk and never touches
them. Used to populate the committed sample's ``renders/`` dir.

Rendering runs on the GPU across a pool of worker processes that each own a
headless EGL context and share the one GPU (see ``cadquarry.render``); the
throughput-limiting step is CPU-side PNG encoding, so this scales with cores.

Usage:
    python scripts/render_sample.py <dataset_dir> [--workers N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make ``cadquarry`` importable when run directly as
# ``python scripts/render_sample.py`` (sys.path[0] is then scripts/, not root).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="Corpus directory (contains geometry/*.stl)")
    ap.add_argument("--workers", type=int, default=0, help="Render processes (0 = auto)")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    geo_dir = dataset / "geometry"
    renders_dir = dataset / "renders"
    stls = sorted(geo_dir.glob("*.stl"))
    if not stls:
        print(f"error: no STL files in {geo_dir}", file=sys.stderr)
        return 1

    from cadquarry import render as _render
    from cadquarry.progress import progress_bar

    renders_dir.mkdir(parents=True, exist_ok=True)
    workers = _render.default_render_workers(args.workers)
    print(f"Rendering {len(stls)} parts × 8 views into {renders_dir}/ "
          f"({workers} GPU workers)…")

    tasks = [(str(p), str(renders_dir / p.stem), None) for p in stls]
    t0 = time.time()
    bar = progress_bar(total=len(tasks), desc="render", unit="part")
    results = _render.render_stls(tasks, n_workers=workers, progress_cb=bar.update)
    bar.close()

    failed = [(pid, err) for pid, (ok, err) in results.items() if not ok]
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
