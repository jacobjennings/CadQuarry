"""Render the 8 standard-view PNGs for every part in a corpus from its STLs.

Unlike ``cadquarry export --formats render`` (which re-executes CadQuery for
every part and removes the STL afterwards unless ``stl`` is also requested),
this renders directly off the STL files already on disk and never touches
them. Used to populate the committed sample's ``renders/`` dir.

Rendering runs on the GPU through a single reused headless EGL context (see
``cadquarry.render``), so this is a single-process loop — one GPU context is
far faster and lighter than spawning one per CPU core.

Usage:
    python scripts/render_sample.py <dataset_dir>
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
    args = ap.parse_args()

    dataset = Path(args.dataset)
    geo_dir = dataset / "geometry"
    renders_dir = dataset / "renders"
    stls = sorted(geo_dir.glob("*.stl"))
    if not stls:
        print(f"error: no STL files in {geo_dir}", file=sys.stderr)
        return 1

    from cadquarry.export import export_renders

    renders_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(stls)} parts × 8 views into {renders_dir}/ (GPU)…")

    t0 = time.time()
    failed: list[tuple[str, str]] = []
    for done, stl in enumerate(stls, 1):
        pid = stl.stem
        try:
            export_renders(stl, renders_dir / pid)
        except Exception as exc:  # pragma: no cover - reported to caller
            failed.append((pid, str(exc)))
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
