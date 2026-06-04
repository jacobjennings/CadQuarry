#!/usr/bin/env python3
"""
Export compact binary STL meshes for every part in a CadQuarry corpus.

This is a *convenience* artifact generator: STL is a lossy, derived view of the
canonical parametric ``.py`` source.  Meshes are written binary and coarsely
tessellated so a large sample stays light enough to lazy-load in a browser
preview (see ``sample/<name>/preview.html``) or attach to a HuggingFace upload.

Usage:
    python scripts/export_stl.py CORPUS_DIR [--out DIR] [--tolerance T]
        [--angular A] [--workers N]

Defaults write to ``CORPUS_DIR/stl/{part_id}.stl``.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Tessellation knobs tuned for small files that still read clearly at thumbnail
# size.  Raise --tolerance / --angular for even smaller meshes.
_DEFAULT_TOL = 0.3
_DEFAULT_ANG = 0.5


def _export_one(args: tuple[str, str, float, float]) -> tuple[str, bool, str]:
    py_file, out_file, tol, ang = args
    try:
        import cadquery as cq  # noqa: F401 (imported once per worker)

        ns: dict = {}
        with open(py_file, encoding="utf-8") as f:
            exec(compile(f.read(), py_file, "exec"), ns)
        params = {k: v["default"] for k, v in ns["PARAMS"].items()}
        result = ns["build"](params)

        shape = result.val()
        # Binary STL keeps files ~3-5x smaller than ASCII.
        shape.exportStl(out_file, tolerance=tol, angularTolerance=ang, ascii=False)
        return (out_file, True, "")
    except Exception as exc:  # noqa: BLE001 - report, never crash the pool
        return (out_file, False, str(exc))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export compact binary STLs for a corpus.")
    ap.add_argument("corpus", help="Corpus directory (contains parts/*.py)")
    ap.add_argument("--out", default=None, help="Output dir (default: <corpus>/stl)")
    ap.add_argument("--tolerance", type=float, default=_DEFAULT_TOL, help="Linear deflection (mm)")
    ap.add_argument("--angular", type=float, default=_DEFAULT_ANG, help="Angular deflection (rad)")
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (0 = cpu count, capped 24)")
    args = ap.parse_args(argv)

    corpus = Path(args.corpus)
    parts_dir = corpus / "parts"
    if not parts_dir.is_dir():
        print(f"error: {parts_dir} not found", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else corpus / "stl"
    out_dir.mkdir(parents=True, exist_ok=True)

    py_files = sorted(parts_dir.glob("*.py"))
    jobs = [
        (str(p), str(out_dir / f"{p.stem}.stl"), args.tolerance, args.angular)
        for p in py_files
    ]

    n_workers = args.workers or min(os.cpu_count() or 4, 24)
    print(f"Exporting {len(jobs)} STLs to {out_dir} ({n_workers} workers, "
          f"tol={args.tolerance} ang={args.angular}) …")

    ok = fail = 0
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, (out_file, success, err) in enumerate(ex.map(_export_one, jobs), 1):
            if success:
                ok += 1
            else:
                fail += 1
                failures.append(f"{Path(out_file).stem}: {err}")
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] ok={ok} fail={fail}")

    if failures:
        print("\nFailures:")
        for f in failures[:20]:
            print(f"  {f}")
        if len(failures) > 20:
            print(f"  … and {len(failures) - 20} more")

    total = sum(f.stat().st_size for f in out_dir.glob("*.stl"))
    print(f"\nDone. {ok} STLs written, {fail} failed. Total {total / 1e6:.1f} MB.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
