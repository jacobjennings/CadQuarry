#!/usr/bin/env python3
"""
Execution-yield harness for the ``compound`` family (and a template for any
family whose samplers can fail to build under CadQuery).

Generation already rejects-and-resamples parts that don't build, so a low
yield never corrupts a corpus — but it *does* waste attempts and can hide a
genuinely broken archetype (one that builds 0% of the time). This harness
surfaces that signal directly: it samples each compound archetype across a
band of seeds, plus a mixed ``compose()`` sample that exercises the real
config-driven weighted dispatch, executes every emitted program through the
persistent CadQuery worker pool, and reports per-archetype acceptance yields.

For every built part it also checks ``n_solids`` (now returned by the worker):
a valid part must be a single connected solid, so any part that builds into
>1 solid is flagged as *disconnected* — the failure mode the AttachOp
``CenterOfBoundBox`` fix addressed for multi-port manifolds.

Use it after changing any compound sampler (or the shared ``_sample_attach_section``
/ ``_maybe_bore`` helpers) to confirm:

  * every archetype still builds at a healthy rate (≈ all but tier-3
    fillet/chamfer fragility, which is shared with plate/block),
  * no archetype has silently regressed to 0% (the failure mode the
    ``stepped_shaft`` RevolveOp→ExtrudeOp fix addressed), and
  * no archetype produces disconnected (multi-solid) geometry.

It imports CadQuery indirectly through the worker pool, so the first run pays
the usual ~1-1.5s import cost. Nothing here is part of the shipped package.

Usage:
    .venv/bin/python scripts/validate_compound.py
    .venv/bin/python scripts/validate_compound.py --per-arch 60 --mixed 200 --workers 24
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadquarry import compose as C
from cadquarry.emit import emit_source
from cadquarry.execute import WorkerPool


def build_jobs(per_arch: int, mixed: int):
    """Return ``(jobs, meta)`` for the worker pool: one analyze job per part."""
    arches = list(C._COMPOUND_SAMPLERS)
    jobs: list[tuple[tuple, dict]] = []
    meta: dict[tuple, tuple] = {}

    # Per-archetype: call each sampler directly so a single weak archetype can't
    # be masked by the others. Tiers are sampled exactly as compose() would.
    for arch in arches:
        sampler = C._COMPOUND_SAMPLERS[arch]
        for i in range(per_arch):
            seed = 100_000 + (hash(arch) % 1000) * 1000 + i
            rng = Random(seed)
            tier = C._sample_tier(rng, "compound", {})
            part = sampler(rng, tier, {"_symmetry": "none"}, seed, i)
            key = (arch, i)
            jobs.append((key, {"op": "analyze", "code": emit_source(part)}))
            meta[key] = (arch, tier)

    # Mixed sample through the real compose() path (weighted archetype dispatch).
    for i in range(mixed):
        seed = 7_000_000 + i
        part = C.compose(seed, index=i, family="compound")
        key = ("mixed", i)
        jobs.append((key, {"op": "analyze", "code": emit_source(part)}))
        meta[key] = ("mixed", part.metadata.tier)

    return jobs, meta, arches


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compound-family execution-yield harness.")
    ap.add_argument("--per-arch", type=int, default=40, help="Parts sampled per archetype")
    ap.add_argument("--mixed", type=int, default=120, help="Parts sampled via compose()")
    ap.add_argument("--workers", type=int, default=16, help="CadQuery worker processes")
    ap.add_argument("--timeout", type=float, default=30.0, help="Per-part timeout (s)")
    args = ap.parse_args(argv)

    jobs, meta, arches = build_jobs(args.per_arch, args.mixed)
    print(f"Executing {len(jobs)} parts across {len(arches)} archetypes + mixed …",
          flush=True)
    with WorkerPool(n_workers=args.workers, timeout=args.timeout) as pool:
        results = pool.map(jobs)

    ok: Counter = Counter()
    tot: Counter = Counter()
    disc: Counter = Counter()   # built but disconnected (>1 solid)
    fails: list[tuple] = []
    for key, res in results.items():
        bucket = key[0]
        tot[bucket] += 1
        if not res.get("success"):
            fails.append((key, meta[key], res.get("error", "")[:160]))
            continue
        n_solids = res.get("n_solids", 1)
        if n_solids != 1:
            disc[bucket] += 1
            fails.append((key, meta[key], f"disconnected: {n_solids} solids"))
            continue
        ok[bucket] += 1

    print("\n=== Per-archetype yields (single-solid builds) ===")
    for arch in arches:
        extra = f"  [{disc[arch]} disconnected]" if disc[arch] else ""
        print(f"  {arch:18s} {ok[arch]:3d}/{tot[arch]:<3d}{extra}")
    mextra = f"  [{disc['mixed']} disconnected]" if disc["mixed"] else ""
    print(f"  {'mixed(compose)':18s} {ok['mixed']:3d}/{tot['mixed']:<3d}{mextra}")

    total_ok, total = sum(ok.values()), sum(tot.values())
    total_disc = sum(disc.values())
    print(f"\nOverall single-solid: {total_ok}/{total} = "
          f"{100 * total_ok / max(total, 1):.1f}%  (disconnected: {total_disc})")

    if fails:
        print(f"\n=== {len(fails)} non-clean parts (first 25) ===")
        for key, (label, tier), err in fails[:25]:
            print(f"  {key} tier={tier} :: {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
