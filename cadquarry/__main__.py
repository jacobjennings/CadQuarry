"""
CadQuarry CLI

  cadquarry generate  — generate a corpus
  cadquarry build     — generate + export every corpus in the seed ladder
  cadquarry publish   — pack + upload the built corpora to Hugging Face
  cadquarry run       — execute a single part with optional param overrides
  cadquarry serve     — launch live customizer for a part (or gallery for corpus)
  cadquarry export    — export geometry artifacts for an existing corpus
  cadquarry verify    — re-execute corpus and check validity + signatures
  cadquarry info      — print part metadata / parameter schema
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from . import __version__


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(config_path: str | None) -> dict:
    if config_path is None:
        default = Path(__file__).parent.parent / "configs" / "default.toml"
        if default.exists():
            config_path = str(default)
        else:
            return {}
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as exc:
        print(f"[cadquarry] warning: could not load config {config_path!r}: {exc}")
        return {}


def _default_seeds_path() -> Path:
    """
    The published seed list matching the current generator.

    Selects the ``seeds/v*.toml`` whose ``[meta].generator_version`` equals
    ``cadquarry.__version__`` (so a generator bump automatically tracks its
    matching list); falls back to the highest-numbered list, then ``v1.toml``.
    """
    seeds_dir = Path(__file__).parent.parent / "seeds"

    def _vnum(p: Path) -> int:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return int(digits) if digits else 0

    candidates = sorted(seeds_dir.glob("v*.toml"), key=_vnum)
    match = None
    for p in candidates:
        try:
            with open(p, "rb") as f:
                meta = tomllib.load(f).get("meta", {})
        except Exception:
            continue
        if str(meta.get("generator_version")) == __version__:
            match = p
    if match is not None:
        return match
    if candidates:
        return candidates[-1]
    return seeds_dir / "v1.toml"


def _load_publish_ladder(seeds_path: Path) -> dict[str, dict]:
    """Return {tag: {"count": int, "seed": int}} from [[publish.corpus]]."""
    with open(seeds_path, "rb") as f:
        data = tomllib.load(f)
    ladder: dict[str, dict] = {}
    for entry in data.get("publish", {}).get("corpus", []):
        ladder[str(entry["tag"])] = {
            "count": int(entry["count"]),
            "seed": int(entry["seed"]),
        }
    return ladder


def _tag_to_int(tag: str) -> int:
    """Parse a size tag like '50k' / '1m' into an int, for sorting."""
    t = tag.lower().strip()
    mult = 1
    if t.endswith("k"):
        mult, t = 1_000, t[:-1]
    elif t.endswith("m"):
        mult, t = 1_000_000, t[:-1]
    try:
        return int(float(t) * mult)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> int:
    from .compose import compose
    from .emit import emit_source, write_part, emit_meta_json
    from .execute import ExecuteResult, WorkerPool, default_worker_count
    from .filter import is_valid, compute_signature, DedupStore, QualityConfig
    from .dataset import DatasetWriter
    from .progress import progress_bar

    config = _load_config(args.config)
    out_dir = Path(args.out)
    seed = args.seed
    count = args.count
    timeout = args.timeout
    no_exec = args.no_exec
    family = args.family or None
    tier = args.tier if args.tier is not None else None
    verbose = args.verbose
    n_workers = default_worker_count(getattr(args, "workers", None))

    qcfg = QualityConfig(
        min_volume_mm3=config.get("quality", {}).get("min_volume_mm3", 1.0),
        max_bbox_ratio=config.get("quality", {}).get("max_bbox_ratio", 20.0),
    )

    writer = DatasetWriter(out_dir, seed)
    dedup = DedupStore()

    accepted = 0
    attempted = 0
    skipped_invalid = 0
    skipped_dup = 0

    print(f"CadQuarry v{__version__} — generating {count} parts (seed={seed})")
    print(f"Output: {out_dir}")
    if no_exec:
        print("  (--no-exec: skipping execution validation)")
    else:
        print(f"  (executing in {n_workers} persistent workers)")

    def compose_attempt(i: int):
        part_seed = (seed * 1_000_003 + i) & 0xFFFFFFFF
        return compose(part_seed, index=i, config=config, family=family, tier=tier)

    # The accept/reject/dedup decision is intrinsically sequential (a part is a
    # duplicate of an *earlier accepted* part), and the manifest order must not
    # depend on completion order.  We therefore keep the decision loop strictly
    # in attempt order, but feed it execution results that were computed ahead
    # of time, in parallel.  Execution results are pure functions of the source,
    # so this is bit-for-bit identical to the old sequential path — only faster.
    pbar = progress_bar(total=count, desc="  generate", unit="part")

    def accept_one(part, geo_sig) -> None:
        nonlocal accepted
        paths = write_part(part, out_dir, geometry_signature=geo_sig)
        meta = emit_meta_json(part, geo_sig)
        writer.add_record(meta, paths, geo_sig)
        accepted += 1
        pbar.update(1)
        if verbose:
            pbar.write(
                f"  [{accepted:>{len(str(count))}}/{count}] "
                f"{part.id}  family={part.metadata.family}  tier={part.metadata.tier}"
            )

    if no_exec:
        # No execution → nothing to parallelise; identical to the legacy loop.
        while accepted < count:
            part = compose_attempt(attempted)
            attempted += 1
            ir_hash = part.ir_hash()
            if dedup.is_duplicate(ir_hash):
                skipped_dup += 1
                continue
            dedup.register(ir_hash)
            accept_one(part, None)
            if attempted > count * 10 and accepted < count // 2:
                print(
                    f"[cadquarry] warning: high rejection rate "
                    f"({skipped_invalid} invalid, {skipped_dup} dup of {attempted} attempts). "
                    "Consider relaxing the quality config or widening parameter ranges."
                )
                break
    else:
        pool = WorkerPool(n_workers, timeout=timeout)
        stop = False
        try:
            while accepted < count and not stop:
                # Speculatively compose + execute a window of upcoming attempts.
                # Over-shooting only wastes (cheap) executions; it never changes
                # which parts are accepted, because the decision loop below is
                # the sole authority and runs in strict attempt order.
                remaining = count - accepted
                batch_n = max(n_workers, remaining + remaining // 3 + n_workers)
                batch = []
                jobs = []
                for j in range(batch_n):
                    i = attempted + j
                    part = compose_attempt(i)
                    ir_hash = part.ir_hash()
                    batch.append((i, part, ir_hash))
                    # Pre-skip exact IR duplicates already accepted in a prior
                    # batch to avoid wasted execution (the decision loop applies
                    # the same check, so this is purely an optimisation).
                    if dedup.is_duplicate(ir_hash):
                        continue
                    code = emit_source(part)
                    jobs.append((i, {"op": "analyze", "code": code, "overrides": {}, "stl_out": ""}))

                results = pool.map(jobs)

                for (i, part, ir_hash) in batch:
                    attempted += 1
                    if dedup.is_duplicate(ir_hash):
                        skipped_dup += 1
                        continue
                    res_dict = results.get(i)
                    if res_dict is None:
                        # Should not happen (every non-pre-skipped attempt was
                        # executed); treat defensively as a failed execution.
                        res_dict = {"success": False, "error": "missing execution result"}
                    result = ExecuteResult.from_dict(res_dict)
                    valid, reason = is_valid(result, qcfg)
                    if not valid:
                        skipped_invalid += 1
                        if verbose:
                            print(f"  REJECT {part.id}: {reason}")
                        continue
                    geo_sig_obj = compute_signature(result)
                    if dedup.is_duplicate(ir_hash, geo_sig_obj):
                        skipped_dup += 1
                        continue
                    dedup.register(ir_hash, geo_sig_obj)
                    accept_one(part, geo_sig_obj.to_dict())

                    if accepted >= count:
                        stop = True
                        break

                    # Matches the legacy loop: the early-stop check is only
                    # reachable immediately after an acceptance (every reject /
                    # dup path `continue`s past it).
                    if attempted > count * 10 and accepted < count // 2:
                        print(
                            f"[cadquarry] warning: high rejection rate "
                            f"({skipped_invalid} invalid, {skipped_dup} dup of {attempted} attempts). "
                            "Consider relaxing the quality config or widening parameter ranges."
                        )
                        stop = True
                        break
        finally:
            pool.close()

    pbar.close()
    manifest = writer.finalize()
    print(
        f"\nDone. {accepted} parts written to {out_dir}\n"
        f"  attempts={attempted}  invalid={skipped_invalid}  duplicates={skipped_dup}\n"
        f"  manifest: {manifest}"
    )
    return 0


# ---------------------------------------------------------------------------
# build — generate (and export) every corpus in the seed ladder in one pass
# ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> int:
    from .export import export_corpus_geometry

    seeds_path = Path(args.seeds) if args.seeds else _default_seeds_path()
    if not seeds_path.exists():
        print(f"error: seed list {seeds_path} not found", file=sys.stderr)
        return 1

    ladder = _load_publish_ladder(seeds_path)
    if not ladder:
        print(f"error: no [[publish.corpus]] entries in {seeds_path}", file=sys.stderr)
        return 1

    tags = args.sizes or list(ladder)
    unknown = [t for t in tags if t not in ladder]
    if unknown:
        print(
            f"error: unknown size(s): {unknown}. Available: {', '.join(ladder)}",
            file=sys.stderr,
        )
        return 1
    tags = sorted(tags, key=_tag_to_int)

    base_out = Path(args.out)
    do_export = not args.no_export
    formats = [f.strip() for f in args.formats.split(",")] if do_export else None

    print(
        f"CadQuarry v{__version__} — building {len(tags)} corpora "
        f"({', '.join(tags)}) into {base_out}/"
    )
    if do_export:
        print(f"  export formats: {formats}")
    else:
        print("  (--no-export: generation only)")

    failures: list[str] = []
    for tag in tags:
        spec = ladder[tag]
        out_dir = base_out / tag
        print(f"\n=== {tag}: {spec['count']:,} parts (seed {spec['seed']}) -> {out_dir} ===")

        # Reuse an already-complete corpus unless --force; generation is fully
        # seeded, so an existing manifest of the right size is bit-identical —
        # but only if it was produced by THIS generator version, otherwise a
        # stale corpus from an older generator would be silently kept.
        manifest = out_dir / "manifest.jsonl"
        skip_gen = False
        if manifest.exists() and not args.force:
            lines = manifest.open(encoding="utf-8").read().splitlines()
            n = len(lines)
            stale_version = None
            if lines:
                try:
                    gv = json.loads(lines[0]).get("generator_version")
                    if gv != __version__:
                        stale_version = gv
                except Exception:
                    pass
            if stale_version is not None:
                print(
                    f"  · regenerating (existing corpus is generator "
                    f"{stale_version}, current is {__version__})"
                )
            elif n >= spec["count"]:
                print(f"  · reusing existing corpus ({n} parts)")
                skip_gen = True

        if not skip_gen:
            gen_args = argparse.Namespace(
                count=spec["count"], seed=spec["seed"], out=str(out_dir),
                config=args.config, timeout=args.timeout, no_exec=False,
                workers=args.workers, family=None, tier=None, verbose=args.verbose,
            )
            rc = cmd_generate(gen_args)
            if rc != 0:
                failures.append(f"{tag} (generate)")
                continue

        if do_export:
            print(f"  · exporting {formats} …")
            try:
                counts = export_corpus_geometry(
                    out_dir, formats=formats, n_workers=args.workers,
                    timeout=args.timeout, verbose=args.verbose,
                )
                for fmt, cnt in counts.items():
                    print(f"    {fmt}: {cnt} files written")
            except Exception as exc:
                print(f"  export failed for {tag}: {exc}", file=sys.stderr)
                failures.append(f"{tag} (export)")

    if failures:
        print(f"\nDone with errors: {', '.join(failures)}")
        return 1
    print(f"\nDone. Built {len(tags)} corpora under {base_out}/")
    return 0


# ---------------------------------------------------------------------------
# publish — pack the built corpora and upload them to Hugging Face
# ---------------------------------------------------------------------------

def cmd_publish(args: argparse.Namespace) -> int:
    """Thin wrapper over scripts/publish_to_hf.py; defaults to publishing all sizes."""
    import importlib.util

    script = Path(__file__).parent.parent / "scripts" / "publish_to_hf.py"
    if not script.exists():
        print(
            f"error: publisher not found at {script}. "
            "Publishing runs from a repo checkout (the scripts/ dir).",
            file=sys.stderr,
        )
        return 1

    spec = importlib.util.spec_from_file_location("_cadquarry_publish_hf", script)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        print(f"error: could not load publisher: {exc}", file=sys.stderr)
        return 1

    # Translate our args into the publisher's argv. Default (no --sizes) is the
    # whole ladder, so a bare `cadquarry publish` uploads everything that was
    # built — the natural pairing with a bare `cadquarry build`.
    argv: list[str] = ["--all"] if not args.sizes else ["--sizes", *args.sizes]
    if args.repo_id:
        argv += ["--repo-id", args.repo_id]
    if args.corpus_root:
        argv += ["--corpus-root", args.corpus_root]
    if args.code_only:
        argv.append("--code-only")
    if args.no_renders:
        argv.append("--no-renders")
    if args.no_stl:
        argv.append("--no-stl")
    if args.no_step:
        argv.append("--no-step")
    if args.private:
        argv.append("--private")
    if args.dry_run:
        argv.append("--dry-run")
    if args.verbose:
        argv.append("--verbose")

    return int(module.main(argv))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    from .execute import execute_file, get_stl

    py_path = Path(args.part)
    if not py_path.exists():
        print(f"error: {py_path} not found", file=sys.stderr)
        return 1

    overrides: dict = {}
    for kv in args.set or []:
        if "=" not in kv:
            print(f"error: --set must be KEY=VALUE, got {kv!r}", file=sys.stderr)
            return 1
        k, v = kv.split("=", 1)
        # Try to parse as number / bool
        try:
            overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            overrides[k] = v

    export_fmt = args.export
    export_out = Path(args.out) if args.out else None

    if export_fmt:
        if export_fmt.lower() == "stl":
            stl = get_stl(py_path, params=overrides or None, timeout=args.timeout)
            if stl is None:
                print("error: execution failed or produced no geometry", file=sys.stderr)
                return 1
            out_path = export_out or py_path.with_suffix(".stl")
            out_path.write_bytes(stl)
            print(f"Written {out_path} ({len(stl):,} bytes)")
            return 0
        else:
            from .export import export_part
            out_dir = export_out or py_path.parent / "geometry"
            exported = export_part(
                py_path, out_dir, formats=[export_fmt],
                params=overrides or None, timeout=args.timeout,
            )
            for fmt, path in exported.items():
                print(f"Written {path}")
            return 0

    # Just execute and report
    result = execute_file(py_path, params=overrides or None, timeout=args.timeout)
    if not result.success:
        print(f"FAILED: {result.error}", file=sys.stderr)
        if result.traceback:
            print(result.traceback, file=sys.stderr)
        return 1

    dims = result.bbox_dims
    print(f"OK  volume={result.volume:.2f} mm³  "
          f"bbox={dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f} mm  "
          f"faces={result.n_faces}  edges={result.n_edges}")
    return 0


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    from .viewer.server import serve_customizer, serve_gallery
    from .dataset import load_manifest

    port = args.port

    if args.part:
        py_path = Path(args.part)
        if not py_path.exists():
            print(f"error: {py_path} not found", file=sys.stderr)
            return 1
        print(f"Starting CadQuarry customizer at http://localhost:{port}")
        print(f"  Part: {py_path}")
        print("  Press Ctrl+C to stop.")
        serve_customizer(py_path, port=port)
    elif args.dataset:
        dataset_dir = Path(args.dataset)
        if not dataset_dir.exists():
            print(f"error: {dataset_dir} not found", file=sys.stderr)
            return 1
        records = load_manifest(dataset_dir)
        print(f"Starting CadQuarry gallery at http://localhost:{port}")
        print(f"  Dataset: {dataset_dir}  ({len(records)} parts)")
        print("  Press Ctrl+C to stop.")
        serve_gallery(dataset_dir, port=port)
    else:
        print("error: supply --part or --dataset", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> int:
    from .export import export_corpus_geometry

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"error: {dataset_dir} not found", file=sys.stderr)
        return 1

    formats = [f.strip() for f in args.formats.split(",")]
    print(f"Exporting {formats} for corpus in {dataset_dir} …")
    counts = export_corpus_geometry(
        dataset_dir, formats=formats,
        n_workers=args.workers, timeout=args.timeout, verbose=args.verbose,
    )
    for fmt, n in counts.items():
        print(f"  {fmt}: {n} files written")
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> int:
    from .execute import execute_file
    from .filter import is_valid, compute_signature, DedupStore, QualityConfig
    from .dataset import load_manifest

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"error: {dataset_dir} not found", file=sys.stderr)
        return 1

    records = load_manifest(dataset_dir)
    print(f"Verifying {len(records)} parts in {dataset_dir} …")

    qcfg = QualityConfig()
    dedup = DedupStore()
    ok = fail = dup = 0

    for rec in records:
        py_rel = rec.get("paths", {}).get("py")
        if not py_rel:
            continue
        py_path = dataset_dir / py_rel
        result = execute_file(py_path, timeout=args.timeout)
        valid, reason = is_valid(result, qcfg)
        if not valid:
            fail += 1
            print(f"  FAIL {rec['part_id']}: {reason}")
            continue
        sig = compute_signature(result)
        if dedup.is_duplicate(rec.get("ir_hash", ""), sig):
            dup += 1
            print(f"  DUP  {rec['part_id']}")
            continue
        dedup.register(rec.get("ir_hash", ""), sig)
        ok += 1

    print(f"\nResults: {ok} ok, {fail} invalid, {dup} duplicates")
    return 0 if fail == 0 else 1


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> int:
    import ast
    from pathlib import Path

    py_path = Path(args.part)
    if not py_path.exists():
        print(f"error: {py_path} not found", file=sys.stderr)
        return 1

    # Parse PARAMS from source without executing it
    source = py_path.read_text(encoding="utf-8")
    ns: dict = {}
    try:
        exec(compile(source, str(py_path), "exec"), ns)
    except Exception as exc:
        print(f"error parsing {py_path}: {exc}", file=sys.stderr)
        return 1

    params = ns.get("PARAMS", {})
    print(f"Part: {py_path.stem}")
    print(f"Parameters ({len(params)}):")
    groups: dict[str, list] = {}
    for name, spec in params.items():
        g = spec.get("group", "Body")
        groups.setdefault(g, []).append((name, spec))
    for group, entries in groups.items():
        print(f"\n  [{group}]")
        for name, spec in entries:
            typ = spec.get("type", "?")
            default = spec.get("default")
            label = spec.get("label", name)
            if typ in ("float", "int"):
                lo, hi = spec.get("min", "?"), spec.get("max", "?")
                step = spec.get("step", "")
                print(f"    {name:<20} {label:<30} default={default}  range=[{lo}, {hi}]  step={step}")
            elif typ == "bool":
                print(f"    {name:<20} {label:<30} default={default}")
            elif typ == "enum":
                choices = spec.get("choices", [])
                print(f"    {name:<20} {label:<30} default={default}  choices={choices}")
    return 0


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cadquarry",
        description="CadQuarry — parametric CAD program generator",
    )
    parser.add_argument("--version", action="version", version=f"cadquarry {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # generate
    gen = sub.add_parser("generate", help="Generate a corpus of parametric CAD parts")
    gen.add_argument("--count", type=int, default=100, metavar="N", help="Parts to generate (default: 100)")
    gen.add_argument("--seed", type=int, default=42, metavar="SEED", help="Random seed (default: 42)")
    gen.add_argument("--out", default="dataset", metavar="DIR", help="Output directory")
    gen.add_argument("--config", default=None, metavar="TOML", help="Config file (default: configs/default.toml)")
    gen.add_argument("--timeout", type=float, default=120.0, metavar="SEC", help="Execution timeout per part (hang detection; generous for slow gear/threaded parts)")
    gen.add_argument("--no-exec", action="store_true", help="Skip execution validation (faster, less safe)")
    gen.add_argument("--workers", type=int, default=0, metavar="N", help="Parallel execution workers (0 = auto)")
    gen.add_argument("--family", default=None, help="Force a part family (plate, revolved, block, …)")
    gen.add_argument("--tier", type=int, default=None, help="Force a complexity tier (0-3)")
    gen.add_argument("--verbose", "-v", action="store_true")

    # build
    bld = sub.add_parser(
        "build",
        help="Generate (and export geometry for) every corpus in the seed ladder in one pass",
    )
    bld.add_argument("--sizes", nargs="+", metavar="TAG", help="Subset of ladder tags to build (default: all)")
    bld.add_argument("--out", default="datasets", metavar="DIR", help="Base output dir; each corpus -> <DIR>/<tag>/ (default: datasets)")
    bld.add_argument("--formats", default="step,stl,render", help="Export formats per corpus (default: step,stl,render)")
    bld.add_argument("--no-export", action="store_true", help="Generate only; skip geometry export")
    bld.add_argument("--config", default=None, metavar="TOML", help="Config file (default: configs/default.toml)")
    bld.add_argument("--seeds", default=None, metavar="TOML", help="Seed list (default: the seeds/v*.toml matching this generator version)")
    bld.add_argument("--timeout", type=float, default=120.0, metavar="SEC", help="Per-part timeout for generate and export (hang detection; generous for slow gear/threaded parts)")
    bld.add_argument("--workers", type=int, default=0, metavar="N", help="Workers for generation and export (0 = auto)")
    bld.add_argument("--force", action="store_true", help="Regenerate even if a corpus already exists")
    bld.add_argument("--verbose", "-v", action="store_true")

    # publish
    pub = sub.add_parser(
        "publish",
        help="Pack and upload the built corpora to Hugging Face (defaults to all sizes)",
    )
    pub.add_argument("--sizes", nargs="+", metavar="TAG", help="Subset of ladder tags to publish (default: all built sizes)")
    pub.add_argument("--repo-id", default=None, metavar="REPO", help="HF dataset repo id (default: from the seed list / CADQUARRY_HF_REPO env)")
    pub.add_argument("--out", dest="corpus_root", default="datasets", metavar="DIR", help="Dir holding built corpora as <DIR>/<tag>/ (default: datasets)")
    pub.add_argument("--code-only", action="store_true", help="Publish only code+metadata (skip geometry variants)")
    pub.add_argument("--no-renders", action="store_true", help="Skip render variants")
    pub.add_argument("--no-stl", action="store_true", help="Skip STL variants")
    pub.add_argument("--no-step", action="store_true", help="Skip STEP variants")
    pub.add_argument("--private", action="store_true", help="Create the dataset repo as private")
    pub.add_argument("--dry-run", action="store_true", help="Pack locally without uploading")
    pub.add_argument("--verbose", "-v", action="store_true")

    # run
    run = sub.add_parser("run", help="Execute a part with optional parameter overrides")
    run.add_argument("part", help="Path to the .py part file")
    run.add_argument("--set", action="append", metavar="KEY=VALUE", help="Parameter override; repeat for multiple (e.g. --set plate_w=60 --set n_holes=6)")
    run.add_argument("--export", metavar="FORMAT", help="Export format: stl, step, svg, render, pointcloud")
    run.add_argument("--out", default=None, metavar="PATH", help="Output path for export")
    run.add_argument("--timeout", type=float, default=120.0)

    # serve
    srv = sub.add_parser("serve", help="Launch live customizer (part) or gallery (dataset)")
    srv_group = srv.add_mutually_exclusive_group(required=True)
    srv_group.add_argument("--part", metavar="PART.py", help="Part to open in customizer")
    srv_group.add_argument("--dataset", metavar="DIR", help="Dataset directory to browse")
    srv.add_argument("--port", type=int, default=8765)

    # export
    exp = sub.add_parser("export", help="Export geometry artifacts for an existing corpus")
    exp.add_argument("dataset", metavar="DIR")
    exp.add_argument("--formats", default="step,stl,render", help="Comma-separated: step,stl,svg,pointcloud,render (default: step,stl,render)")
    exp.add_argument("--timeout", type=float, default=120.0)
    exp.add_argument("--workers", type=int, default=0, metavar="N", help="Parallel export workers (0 = auto)")
    exp.add_argument("--verbose", "-v", action="store_true")

    # verify
    ver = sub.add_parser("verify", help="Re-execute corpus and check validity + signatures")
    ver.add_argument("dataset", metavar="DIR")
    ver.add_argument("--timeout", type=float, default=120.0)

    # info
    inf = sub.add_parser("info", help="Print parameter schema for a part")
    inf.add_argument("part", metavar="PART.py")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "generate": cmd_generate,
        "build": cmd_build,
        "publish": cmd_publish,
        "run": cmd_run,
        "serve": cmd_serve,
        "export": cmd_export,
        "verify": cmd_verify,
        "info": cmd_info,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
