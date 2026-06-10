"""
CadQuarry CLI

  cadquarry generate     — generate a corpus
  cadquarry build        — generate + export every corpus in the seed ladder
  cadquarry build_sample — rebuild the committed GitHub Pages sample corpus
  cadquarry publish      — pack + upload the built corpora to Hugging Face
  cadquarry run          — execute a single part with optional param overrides
  cadquarry serve        — launch live customizer for a part (or gallery for corpus)
  cadquarry export       — export geometry artifacts for an existing corpus
  cadquarry annotate     — add procedural dimension metadata to an existing corpus
  cadquarry verify       — re-execute corpus and check validity + signatures
  cadquarry info         — print part metadata / parameter schema

Commands can be chained to run in sequence, e.g. a full Pages + dataset refresh:

  cadquarry build_sample build publish

Chained commands run with their defaults (don't interleave per-command flags).
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


def _render_opts(args: argparse.Namespace) -> dict:
    """
    Resolve the render view/pass/edge CLI flags into export_corpus_geometry
    kwargs.  Returns ``{}`` for commands that don't expose them so callers can
    splat it unconditionally.
    """
    from .export import resolve_passes, resolve_views

    if not hasattr(args, "render_views"):
        return {}
    return {
        "render_views": resolve_views(args.render_views),
        "render_passes": resolve_passes(args.render_passes),
        "edge_crispness": args.edge_crispness,
    }


def _add_render_flags(p: argparse.ArgumentParser, default_views: str) -> None:
    """Attach the shared render view/pass/edge selection flags to a subparser."""
    p.add_argument(
        "--render-views", default=default_views, metavar="SPEC",
        help="View set for renders: a preset (all, iso-corners, ortho, cad) or a "
             f"comma list of view names (default: {default_views})",
    )
    p.add_argument(
        "--render-passes", default="shaded,normal,depth,edge", metavar="LIST",
        help="Render passes to emit per view, comma-separated: "
             "shaded,normal,depth,edge (default: all four)",
    )
    p.add_argument(
        "--edge-crispness", type=float, default=0.6, metavar="0..1",
        help="Feature-edge crispness; lower for noisy real scans (default: 0.6)",
    )


def _default_seeds_path() -> Path:
    """The active seed list (``seeds/seeds.toml``); shared with the publisher."""
    from .publish import default_seeds_path
    return default_seeds_path()


def _load_publish_ladder(seeds_path: Path) -> dict[str, dict]:
    """
    Return {tag: {"count": int, "seed": int}} from [[publish.corpus]].  In
    ``mode = "prefix"`` seed lists the per-tag seed defaults to ``base_seed``
    (all sizes are prefixes of one base corpus).
    """
    with open(seeds_path, "rb") as f:
        data = tomllib.load(f)
    pub = data.get("publish", {})
    base_seed = pub.get("base_seed")
    ladder: dict[str, dict] = {}
    for entry in pub.get("corpus", []):
        ladder[str(entry["tag"])] = {
            "count": int(entry["count"]),
            "seed": int(entry.get("seed", base_seed)),
        }
    return ladder


def _load_publish_meta(seeds_path: Path) -> dict:
    """
    Return the publish-ladder mode metadata: ``{mode, base_seed, base_tag,
    base_count}``.  ``mode`` is ``"prefix"`` (sizes are nested prefixes of one
    base corpus) or ``"independent"`` (legacy: each tag its own seed).
    """
    with open(seeds_path, "rb") as f:
        pub = tomllib.load(f).get("publish", {})
    counts = [int(e["count"]) for e in pub.get("corpus", [])]
    return {
        "mode": pub.get("mode", "independent"),
        "base_seed": pub.get("base_seed"),
        "base_tag": pub.get("base_tag", "base"),
        "base_count": max(counts) if counts else 0,
    }


def _load_corpus_list(seeds_path: Path) -> dict[str, dict]:
    """
    Return {name: {"seed", "count", "config"}} from the [[corpus]] entries.  The
    canonical sample corpora share one ``[meta].sample_seed`` (only count varies,
    so each size is a prefix of the next); a per-entry ``seed`` still overrides.
    """
    with open(seeds_path, "rb") as f:
        data = tomllib.load(f)
    meta = data.get("meta", {})
    default_config = meta.get("config")
    sample_seed = meta.get("sample_seed")
    corpora: dict[str, dict] = {}
    for entry in data.get("corpus", []):
        corpora[str(entry["name"])] = {
            "seed": int(entry.get("seed", sample_seed)),
            "count": int(entry["count"]),
            "config": entry.get("config", default_config),
        }
    return corpora


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

    # --extend: resume an existing corpus and append up to the new (larger)
    # --count.  The attempt stream is a pure function of (seed, i) and dedup runs
    # in strict attempt order, so replaying from the recorded high-water attempt
    # index with the existing dedup state appends parts that are bit-identical to
    # a from-scratch count=N run — the existing parts are never recomputed.
    if getattr(args, "extend", False):
        from .dataset import load_manifest, load_state
        from .filter import GeometrySignature

        state = load_state(out_dir)
        records = load_manifest(out_dir)
        if state is None or not records:
            print(f"error: --extend given but no resumable corpus in {out_dir} "
                  f"(need manifest.jsonl + state.json)", file=sys.stderr)
            return 1
        if not getattr(args, "force", False):
            if int(state.get("seed", seed)) != seed:
                print(f"error: existing corpus seed {state.get('seed')} != {seed}; "
                      f"refusing to extend (use --force to override)", file=sys.stderr)
                return 1
            if str(state.get("generator_version")) != __version__:
                print(f"error: existing corpus generator {state.get('generator_version')} "
                      f"!= {__version__}; refusing to extend (use --force)", file=sys.stderr)
                return 1
        for rec in records:
            gs = rec.get("geometry_signature")
            geo = GeometrySignature.from_dict(gs) if gs else None
            if rec.get("ir_hash"):
                dedup.register(rec["ir_hash"], geo)
        writer.preload(records)
        accepted = len(records)
        attempted = int(state.get("attempted", accepted))
        if accepted >= count:
            print(f"Corpus in {out_dir} already has {accepted} parts "
                  f"(>= target {count}); nothing to do.")
            return 0
        print(f"CadQuarry v{__version__} — extending {out_dir} from {accepted} "
              f"to {count} parts (seed={seed})")
    else:
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
    if accepted:
        pbar.update(accepted)  # extend: bar starts at the preloaded count

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
    manifest = writer.finalize(attempted=attempted)
    print(
        f"\nDone. {accepted} parts written to {out_dir}\n"
        f"  attempts={attempted}  invalid={skipped_invalid}  duplicates={skipped_dup}\n"
        f"  manifest: {manifest}"
    )
    return 0


# ---------------------------------------------------------------------------
# build — generate (and export) every corpus in the seed ladder in one pass
# ---------------------------------------------------------------------------

def _corpus_state(out_dir: Path) -> tuple[int, str | None]:
    """Return (existing part count, stale generator version or None) for a corpus dir."""
    manifest = out_dir / "manifest.jsonl"
    if not manifest.exists():
        return 0, None
    lines = manifest.open(encoding="utf-8").read().splitlines()
    stale = None
    if lines:
        try:
            gv = json.loads(lines[0]).get("generator_version")
            if gv != __version__:
                stale = gv
        except Exception:
            pass
    return len(lines), stale


def _build_prefix_base(args: argparse.Namespace, meta: dict, ladder: dict) -> int:
    """
    Build (or grow) the single base corpus that every size tag is a prefix of.
    ``--extend-to N`` sets the target part count (for an initial build or to
    grow an existing base); a bare build targets the largest ladder tag.
    """
    from .export import export_corpus_geometry

    base_tag = meta["base_tag"]
    base_seed = meta["base_seed"]
    target = getattr(args, "extend_to", None) or meta["base_count"]
    out_dir = Path(args.out) / base_tag
    do_export = not args.no_export
    formats = [f.strip() for f in args.formats.split(",")] if do_export else None

    print(
        f"CadQuarry v{__version__} — prefix ladder: building base corpus "
        f"'{base_tag}' to {target:,} parts (seed {base_seed}) -> {out_dir}"
    )
    if do_export:
        print(f"  export formats: {formats}")

    existing_n, stale = _corpus_state(out_dir)
    skip_gen = False
    extend = False
    if stale is not None and not args.force:
        print(f"  · regenerating from scratch (existing base is generator "
              f"{stale}, current is {__version__})")
    elif existing_n >= target and not args.force:
        print(f"  · reusing existing base ({existing_n:,} parts >= {target:,})")
        skip_gen = True
    elif existing_n > 0:
        print(f"  · extending base from {existing_n:,} to {target:,} parts "
              f"(reusing existing parts + geometry)")
        extend = True

    if not skip_gen:
        gen_args = argparse.Namespace(
            count=target, seed=base_seed, out=str(out_dir),
            config=args.config, timeout=args.timeout, no_exec=False,
            workers=args.workers, family=None, tier=None, verbose=args.verbose,
            extend=extend, force=args.force,
        )
        if cmd_generate(gen_args) != 0:
            print("error: base generation failed", file=sys.stderr)
            return 1

    if do_export:
        print(f"  · exporting {formats} (incremental — skips already-exported parts) …")
        try:
            counts = export_corpus_geometry(
                out_dir, formats=formats, n_workers=args.workers,
                timeout=args.timeout, verbose=args.verbose, **_render_opts(args),
            )
            for fmt, cnt in counts.items():
                print(f"    {fmt}: {cnt} files written")
        except Exception as exc:
            print(f"error: base export failed: {exc}", file=sys.stderr)
            return 1

    sizes = ", ".join(sorted(ladder, key=_tag_to_int))
    print(
        f"\nDone. Base corpus '{base_tag}' built at {out_dir}.\n"
        f"  `cadquarry publish` slices the size ladder ({sizes}) from this one "
        f"corpus — no per-size rebuild.\n"
        f"  Grow it later with `cadquarry build --extend-to <N>` (reuses everything)."
    )
    return 0


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

    # Prefix-mode ladders are one base corpus + zero-cost slices: build the base
    # once (optionally growing it with --extend-to) instead of per-tag corpora.
    meta = _load_publish_meta(seeds_path)
    if meta["mode"] == "prefix":
        return _build_prefix_base(args, meta, ladder)

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
                    timeout=args.timeout, verbose=args.verbose, **_render_opts(args),
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
# build_sample — regenerate the committed GitHub Pages sample corpus
# ---------------------------------------------------------------------------

def cmd_build_sample(args: argparse.Namespace) -> int:
    """
    Rebuild the committed sample corpus that backs the GitHub Pages preview.

    GitHub Pages serves the repo as-is: ``index.html`` redirects to
    ``sample/<name>/preview.html``, a self-contained gallery that fetches the
    sibling ``manifest.jsonl`` + ``geometry/*.stl`` + ``renders/`` at runtime.
    So "updating the page" means regenerating that committed corpus and its
    geometry/renders. This command does exactly that, deterministically, from
    the ``[[corpus]]`` entry in the active seed list.
    """
    from .export import export_corpus_geometry

    seeds_path = Path(args.seeds) if args.seeds else _default_seeds_path()
    if not seeds_path.exists():
        print(f"error: seed list {seeds_path} not found", file=sys.stderr)
        return 1

    corpora = _load_corpus_list(seeds_path)
    spec = corpora.get(args.name)
    if spec is None:
        avail = ", ".join(corpora) or "(none)"
        print(
            f"error: no [[corpus]] named {args.name!r} in {seeds_path}. "
            f"Available: {avail}",
            file=sys.stderr,
        )
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out) if args.out else repo_root / "sample" / args.name
    config = args.config or spec["config"]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    print(
        f"CadQuarry v{__version__} — rebuilding sample {args.name!r} "
        f"({spec['count']:,} parts, seed {spec['seed']}) -> {out_dir}"
    )
    if config:
        print(f"  config: {config}")
    print(f"  export formats: {formats}")

    gen_args = argparse.Namespace(
        count=spec["count"], seed=spec["seed"], out=str(out_dir),
        config=config, timeout=args.timeout, no_exec=False,
        workers=args.workers, family=None, tier=None, verbose=args.verbose,
    )
    rc = cmd_generate(gen_args)
    if rc != 0:
        print("error: sample generation failed", file=sys.stderr)
        return rc

    if formats:
        print(f"\n  · exporting {formats} …")
        try:
            counts = export_corpus_geometry(
                out_dir, formats=formats, n_workers=args.workers,
                timeout=args.timeout, verbose=args.verbose, **_render_opts(args),
            )
            for fmt, cnt in counts.items():
                print(f"    {fmt}: {cnt} files written")
        except Exception as exc:
            print(f"error: sample export failed: {exc}", file=sys.stderr)
            return 1

    print(
        f"\nDone. Sample {args.name!r} rebuilt at {out_dir}.\n"
        f"  GitHub Pages serves sample/{args.name}/preview.html — commit the "
        f"changes to publish."
    )
    return 0


# ---------------------------------------------------------------------------
# publish — pack the built corpora and upload them to Hugging Face
# ---------------------------------------------------------------------------

def cmd_publish(args: argparse.Namespace) -> int:
    """Pack the built corpora and upload them to HuggingFace (defaults to all sizes)."""
    from .publish import publish, PublishError

    # Accept both space- and comma-separated sizes: --sizes 1k 2k or --sizes 1k,2k.
    sizes = None
    if args.sizes:
        sizes = [s for tok in args.sizes for s in tok.split(",") if s]

    try:
        return publish(
            sizes=sizes,
            all_sizes=not sizes,  # bare `publish` -> every built size
            repo_id=args.repo_id,
            corpus_root=args.corpus_root,
            code_only=args.code_only,
            no_renders=args.no_renders,
            no_stl=args.no_stl,
            no_step=args.no_step,
            private=args.private,
            dry_run=args.dry_run,
            upload_only=args.upload_only,
            verbose=args.verbose,
        )
    except PublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


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
        **_render_opts(args),
    )
    for fmt, n in counts.items():
        print(f"  {fmt}: {n} files written")
    return 0


# ---------------------------------------------------------------------------
# annotate — supplement an existing corpus with dimension metadata
# ---------------------------------------------------------------------------

def cmd_annotate(args: argparse.Namespace) -> int:
    """
    Supplement an existing corpus with the procedural dimension metadata
    *in place* — and upgrade it to the current generator version — without
    re-running any geometry export (STEP / STL / renders / point clouds).

    Each part is recomposed deterministically from its stored seed + index
    (composition is a pure function of seed + config), so no CAD execution is
    needed.  For every part we re-emit the cheap text artifacts — parts/*.py,
    params/*.params.json, meta/*.meta.json (now carrying a ``dimensions`` block)
    and a new dims/*.txt sidecar — while preserving the geometry artifacts and
    their manifest paths untouched.

    The corpus must be annotated with the same --config it was generated with.
    Re-composition is validated against the stored ir_hash (ignoring the
    generator-version field, which the upgrade is expected to change); a part
    whose geometry-relevant IR diverges is left as-is and reported, so a wrong
    --config can never overwrite good source with mismatched dimensions.
    """
    import json as _json

    from .compose import compose
    from .dataset import load_manifest
    from .emit import emit_meta_json, emit_params_json, emit_source
    from .progress import progress_bar

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"error: {dataset_dir} not found", file=sys.stderr)
        return 1

    records = load_manifest(dataset_dir)
    if not records:
        print(f"error: no manifest.jsonl records in {dataset_dir}", file=sys.stderr)
        return 1

    config = _load_config(args.config)
    for sub in ("parts", "params", "meta", "dims"):
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)
    # Geometry path keys are preserved verbatim; the text artifacts are re-emitted.
    _TEXT_KEYS = ("py", "params", "meta", "dims")

    print(
        f"CadQuarry v{__version__} — annotating {len(records)} parts in "
        f"{dataset_dir} with dimension metadata (no geometry re-export)"
    )

    annotated = 0
    mismatched = 0
    skipped = 0
    new_records: list[dict] = []
    pbar = progress_bar(total=len(records), desc="  annotate", unit="part")

    for rec in records:
        pid = rec.get("part_id")
        seed = rec.get("seed")
        if not pid or seed is None:
            skipped += 1
            new_records.append(rec)
            pbar.update(1)
            continue

        try:
            index = int(str(pid).rsplit("_", 1)[1])
        except (IndexError, ValueError):
            index = 0

        try:
            part = compose(int(seed), index=index, config=config)
        except Exception as exc:
            if args.verbose:
                pbar.write(f"  SKIP {pid}: recompose failed ({exc})")
            skipped += 1
            new_records.append(rec)
            pbar.update(1)
            continue

        # Guard: a divergent ir_hash means we recomposed a different part than
        # what is on disk (usually the wrong --config), so the dims would lie.
        # ir_hash includes generator_version, which legitimately changes across a
        # version bump even though the geometry IR is identical — so compare
        # against the stored version label to isolate a true geometry divergence.
        stored_hash = rec.get("ir_hash")
        stored_gv = rec.get("generator_version") or rec.get("cadquarry_version")
        if stored_hash:
            current_gv = part.metadata.generator_version
            if stored_gv:
                part.metadata.generator_version = stored_gv
            geom_hash = part.ir_hash()
            part.metadata.generator_version = current_gv
            if geom_hash != stored_hash:
                mismatched += 1
                new_records.append(rec)
                if args.verbose:
                    pbar.write(f"  MISMATCH {pid}: ir_hash differs — left unannotated")
                pbar.update(1)
                continue

        geo_sig = rec.get("geometry_signature")
        meta = emit_meta_json(part, geo_sig)

        # Re-emit the text artifacts in place (geometry exports are left alone).
        (dataset_dir / "parts" / f"{pid}.py").write_text(
            emit_source(part), encoding="utf-8"
        )
        (dataset_dir / "params" / f"{pid}.params.json").write_text(
            _json.dumps(emit_params_json(part), indent=2), encoding="utf-8"
        )
        (dataset_dir / "meta" / f"{pid}.meta.json").write_text(
            _json.dumps(meta, indent=2), encoding="utf-8"
        )
        (dataset_dir / "dims" / f"{pid}.txt").write_text(
            meta["dimensions"]["text"] + "\n", encoding="utf-8"
        )

        # Rebuild the manifest record: recomposed meta + preserved geometry
        # paths + refreshed text-artifact paths.
        new_rec = dict(meta)
        paths = {k: v for k, v in rec.get("paths", {}).items() if k not in _TEXT_KEYS}
        paths["py"] = f"parts/{pid}.py"
        paths["params"] = f"params/{pid}.params.json"
        paths["meta"] = f"meta/{pid}.meta.json"
        paths["dims"] = f"dims/{pid}.txt"
        new_rec["paths"] = paths
        new_records.append(new_rec)
        annotated += 1
        pbar.update(1)

    pbar.close()

    # Rewrite the manifest atomically (write then replace).
    manifest_path = dataset_dir / "manifest.jsonl"
    tmp = manifest_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in new_records:
            f.write(_json.dumps(rec) + "\n")
    tmp.replace(manifest_path)

    print(
        f"\nDone. annotated={annotated}  skipped={skipped}  "
        f"ir_hash mismatches={mismatched}\n"
        f"  dims sidecars: {dataset_dir / 'dims'}/\n"
        f"  manifest: {manifest_path}"
    )
    if mismatched:
        print(
            f"  note: {mismatched} parts were left unannotated because their "
            f"recomposed (geometry) IR differed — re-run with the --config used "
            f"to generate this corpus."
        )
    return 0 if mismatched == 0 else 1


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
    gen.add_argument("--extend", action="store_true",
                     help="Resume an existing corpus in --out and append parts up to --count "
                          "(reuses prior parts; replays the deterministic attempt stream)")
    gen.add_argument("--force", action="store_true",
                     help="With --extend, proceed even if the existing corpus was made by a "
                          "different seed or generator version")
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
    bld.add_argument("--seeds", default=None, metavar="TOML", help="Seed list (default: seeds/seeds.toml)")
    bld.add_argument("--timeout", type=float, default=120.0, metavar="SEC", help="Per-part timeout for generate and export (hang detection; generous for slow gear/threaded parts)")
    bld.add_argument("--workers", type=int, default=0, metavar="N", help="Workers for generation and export (0 = auto)")
    bld.add_argument("--force", action="store_true", help="Regenerate even if a corpus already exists")
    bld.add_argument("--extend-to", type=int, default=None, metavar="N",
                     help="Prefix-ladder mode: set the base-corpus target part count "
                          "(builds or grows the base to N, reusing existing parts)")
    _add_render_flags(bld, default_views="all")
    bld.add_argument("--verbose", "-v", action="store_true")

    # build_sample
    smp = sub.add_parser(
        "build_sample",
        help="Rebuild the committed sample corpus behind the GitHub Pages preview",
    )
    smp.add_argument("--name", default="demo-1k", metavar="NAME", help="[[corpus]] entry to rebuild (default: demo-1k)")
    smp.add_argument("--out", default=None, metavar="DIR", help="Output dir (default: sample/<name>/ in the repo)")
    smp.add_argument("--formats", default="step,stl,render", help="Export formats (default: step,stl,render)")
    smp.add_argument("--config", default=None, metavar="TOML", help="Config file (default: the corpus entry's config)")
    smp.add_argument("--seeds", default=None, metavar="TOML", help="Seed list (default: seeds/seeds.toml)")
    smp.add_argument("--timeout", type=float, default=120.0, metavar="SEC", help="Per-part timeout for generate and export")
    smp.add_argument("--workers", type=int, default=0, metavar="N", help="Workers for generation and export (0 = auto)")
    # The Pages preview gallery references the full eight-view set, so keep it.
    _add_render_flags(smp, default_views="all")
    smp.add_argument("--verbose", "-v", action="store_true")

    # publish
    pub = sub.add_parser(
        "publish",
        help="Pack and upload the built corpora to Hugging Face (defaults to all sizes)",
    )
    pub.add_argument("--sizes", nargs="+", metavar="TAG", help="Subset of ladder tags to publish, space- or comma-separated (default: all built sizes)")
    pub.add_argument("--repo-id", default=None, metavar="REPO", help="HF dataset repo id (default: from the seed list / CADQUARRY_HF_REPO env)")
    pub.add_argument("--out", dest="corpus_root", default="datasets", metavar="DIR", help="Dir holding built corpora as <DIR>/<tag>/ (default: datasets)")
    pub.add_argument("--code-only", action="store_true", help="Publish only code+metadata (skip geometry variants)")
    pub.add_argument("--no-renders", action="store_true", help="Skip render variants")
    pub.add_argument("--no-stl", action="store_true", help="Skip STL variants")
    pub.add_argument("--no-step", action="store_true", help="Skip STEP variants")
    pub.add_argument("--private", action="store_true", help="Create the dataset repo as private")
    pub.add_argument("--dry-run", action="store_true", help="Pack locally without uploading")
    pub.add_argument("--upload-only", action="store_true", help="Skip packing; upload the already-packed staging tree as-is (resume an interrupted upload)")
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
    _add_render_flags(exp, default_views="all")
    exp.add_argument("--verbose", "-v", action="store_true")

    # annotate
    ann = sub.add_parser(
        "annotate",
        help="Supplement an existing corpus with procedural dimension metadata (no geometry re-export)",
    )
    ann.add_argument("dataset", metavar="DIR")
    ann.add_argument("--config", default=None, metavar="TOML", help="Config used to generate the corpus (default: configs/default.toml)")
    ann.add_argument("--verbose", "-v", action="store_true")

    # verify
    ver = sub.add_parser("verify", help="Re-execute corpus and check validity + signatures")
    ver.add_argument("dataset", metavar="DIR")
    ver.add_argument("--timeout", type=float, default=120.0)

    # info
    inf = sub.add_parser("info", help="Print parameter schema for a part")
    inf.add_argument("part", metavar="PART.py")

    return parser


HANDLERS = {
    "generate": cmd_generate,
    "build": cmd_build,
    "build_sample": cmd_build_sample,
    "publish": cmd_publish,
    "run": cmd_run,
    "serve": cmd_serve,
    "export": cmd_export,
    "annotate": cmd_annotate,
    "verify": cmd_verify,
    "info": cmd_info,
}


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def main() -> None:
    parser = build_parser()
    subparsers = _subparsers(parser)
    argv = sys.argv[1:]

    # Single-command / global path: no args, a global flag (e.g. --version), or
    # a lone command. Let argparse own help, --version, and error reporting.
    if not argv or argv[0] not in subparsers:
        args = parser.parse_args(argv)
        if args.command is None:
            parser.print_help()
            sys.exit(0)
        sys.exit(HANDLERS[args.command](args))

    # Command-chaining path: `cadquarry build_sample build publish` runs each in
    # order. We peel one command at a time; whatever a subparser doesn't consume
    # is treated as the start of the next command. Chained commands run with
    # their defaults — don't interleave per-command flags in a chain.
    chain: list[argparse.Namespace] = []
    rest = argv
    while rest:
        cmd = rest[0]
        if cmd not in subparsers:
            # Leftover that isn't a command (typo / stray flag). Re-parse the
            # whole thing so argparse emits its standard, helpful error.
            parser.parse_args(argv)
            sys.exit(2)  # unreachable: parse_args exits non-zero first
        ns, rest = subparsers[cmd].parse_known_args(rest[1:])
        ns.command = cmd
        chain.append(ns)

    multi = len(chain) > 1
    for ns in chain:
        if multi:
            print(f"\n==> cadquarry {ns.command}\n")
        rc = HANDLERS[ns.command](ns) or 0
        if rc != 0:
            sys.exit(rc)
    sys.exit(0)


if __name__ == "__main__":
    main()
