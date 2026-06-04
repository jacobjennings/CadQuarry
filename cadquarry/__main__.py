"""
CadQuarry CLI

  cadquarry generate  — generate a corpus
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


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> int:
    from .compose import compose
    from .emit import emit_source, write_part, emit_meta_json, emit_params_json
    from .execute import execute_source
    from .filter import is_valid, compute_signature, DedupStore, QualityConfig
    from .dataset import DatasetWriter

    config = _load_config(args.config)
    out_dir = Path(args.out)
    seed = args.seed
    count = args.count
    timeout = args.timeout
    no_exec = args.no_exec
    family = args.family or None
    tier = args.tier if args.tier is not None else None
    verbose = args.verbose

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

    while accepted < count:
        # Derive a per-part seed from the global seed + attempt index
        part_seed = (seed * 1_000_003 + attempted) & 0xFFFFFFFF
        part = compose(part_seed, index=attempted, config=config, family=family, tier=tier)
        attempted += 1

        ir_hash = part.ir_hash()
        if dedup.is_duplicate(ir_hash):
            skipped_dup += 1
            continue

        code = emit_source(part)

        geo_sig = None
        if not no_exec:
            result = execute_source(code, timeout=timeout)
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
            geo_sig = geo_sig_obj.to_dict()
        else:
            dedup.register(ir_hash)

        paths = write_part(part, out_dir, geometry_signature=geo_sig)
        meta = emit_meta_json(part, geo_sig)
        writer.add_record(meta, paths, geo_sig)
        accepted += 1

        if verbose or accepted % max(1, count // 20) == 0:
            print(
                f"  [{accepted:>{len(str(count))}}/{count}] "
                f"{part.id}  family={part.metadata.family}  tier={part.metadata.tier}"
            )

        if attempted > count * 10 and accepted < count // 2:
            print(
                f"[cadquarry] warning: high rejection rate "
                f"({skipped_invalid} invalid, {skipped_dup} dup of {attempted} attempts). "
                "Consider relaxing the quality config or widening parameter ranges."
            )
            break

    manifest = writer.finalize()
    print(
        f"\nDone. {accepted} parts written to {out_dir}\n"
        f"  attempts={attempted}  invalid={skipped_invalid}  duplicates={skipped_dup}\n"
        f"  manifest: {manifest}"
    )
    return 0


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
        timeout=args.timeout, verbose=args.verbose,
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
    gen.add_argument("--timeout", type=float, default=30.0, metavar="SEC", help="Execution timeout per part")
    gen.add_argument("--no-exec", action="store_true", help="Skip execution validation (faster, less safe)")
    gen.add_argument("--family", default=None, help="Force a part family (plate, revolved, block, …)")
    gen.add_argument("--tier", type=int, default=None, help="Force a complexity tier (0-3)")
    gen.add_argument("--verbose", "-v", action="store_true")

    # run
    run = sub.add_parser("run", help="Execute a part with optional parameter overrides")
    run.add_argument("part", help="Path to the .py part file")
    run.add_argument("--set", action="append", metavar="KEY=VALUE", help="Parameter override; repeat for multiple (e.g. --set plate_w=60 --set n_holes=6)")
    run.add_argument("--export", metavar="FORMAT", help="Export format: stl, step")
    run.add_argument("--out", default=None, metavar="PATH", help="Output path for export")
    run.add_argument("--timeout", type=float, default=30.0)

    # serve
    srv = sub.add_parser("serve", help="Launch live customizer (part) or gallery (dataset)")
    srv_group = srv.add_mutually_exclusive_group(required=True)
    srv_group.add_argument("--part", metavar="PART.py", help="Part to open in customizer")
    srv_group.add_argument("--dataset", metavar="DIR", help="Dataset directory to browse")
    srv.add_argument("--port", type=int, default=8765)

    # export
    exp = sub.add_parser("export", help="Export geometry artifacts for an existing corpus")
    exp.add_argument("dataset", metavar="DIR")
    exp.add_argument("--formats", default="step,stl", help="Comma-separated: step,stl,svg,pointcloud,render")
    exp.add_argument("--timeout", type=float, default=60.0)
    exp.add_argument("--verbose", "-v", action="store_true")

    # verify
    ver = sub.add_parser("verify", help="Re-execute corpus and check validity + signatures")
    ver.add_argument("dataset", metavar="DIR")
    ver.add_argument("--timeout", type=float, default=30.0)

    # info
    inf = sub.add_parser("info", help="Print parameter schema for a part")
    inf.add_argument("part", metavar="PART.py")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "generate": cmd_generate,
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
