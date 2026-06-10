"""
Export geometry artifacts (STEP, STL, point cloud, render).

All geometry is exported by running the part's .py source in a subprocess;
this keeps the main process free of CadQuery imports and makes exports
independent of the generation pipeline.

Point cloud sampling uses trimesh (optional dependency).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Subprocess script for exporting a single part to multiple formats.
_EXPORT_RUNNER = """
import sys, json, os
import cadquery as cq

def _export():
    part_file = sys.argv[1]
    formats   = json.loads(sys.argv[2])
    out_dir   = sys.argv[3]
    overrides = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}

    ns = {}
    with open(part_file, encoding="utf-8") as f:
        exec(f.read(), ns)

    params = {k: v["default"] for k, v in ns["PARAMS"].items()}
    params.update(overrides)
    result = ns["build"](params)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(part_file))[0]
    exported = []

    if "step" in formats:
        p = os.path.join(out_dir, stem + ".step")
        cq.exporters.export(result, p)
        exported.append(p)
    if "stl" in formats:
        p = os.path.join(out_dir, stem + ".stl")
        cq.exporters.export(result, p)
        exported.append(p)
    if "svg" in formats:
        p = os.path.join(out_dir, stem + ".svg")
        cq.exporters.export(result, p, opt={"width": 400, "height": 300})
        exported.append(p)

    print(json.dumps({"success": True, "files": exported}))

try:
    _export()
except Exception as e:
    import traceback
    print(json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()}))
"""


# Formats produced directly from the B-rep in the CadQuery subprocess.
_BREP_FORMATS = {"step", "stl", "svg"}
# Formats derived from a triangulated mesh (need an STL first).
_MESH_FORMATS = {"pointcloud", "render"}

ALL_FORMATS = sorted(_BREP_FORMATS | _MESH_FORMATS)

# Default geometry formats produced by `cadquarry export` and export_part().
DEFAULT_FORMATS = ["step", "stl", "render"]

# Named viewpoints for multi-angle renders: (elevation_deg, azimuth_deg) for
# matplotlib's view_init.  Four canonical CAD views (front/top/right + a
# standard isometric) plus the four isometric corners — eight angles per part.
STANDARD_VIEWS: dict[str, tuple[float, float]] = {
    "front":  (0.0,   -90.0),
    "top":    (90.0,  -90.0),
    "right":  (0.0,     0.0),
    "iso":    (28.0,  -55.0),
    "iso_fr": (30.0,  -45.0),
    "iso_fl": (30.0, -135.0),
    "iso_br": (30.0,   45.0),
    "iso_bl": (30.0,  135.0),
}

# Named view sets selectable from the CLI / dataset path.  ``iso-corners`` is the
# default for model input: the four three-quarter corners are the most legible,
# least-degenerate angles (unlike the straight-on front/top/right trio, where a
# part collapses to a flat silhouette and edges are impossible to read).
VIEW_PRESETS: dict[str, list[str]] = {
    "all": list(STANDARD_VIEWS),
    "iso-corners": ["iso_fr", "iso_fl", "iso_br", "iso_bl"],
    "ortho": ["front", "top", "right"],
    "cad": ["front", "top", "right", "iso"],
}

# Output passes the renderer can emit per view (see cadquarry.render.PASSES).
RENDER_PASSES = ("shaded", "normal", "depth", "edge")


def resolve_views(spec: str | None) -> dict[str, tuple[float, float]]:
    """
    Resolve a view selection into a ``{name: (elev, azim)}`` mapping.

    ``spec`` may be a preset name (``all`` / ``iso-corners`` / ``ortho`` /
    ``cad``), a comma-separated list of individual view names, or None (=>
    ``all``).  Unknown names raise ValueError.
    """
    if spec is None:
        return dict(STANDARD_VIEWS)
    key = spec.strip().lower()
    if key in VIEW_PRESETS:
        names = VIEW_PRESETS[key]
    else:
        names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in STANDARD_VIEWS]
    if unknown:
        raise ValueError(
            f"Unknown view(s): {unknown}. Valid: {sorted(STANDARD_VIEWS)} "
            f"or presets {sorted(VIEW_PRESETS)}"
        )
    return {n: STANDARD_VIEWS[n] for n in names}


def resolve_passes(spec: str | None) -> tuple[str, ...]:
    """Resolve a comma-separated pass selection; None => ``shaded`` only."""
    if spec is None:
        return ("shaded",)
    names = [n.strip().lower() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in RENDER_PASSES]
    if unknown:
        raise ValueError(f"Unknown render pass(es): {unknown}. Valid: {list(RENDER_PASSES)}")
    # Preserve canonical order, de-duped.
    return tuple(p for p in RENDER_PASSES if p in names)


def render_filename(view: str, pass_name: str) -> str:
    """PNG filename for a (view, pass).  Shaded keeps the bare ``{view}.png`` name
    so existing galleries/manifests/parquet columns stay valid; other passes get
    a ``{view}_{pass}.png`` sibling."""
    return f"{view}.png" if pass_name == "shaded" else f"{view}_{pass_name}.png"


def render_deps_available() -> bool:
    """True if the optional GPU render stack (moderngl + EGL + trimesh) works."""
    from . import render as _render
    return _render.render_available()


def export_part(
    py_path: Path,
    out_dir: Path,
    formats: list[str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 60.0,
    n_points: int = 2048,
) -> dict[str, Path]:
    """
    Export geometry for a part.

    py_path:  Path to the generated .py file.
    out_dir:  Destination directory.
    formats:  Any of 'step', 'stl', 'svg', 'pointcloud', 'render'.
    params:   Parameter overrides; defaults if omitted.
    Returns:  {format: path} for successfully-written files.

    'pointcloud' and 'render' are derived from a triangulated mesh, so an STL
    is produced internally (and removed afterwards if not requested directly).

    'render' writes one PNG per standard viewpoint into a per-part directory
    (renders/{id}/{view}.png); the returned dict maps 'render' to that dir.
    """
    if formats is None:
        formats = list(DEFAULT_FORMATS)

    brep = [f for f in formats if f in _BREP_FORMATS]
    mesh = [f for f in formats if f in _MESH_FORMATS]
    unknown = [f for f in formats if f not in ALL_FORMATS]
    if unknown:
        raise ValueError(f"Unknown export format(s): {unknown}. Valid: {ALL_FORMATS}")

    # We need an STL on disk if the user asked for it or any mesh-derived format.
    stl_requested = "stl" in brep
    need_stl = stl_requested or bool(mesh)
    run_formats = list(brep)
    if need_stl and "stl" not in run_formats:
        run_formats.append("stl")

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as tf:
        tf.write(_EXPORT_RUNNER)
        script = tf.name

    out_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    try:
        if run_formats:
            proc = subprocess.run(
                [
                    sys.executable, script,
                    str(py_path),
                    json.dumps(run_formats),
                    str(out_dir),
                    json.dumps(params or {}),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = proc.stdout.strip()
            if not stdout:
                raise RuntimeError(f"No output from export subprocess. stderr: {proc.stderr[:300]}")
            data = json.loads(stdout)
            if not data.get("success"):
                raise RuntimeError(data.get("error", "unknown export error"))
            for p in data.get("files", []):
                exported[Path(p).suffix.lstrip(".")] = Path(p)

        # Derive mesh-based artifacts from the STL.
        if mesh:
            stl_path = exported.get("stl")
            if stl_path is None or not stl_path.exists():
                raise RuntimeError("STL needed for mesh export was not produced")
            stem = py_path.stem
            if "pointcloud" in mesh:
                pc_out = out_dir.parent / "pointclouds" / f"{stem}.ply" \
                    if out_dir.name == "geometry" else out_dir / f"{stem}.ply"
                exported["pointcloud"] = export_pointcloud(stl_path, pc_out, n_points=n_points)
            if "render" in mesh:
                rn_out = out_dir.parent / "renders" / stem \
                    if out_dir.name == "geometry" else out_dir / "renders" / stem
                export_renders(stl_path, rn_out)
                exported["render"] = rn_out

            # Drop the STL if it was only an intermediate.
            if not stl_requested and stl_path.exists():
                try:
                    stl_path.unlink()
                except OSError:
                    pass
                exported.pop("stl", None)

        return exported
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


def export_pointcloud(
    stl_path: Path,
    out_path: Path,
    n_points: int = 2048,
) -> Path:
    """
    Sample a point cloud from an STL file using trimesh.
    Requires the 'trimesh' optional dependency.
    """
    try:
        import trimesh
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Point cloud export requires trimesh: pip install trimesh"
        ) from exc

    mesh = trimesh.load_mesh(str(stl_path))
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write as PLY (binary) — compact and widely supported.
    pc = trimesh.PointCloud(pts)
    pc.export(str(out_path))
    return out_path


def export_renders(
    stl_path: Path,
    out_dir: Path,
    views: dict[str, tuple[float, float]] | None = None,
    size: int = 512,
    ssaa: int = 2,
    base_color: tuple[float, float, float] | None = None,
    passes: tuple[str, ...] = ("shaded",),
    edge_crispness: float = 0.6,
) -> dict[str, Path]:
    """
    Render thumbnail PNGs of a mesh from multiple viewpoints, headless.

    One PNG is written per (view, pass) into out_dir.  ``views`` maps a view name
    to (elevation_deg, azimuth_deg); defaults to ``STANDARD_VIEWS``.  ``passes``
    is any subset of ``RENDER_PASSES`` (``shaded``/``normal``/``depth``/``edge``);
    the shaded pass keeps the bare ``{view}.png`` name while the others land as
    ``{view}_{pass}.png`` siblings (see :func:`render_filename`).
    ``base_color`` overrides the surface color; when None a deterministic
    per-part color is derived from the file stem.  ``edge_crispness`` (0..1) tunes
    the feature-edge pass — lower it for noisy real scans.  Returns
    {key: png_path} keyed by view (shaded) or ``view_pass``.

    Rendering is done on the GPU via a headless EGL OpenGL context (moderngl):
    an orthographic deferred pipeline with a real depth buffer, screen-space
    ambient occlusion and soft hemispherical + positional-key lighting,
    supersampled for clean edges.  The G-buffer (view-space normals + depth) is
    reused across passes, and the mesh is uploaded once with only the camera
    moving between views, so extra views/passes cost little.  See
    :mod:`cadquarry.render`.
    """
    try:
        import trimesh
        from PIL import Image
        from . import render as _render
    except ImportError as exc:
        raise ImportError(
            "Render export requires moderngl, trimesh and Pillow: "
            "pip install 'cadquarry[export]'"
        ) from exc

    if views is None:
        views = STANDARD_VIEWS
    if base_color is None:
        base_color = _render.color_for(Path(stl_path).stem)

    mesh = trimesh.load_mesh(str(stl_path))
    # An STL is a flat triangle soup, but be defensive about scene-style loads.
    if hasattr(mesh, "dump"):
        mesh = mesh.dump(concatenate=True)

    renderer = _render.get_renderer(size=size, ssaa=ssaa)
    images = renderer.render_mesh(
        mesh.vertices, mesh.faces, views=views, base_color=base_color,
        passes=passes, edge_crispness=edge_crispness,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for view_name, by_pass in images.items():
        for pass_name, arr in by_pass.items():
            out_path = out_dir / render_filename(view_name, pass_name)
            Image.fromarray(arr, "RGBA").save(str(out_path))
            key = view_name if pass_name == "shaded" else f"{view_name}_{pass_name}"
            written[key] = out_path
    return written


def export_render(
    stl_path: Path,
    out_path: Path,
    size: int = 512,
    elev: float = 28.0,
    azim: float = -55.0,
) -> Path:
    """
    Render a single shaded thumbnail PNG of a mesh, headless.

    Thin single-view wrapper around :func:`export_renders`; kept for callers
    that want one fixed angle at an explicit file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_renders(stl_path, out_path.parent, views={out_path.stem: (elev, azim)}, size=size)
    return out_path


def export_corpus_geometry(
    dataset_dir: Path,
    formats: list[str] | None = None,
    n_workers: int = 0,
    timeout: float = 60.0,
    verbose: bool = False,
    n_points: int = 2048,
    render_views: dict[str, tuple[float, float]] | None = None,
    render_passes: tuple[str, ...] = ("shaded",),
    edge_crispness: float = 0.6,
    skip_existing: bool = True,
) -> dict[str, int]:
    """
    Export geometry for all parts in a dataset directory.

    B-rep formats (step/stl/svg) land in <dataset>/geometry/, point clouds in
    <dataset>/pointclouds/, and multi-angle renders in
    <dataset>/renders/{id}/{view}.png (one PNG per STANDARD_VIEWS angle).
    Returns counts of {format: n_exported}; 'render' counts parts rendered.

    The expensive per-part cost is importing cadquery and rebuilding the solid;
    we amortise it across a pool of persistent workers (see
    ``execute.WorkerPool``) so the whole corpus shares one set of imports.
    """
    from .execute import WorkerPool, default_worker_count

    if formats is None:
        formats = list(DEFAULT_FORMATS)

    brep = [f for f in formats if f in _BREP_FORMATS]
    mesh = [f for f in formats if f in _MESH_FORMATS]
    unknown = [f for f in formats if f not in ALL_FORMATS]
    if unknown:
        raise ValueError(f"Unknown export format(s): {unknown}. Valid: {ALL_FORMATS}")

    # 'render' is on by default but needs optional deps; degrade gracefully with
    # a single clear warning instead of failing every part silently.
    if "render" in mesh and not render_deps_available():
        print(
            "[cadquarry] warning: skipping renders — GPU render stack "
            "unavailable (needs moderngl + an EGL-capable GL driver). Install "
            "with pip install 'cadquarry[export]'"
        )
        mesh = [f for f in mesh if f != "render"]
        formats = [f for f in formats if f != "render"]

    parts_dir = dataset_dir / "parts"
    geo_dir = dataset_dir / "geometry"
    py_files = sorted(parts_dir.glob("*.py"))

    counts: dict[str, int] = {f: 0 for f in formats}

    # Each part needs an STL on disk if the user asked for it or if any
    # mesh-derived format (pointcloud/render) was requested.
    stl_requested = "stl" in brep
    need_stl = stl_requested or bool(mesh)
    run_formats = list(brep)
    if need_stl and "stl" not in run_formats:
        run_formats.append("stl")

    # Incremental export (default): skip parts whose requested artifacts already
    # exist, so extending a corpus only exports/renders the newly-appended tail.
    # A part is re-exported if *any* requested artifact is missing, so partial or
    # interrupted prior exports self-heal.
    renders_dir = dataset_dir / "renders"
    pc_dir = dataset_dir / "pointclouds"

    def _needs_export(stem: str) -> bool:
        if "step" in brep and not (geo_dir / f"{stem}.step").exists():
            return True
        if stl_requested and not (geo_dir / f"{stem}.stl").exists():
            return True
        if "svg" in brep and not (geo_dir / f"{stem}.svg").exists():
            return True
        if "render" in mesh and not (renders_dir / stem).is_dir():
            return True
        if "pointcloud" in mesh and not (pc_dir / f"{stem}.ply").exists():
            return True
        return False

    if skip_existing:
        kept = [p for p in py_files if _needs_export(p.stem)]
        n_skip = len(py_files) - len(kept)
        if n_skip:
            print(f"  · skipping {n_skip} parts already exported "
                  f"({len(kept)} to export)")
        py_files = kept

    geo_dir.mkdir(parents=True, exist_ok=True)

    from .progress import progress_bar

    n = default_worker_count(n_workers)
    pool = WorkerPool(n, timeout=timeout)
    try:
        # Phase 1: run every part's B-rep export in parallel across workers.
        results: dict[str, dict] = {}
        if run_formats:
            jobs = [
                (
                    py_path.stem,
                    {
                        "op": "export",
                        "part_file": str(py_path),
                        "formats": run_formats,
                        "out_dir": str(geo_dir),
                        "overrides": {},
                    },
                )
                for py_path in py_files
            ]
            bar = progress_bar(total=len(jobs), desc="  step/stl", unit="part")
            results = pool.map(jobs, progress=bar.update)
            bar.close()
    finally:
        pool.close()

    # Phase 2a: tally B-rep outputs and collect the STLs that mesh-derived
    # formats (render / pointcloud) will consume.
    stl_for: dict[str, Path] = {}
    for py_path in py_files:
        pid = py_path.stem
        data = results.get(pid)
        if data is None and run_formats:
            if verbose:
                print(f"  FAILED {pid}: no export result")
            continue
        exported: dict[str, Path] = {}
        if data is not None:
            if not data.get("success"):
                if verbose:
                    print(f"  FAILED {pid}: {data.get('error', 'unknown export error')}")
                continue
            for p in data.get("files", []):
                exported[Path(p).suffix.lstrip(".")] = Path(p)

        for fmt, path in exported.items():
            if fmt in counts and fmt in _BREP_FORMATS and path.exists():
                counts[fmt] += 1
                if verbose:
                    print(f"  exported {pid}.{fmt}")

        stl_path = exported.get("stl")
        if mesh and stl_path is not None and stl_path.exists():
            stl_for[pid] = stl_path

    # Phase 2b: point clouds (trimesh, CPU) — uncommon, kept sequential.
    if "pointcloud" in mesh and stl_for:
        pc_bar = progress_bar(total=len(stl_for), desc="  pointcloud", unit="part")
        for pid, stl_path in stl_for.items():
            try:
                pc_out = geo_dir.parent / "pointclouds" / f"{pid}.ply"
                export_pointcloud(stl_path, pc_out, n_points=n_points)
                if pc_out.exists():
                    counts["pointcloud"] += 1
            except Exception as exc:
                if verbose:
                    print(f"  FAILED {pid} (pointcloud): {exc}")
            pc_bar.update()
        pc_bar.close()

    # Phase 2c: renders, parallelised across GPU worker processes.  This is the
    # heavy step; throughput is CPU-bound (downsample + PNG encode), so it scales
    # with processes while they share the one GPU.
    if "render" in mesh and stl_for:
        from . import render as _render

        rn_workers = _render.default_render_workers(n_workers)
        tasks = [
            (str(stl_path), str(geo_dir.parent / "renders" / pid), None)
            for pid, stl_path in stl_for.items()
        ]
        rn_bar = progress_bar(total=len(tasks), desc="  render", unit="part")
        rn_results = _render.render_stls(
            tasks, n_workers=rn_workers, progress_cb=rn_bar.update,
            views=render_views, passes=render_passes, edge_crispness=edge_crispness,
        )
        rn_bar.close()
        for pid, (ok, err) in rn_results.items():
            if ok:
                counts["render"] += 1
            elif verbose:
                print(f"  FAILED {pid} (render): {err}")

    # Phase 2d: drop intermediate STLs that were only produced to feed meshing.
    if mesh and not stl_requested:
        for stl_path in stl_for.values():
            try:
                stl_path.unlink()
            except OSError:
                pass

    return counts
