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
    """
    if formats is None:
        formats = ["step", "stl"]

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
                rn_out = out_dir.parent / "renders" / f"{stem}.png" \
                    if out_dir.name == "geometry" else out_dir / f"{stem}.png"
                exported["render"] = export_render(stl_path, rn_out)

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


def export_render(
    stl_path: Path,
    out_path: Path,
    size: int = 512,
    elev: float = 28.0,
    azim: float = -55.0,
) -> Path:
    """
    Render a shaded thumbnail PNG of a mesh, headless.

    Uses trimesh to load the STL and matplotlib (Agg backend) to draw a shaded
    triangle surface.  Both are optional dependencies kept out of the core data
    path; a tool's license never reaches generated output regardless.
    """
    try:
        import numpy as np
        import trimesh
        import matplotlib
        matplotlib.use("Agg")  # headless, no display required
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:
        raise ImportError(
            "Render export requires trimesh and matplotlib: "
            "pip install 'cadquarry[export]' matplotlib"
        ) from exc

    mesh = trimesh.load_mesh(str(stl_path))
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    tris = verts[faces]

    # Simple Lambert shading from a fixed light direction.
    normals = np.asarray(mesh.face_normals)
    light = np.array([0.3, -0.5, 0.8])
    light = light / np.linalg.norm(light)
    intensity = np.clip(normals @ light, 0.0, 1.0) * 0.7 + 0.3
    base = np.array([0.40, 0.55, 0.85])
    facecolors = np.clip(intensity[:, None] * base[None, :], 0, 1)

    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(tris, facecolors=facecolors, edgecolors="none")
    ax.add_collection3d(coll)

    # Equal aspect cube around the part.
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = (mins + maxs) / 2
    span = float((maxs - mins).max()) * 0.6 or 1.0
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


def export_corpus_geometry(
    dataset_dir: Path,
    formats: list[str] | None = None,
    n_workers: int = 0,
    timeout: float = 60.0,
    verbose: bool = False,
    n_points: int = 2048,
) -> dict[str, int]:
    """
    Export geometry for all parts in a dataset directory.

    B-rep formats (step/stl/svg) land in <dataset>/geometry/, point clouds in
    <dataset>/pointclouds/, and renders in <dataset>/renders/.
    Returns counts of {format: n_exported}.

    The expensive per-part cost is importing cadquery and rebuilding the solid;
    we amortise it across a pool of persistent workers (see
    ``execute.WorkerPool``) so the whole corpus shares one set of imports.
    """
    from .execute import WorkerPool, default_worker_count

    if formats is None:
        formats = ["step", "stl"]

    brep = [f for f in formats if f in _BREP_FORMATS]
    mesh = [f for f in formats if f in _MESH_FORMATS]
    unknown = [f for f in formats if f not in ALL_FORMATS]
    if unknown:
        raise ValueError(f"Unknown export format(s): {unknown}. Valid: {ALL_FORMATS}")

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

    geo_dir.mkdir(parents=True, exist_ok=True)

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
            results = pool.map(jobs)
    finally:
        pool.close()

    # Phase 2: tally B-rep outputs and derive mesh-based artifacts (trimesh /
    # matplotlib run in this process; they never import cadquery).
    for py_path in py_files:
        pid = py_path.stem
        exported: dict[str, Path] = {}
        data = results.get(pid)
        if data is None and run_formats:
            if verbose:
                print(f"  FAILED {pid}: no export result")
            continue
        if data is not None:
            if not data.get("success"):
                if verbose:
                    print(f"  FAILED {pid}: {data.get('error', 'unknown export error')}")
                continue
            for p in data.get("files", []):
                exported[Path(p).suffix.lstrip(".")] = Path(p)

        try:
            if mesh:
                stl_path = exported.get("stl")
                if stl_path is None or not stl_path.exists():
                    raise RuntimeError("STL needed for mesh export was not produced")
                if "pointcloud" in mesh:
                    pc_out = geo_dir.parent / "pointclouds" / f"{pid}.ply"
                    exported["pointcloud"] = export_pointcloud(stl_path, pc_out, n_points=n_points)
                if "render" in mesh:
                    rn_out = geo_dir.parent / "renders" / f"{pid}.png"
                    exported["render"] = export_render(stl_path, rn_out)
                if not stl_requested and stl_path.exists():
                    try:
                        stl_path.unlink()
                    except OSError:
                        pass
                    exported.pop("stl", None)
        except Exception as exc:
            if verbose:
                print(f"  FAILED {pid} (mesh): {exc}")

        for fmt, path in exported.items():
            if fmt in counts and path.exists():
                counts[fmt] = counts.get(fmt, 0) + 1
                if verbose:
                    print(f"  exported {pid}.{fmt}")

    return counts
