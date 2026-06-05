"""
Persistent CadQuery worker process.

A single worker imports cadquery/OCP ONCE and then services many jobs over a
pipe, amortising the ~1-1.5s import cost that otherwise dominates per-part
execution.  This is the engine behind the parallel generation/export path in
``cadquarry.execute.WorkerPool``.

Protocol (line-delimited JSON, one object per line):

    parent  -> worker : request object on stdin
    worker  -> parent : response object on the fd given by ``CQ_RESULT_FD``

stdout/stderr of this process are redirected to /dev/null so that any chatter
from OCP/cadquery (or user code) can never corrupt the result stream.

Requests:

    {"op": "analyze", "code": <str>, "overrides": {...}, "stl_out": <str>}
    {"op": "export",  "part_file": <str>, "formats": [...], "out_dir": <str>,
     "overrides": {...}}

The ``analyze`` response carries the exact same fields the legacy in-line
``_RUNNER`` produced, computed by the identical call sequence, so geometry
signatures are bit-for-bit identical to the one-subprocess-per-part path.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any


def _analyze(req: dict[str, Any]) -> dict[str, Any]:
    code = req["code"]
    overrides = req.get("overrides") or {}
    stl_out = req.get("stl_out") or ""

    ns: dict[str, Any] = {}
    exec(compile(code, "<cadquarry-part>", "exec"), ns)

    PARAMS = ns["PARAMS"]
    build = ns["build"]

    params = {k: v["default"] for k, v in PARAMS.items()}
    params.update(overrides)

    result = build(params)
    solid = result.val()

    # Volume and surface area (try modern CQ API, fall back to OCC).
    try:
        vol = solid.Volume()
        area = solid.Area()
    except AttributeError:
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop
        vp = GProp_GProps()
        brepgprop.VolumeProperties(solid.wrapped, vp)
        vol = vp.Mass()
        sp = GProp_GProps()
        brepgprop.SurfaceProperties(solid.wrapped, sp)
        area = sp.Mass()

    bb = result.val().BoundingBox()
    bbox = [bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax]

    # Principal moments of inertia about the centroid — pose-invariant
    # signature component (see filter.compute_signature).
    moments: list[float] = []
    try:
        from OCP.GProp import GProp_GProps as _CQ_GP
        from OCP.BRepGProp import BRepGProp as _CQ_BG
        mp = _CQ_GP()
        _CQ_BG.VolumeProperties_s(solid.wrapped, mp)
        moments = [float(x) for x in mp.PrincipalProperties().Moments()]
    except Exception:
        moments = []

    n_faces = len(result.faces("").vals())
    n_edges = len(result.edges("").vals())
    n_verts = len(result.vertices("").vals())
    n_solids = len(result.solids().vals())

    if stl_out:
        import cadquery as _cq
        _cq.exporters.export(result, stl_out)

    return {
        "success": True,
        "volume": vol,
        "surface_area": area,
        "bbox": bbox,
        "principal_moments": moments,
        "n_faces": n_faces,
        "n_edges": n_edges,
        "n_vertices": n_verts,
        "n_solids": n_solids,
    }


def _export(req: dict[str, Any]) -> dict[str, Any]:
    import cadquery as cq

    part_file = req["part_file"]
    formats = req["formats"]
    out_dir = req["out_dir"]
    overrides = req.get("overrides") or {}

    ns: dict[str, Any] = {}
    with open(part_file, encoding="utf-8") as f:
        exec(f.read(), ns)

    params = {k: v["default"] for k, v in ns["PARAMS"].items()}
    params.update(overrides)
    result = ns["build"](params)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(part_file))[0]
    exported: list[str] = []

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

    return {"success": True, "files": exported}


def _dispatch(req: dict[str, Any]) -> dict[str, Any]:
    op = req.get("op")
    if op == "analyze":
        return _analyze(req)
    if op == "export":
        return _export(req)
    return {"success": False, "error": f"unknown op: {op!r}"}


def main() -> None:
    result_fd = int(os.environ["CQ_RESULT_FD"])
    out = os.fdopen(result_fd, "wb", buffering=0)
    inp = sys.stdin.buffer

    # Silence anything written to the real stdout/stderr (e.g. OCP warnings)
    # so it can never interleave with the JSON result stream the parent reads.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    while True:
        line = inp.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = _dispatch(req)
        except Exception as exc:  # never let one bad job kill the worker
            resp = {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        out.write((json.dumps(resp) + "\n").encode("utf-8"))
        out.flush()


if __name__ == "__main__":
    main()
