"""
Sandboxed execution of generated CadQuery programs.

Each execution runs in a child subprocess with a hard timeout so a bad
program can't hang the generator.  The child writes a JSON result record to
stdout; we parse it back in the parent.

Design:
- Parent calls execute_source(code, params, timeout) → ExecuteResult
- Child runs the code via exec(), calls build(params), inspects the solid,
  then prints a JSON blob and exits.
- The child also optionally produces binary STL to a temp file whose path
  is passed back in the JSON, so the viewer can display geometry.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Runner code appended to the user's program inside the subprocess.
# We use a distinct variable prefix (_cq_) to minimise collision risk.
_RUNNER = """
import json as _cq_json
import sys as _cq_sys
import traceback as _cq_tb

def _cq_run():
    _overrides_raw = _cq_sys.argv[1] if len(_cq_sys.argv) > 1 else "{}"
    _stl_out = _cq_sys.argv[2] if len(_cq_sys.argv) > 2 else ""
    _overrides = _cq_json.loads(_overrides_raw)

    _params = {k: v["default"] for k, v in PARAMS.items()}
    _params.update(_overrides)

    try:
        _result = build(_params)
        _solid = _result.val()

        # Volume and surface area (try modern CQ API, fall back to OCC)
        try:
            _vol = _solid.Volume()
            _area = _solid.Area()
        except AttributeError:
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop
            _vp = GProp_GProps()
            brepgprop.VolumeProperties(_solid.wrapped, _vp)
            _vol = _vp.Mass()
            _sp = GProp_GProps()
            brepgprop.SurfaceProperties(_solid.wrapped, _sp)
            _area = _sp.Mass()

        _bb = _result.val().BoundingBox()
        _bbox = [_bb.xmin, _bb.ymin, _bb.zmin, _bb.xmax, _bb.ymax, _bb.zmax]

        # Principal moments of inertia about the centroid — invariant to
        # arbitrary rotation and (about the centroid) translation, so they make
        # a strong pose-invariant geometry-signature component.
        _moments = []
        try:
            from OCP.GProp import GProp_GProps as _CQ_GP
            from OCP.BRepGProp import BRepGProp as _CQ_BG
            _mp = _CQ_GP()
            _CQ_BG.VolumeProperties_s(_solid.wrapped, _mp)
            _moments = [float(x) for x in _mp.PrincipalProperties().Moments()]
        except Exception:
            _moments = []

        # Topology counts
        _n_faces = len(_result.faces("").vals())
        _n_edges = len(_result.edges("").vals())
        _n_verts = len(_result.vertices("").vals())

        # Optional STL export
        if _stl_out:
            import cadquery as _cq
            _cq.exporters.export(_result, _stl_out)

        print(_cq_json.dumps({
            "success": True,
            "volume": _vol,
            "surface_area": _area,
            "bbox": _bbox,
            "principal_moments": _moments,
            "n_faces": _n_faces,
            "n_edges": _n_edges,
            "n_vertices": _n_verts,
        }))
    except Exception as _e:
        print(_cq_json.dumps({
            "success": False,
            "error": str(_e),
            "traceback": _cq_tb.format_exc(),
        }))

_cq_run()
"""


@dataclass
class ExecuteResult:
    success: bool
    error: str = ""
    traceback: str = ""
    volume: float = 0.0
    surface_area: float = 0.0
    bbox: list[float] = field(default_factory=lambda: [0.0] * 6)
    principal_moments: list[float] = field(default_factory=list)
    n_faces: int = 0
    n_edges: int = 0
    n_vertices: int = 0
    stl_path: Path | None = None

    @property
    def bbox_dims(self) -> tuple[float, float, float]:
        x0, y0, z0, x1, y1, z1 = self.bbox
        return (x1 - x0, y1 - y0, z1 - z0)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExecuteResult":
        return cls(
            success=d.get("success", False),
            error=d.get("error", ""),
            traceback=d.get("traceback", ""),
            volume=d.get("volume", 0.0),
            surface_area=d.get("surface_area", 0.0),
            bbox=d.get("bbox", [0.0] * 6),
            principal_moments=d.get("principal_moments", []),
            n_faces=d.get("n_faces", 0),
            n_edges=d.get("n_edges", 0),
            n_vertices=d.get("n_vertices", 0),
        )


def execute_source(
    code: str,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
    export_stl: bool = False,
) -> ExecuteResult:
    """
    Execute generated CadQuery source in a subprocess.

    code:       Full .py source (including PARAMS + build(p)).
    params:     Parameter overrides; defaults are used for omitted keys.
    timeout:    Hard wall-clock timeout in seconds.
    export_stl: If True, write an STL to a temp file and attach its path.
    """
    overrides = json.dumps(params or {})
    full_code = code + "\n" + _RUNNER

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as tf:
        tf.write(full_code)
        script = tf.name

    stl_path: Path | None = None
    stl_arg = ""
    if export_stl:
        stl_fd, stl_tmp = tempfile.mkstemp(suffix=".stl")
        os.close(stl_fd)
        stl_path = Path(stl_tmp)
        stl_arg = str(stl_tmp)

    try:
        proc = subprocess.run(
            [sys.executable, script, overrides, stl_arg],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout.strip()
        if not stdout:
            return ExecuteResult(
                success=False,
                error=f"No output from subprocess (stderr: {proc.stderr[:500]})",
            )
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return ExecuteResult(
                success=False,
                error=f"Malformed JSON from subprocess: {stdout[:200]}",
            )
        result = ExecuteResult.from_dict(data)
        if export_stl and stl_path and stl_path.exists() and stl_path.stat().st_size > 0:
            result.stl_path = stl_path
        return result
    except subprocess.TimeoutExpired:
        return ExecuteResult(success=False, error=f"Execution timed out after {timeout}s")
    except Exception as exc:
        return ExecuteResult(success=False, error=str(exc))
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


def execute_file(
    path: Path,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
    export_stl: bool = False,
) -> ExecuteResult:
    """Execute a part .py file from disk."""
    return execute_source(
        path.read_text(encoding="utf-8"),
        params=params,
        timeout=timeout,
        export_stl=export_stl,
    )


def get_stl(path: Path, params: dict[str, Any] | None = None, timeout: float = 30.0) -> bytes | None:
    """Execute a part and return raw STL bytes, or None on failure."""
    result = execute_file(path, params=params, timeout=timeout, export_stl=True)
    if result.success and result.stl_path and result.stl_path.exists():
        data = result.stl_path.read_bytes()
        try:
            result.stl_path.unlink()
        except OSError:
            pass
        return data
    return None
