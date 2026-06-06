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
import queue
import select
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
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
        _n_solids = len(_result.solids().vals())

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
            "n_solids": _n_solids,
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
    n_solids: int = 1
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
            n_solids=d.get("n_solids", 1),
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


# ---------------------------------------------------------------------------
# Persistent worker pool
# ---------------------------------------------------------------------------
#
# The one-subprocess-per-part path above is fine for single, interactive
# executions (viewer / `cadquarry run`).  For batch generation/export the
# ~1-1.5s cadquery import cost dominates, so we keep a pool of long-lived
# workers (cadquarry._worker) that import cadquery once and then service many
# jobs over a pipe.
#
# Determinism: a worker's response is a pure function of the request (same
# source + overrides => identical metrics), independent of which worker runs
# it or in what order, so parallel execution does not change any output.
#
# Killability: each worker handles one job at a time.  A job that hangs OCC is
# bounded by the same hard wall-clock timeout as before — on timeout the parent
# SIGKILLs that worker and respawns a fresh one, losing only the single bad
# job rather than the batch.
#
# Memory: OpenCASCADE/OCP retains a few MB per *distinct* part it processes
# (BRep caches + an allocator that holds freed blocks in free-lists rather than
# returning them to the OS), so a worker that lives for a whole corpus grows
# without bound — at corpus scale, times the worker count, this is enough to
# exhaust RAM and spill to swap.  To cap it, each worker is recycled (its
# subprocess gracefully torn down and respawned) after a bounded number of
# jobs.  Because a response is a pure function of the request, respawning
# mid-corpus is bit-for-bit identical to never doing so — it only frees memory.

# Recycle a worker subprocess after this many jobs to bound peak RSS.  Override
# with CADQUARRY_MAX_JOBS_PER_WORKER (0 disables recycling).  The ~1-1.5s
# respawn cost is amortised over the budget, so a few hundred keeps the import
# overhead negligible while still releasing memory frequently.
def _default_max_jobs_per_worker() -> int:
    raw = os.environ.get("CADQUARRY_MAX_JOBS_PER_WORKER")
    if raw is None:
        return 256
    try:
        return max(0, int(raw))
    except ValueError:
        return 256


class _Worker:
    """A single long-lived cadquery worker subprocess."""

    def __init__(self, timeout: float, max_jobs: int = 0) -> None:
        self.timeout = timeout
        self.max_jobs = max(0, int(max_jobs))
        self._jobs_done = 0
        self.proc: subprocess.Popen | None = None
        self._r_fd: int = -1
        self._buf = b""
        self._spawn()

    def _spawn(self) -> None:
        r_fd, w_fd = os.pipe()
        env = dict(os.environ)
        env["CQ_RESULT_FD"] = str(w_fd)
        # Make the write end inheritable across exec.
        os.set_inheritable(w_fd, True)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "cadquarry._worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(w_fd,),
            env=env,
        )
        os.close(w_fd)  # only the child writes to it
        self._r_fd = r_fd
        self._buf = b""
        self._jobs_done = 0

    def _readline(self, deadline: float) -> bytes | None:
        """Read one newline-terminated line, or None on timeout/EOF."""
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self._r_fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(self._r_fd, 65536)
            if not chunk:
                return None  # worker died / closed pipe
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def run(self, req: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one request, enforcing the hard timeout via kill+respawn."""
        payload = (json.dumps(req) + "\n").encode("utf-8")
        try:
            assert self.proc is not None and self.proc.stdin is not None
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._kill_respawn()
            return {"success": False, "error": "worker pipe broken before dispatch"}

        deadline = time.monotonic() + self.timeout
        line = self._readline(deadline)
        if line is None:
            self._kill_respawn()
            return {"success": False, "error": f"Execution timed out after {self.timeout}s"}
        try:
            resp = json.loads(line.decode("utf-8"))
        except Exception as exc:  # malformed — treat as failure, recycle worker
            self._kill_respawn()
            return {"success": False, "error": f"malformed worker response: {exc}"}

        # Bound peak RSS: OCP/OCC retains memory per distinct part, so recycle
        # the subprocess once it has served its job budget.  (A respawn already
        # happened above on timeout/crash, resetting the counter.)
        self._jobs_done += 1
        if self.max_jobs and self._jobs_done >= self.max_jobs:
            self._recycle()
        return resp

    def _recycle(self) -> None:
        """Gracefully tear down the worker subprocess and start a fresh one,
        releasing all memory it accumulated.  Used to bound peak RSS over a
        long batch; unlike ``_kill_respawn`` this is a clean shutdown, not a
        reaction to a hung/broken worker."""
        self.close()
        self._spawn()

    def _kill_respawn(self) -> None:
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        if self._r_fd >= 0:
            try:
                os.close(self._r_fd)
            except OSError:
                pass
            self._r_fd = -1
        self._spawn()

    def close(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self._r_fd >= 0:
            try:
                os.close(self._r_fd)
            except OSError:
                pass
            self._r_fd = -1


class WorkerPool:
    """
    A pool of persistent cadquery workers.

    Use ``map(jobs)`` with ``jobs`` an iterable of ``(key, request)`` pairs;
    it returns ``{key: response_dict}`` once every job has completed.  Each
    request is a dict understood by ``cadquarry._worker`` (op="analyze" or
    op="export").  Jobs run concurrently across workers; ordering of results
    is the caller's responsibility (keys are preserved).

    Each worker subprocess is recycled after ``max_jobs_per_worker`` jobs to
    keep OCC/OCP memory from accumulating across a whole corpus; since a
    response is a pure function of its request, recycling never changes output.
    """

    def __init__(
        self, n_workers: int, timeout: float = 30.0, max_jobs_per_worker: int | None = None
    ) -> None:
        self.n_workers = max(1, int(n_workers))
        self.timeout = timeout
        # Recycle each worker after this many jobs to bound peak RSS (see the
        # module note above).  ``None`` -> env-configurable default; 0 disables.
        self.max_jobs_per_worker = (
            _default_max_jobs_per_worker() if max_jobs_per_worker is None
            else max(0, int(max_jobs_per_worker))
        )
        self._job_q: queue.Queue = queue.Queue()
        self._workers = [
            _Worker(timeout, max_jobs=self.max_jobs_per_worker)
            for _ in range(self.n_workers)
        ]
        self._threads: list[threading.Thread] = []
        for w in self._workers:
            t = threading.Thread(target=self._worker_loop, args=(w,), daemon=True)
            t.start()
            self._threads.append(t)

    def _worker_loop(self, worker: _Worker) -> None:
        while True:
            item = self._job_q.get()
            try:
                if item is None:
                    return
                key, req, results, on_done = item
                results[key] = worker.run(req)
                on_done()
            finally:
                self._job_q.task_done()

    def map(
        self,
        jobs: list[tuple[Any, dict[str, Any]]],
        progress=None,
    ) -> dict[Any, dict[str, Any]]:
        """Run jobs across workers. ``progress`` (if given) is called once per
        completed job, on the completing worker thread (serialised under a lock)."""
        results: dict[Any, dict[str, Any]] = {}
        jobs = list(jobs)
        if not jobs:
            return results
        remaining = {"n": len(jobs)}
        lock = threading.Lock()
        done_ev = threading.Event()

        def on_done() -> None:
            with lock:
                if progress is not None:
                    progress()
                remaining["n"] -= 1
                if remaining["n"] == 0:
                    done_ev.set()

        for key, req in jobs:
            self._job_q.put((key, req, results, on_done))
        done_ev.wait()
        return results

    def close(self) -> None:
        for _ in self._workers:
            self._job_q.put(None)
        for t in self._threads:
            t.join(timeout=10)
        for w in self._workers:
            w.close()

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def default_worker_count(requested: int | None = None) -> int:
    """Pick a sensible worker count: explicit value, else ~CPU-bound default."""
    if requested and requested > 0:
        return requested
    cpu = os.cpu_count() or 4
    return max(1, min(cpu, 24))
