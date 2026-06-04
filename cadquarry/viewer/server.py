"""
HTTP server for the CadQuarry customizer and gallery.

Two entry points:
  serve_customizer(py_path, port) — opens a single-part live customizer
  serve_gallery(dataset_dir, port) — opens a browseable gallery

Both serve the same index.html; the mode is determined by the URL:
  /             → gallery (requires dataset mode) or redirect to customizer
  /customizer   → customizer for the current / specified part
  /api/*        → JSON + binary API endpoints
"""
from __future__ import annotations

import json
import mimetypes
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..dataset import load_manifest
from ..execute import get_stl, execute_file

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    # Injected by the serve_* functions
    _py_path: Path | None = None
    _dataset_dir: Path | None = None

    def log_message(self, fmt: str, *args: Any) -> None:
        # Suppress default access log noise; uncomment for debugging.
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        data = path.read_bytes()
        self._send_bytes(data, mime)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # -----------------------------------------------------------------------
    # Route dispatch
    # -----------------------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # Static files
        if path == "/" or path == "/customizer":
            self._send_file(STATIC_DIR / "index.html")
            return

        static_candidate = STATIC_DIR / path.lstrip("/")
        if static_candidate.exists() and static_candidate.is_file():
            self._send_file(static_candidate)
            return

        # API routes
        if path == "/api/params":
            self._api_get_params(parsed)
        elif path == "/api/stl":
            self._api_get_stl(parsed)
        elif path.startswith("/api/stl/"):
            part_id = path[len("/api/stl/"):]
            self._api_get_stl_by_id(part_id)
        elif path == "/api/manifest":
            self._api_get_manifest()
        elif path == "/api/info":
            self._api_get_info(parsed)
        elif path.startswith("/api/part/"):
            part_id = path[len("/api/part/"):]
            self._api_get_part_info(part_id)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/execute":
            self._api_post_execute()
        else:
            self._send_json({"error": "not found"}, 404)

    # -----------------------------------------------------------------------
    # API implementations
    # -----------------------------------------------------------------------

    def _api_get_params(self, parsed) -> None:
        py = self._resolve_part(parsed)
        if py is None:
            self._send_json({"error": "no part loaded"}, 400)
            return
        try:
            ns: dict = {}
            exec(py.read_text(encoding="utf-8"), ns)
            self._send_json(ns.get("PARAMS", {}))
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def _api_get_stl(self, parsed) -> None:
        py = self._resolve_part(parsed)
        if py is None:
            self._send_json({"error": "no part loaded"}, 400)
            return
        data = get_stl(py, timeout=30.0)
        if data is None:
            self._send_json({"error": "execution failed"}, 500)
            return
        self._send_bytes(data, "model/stl")

    def _api_get_stl_by_id(self, part_id: str) -> None:
        if self._dataset_dir is None:
            self._send_json({"error": "no dataset loaded"}, 400)
            return
        py = self._dataset_dir / "parts" / f"{part_id}.py"
        if not py.exists():
            # Try geometry directory first (pre-generated)
            stl = self._dataset_dir / "geometry" / f"{part_id}.stl"
            if stl.exists():
                self._send_bytes(stl.read_bytes(), "model/stl")
                return
            self._send_json({"error": "part not found"}, 404)
            return
        data = get_stl(py, timeout=30.0)
        if data is None:
            self._send_json({"error": "execution failed"}, 500)
            return
        self._send_bytes(data, "model/stl")

    def _api_get_manifest(self) -> None:
        if self._dataset_dir is None:
            self._send_json([], 200)
            return
        records = load_manifest(self._dataset_dir)
        self._send_json(records)

    def _api_get_info(self, parsed) -> None:
        py = self._resolve_part(parsed)
        if py is None:
            self._send_json({"error": "no part loaded"}, 400)
            return
        meta_path = py.parent.parent / "meta" / (py.stem + ".meta.json")
        if meta_path.exists():
            self._send_json(json.loads(meta_path.read_text()))
            return
        # Minimal info from filename
        self._send_json({"part_id": py.stem})

    def _api_get_part_info(self, part_id: str) -> None:
        if self._dataset_dir is None:
            self._send_json({"error": "no dataset"}, 400)
            return
        meta_path = self._dataset_dir / "meta" / f"{part_id}.meta.json"
        if meta_path.exists():
            self._send_json(json.loads(meta_path.read_text()))
        else:
            self._send_json({"part_id": part_id})

    def _api_post_execute(self) -> None:
        py = self._resolve_part(None)
        if py is None:
            self._send_json({"error": "no part loaded"}, 400)
            return

        body = self._read_body()
        try:
            overrides = json.loads(body) if body else {}
        except json.JSONDecodeError:
            overrides = {}

        # Return STL bytes directly for three.js
        data = get_stl(py, params=overrides or None, timeout=30.0)
        if data is None:
            self._send_json({"error": "execution failed or geometry invalid"}, 500)
            return
        self._send_bytes(data, "model/stl")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _resolve_part(self, parsed) -> Path | None:
        if parsed is not None:
            qs = parse_qs(parsed.query)
            if "part" in qs:
                return Path(qs["part"][0])
        return self.__class__._py_path


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _make_handler(py_path: Path | None = None, dataset_dir: Path | None = None):
    class H(_Handler):
        pass
    H._py_path = py_path
    H._dataset_dir = dataset_dir
    return H


def serve_customizer(py_path: Path, port: int = 8765) -> None:
    handler = _make_handler(py_path=py_path)
    httpd = HTTPServer(("localhost", port), handler)
    url = f"http://localhost:{port}/customizer"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def serve_gallery(dataset_dir: Path, port: int = 8765) -> None:
    handler = _make_handler(dataset_dir=dataset_dir)
    httpd = HTTPServer(("localhost", port), handler)
    url = f"http://localhost:{port}/"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
