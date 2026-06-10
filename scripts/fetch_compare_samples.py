"""Fetch a small sample of meshes from third-party CAD/geometry datasets for the
comparison page (``sample/compare/compare.html``).

These meshes are **NOT** CadQuarry data and are **NOT** CC0 — each retains the
license of its origin dataset / original author. They are pulled here only to
populate an apples-to-apples visual comparison. See
``sample/compare/THIRD_PARTY_LICENSES.md``. Only non-gated Hugging Face mirrors
that expose individual mesh files are supported; gated datasets (ShapeNet) and
mesh-less command-sequence datasets (DeepCAD) are intentionally not fetched.

Usage:
    python scripts/fetch_compare_samples.py            # all configured sources
    python scripts/fetch_compare_samples.py thingi10k  # one source
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "sample" / "compare" / "models"

# Per-source config.
#   type "hf_files":  pull individual mesh files from a non-gated HF dataset repo
#                     (``subdir`` is the dir inside the repo that holds them).
#   type "archive7z": download a .7z chunk, pick the N smallest meshes of ``ext``,
#                     extract just those, and convert them to STL.
# ``size_cap`` bounds a single committed mesh (keeps the repo sane); ``count`` is
# how many to keep.
SOURCES: dict[str, dict] = {
    "thingi10k": {
        "type": "hf_files",
        "repo_id": "Thingi10K/Thingi10K",
        "subdir": "raw_meshes",
        "ext": ".stl",
        "count": 100,
        "size_cap": 400_000,
    },
    "fusion360": {
        "type": "hf_files",
        "repo_id": "maksimko123/fusion360_test_mesh",
        "subdir": "",
        "ext": ".stl",
        "count": 100,
        "size_cap": 400_000,
    },
    "abc": {
        "type": "archive7z",
        # ABC obj chunk 0 (abc_0000_obj_v00.7z, ~7.9 GB) from the NYU archive.
        "url": "https://archive.nyu.edu/rest/bitstreams/89085/retrieve",
        "ext": ".obj",
        "count": 100,
        "size_cap": 600_000,
    },
    "cad_recode": {
        # CadQuery .py programs; we execute each and export the resulting solid to
        # STL (CAD-Recode, like CadQuarry, emits CadQuery code rather than meshes).
        "type": "cqcode",
        "repo_id": "filapro/cad-recode",
        "subdir": "val",
        "ext": ".py",
        "count": 100,
        "size_cap": 600_000,
    },
}


def list_tree(repo_id: str, subdir: str) -> list[dict]:
    path = f"/{subdir}" if subdir else ""
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main{path}?recursive=false"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _write_index(out_dir: Path, index: list[dict]) -> None:
    (out_dir / "index.json").write_text(json.dumps(index, indent=0))
    print(f"[{out_dir.name}] wrote {len(index)} -> {out_dir.relative_to(REPO_ROOT)}/index.json")


def fetch_archive7z(name: str, cfg: dict) -> None:
    """Download a .7z chunk, extract the N smallest meshes, convert each to STL."""
    import trimesh  # noqa: F401  (import early so a missing dep fails fast)

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"abc_{name}_") as td:
        tmp = Path(td)
        archive = tmp / "chunk.7z"
        print(f"[{name}] downloading {cfg['url']} (large)…")
        urllib.request.urlretrieve(cfg["url"], archive)
        print(f"[{name}] {archive.stat().st_size/1e9:.2f} GB downloaded; listing…")

        # `7z l -slt` prints a Path/Size block per entry.
        listing = subprocess.run(
            ["7z", "l", "-slt", str(archive)], capture_output=True, text=True, check=True
        ).stdout
        meshes: list[tuple[str, int]] = []
        path, size = None, None
        for line in listing.splitlines():
            if line.startswith("Path = "):
                path = line[7:]
            elif line.startswith("Size = "):
                size = int(line[7:] or 0)
                if path and path.lower().endswith(cfg["ext"]) and 0 < size <= cfg["size_cap"]:
                    meshes.append((path, size))
                path, size = None, None
        meshes.sort(key=lambda x: x[1])
        meshes = meshes[: cfg["count"]]
        print(f"[{name}] extracting {len(meshes)} {cfg['ext']} files…")

        index = []
        for i, (mpath, _) in enumerate(meshes, 1):
            subprocess.run(
                ["7z", "e", "-y", f"-o{tmp/'ex'}", str(archive), mpath],
                capture_output=True, check=True,
            )
            src = tmp / "ex" / Path(mpath).name
            # ABC names every model's file the same inside its own dir; key by the
            # parent dir (the ABC model id) to keep filenames unique.
            mid = Path(mpath).parent.name or Path(mpath).stem
            dest = out_dir / f"{mid}.stl"
            mesh = trimesh.load(src, force="mesh")
            mesh.export(dest)
            src.unlink(missing_ok=True)
            index.append({"file": dest.name, "id": mid})
            if i % 20 == 0 or i == len(meshes):
                print(f"  {i}/{len(meshes)}")
    _write_index(out_dir, index)


def fetch_cqcode(name: str, cfg: dict) -> None:
    """Download CadQuery .py programs and execute each one into an STL mesh."""
    import cadquery as cq

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = list_tree(cfg["repo_id"], cfg["subdir"])
    pys = [e for e in entries if e.get("type") == "file" and e["path"].lower().endswith(cfg["ext"])]
    pys.sort(key=lambda e: int(Path(e["path"]).stem) if Path(e["path"]).stem.isdigit() else 1 << 30)
    print(f"[{name}] {cfg['repo_id']}: executing CadQuery programs until {cfg['count']} succeed…")

    base = f"https://huggingface.co/datasets/{cfg['repo_id']}/resolve/main/"
    index, tried = [], 0
    for e in pys:
        if len(index) >= cfg["count"]:
            break
        tried += 1
        stem = Path(e["path"]).stem
        try:
            code = urllib.request.urlopen(base + e["path"], timeout=30).read().decode()
            ns: dict = {}
            exec(code, ns)  # noqa: S102 — trusted-as-much-as-running-the-generator
            obj = ns.get("r") or ns.get("result")
            if obj is None:
                continue
            dest = out_dir / f"{stem}.stl"
            cq.exporters.export(obj, str(dest))
            if not dest.exists() or dest.stat().st_size == 0 or dest.stat().st_size > cfg["size_cap"]:
                dest.unlink(missing_ok=True)
                continue
            index.append({"file": dest.name, "id": stem})
            if len(index) % 20 == 0:
                print(f"  {len(index)}/{cfg['count']} (tried {tried})")
        except Exception:
            continue
    print(f"[{name}] {len(index)} succeeded from {tried} programs")
    _write_index(out_dir, index)


def fetch_source(name: str, cfg: dict) -> None:
    if cfg["type"] == "archive7z":
        fetch_archive7z(name, cfg)
        return
    if cfg["type"] == "cqcode":
        fetch_cqcode(name, cfg)
        return

    from huggingface_hub import hf_hub_download

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = list_tree(cfg["repo_id"], cfg["subdir"])
    files = [
        e for e in entries
        if e.get("type") == "file"
        and e["path"].lower().endswith(cfg["ext"])
        and 0 < (e.get("size") or 0) <= cfg["size_cap"]
    ]
    files.sort(key=lambda e: e["path"])
    files = files[: cfg["count"]]
    print(f"[{name}] {cfg['repo_id']}: {len(files)} meshes (<= {cfg['size_cap']} bytes)")

    index = []
    for i, e in enumerate(files, 1):
        src_path = e["path"]
        stem = Path(src_path).name
        dest = out_dir / stem
        if not dest.exists():
            cached = hf_hub_download(
                repo_id=cfg["repo_id"], filename=src_path, repo_type="dataset"
            )
            shutil.copy(cached, dest)
        index.append({"file": stem, "id": Path(stem).stem})
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")
    _write_index(out_dir, index)


def main() -> int:
    which = sys.argv[1:] or list(SOURCES)
    unknown = [w for w in which if w not in SOURCES]
    if unknown:
        print(f"unknown source(s): {unknown}. Known: {list(SOURCES)}", file=sys.stderr)
        return 2
    for name in which:
        fetch_source(name, SOURCES[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
