#!/usr/bin/env python3
"""
Generate the CadQuarry corpus size-ladder and publish it to a HuggingFace
dataset repository.

Design goals
------------
* **No secrets in the repo.** The HF token is read from the environment
  (``HF_TOKEN``, falling back to ``HUGGING_FACE_HUB_TOKEN``) or from a prior
  ``huggingface-cli login``. It is never printed, logged, or written to disk.
* **Reproducible.** Every size is pinned to a ``(count, seed)`` in
  ``seeds/v1.toml`` under ``[[publish.corpus]]``. The generator version + seed
  regenerate each corpus bit-for-bit, so the upload is a convenience artifact.
* **Reusable by anyone.** Point it at your own repo with ``--repo-id`` (or the
  ``CADQUARRY_HF_REPO`` env var). With no namespace it defaults to
  ``<your-username>/cadquarry`` resolved from your token.

What it uploads
---------------
One dataset repo with a folder per size, each containing a single
``corpus.jsonl`` whose rows carry the parametric ``source`` and ``params``
inline (browsable in the HF dataset viewer, loadable via ``load_dataset``).
A generated ``README.md`` declares one ``load_dataset`` config per size.

Examples
--------
    # Build + upload just the small configs (fast)
    python scripts/publish_to_hf.py --sizes 1k 2k 5k

    # Everything, to your own repo
    python scripts/publish_to_hf.py --all --repo-id me/cadquarry

    # Generate + pack locally but do not upload (inspect .hf_build/)
    python scripts/publish_to_hf.py --sizes 1k --dry-run

    # Also export + upload compact STL meshes alongside each corpus
    python scripts/publish_to_hf.py --sizes 1k --with-geometry
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cadquarry import __version__ as GEN_VERSION  # noqa: E402
from cadquarry.dataset import pack_corpus_jsonl  # noqa: E402

SEEDS_FILE = REPO_ROOT / "seeds" / "v1.toml"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"


# ---------------------------------------------------------------------------
# Seed ladder
# ---------------------------------------------------------------------------

def load_publish_ladder() -> tuple[str, dict[str, dict]]:
    """Return (default_repo_id, {tag: {count, seed}}) from seeds/v1.toml."""
    with open(SEEDS_FILE, "rb") as f:
        data = tomllib.load(f)
    pub = data.get("publish", {})
    repo_id = pub.get("repo_id", "cadquarry")
    ladder: dict[str, dict] = {}
    for entry in pub.get("corpus", []):
        ladder[str(entry["tag"])] = {"count": int(entry["count"]), "seed": int(entry["seed"])}
    if not ladder:
        raise SystemExit("No [[publish.corpus]] entries found in seeds/v1.toml")
    return repo_id, ladder


def tag_to_int(tag: str) -> int:
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
# Token (never printed)
# ---------------------------------------------------------------------------

def resolve_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------

def generate_corpus(out_dir: Path, count: int, seed: int, workers: int, force: bool) -> None:
    manifest = out_dir / "manifest.jsonl"
    if manifest.exists() and not force:
        n = sum(1 for _ in manifest.open())
        if n >= count:
            print(f"  · reusing existing corpus ({n} parts) at {out_dir}")
            return
    cmd = [
        sys.executable, "-m", "cadquarry", "generate",
        "--seed", str(seed), "--count", str(count),
        "--out", str(out_dir), "--config", str(DEFAULT_CONFIG),
    ]
    if workers:
        cmd += ["--workers", str(workers)]
    print(f"  · generating {count} parts (seed={seed}) …")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def export_geometry(corpus_dir: Path, out_dir: Path, workers: int) -> None:
    script = REPO_ROOT / "scripts" / "export_stl.py"
    cmd = [sys.executable, str(script), str(corpus_dir), "--out", str(out_dir)]
    if workers:
        cmd += ["--workers", str(workers)]
    print("  · exporting compact STL meshes …")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def write_size_card(folder: Path, tag: str, count: int, seed: int, n_rows: int, repo_id: str) -> None:
    card = f"""\
# CadQuarry — `{tag}` ({count:,} parts)

Generator: **CadQuarry v{GEN_VERSION}** · Seed: **{seed}** · Rows: **{n_rows:,}**
License: **CC0-1.0** (data) · Generator code: **Apache-2.0**

Regenerate this exact corpus from the [generator]\
(https://github.com/jacobjennings/CadQuarry):

```bash
cadquarry generate --seed {seed} --count {count} \\
    --config configs/default.toml --out {tag}/
python scripts/publish_to_hf.py --sizes {tag}   # pack + upload
```

`corpus.jsonl` has one row per part with the parametric `source` and `params`
inlined. Load it with:

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", "{tag}", split="train")
```
"""
    (folder / "README.md").write_text(card, encoding="utf-8")


def build_dataset_readme(repo_id: str, tags: list[str], ladder: dict[str, dict]) -> str:
    tags = sorted(tags, key=tag_to_int)
    config_lines = []
    for i, tag in enumerate(tags):
        config_lines.append(f"  - config_name: \"{tag}\"")
        config_lines.append(f"    data_files: \"{tag}/corpus.jsonl\"")
        if i == 0:
            config_lines.append("    default: true")
    configs_yaml = "\n".join(config_lines)

    table = "\n".join(
        f"| `{t}` | {ladder.get(t, {}).get('count', tag_to_int(t)):,} | "
        f"{ladder.get(t, {}).get('seed', '—')} |"
        for t in tags
    )

    largest = max((tag_to_int(t) for t in tags), default=0)
    if largest >= 1_000_000:
        size_cat = "1M<n<10M"
    elif largest >= 100_000:
        size_cat = "100K<n<1M"
    elif largest >= 10_000:
        size_cat = "10K<n<100K"
    elif largest >= 1_000:
        size_cat = "1K<n<10K"
    else:
        size_cat = "n<1K"

    return f"""\
---
license: cc0-1.0
pretty_name: CadQuarry
tags:
  - cad
  - cadquery
  - procedural-generation
  - parametric
  - 3d
size_categories:
  - {size_cat}
configs:
{configs_yaml}
---

# CadQuarry

Procedurally generated, **execution-validated**, fully **parametric** CadQuery
programs and the geometry signatures they produce. Every part is a pure Python
function of typed, range-bounded parameters and is reproducible **bit-for-bit**
from a single seed.

- **Generator (canonical source):** https://github.com/jacobjennings/CadQuarry
  — generator version **v{GEN_VERSION}**, config `configs/default.toml`.
- **Code license:** Apache-2.0 · **Data license:** CC0-1.0 (this dataset).
- **Live preview of a 1k sample:** see the repo `sample/demo-1k/preview.html`.

This dataset is a **convenience artifact**: the generator plus the seed ladder
below is the actual deliverable. Anything here can be regenerated or extended.

## Configs (size ladder)

| Config | Parts | Seed |
|--------|-------|------|
{table}

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", "{tags[0] if tags else '1k'}", split="train")
print(ds[0]["source"])   # the parametric CadQuery program
```

## Schema

Each row: `part_id`, `family`, `tier`, `seed`, `symmetry`, `op_count`,
`ir_hash`, `generator_version`, `license`, `geometry_signature`
(volume, surface area, principal moments, face/edge/vertex counts),
`params` (the typed PARAMS schema), and `source` (the full `.py` program).

## Reproducibility

```bash
pip install -e ".[dev]"   # from the generator repo
cadquarry generate --seed <seed> --count <count> \\
    --config configs/default.toml --out <tag>/
```

Same generator version + seed ⇒ identical corpus and identical geometry
signatures.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    default_repo, ladder = load_publish_ladder()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", metavar="TAG",
                    help=f"Sizes to build/upload (default: 1k). Available: {', '.join(ladder)}")
    ap.add_argument("--all", action="store_true", help="Build/upload every size in the ladder")
    ap.add_argument("--repo-id", default=os.environ.get("CADQUARRY_HF_REPO") or default_repo,
                    help="HF dataset repo id (default: from seeds/v1.toml or CADQUARRY_HF_REPO)")
    ap.add_argument("--workdir", default=str(REPO_ROOT / ".hf_build"),
                    help="Staging dir for generated corpora (default: .hf_build)")
    ap.add_argument("--workers", type=int, default=0, help="Generation/export workers (0 = auto)")
    ap.add_argument("--with-geometry", action="store_true",
                    help="Also export + upload compact STL meshes per part (large)")
    ap.add_argument("--private", action="store_true", help="Create the dataset repo as private")
    ap.add_argument("--force", action="store_true", help="Regenerate even if a corpus already exists")
    ap.add_argument("--dry-run", action="store_true", help="Generate + pack locally, do not upload")
    args = ap.parse_args(argv)

    if args.all:
        tags = list(ladder)
    elif args.sizes:
        tags = args.sizes
    else:
        tags = ["1k"]

    unknown = [t for t in tags if t not in ladder]
    if unknown:
        raise SystemExit(f"Unknown size(s): {unknown}. Available: {', '.join(ladder)}")

    # Resolve repo namespace + token unless this is a pure local dry run.
    token = resolve_token()
    repo_id = args.repo_id
    api = None
    if not args.dry_run:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise SystemExit(
                "huggingface_hub is required to upload. Install it:\n"
                "    pip install -e \".[publish]\"   (or: pip install huggingface_hub)\n"
                "Or run with --dry-run to only build locally."
            )
        if not token:
            raise SystemExit(
                "No HuggingFace token found. Set HF_TOKEN in your environment "
                "(or run `huggingface-cli login`). The token is never printed or stored by this script."
            )
        api = HfApi(token=token)
        whoami = api.whoami()
        username = whoami.get("name")
        if "/" not in repo_id:
            repo_id = f"{username}/{repo_id}"
        print(f"Authenticated as '{username}'. Target dataset repo: {repo_id}")

    staging = Path(args.workdir)
    upload_root = staging / "upload"
    upload_root.mkdir(parents=True, exist_ok=True)

    built: list[str] = []
    for tag in sorted(tags, key=tag_to_int):
        spec = ladder[tag]
        print(f"\n=== {tag}: {spec['count']:,} parts (seed {spec['seed']}) ===")
        corpus_dir = staging / "corpora" / tag
        generate_corpus(corpus_dir, spec["count"], spec["seed"], args.workers, args.force)

        dest = upload_root / tag
        dest.mkdir(parents=True, exist_ok=True)
        print("  · packing corpus.jsonl …")
        n_rows = pack_corpus_jsonl(corpus_dir, dest / "corpus.jsonl", include_source=True)
        write_size_card(dest, tag, spec["count"], spec["seed"], n_rows, repo_id)
        if args.with_geometry:
            export_geometry(corpus_dir, dest / "stl", args.workers)
        built.append(tag)
        print(f"  · packed {n_rows:,} rows -> {dest / 'corpus.jsonl'}")

    # Card lists every config present in the repo (existing + newly built).
    existing: set[str] = set()
    if api is not None:
        try:
            for f in api.list_repo_files(repo_id, repo_type="dataset"):
                if f.endswith("/corpus.jsonl"):
                    existing.add(f.split("/", 1)[0])
        except Exception:
            pass
    all_tags = sorted(set(built) | existing, key=tag_to_int)
    readme = build_dataset_readme(repo_id, all_tags, ladder)
    (upload_root / "README.md").write_text(readme, encoding="utf-8")

    if args.dry_run:
        print(f"\n[dry-run] Built {built} under {upload_root}. Skipping upload.")
        return 0

    assert api is not None
    print(f"\nEnsuring dataset repo {repo_id} exists …")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    print("Uploading … (this can take a while for large sizes)")
    # Only push the sizes built in THIS run (plus the regenerated card). The
    # staging dir may hold leftovers from earlier/aborted runs; uploading the
    # whole folder would sweep those in, so we restrict to the built tags.
    allow = ["README.md"] + [f"{t}/**" for t in built]
    api.upload_folder(
        folder_path=str(upload_root),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow,
        commit_message=f"Publish CadQuarry v{GEN_VERSION} corpora: {', '.join(built)}",
    )
    print(f"\nDone. https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
