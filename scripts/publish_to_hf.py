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
* **Six content variants per corpus.** Each HuggingFace config gives consumers
  exactly the columns they need without fetching data they don't.
* **Two complexity-tier slices per corpus.** Each content variant is published
  for *all tiers* (canonical names) and again limited to *tiers 0–2 inclusive*
  (a ``-t0-2`` config suffix), so consumers can opt out of the highest-complexity
  parts without post-filtering.

Variants published per corpus size
-----------------------------------
  {tag}              — CadQuery source + metadata only (JSONL; fastest to load)
  {tag}-renders      — + 8-view render images (Parquet, ``dtype: image`` columns)
  {tag}-stl          — + binary STL mesh (Parquet, ``dtype: binary`` column)
  {tag}-step         — + binary STEP B-rep (Parquet, ``dtype: binary`` column)
  {tag}-geo          — + renders + STL (no STEP)
  {tag}-full         — + renders + STL + STEP (everything)

Each of the above is also published as ``{tag}-t0-2[…]`` (tiers 0–2 only),
with data nested under ``{tag}/tier0-2/``.

Examples
--------
    # Build + upload just the small configs (fast, code-only)
    python scripts/publish_to_hf.py --sizes 1k 2k 5k --code-only

    # Everything, to your own repo
    python scripts/publish_to_hf.py --all --repo-id me/cadquarry

    # Generate + pack locally but do not upload (inspect .hf_build/)
    python scripts/publish_to_hf.py --sizes 1k --dry-run

    # Full pipeline with all geometry variants
    python scripts/publish_to_hf.py --sizes 1k 10k
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cadquarry import __version__ as GEN_VERSION          # noqa: E402
from cadquarry.dataset import (                            # noqa: E402
    pack_corpus_jsonl,
    pack_corpus_parquet,
    RENDER_VIEWS,
)
from cadquarry.export import export_corpus_geometry        # noqa: E402

SEEDS_FILE = REPO_ROOT / "seeds" / "v1.toml"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"

# ── variant definitions ────────────────────────────────────────────────────────
# Each entry: (suffix, include_renders, include_stl, include_step)
# "" suffix = the primary code-only JSONL config.
VARIANTS: list[tuple[str, bool, bool, bool]] = [
    ("",         False, False, False),  # code + metadata only  (JSONL)
    ("-renders", True,  False, False),  # + renders             (Parquet)
    ("-stl",     False, True,  False),  # + STL                 (Parquet)
    ("-step",    False, False, True),   # + STEP                (Parquet)
    ("-geo",     True,  True,  False),  # + renders + STL       (Parquet)
    ("-full",    True,  True,  True),   # + renders + STL + STEP (Parquet)
]

# ── tier slices ─────────────────────────────────────────────────────────────────
# A second, orthogonal axis: every content variant is published once for all
# tiers and once limited to tiers 0–2 (inclusive).
# Each entry: (subdir, config_part, tier_max)
#   subdir      — files nested under "{tag}/{subdir}/" ("" = directly in "{tag}/")
#   config_part — inserted between {tag} and the content suffix in the config name
#   tier_max    — inclusive complexity-tier cap (None = keep every tier)
# The all-tiers slice keeps the canonical unlabeled config names (and the default).
TIER_SLICES: list[tuple[str, str, int | None]] = [
    ("",        "",      None),
    ("tier0-2", "-t0-2", 2),
]


# ── seed ladder ────────────────────────────────────────────────────────────────

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


# ── auth ───────────────────────────────────────────────────────────────────────

def resolve_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


# ── build steps ───────────────────────────────────────────────────────────────

def generate_corpus(
    out_dir: Path, count: int, seed: int, workers: int, force: bool, timeout: float,
) -> None:
    import subprocess
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
        "--timeout", str(timeout),
    ]
    if workers:
        cmd += ["--workers", str(workers)]
    print(f"  · generating {count} parts (seed={seed}) …")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def export_geometry(
    corpus_dir: Path,
    workers: int,
    formats: list[str],
    timeout: float,
    verbose: bool,
) -> None:
    """Export geometry artifacts for a corpus using the cadquarry export pipeline."""
    n = default_worker_count(workers)
    print(f"  · exporting geometry ({', '.join(formats)}, {n} workers) …")
    counts = export_corpus_geometry(
        corpus_dir, formats=formats, n_workers=n, timeout=timeout, verbose=verbose,
    )
    for fmt, cnt in counts.items():
        print(f"    {fmt}: {cnt} files")


def default_worker_count(n: int) -> int:
    import os as _os
    return n or min(_os.cpu_count() or 4, 16)


# ── packing ───────────────────────────────────────────────────────────────────

def pack_variant(
    corpus_dir: Path,
    tag: str,
    tag_dir: Path,
    tier_subdir: str,
    tier_part: str,
    tier_max: int | None,
    suffix: str,
    include_renders: bool,
    include_stl: bool,
    include_step: bool,
) -> tuple[str, str] | None:
    """
    Pack one (tier slice × content variant) into the upload tree.  Returns
    (config_name, relative_data_file) on success, or None if the required
    geometry files are missing.

    ``tag_dir`` is ``upload_root/{tag}``; ``tier_subdir`` ("" or e.g. "tier0-2")
    nests the data file, ``tier_part`` ("" or "-t0-2") labels the config name, and
    ``tier_max`` caps the included complexity tier (None = all tiers).
    """
    config_name = f"{tag}{tier_part}{suffix}"

    # Detect whether the needed geometry is actually present.
    if include_renders:
        render_base = corpus_dir / "renders"
        if not render_base.is_dir() or not any(render_base.iterdir()):
            print(f"    · skipping {config_name!r} — no renders found")
            return None
    if include_stl:
        geo_dir = corpus_dir / "geometry"
        if not any(geo_dir.glob("*.stl")):
            print(f"    · skipping {config_name!r} — no STL files found")
            return None
    if include_step:
        geo_dir = corpus_dir / "geometry"
        if not any(geo_dir.glob("*.step")):
            print(f"    · skipping {config_name!r} — no STEP files found")
            return None

    dest = tag_dir / tier_subdir if tier_subdir else tag_dir
    dest.mkdir(parents=True, exist_ok=True)
    rel_prefix = f"{tag}/{tier_subdir}/" if tier_subdir else f"{tag}/"

    if not suffix:
        # Code-only variant: JSONL (HF dataset viewer can browse it inline).
        out = dest / "corpus.jsonl"
        n = pack_corpus_jsonl(corpus_dir, out, include_source=True, tier_max=tier_max)
        data_file = f"{rel_prefix}corpus.jsonl"
    else:
        # Geometry variant: Parquet with binary columns.
        fname = f"corpus{suffix}.parquet"
        out = dest / fname
        n = pack_corpus_parquet(
            corpus_dir, out,
            include_source=True,
            include_renders=include_renders,
            include_stl=include_stl,
            include_step=include_step,
            tier_max=tier_max,
        )
        data_file = f"{rel_prefix}{fname}"

    mb = out.stat().st_size / 1e6 if out.exists() else 0
    print(f"    · {config_name}: {n:,} rows → {data_file} ({mb:.1f} MB)")
    return config_name, data_file


# ── README / dataset card ─────────────────────────────────────────────────────

def _features_yaml(include_renders: bool, include_stl: bool, include_step: bool) -> str:
    """
    Return the `  features:` YAML block for a Parquet config entry.
    Items are at 2-space indent from the config `- ` bullet.
    """
    base = [
        "  features:",
        "  - name: part_id",
        "    dtype: string",
        "  - name: family",
        "    dtype: string",
        "  - name: tier",
        "    dtype: int32",
        "  - name: seed",
        "    dtype: int64",
        "  - name: symmetry",
        "    dtype: string",
        "  - name: op_count",
        "    dtype: int32",
        "  - name: ir_hash",
        "    dtype: string",
        "  - name: generator_version",
        "    dtype: string",
        "  - name: license",
        "    dtype: string",
        "  - name: geometry_signature",
        "    dtype: string",
        "  - name: source",
        "    dtype: string",
        "  - name: params",
        "    dtype: string",
    ]
    if include_renders:
        for v in RENDER_VIEWS:
            base += [f"  - name: render_{v}", "    dtype: image"]
    if include_stl:
        base += ["  - name: stl_bytes", "    dtype: binary"]
    if include_step:
        base += ["  - name: step_bytes", "    dtype: binary"]
    return "\n".join(base)


def _configs_yaml(
    built: dict[str, list[tuple[str, str, bool, bool, bool]]],
    all_tags: list[str],
) -> str:
    """
    Build the `configs:` YAML block.  Each entry starts with `- ` (no extra
    indent) so it nests correctly under `configs:` in the front-matter.
    """
    lines: list[str] = []
    first_code = True
    for tag in all_tags:
        for config_name, data_file, inc_r, inc_s, inc_sp in built[tag]:
            is_code_only = not (inc_r or inc_s or inc_sp)
            lines.append(f'- config_name: "{config_name}"')
            lines.append(f'  data_files: "{data_file}"')
            if first_code and is_code_only:
                lines.append("  default: true")
                first_code = False
            if not is_code_only:
                lines.append(_features_yaml(inc_r, inc_s, inc_sp))
    return "\n".join(lines)


def build_dataset_readme(
    repo_id: str,
    built: dict[str, list[tuple[str, str, bool, bool, bool]]],  # {tag: [(config, file, r, s, sp), …]}
    ladder: dict[str, dict],
) -> str:
    all_tags = sorted(built.keys(), key=tag_to_int)

    largest = max((tag_to_int(t) for t in all_tags), default=0)
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

    table_rows = []
    for tag in all_tags:
        n = ladder.get(tag, {}).get("count", tag_to_int(tag))
        seed = ladder.get(tag, {}).get("seed", "—")
        cfgs = ", ".join(f"`{cn}`" for cn, *_ in built[tag])
        table_rows.append(f"| `{tag}` | {n:,} | {seed} | {cfgs} |")
    table = "\n".join(table_rows)

    ex_tag = all_tags[0] if all_tags else "1k"
    has_renders = any(r          for t in all_tags for _, _, r, s, sp in built[t])
    has_stl     = any(s          for t in all_tags for _, _, r, s, sp in built[t])
    has_full    = any(r and s and sp for t in all_tags for _, _, r, s, sp in built[t])

    # ── YAML front-matter (built as a plain string — no textwrap.dedent) ─────
    fm_lines = [
        "---",
        "license: cc0-1.0",
        "pretty_name: CadQuarry",
        "language:",
        "  - code",
        "tags:",
        "  - cad",
        "  - cadquery",
        "  - procedural-generation",
        "  - parametric",
        "  - 3d",
        "  - synthetic",
        "size_categories:",
        f"  - {size_cat}",
        "configs:",
        _configs_yaml(built, all_tags),
        "---",
        "",
    ]
    front_matter = "\n".join(fm_lines)

    # ── quickstart snippets ──────────────────────────────────────────────────
    render_snip = (
        f'\n# With 8-view renders (PIL Images):\n'
        f'ds = load_dataset("{repo_id}", "{ex_tag}-renders", split="train")\n'
        f'ds[0]["render_iso"].show()         # isometric view\n'
        f'ds[0]["render_front"].show()       # front orthographic\n'
    ) if has_renders else ""

    stl_snip = (
        f'\n# With STL mesh:\n'
        f'import trimesh, io\n'
        f'ds = load_dataset("{repo_id}", "{ex_tag}-stl", split="train")\n'
        f'mesh = trimesh.load(io.BytesIO(ds[0]["stl_bytes"]), file_type="stl")\n'
        f'mesh.show()\n'
    ) if has_stl else ""

    full_snip = (
        f'\n# Full corpus (renders + STL + STEP):\n'
        f'ds = load_dataset("{repo_id}", "{ex_tag}-full", split="train")\n'
        f'with open("part.step", "wb") as f:\n'
        f'    f.write(ds[0]["step_bytes"])\n'
    ) if has_full else ""

    # ── body ─────────────────────────────────────────────────────────────────
    body = f"""\
# CadQuarry

Procedurally generated, **execution-validated**, fully **parametric** CadQuery
programs and the geometry they produce. Every part is a pure Python function of
typed, range-bounded parameters and is reproducible **bit-for-bit** from a seed.

- **Generator (canonical source):** https://github.com/jacobjennings/CadQuarry
  generator version **v{GEN_VERSION}**, config `configs/default.toml`
- **Code license:** Apache-2.0 · **Data license:** CC0-1.0

Each corpus in this dataset is a **convenience artifact**: the generator plus
the seed ladder is the actual deliverable. Any corpus can be regenerated locally.

---

## Quick start

### Python (`datasets` library)

```bash
pip install datasets
```

```python
from datasets import load_dataset

# Code + metadata only (fastest):
ds = load_dataset("{repo_id}", "{ex_tag}", split="train")
print(ds[0]["source"])   # full parametric CadQuery program
print(ds[0]["family"])   # e.g. "plate", "revolved", "block"
{render_snip}{stl_snip}{full_snip}
```

### HuggingFace CLI

```bash
pip install huggingface_hub

# Code-only corpus:
huggingface-cli download {repo_id} \\
    --repo-type dataset \\
    --include "{ex_tag}/corpus.jsonl" \\
    --local-dir ./cadquarry-{ex_tag}

# All variants for one size:
huggingface-cli download {repo_id} \\
    --repo-type dataset \\
    --include "{ex_tag}/*" \\
    --local-dir ./cadquarry-{ex_tag}
```

### Python Hub API (selective download)

```python
from huggingface_hub import snapshot_download

local = snapshot_download(
    "{repo_id}",
    repo_type="dataset",
    allow_patterns=["{ex_tag}/corpus.jsonl", "{ex_tag}/corpus-stl.parquet"],
)
```

---

## Content variants

Each corpus size ships as six HuggingFace configs so you only fetch what you need:

| Config suffix | Content | File format |
|---------------|---------|-------------|
| *(none)* | CadQuery source + metadata | JSONL |
| `-renders` | + 8 shaded render images | Parquet (`render_*` = `image`) |
| `-stl` | + binary STL mesh | Parquet (`stl_bytes` = `binary`) |
| `-step` | + binary STEP B-rep | Parquet (`step_bytes` = `binary`) |
| `-geo` | + renders + STL | Parquet |
| `-full` | + renders + STL + STEP | Parquet |

---

## Complexity-tier slices

Every content variant above is published along a second axis so you can exclude
the highest-complexity parts without filtering yourself:

| Tier slice | Config form | Tiers included |
|------------|-------------|----------------|
| All tiers | `{{tag}}{{suffix}}` (e.g. `1k`, `1k-renders`) | 0–3 |
| Tiers 0–2 | `{{tag}}-t0-2{{suffix}}` (e.g. `1k-t0-2`, `1k-t0-2-renders`) | 0, 1, 2 |

```python
# All tiers (default naming):
ds = load_dataset("{repo_id}", "{ex_tag}", split="train")

# Tiers 0–2 only:
ds = load_dataset("{repo_id}", "{ex_tag}-t0-2", split="train")
```

The two slices share the same schema; the `-t0-2` slice simply omits every row
with `tier == 3`. Data files live under `{{tag}}/` (all tiers) and
`{{tag}}/tier0-2/` (tiers 0–2).

---

## Corpus sizes

| Tag | Parts | Seed | Available configs |
|-----|-------|------|-------------------|
{table}

---

## Schema

**All configs** include:

| Column | Type | Description |
|--------|------|-------------|
| `part_id` | string | Unique deterministic identifier |
| `family` | string | `plate`, `bracket`, `revolved`, `block`, `compound`, `enclosure`, `flanged`, `ribbed`, `profiled` |
| `tier` | int32 | Complexity tier 0–3 |
| `seed` | int64 | Per-part seed |
| `symmetry` | string | Detected symmetry class |
| `op_count` | int32 | Number of CadQuery operations |
| `ir_hash` | string | SHA-256 of the canonical IR (dedup key) |
| `generator_version` | string | CadQuarry version |
| `license` | string | Always `CC0-1.0` |
| `geometry_signature` | string (JSON) | Volume, surface area, face/edge/vertex counts |
| `source` | string | Full `.py` CadQuery program |
| `params` | string (JSON) | Typed parameter schema with ranges and defaults |

**Geometry columns** (geometry variants only):

| Column | Type | Present in |
|--------|------|-----------|
| `render_front` … `render_iso_bl` | image | `-renders`, `-geo`, `-full` |
| `stl_bytes` | binary | `-stl`, `-geo`, `-full` |
| `step_bytes` | binary | `-step`, `-full` |

Render views: `front`, `top`, `right`, `iso`, `iso_fr`, `iso_fl`, `iso_br`, `iso_bl`.

---

## Reproducibility

Same generator version + seed ⇒ identical corpus bit-for-bit:

```bash
pip install -e ".[dev]"   # from the generator repo
cadquarry generate \\
    --seed <seed> --count <count> \\
    --config configs/default.toml \\
    --out <output_dir>
```

To regenerate with all geometry:

```bash
cadquarry build --sizes <tag> --formats step,stl,render --out datasets/
```

---

## Working with the parametric source

Each `source` is a standalone Python module with two exports:

```python
PARAMS: dict[str, dict]          # typed parameter schema
def build(p: dict) -> cq.Workplane: ...
```

Run a part with CadQuery installed:

```python
import cadquery as cq, json

source = ds[0]["source"]
params = json.loads(ds[0]["params"])
defaults = {{k: v["default"] for k, v in params.items()}}

ns = {{}}
exec(compile(source, "<part>", "exec"), ns)
result = ns["build"](defaults)
cq.exporters.export(result, "part.step")
```
"""
    return front_matter + "\n" + body


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    default_repo, ladder = load_publish_ladder()

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--sizes", nargs="+", metavar="TAG",
        help=f"Sizes to build/upload (default: 1k). Available: {', '.join(ladder)}",
    )
    ap.add_argument("--all", action="store_true", help="Build/upload every size in the ladder")
    ap.add_argument(
        "--repo-id", default=os.environ.get("CADQUARRY_HF_REPO") or default_repo,
        help="HF dataset repo id (default: from seeds/v1.toml or CADQUARRY_HF_REPO)",
    )
    ap.add_argument(
        "--workdir", default=str(REPO_ROOT / ".hf_build"),
        help="Staging dir for generated corpora (default: .hf_build)",
    )
    ap.add_argument("--workers", type=int, default=0, help="Generation/export workers (0 = auto)")
    ap.add_argument("--timeout", type=float, default=60.0, help="Per-part timeout (default: 60s)")
    ap.add_argument(
        "--code-only", action="store_true",
        help="Skip geometry export; publish only the code+metadata (JSONL) variant",
    )
    ap.add_argument("--no-renders", action="store_true", help="Skip render generation and render variants")
    ap.add_argument("--no-stl",     action="store_true", help="Skip STL export and STL variants")
    ap.add_argument("--no-step",    action="store_true", help="Skip STEP export and STEP variants")
    ap.add_argument("--private", action="store_true", help="Create the dataset repo as private")
    ap.add_argument("--force", action="store_true", help="Regenerate even if a corpus already exists")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Generate + pack locally, do not upload",
    )
    ap.add_argument("--verbose", "-v", action="store_true")
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

    # Resolve which content types to export based on flags.
    want_renders = not args.code_only and not args.no_renders
    want_stl     = not args.code_only and not args.no_stl
    want_step    = not args.code_only and not args.no_step

    # Geometry formats we actually need to export.
    geo_formats: list[str] = []
    if want_renders:
        geo_formats.append("render")
    if want_stl:
        geo_formats.append("stl")
    if want_step:
        geo_formats.append("step")

    # Active variant set: code always, geometry variants only when wanted.
    active_variants = [
        (sfx, r, s, sp) for sfx, r, s, sp in VARIANTS
        if not (r and not want_renders)
        and not (s and not want_stl)
        and not (sp and not want_step)
    ]

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
                "    pip install -e \".[publish]\"\n"
                "Or run with --dry-run to only build locally."
            )
        if not token:
            raise SystemExit(
                "No HuggingFace token found. Set HF_TOKEN in your environment "
                "(or run `huggingface-cli login`)."
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

    # {tag: [(config_name, data_file, inc_r, inc_s, inc_sp), …]} — across all tags.
    built: dict[str, list[tuple[str, str, bool, bool, bool]]] = {}

    for tag in sorted(tags, key=tag_to_int):
        spec = ladder[tag]
        print(f"\n=== {tag}: {spec['count']:,} parts (seed {spec['seed']}) ===")
        corpus_dir = staging / "corpora" / tag

        generate_corpus(
            corpus_dir, spec["count"], spec["seed"],
            args.workers, args.force, args.timeout,
        )

        if geo_formats:
            export_geometry(
                corpus_dir, args.workers, geo_formats, args.timeout, args.verbose,
            )

        tag_dir = upload_root / tag
        built[tag] = []
        print("  · packing variants …")
        for tier_subdir, tier_part, tier_max in TIER_SLICES:
            for sfx, inc_r, inc_s, inc_sp in active_variants:
                result = pack_variant(
                    corpus_dir, tag, tag_dir,
                    tier_subdir, tier_part, tier_max,
                    sfx, inc_r, inc_s, inc_sp,
                )
                if result is not None:
                    cfg, data_file = result
                    built[tag].append((cfg, data_file, inc_r, inc_s, inc_sp))

    # Merge any existing configs in the remote repo so the README stays complete.
    existing: set[str] = set()
    if api is not None:
        try:
            for f in api.list_repo_files(repo_id, repo_type="dataset"):
                if f.endswith("corpus.jsonl") or f.endswith(".parquet"):
                    tag_part = f.split("/", 1)[0]
                    if tag_part not in built:
                        existing.add(tag_part)
        except Exception:
            pass
    for ext_tag in existing:
        if ext_tag not in built:
            built[ext_tag] = []  # placeholder; no data_file known

    readme = build_dataset_readme(repo_id, {t: v for t, v in built.items() if v}, ladder)
    (upload_root / "README.md").write_text(readme, encoding="utf-8")

    if args.dry_run:
        print(f"\n[dry-run] Built {sorted(built)} under {upload_root}. Skipping upload.")
        return 0

    assert api is not None
    print(f"\nEnsuring dataset repo {repo_id} exists …")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=args.private)

    print("Uploading … (this can take a while for large sizes)")
    built_tags = [t for t, vs in built.items() if vs]
    allow = ["README.md"] + [f"{t}/**" for t in built_tags]
    api.upload_folder(
        folder_path=str(upload_root),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow,
        commit_message=f"Publish CadQuarry v{GEN_VERSION} corpora: {', '.join(built_tags)}",
    )
    print(f"\nDone. https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
