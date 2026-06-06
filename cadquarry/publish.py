"""
Pack pre-built CadQuarry corpora and publish them to a HuggingFace dataset repo.

This module does **not** generate or render anything. Corpus generation and
geometry export are a separate, explicit step — run ``cadquarry build`` first,
which writes one ``datasets/{tag}/`` corpus per size (manifest + parts +
geometry + renders). This publisher then only **packs** those artifacts into
HuggingFace configs and **uploads** them.  It is driven through the CLI:

    cadquarry publish              # all built sizes -> <user>/cadquarry
    cadquarry publish --sizes 1k 2k 5k --dry-run

Design goals
------------
* **No secrets in the repo.** The HF token is read from the environment
  (``HF_TOKEN``, falling back to ``HUGGING_FACE_HUB_TOKEN``) or a prior
  ``huggingface-cli login``. It is never printed, logged, or written to disk.
* **Reproducible.** Every size is pinned to a ``(count, seed)`` in the active
  ``seeds/v*.toml`` under ``[[publish.corpus]]``.
* **Six content variants per corpus** × **two complexity-tier slices** so
  consumers fetch exactly the columns/tiers they need.

Variants published per corpus size
-----------------------------------
  {tag}              — CadQuery source + metadata only (JSONL; fastest to load)
  {tag}-renders      — + 8-view render images (Parquet, ``dtype: image`` columns)
  {tag}-stl          — + binary STL mesh (Parquet, ``dtype: binary`` column)
  {tag}-step         — + binary STEP B-rep (Parquet, ``dtype: binary`` column)
  {tag}-geo          — + renders + STL (no STEP)
  {tag}-full         — + renders + STL + STEP (everything)

Each is also published as ``{tag}-t0-2[…]`` (tiers 0–2 only), with data nested
under ``{tag}/tier0-2/``. Geometry variants are skipped automatically for any
corpus that lacks the corresponding artifacts.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

from . import __version__ as GEN_VERSION
from .dataset import (
    RENDER_VIEWS,
    load_manifest,
    pack_corpus_jsonl,
    pack_corpus_parquet,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS_DIR = REPO_ROOT / "seeds"
DEFAULT_CORPUS_ROOT = REPO_ROOT / "datasets"
DEFAULT_WORKDIR = REPO_ROOT / ".hf_build"


class PublishError(Exception):
    """Raised for user-facing publish failures (missing corpus, token, etc.)."""


# ── seed list resolution ─────────────────────────────────────────────────────

def default_seeds_path() -> Path:
    """
    The published seed list matching the current generator version: the
    ``seeds/v*.toml`` whose ``[meta].generator_version == cadquarry.__version__``,
    else the highest-numbered list (falling back to ``v1.toml``).
    """
    def _vnum(p: Path) -> int:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return int(digits) if digits else 0

    candidates = sorted(SEEDS_DIR.glob("v*.toml"), key=_vnum)
    match = None
    for p in candidates:
        try:
            with open(p, "rb") as f:
                meta = tomllib.load(f).get("meta", {})
        except Exception:
            continue
        if str(meta.get("generator_version")) == GEN_VERSION:
            match = p
    if match is not None:
        return match
    if candidates:
        return candidates[-1]
    return SEEDS_DIR / "v1.toml"


def load_publish_ladder(seeds_path: Path | None = None) -> tuple[str, dict[str, dict]]:
    """Return (default_repo_id, {tag: {count, seed}}) from the active seed list."""
    seeds_path = seeds_path or default_seeds_path()
    with open(seeds_path, "rb") as f:
        data = tomllib.load(f)
    pub = data.get("publish", {})
    repo_id = pub.get("repo_id", "cadquarry")
    ladder: dict[str, dict] = {}
    for entry in pub.get("corpus", []):
        ladder[str(entry["tag"])] = {"count": int(entry["count"]), "seed": int(entry["seed"])}
    if not ladder:
        raise PublishError(f"No [[publish.corpus]] entries found in {seeds_path}")
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


def resolve_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


# ── variant + tier-slice definitions ─────────────────────────────────────────
# Each VARIANTS entry: (suffix, include_renders, include_stl, include_step).
# "" suffix = the primary code-only JSONL config.
VARIANTS: list[tuple[str, bool, bool, bool]] = [
    ("",         False, False, False),  # code + metadata only  (JSONL)
    ("-renders", True,  False, False),  # + renders             (Parquet)
    ("-stl",     False, True,  False),  # + STL                 (Parquet)
    ("-step",    False, False, True),   # + STEP                (Parquet)
    ("-geo",     True,  True,  False),  # + renders + STL       (Parquet)
    ("-full",    True,  True,  True),   # + renders + STL + STEP (Parquet)
]

# Orthogonal axis: every content variant is published once for all tiers and
# once limited to tiers 0–2 (inclusive). Each entry: (subdir, config_part, tier_max).
TIER_SLICES: list[tuple[str, str, int | None]] = [
    ("",        "",      None),
    ("tier0-2", "-t0-2", 2),
]


# ── corpus resolution + packing ──────────────────────────────────────────────

def resolve_corpus_dir(corpus_root: Path, tag: str, expected_count: int) -> Path:
    """
    Return the pre-built corpus directory for ``tag`` under ``corpus_root``,
    raising a clear error (pointing at ``cadquarry build``) if it's missing.
    """
    corpus_dir = corpus_root / tag
    manifest = corpus_dir / "manifest.jsonl"
    if not manifest.exists():
        raise PublishError(
            f"No corpus found at {corpus_dir} (missing manifest.jsonl).\n"
            f"Build it first (separate step):\n"
            f"    cadquarry build --sizes {tag} --formats step,stl,render --out {corpus_root}"
        )
    n = len(load_manifest(corpus_dir))
    if expected_count and n < expected_count:
        print(f"  · warning: {tag} has {n} parts, expected {expected_count:,} (packing what's present)")
    else:
        print(f"  · using pre-built corpus ({n:,} parts) at {corpus_dir}")
    return corpus_dir


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
    Pack one (tier slice × content variant) into the upload tree. Returns
    (config_name, relative_data_file), or None if required geometry is missing.
    """
    config_name = f"{tag}{tier_part}{suffix}"

    if include_renders:
        render_base = corpus_dir / "renders"
        if not render_base.is_dir() or not any(render_base.iterdir()):
            print(f"    · skipping {config_name!r} — no renders found")
            return None
    if include_stl and not any((corpus_dir / "geometry").glob("*.stl")):
        print(f"    · skipping {config_name!r} — no STL files found")
        return None
    if include_step and not any((corpus_dir / "geometry").glob("*.step")):
        print(f"    · skipping {config_name!r} — no STEP files found")
        return None

    dest = tag_dir / tier_subdir if tier_subdir else tag_dir
    dest.mkdir(parents=True, exist_ok=True)
    rel_prefix = f"{tag}/{tier_subdir}/" if tier_subdir else f"{tag}/"

    if not suffix:
        out = dest / "corpus.jsonl"
        n = pack_corpus_jsonl(corpus_dir, out, include_source=True, tier_max=tier_max)
        data_file = f"{rel_prefix}corpus.jsonl"
    else:
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


def discover_packed(
    upload_root: Path,
    tag: str,
    active_variants: list[tuple[str, bool, bool, bool]],
) -> list[tuple[str, str, bool, bool, bool]]:
    """
    Find already-packed variant files for ``tag`` under ``upload_root`` (for
    ``upload_only`` resume). Returns the same (config_name, data_file, inc_r,
    inc_s, inc_sp) tuples ``pack_variant`` would have produced, for every file
    that exists. Filenames are deterministic, so this just mirrors the naming.
    """
    found: list[tuple[str, str, bool, bool, bool]] = []
    for tier_subdir, tier_part, _tier_max in TIER_SLICES:
        for sfx, inc_r, inc_s, inc_sp in active_variants:
            fname = "corpus.jsonl" if not sfx else f"corpus{sfx}.parquet"
            data_file = f"{tag}/{tier_subdir}/{fname}" if tier_subdir else f"{tag}/{fname}"
            if (upload_root / data_file).exists():
                config_name = f"{tag}{tier_part}{sfx}"
                found.append((config_name, data_file, inc_r, inc_s, inc_sp))
    return found


# ── README / dataset card ─────────────────────────────────────────────────────

def _features_yaml(include_renders: bool, include_stl: bool, include_step: bool) -> str:
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
    built: dict[str, list[tuple[str, str, bool, bool, bool]]],
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
| `family` | string | `plate`, `bracket`, `revolved`, `block`, `compound`, `enclosure`, `flanged`, `ribbed`, `profiled`, `gear`, `threaded` |
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


# ── entry point ───────────────────────────────────────────────────────────────

def publish(
    *,
    sizes: list[str] | None = None,
    all_sizes: bool = False,
    repo_id: str | None = None,
    corpus_root: str | Path | None = None,
    workdir: str | Path | None = None,
    code_only: bool = False,
    no_renders: bool = False,
    no_stl: bool = False,
    no_step: bool = False,
    private: bool = False,
    dry_run: bool = False,
    upload_only: bool = False,
    verbose: bool = False,
) -> int:
    """
    Pack the pre-built corpora and upload them to HuggingFace.

    With neither ``sizes`` nor ``all_sizes`` given, publishes **all** sizes in
    the ladder (the natural pairing with a bare ``cadquarry build``).

    ``upload_only`` skips packing entirely and uploads whatever is already in
    the staging tree (``.hf_build/upload/``) — used to resume an interrupted
    upload without re-packing the (large) Parquet variants.
    """
    default_repo, ladder = load_publish_ladder()

    if all_sizes or not sizes:
        tags = list(ladder)
    else:
        tags = sizes

    unknown = [t for t in tags if t not in ladder]
    if unknown:
        raise PublishError(f"Unknown size(s): {unknown}. Available: {', '.join(ladder)}")

    corpus_root = Path(corpus_root) if corpus_root else DEFAULT_CORPUS_ROOT
    repo_id = repo_id or os.environ.get("CADQUARRY_HF_REPO") or default_repo

    # Resolve which content types to pack based on flags.
    want_renders = not code_only and not no_renders
    want_stl     = not code_only and not no_stl
    want_step    = not code_only and not no_step

    active_variants = [
        (sfx, r, s, sp) for sfx, r, s, sp in VARIANTS
        if not (r and not want_renders)
        and not (s and not want_stl)
        and not (sp and not want_step)
    ]

    # Resolve repo namespace + token unless this is a pure local dry run.
    token = resolve_token()
    api = None
    if not dry_run:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise PublishError(
                "huggingface_hub is required to upload. Install it:\n"
                "    uv pip install -e \".[publish]\"\n"
                "Or run with --dry-run to only build locally."
            ) from exc
        if not token:
            raise PublishError(
                "No HuggingFace token found. Set HF_TOKEN in your environment "
                "(or run `huggingface-cli login`)."
            )
        api = HfApi(token=token)
        username = api.whoami().get("name")
        if "/" not in repo_id:
            repo_id = f"{username}/{repo_id}"
        print(f"Authenticated as '{username}'. Target dataset repo: {repo_id}")

    workdir = Path(workdir) if workdir else DEFAULT_WORKDIR
    upload_root = workdir / "upload"
    upload_root.mkdir(parents=True, exist_ok=True)

    # {tag: [(config_name, data_file, inc_r, inc_s, inc_sp), …]}
    built: dict[str, list[tuple[str, str, bool, bool, bool]]] = {}

    for tag in sorted(tags, key=tag_to_int):
        spec = ladder[tag]
        if upload_only:
            # Resume mode: don't re-pack — just register whatever variant files
            # are already staged under upload_root/{tag}/.
            found = discover_packed(upload_root, tag, active_variants)
            built[tag] = found
            if found:
                print(f"\n=== {tag}: {len(found)} staged variant(s) to upload ===")
                for cfg, data_file, *_ in found:
                    mb = (upload_root / data_file).stat().st_size / 1e6
                    print(f"  · {cfg}: {data_file} ({mb:.1f} MB)")
            else:
                print(f"\n=== {tag}: nothing staged under {upload_root / tag} — skipping ===")
            continue

        print(f"\n=== {tag}: {spec['count']:,} parts (seed {spec['seed']}) ===")
        corpus_dir = resolve_corpus_dir(corpus_root, tag, spec["count"])

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

    if upload_only and not any(built.values()):
        raise PublishError(
            f"--upload-only: no packed files found under {upload_root}. "
            f"Run a normal `cadquarry publish` (without --upload-only) first."
        )

    readme = build_dataset_readme(repo_id, {t: v for t, v in built.items() if v}, ladder)
    (upload_root / "README.md").write_text(readme, encoding="utf-8")

    if dry_run:
        print(f"\n[dry-run] Packed {sorted(built)} under {upload_root}. Skipping upload.")
        return 0

    assert api is not None
    print(f"\nEnsuring dataset repo {repo_id} exists …")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)

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
