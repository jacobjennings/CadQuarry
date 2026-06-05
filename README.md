# CadQuarry

**Procedural generator for diverse, valid, parametric CadQuery programs and the geometry they produce.**

Every generated part is a pure Python function of its parameters — tweak any slider and the geometry updates in milliseconds, no regeneration, no ML. CadQuarry exists to be a clean, freely-usable source of CAD program data: permissively licensed code, CC0 data.

---

## Where things live

CadQuarry is split across two homes so that the code/data license split maps
onto the platforms instead of needing prose:

| | Home | License | Contents |
|---|---|---|---|
| **Generator** | this GitHub repo | Apache-2.0 | code, seed lists, docs, a committed **1,000-part sample** in [`sample/demo-1k/`](sample/demo-1k/) |
| **Full corpus** | [Hugging Face dataset](https://huggingface.co/datasets/jacobjennings/cadquarry) | CC0-1.0 | the size ladder (1k → 500k), browsable in the dataset viewer, `load_dataset`-able |

The published corpus is a **convenience artifact** — the generator plus the
seed list ([`seeds/v1.toml`](seeds/v1.toml)) is the canonical source. Everything
is reproducible bit-for-bit from a seed.

### 🔎 Live in-browser preview (no install)

The committed sample renders its real geometry directly in your browser:

- **[▶ Open the demo-1k preview](https://jacobjennings.github.io/CadQuarry/sample/demo-1k/preview.html)** (GitHub Pages)

---

## What it does

- Generates large batches of unique, executable [CadQuery](https://cadquery.readthedocs.io) programs across a broad operation vocabulary (plates, shafts, blocks, enclosures, flanged hubs, ribbed structures, profiled extrusions, L/C/Z angle brackets, and multi-section compound assemblies — manifolds, stepped shafts, standoffs — with round, polygonal, slot, and rectangular cross-sections). Render images and STL/STEP files are included using the default parameters for each object.
- Guarantees validity by execution — every accepted part actually builds to a non-empty solid.
- Every program is **parametric by construction**: it declares a machine-readable `PARAMS` schema (typed, range-bounded, UI-labeled) and is a pure function `build(p)` of those parameters.
- Ships a live **customizer**: open any part, move sliders, the 3D view updates.
- Ships a **gallery**: browse a generated corpus, filter by family/tier, click to open in the customizer.
- Fully **reproducible**: seed + generator version → identical corpus, bit-for-bit.
- **No restrictions on use**: code is Apache-2.0, generated data is CC0-1.0.

---

## Setup

CadQuarry is developed against [uv](https://docs.astral.sh/uv/) and a local
`.venv`. uv resolves the whole stack (including CadQuery and its OCP kernel)
from PyPI, so no conda step is required.

```bash
# 1. Install uv if you don't have it (see https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create a Python 3.11+ virtualenv in ./.venv
uv venv --python 3.11

# 3. Install CadQuarry + extras into it (dev tooling + geometry export deps)
uv pip install -e ".[dev,export]"
```

This pins everything into `./.venv`. The examples below call the venv's
executables directly as `.venv/bin/<cmd>` so they work without activating the
environment — if you prefer, run `source .venv/bin/activate` once and drop the
`.venv/bin/` prefix.

> If CadQuery's wheels don't resolve for your platform, see the
> [CadQuery install docs](https://cadquery.readthedocs.io/en/latest/installation.html);
> everything else in CadQuarry installs cleanly from PyPI.

---

## Quick start

```bash
# Generate 100 parts
.venv/bin/cadquarry generate --count 100 --seed 42 --out dataset/

# Build EVERY corpus in the seed ladder (generate + export geometry) in one pass
.venv/bin/cadquarry build                       # all sizes -> datasets/{tag}/ with STEP+STL+renders
.venv/bin/cadquarry build --sizes 1k 2k 5k      # just a subset
.venv/bin/cadquarry build --no-export           # generate only, skip geometry

# Open the live customizer for a single part
.venv/bin/cadquarry serve --part examples/plate_with_holes.py

# Browse a generated corpus
.venv/bin/cadquarry serve --dataset dataset/

# Execute a part with parameter overrides
.venv/bin/cadquarry run examples/plate_with_holes.py --set plate_w=80 --set thickness=8

# Export to STL
.venv/bin/cadquarry run examples/plate_with_holes.py --export stl --out plate.stl

# Print the parameter schema for a part
.venv/bin/cadquarry info examples/plate_with_holes.py

# Verify all parts in a corpus re-execute correctly
.venv/bin/cadquarry verify dataset/
```

---

## Part families

| Family | Description | Weight |
|---|---|---|
| `plate` | Flat rectangular plates with holes, fillets, pockets | 20% |
| `bracket` | Angle brackets — L (single leg), C/channel (two legs), Z (cranked offset) — with per-leg holes and optional gussets | 16% |
| `revolved` | Shafts, bushings, washers — solid of revolution | 16% |
| `block` | Prismatic blocks/housings with pockets and bosses | 12% |
| `compound` | Multi-section assemblies — manifolds (bored side ports), stepped shafts (+ polygon drive heads), polygon standoffs, pedestals, side spigots, mounting tabs. Sections use round, polygonal (hex/oct), slot, and rect cross-sections | 12% |
| `flanged` | Revolved stub + polar bolt-circle pattern | 7% |
| `ribbed` | Base plate + patterned thin ribs | 6% |
| `enclosure` | Shelled box (hollow, open-top) | 6% |
| `profiled` | Long constant cross-section — rod, tube, slot, or 3/4/5/6/8-sided polygon bar | 5% |

Family weights, tier distributions, and per-family dimension ranges are all tunable via `configs/default.toml`.

---

## Complexity tiers

| Tier | Description | Default weight |
|---|---|---|
| 0 | Single feature / bare primitive | 25% |
| 1 | 2–3 ops: holes, a base + one attached section (spigot, column, tab) | 40% |
| 2 | Fillets, chamfers, counterbore/countersink, drive heads, extra ports | 25% |
| 3 | Pockets, bosses, ribs, additional bored ports, deep multi-section trees | 10% |

Tiers are clamped per family (e.g. `bracket`/`block`/`compound` start at tier 1),
and the `compound` family scales section count with tier — a manifold sprouts
2 → 3 → 4 bored ports as the tier rises.

---

## Performance

Generation validates every part by execution. The expensive step is importing
CadQuery/OCP (~1–1.5s), so CadQuarry runs a pool of **persistent workers** that
import CadQuery once and then build many parts each, fanning the work across
cores. Sampling stays seeded per-part and the accept/dedup decision stays in
strict attempt order, so output is **bit-for-bit identical regardless of worker
count** — the same seed always yields the same corpus.

Measured on an **AMD Ryzen Threadripper 9960X (24C/48T)** with the default
24-worker pool (`generate`, execution-validated, no geometry export):

| Dataset size | Estimated time | Notes |
|---|---|---|
| 5,000   | ~20 s     | |
| 10,000  | ~40 s     | |
| 25,000  | ~1 m 45 s | |
| 50,000  | ~3 m 25 s | |
| 100,000 | ~6 m 50 s | |
| 200,000 | ~13 m 30 s | |

Throughput is roughly **~250 accepted parts/sec** (sustained) after a ~2s
worker warm-up. Anchored on real runs: 200 parts in **2.6s**, 2,000 in **9.9s**.
For reference, the legacy one-subprocess-per-part path managed ~0.85 parts/sec
(~90× slower) — a 50k corpus would have taken **over 16 hours** instead of
minutes.

Tune the pool with `--workers N` (default: `min(cores, 24)`). Estimates scale
roughly linearly with core count and exclude STEP/STL/point-cloud export
(`cadquarry export`, which uses the same worker pool).

---

## Parametric format

Every generated `.py` file is self-contained and customizer-compatible:

```python
import cadquery as cq

# Generated by CadQuarry v0.1.0 — seed 42 — CC0-1.0
PARAMS = {
    "plate_w":  {"type": "float", "default": 80.0, "min": 30.0, "max": 150.0, "step": 1.0,
                 "group": "Body", "label": "Plate width (mm)"},
    "filleted": {"type": "bool",  "default": True,  "group": "Body", "label": "Fillet corners"},
    ...
}

def build(p):
    result = cq.Workplane("XY").box(p["plate_w"], p["plate_d"], p["thickness"])
    if p["filleted"]:
        result = result.edges("|Z").fillet(min(p["plate_w"], p["plate_d"]) * 0.08)
    ...
    return result

result = build({k: v["default"] for k, v in PARAMS.items()})
```

Full format spec: [docs/parametric-format.md](docs/parametric-format.md)

---

## Dataset layout

```
dataset/
├── manifest.jsonl            one JSON record per part (id, family, tier, signature, paths)
├── DATASET_CARD.md           scale, distribution, generator version, license
├── parts/{id}.py             parametric CadQuery source
├── params/{id}.params.json   parameter schema sidecar
├── meta/{id}.meta.json       provenance: seed, family, tier, geometry signature
├── geometry/{id}.step/.stl   (optional, generated by cadquarry export)
├── renders/{id}/{view}.png   (optional) multi-angle PNG renders, one per view
└── pointclouds/{id}.ply      (optional) sampled point cloud
```

`cadquarry generate` writes only the text artifacts (code, params, meta).
Geometry is produced on demand by `cadquarry export`, whose default formats
are **STEP + STL + renders**:

```bash
# STEP, STL, and 8-angle renders for every part (the default)
.venv/bin/cadquarry export dataset/

# Pick formats explicitly
.venv/bin/cadquarry export dataset/ --formats step,stl,render,pointcloud
```

Each part is rendered from **eight standard viewpoints** — `front`, `top`,
`right`, a canonical `iso`, and the four isometric corners (`iso_fr`, `iso_fl`,
`iso_br`, `iso_bl`) — written to `renders/{id}/{view}.png`. Renders need the
optional `trimesh` + `matplotlib` deps (already covered by the `export` extra,
i.e. `uv pip install -e ".[export]"`); if they're missing, `export` prints one
warning and skips renders while still writing STEP/STL.

---

## Build everything in one pass

`cadquarry build` is the single unified command that generates **and** exports
every corpus in the seed ladder (`[[publish.corpus]]` in
[`seeds/v1.toml`](seeds/v1.toml)), so you don't have to script a
generate-then-export loop yourself.

```bash
# Generate + export STEP/STL/renders for every size -> datasets/{tag}/
.venv/bin/cadquarry build

# Build a subset, choose formats, change the base output dir
.venv/bin/cadquarry build --sizes 1k 2k 5k --formats step,stl --out datasets/

# Generate only (no geometry)
.venv/bin/cadquarry build --no-export
```

For each ladder tag it writes `datasets/{tag}/` with the usual corpus layout
plus exported geometry. Builds are **resumable**: a corpus whose
`manifest.jsonl` already has enough parts is reused as-is (it's bit-identical to
a fresh run anyway); pass `--force` to regenerate. `--workers` is shared by both
the generation and export phases.

> Heads up: the ladder goes up to **500k parts**. Run `--sizes` with the
> specific tags you want unless you really intend to build the whole ladder.

---

## Reproducibility

```bash
# Re-generate an exact corpus from its seed
.venv/bin/cadquarry generate --seed 42 --count 5000 --config configs/default.toml --out dataset-repro/

# Verify signatures match
.venv/bin/cadquarry verify dataset-repro/
```

Same seed + same generator version → identical `manifest.jsonl` and identical geometry signatures.

---

## Committed sample (`sample/demo-1k/`)

A pinned 1,000-part corpus is checked into the repo so anyone can inspect the
format, run the customizer, and preview the data without downloading anything:

```
sample/demo-1k/
├── manifest.jsonl       one record per part
├── parts/{id}.py        parametric CadQuery source
├── params/{id}.params.json
├── meta/{id}.meta.json
├── geometry/{id}.step   STEP B-rep solids
├── geometry/{id}.stl    compact binary meshes (for the in-browser preview)
├── DATASET_CARD.md
└── preview.html         self-contained gallery (three.js, lazy-loaded geometry)
```

It is corpus `demo-1k` from `seeds/v1.toml` (seed `1234`). Regenerate it:

```bash
# 1. Regenerate the part sources + params + meta (deterministic from seed 1234)
.venv/bin/cadquarry generate --seed 1234 --count 1000 --out sample/demo-1k/

# 2. Export geometry into geometry/ — STEP B-reps + the compact binary STL
#    meshes the in-browser preview loads
.venv/bin/cadquarry export sample/demo-1k/ --formats step,stl
```

---

## Using the published dataset (Hugging Face)

The full corpus is published at
[`jacobjennings/cadquarry`](https://huggingface.co/datasets/jacobjennings/cadquarry).
You don't need to install CadQuarry or CadQuery to consume it — just the
`datasets` library.

```bash
pip install datasets
```

```python
from datasets import load_dataset

# Code + metadata only (fastest; JSONL-backed):
ds = load_dataset("jacobjennings/cadquarry", "1k", split="train")
print(ds[0]["source"])   # full parametric CadQuery program (the canonical artifact)
print(ds[0]["family"])   # e.g. "plate", "revolved", "compound"
print(ds[0]["params"])   # typed parameter schema (JSON)

# With 8-view shaded renders (PIL images):
ds = load_dataset("jacobjennings/cadquarry", "1k-renders", split="train")
ds[0]["render_iso"].show()

# With binary STL meshes:
import io, trimesh
ds = load_dataset("jacobjennings/cadquarry", "1k-stl", split="train")
mesh = trimesh.load(io.BytesIO(ds[0]["stl_bytes"]), file_type="stl")

# Everything (renders + STL + STEP B-rep):
ds = load_dataset("jacobjennings/cadquarry", "1k-full", split="train")
with open("part.step", "wb") as f:
    f.write(ds[0]["step_bytes"])
```

### Configs

Each corpus size is published as **six** configs, so you fetch only what you
need. Swap the `1k` prefix for any size in the ladder:

| Config | Contents | Format |
|---|---|---|
| `<tag>` | CadQuery source + metadata | JSONL |
| `<tag>-renders` | + 8-view render images | Parquet (`render_*` = image) |
| `<tag>-stl` | + binary STL mesh | Parquet (`stl_bytes` = binary) |
| `<tag>-step` | + binary STEP B-rep | Parquet (`step_bytes` = binary) |
| `<tag>-geo` | + renders + STL | Parquet |
| `<tag>-full` | + renders + STL + STEP | Parquet |

Available `<tag>` sizes (from [`seeds/v1.toml`](seeds/v1.toml)): `1k`, `2k`,
`5k`, `10k`, `20k`, `50k`, `100k`, `200k`, `500k`. For example,
`load_dataset("jacobjennings/cadquarry", "50k-stl")`.

Every part is reproducible bit-for-bit from its seed, so the published data is a
**convenience artifact** — the generator plus `seeds/v1.toml` is the canonical
source.

---

## Publishing the full corpus to Hugging Face

[`scripts/publish_to_hf.py`](scripts/publish_to_hf.py) builds the reproducible
size ladder (1k, 2k, 5k, 10k, 20k, 50k, 100k, 200k, 500k) and uploads each as a
set of `load_dataset` configs of one HF dataset. The code-only variant is packed
into a single `corpus.jsonl` with the parametric `source` and `params` inlined;
the geometry variants (renders/STL/STEP) are packed into Snappy-compressed
Parquet with typed binary columns.

```bash
uv pip install -e ".[publish]"          # adds huggingface_hub + pyarrow

# Build + upload the small configs to your own repo
.venv/bin/python scripts/publish_to_hf.py --sizes 1k 2k 5k --repo-id <user>/cadquarry

# Build everything (large; 500k generation takes a while)
.venv/bin/python scripts/publish_to_hf.py --all

# Generate + pack locally without uploading (inspect .hf_build/)
.venv/bin/python scripts/publish_to_hf.py --sizes 1k --dry-run
```

**Secrets.** The script reads your token from the `HF_TOKEN` environment
variable (or a prior `huggingface-cli login`) and **never prints, logs, or
commits it**. Nothing secret is stored in the repo, so the script works as-is
for anyone with their own HF account who wants to regenerate or fork the data.
The size ladder's seeds live in `seeds/v1.toml` under `[[publish.corpus]]`.

---

## Configuration

All distribution knobs live in `configs/default.toml`. Key sections:

```toml
[distribution.families]
plate    = 0.20
revolved = 0.16
bracket  = 0.16
compound = 0.12
...

[distribution.tiers]
tier0 = 0.25
tier1 = 0.40
tier2 = 0.25
tier3 = 0.10

[families.plate]
width_min = 20.0
width_max = 120.0
...

[fasteners]
clearance_diameters = [3.2, 4.3, 5.3, 6.4, 8.4, 10.5, 13.0]
```

---

## Development

```bash
uv pip install -e ".[dev]"
.venv/bin/pytest                  # runs tests that don't require cadquery
.venv/bin/pytest -k "not exec"   # same (explicit filter)
```

Tests in `tests/` that don't touch the executor run without CadQuery. Execution tests require CadQuery.

---

## License

| What | License |
|---|---|
| Generator source code (`cadquarry/`) | [Apache-2.0](LICENSE) |
| Generated data (`.py`, `.step`, `.stl`, `.params.json`, `.meta.json`) | [CC0-1.0](DATA_LICENSE) |

CC0 makes the intent unambiguous: do whatever you want with the data, including commercial use, no attribution required.
