# CadQuarry

**Procedural generator for diverse, valid, parametric CadQuery programs and the geometry they produce.**

Every generated part is a pure Python function of its parameters — tweak any slider and the geometry updates in milliseconds, no regeneration, no ML. CadQuarry exists to be a clean, freely-usable source of CAD program data: permissively licensed code, CC0 data.

> Most families emit self-contained programs that run on `cadquery` alone. The
> three mechanical families — `gear`, `threaded`, and `tapped` — emit programs
> built on the [build123d](https://github.com/gumyr/build123d) stack
> ([py_gearworks](https://github.com/GarryBGoode/py_gearworks),
> [bd_warehouse](https://github.com/gumyr/bd_warehouse)) and bridged back into
> CadQuery, so they require the optional **`mech`** extra to execute (see
> [Setup](#setup)).

---

## Where things live

CadQuarry is split across two homes so that the code/data license split maps
onto the platforms instead of needing prose:

| | Home | License | Contents |
|---|---|---|---|
| **Generator** | this GitHub repo | Apache-2.0 | code, seed lists, docs, a committed **1,000-part sample** in [`sample/demo-1k/`](sample/demo-1k/) |
| **Full corpus** | [Hugging Face dataset](https://huggingface.co/datasets/jacobjennings/cadquarry) | CC0-1.0 | a nested size ladder (1k → 100k, each size a prefix of the next), browsable in the dataset viewer, `load_dataset`-able |

The published corpus is a **convenience artifact** — the generator plus the
seed list ([`seeds/v3.toml`](seeds/v3.toml)) is the canonical source. Everything
is reproducible bit-for-bit from a seed.

### 🔎 Live in-browser preview (no install)

The committed sample renders its real geometry directly in your browser:

- **[▶ Open the demo-1k preview](https://jacobjennings.github.io/CadQuarry/sample/demo-1k/preview.html)** (GitHub Pages)

---

## What it does

- Generates large batches of unique, executable [CadQuery](https://cadquery.readthedocs.io) programs across a broad operation vocabulary: plates, shafts, blocks, enclosures, flanged hubs, ribbed structures, profiled extrusions, L/C/Z angle brackets, and multi-section compound assemblies (manifolds, stepped shafts, standoffs, lofted adapters) — with round, polygonal, slot, and rectangular cross-sections. Beyond those primitives it emits **freeform sketches** (closed line / arc / Bézier / spline loops, extruded or revolved into organic bodies), **lofts** through stacked cross-sections, **sweeps** along spline paths, draft-tapered extrudes, and angled / multi-face holes. Three real mechanical families add involute/cycloid **gears**, standards-based **threaded** parts (ISO/ACME/trapezoidal), and **tapped** holes with real cut ISO internal threads — all built on build123d and bridged into CadQuery (opt-in `mech` extra). Render images and STL/STEP files are included using the default parameters for each object.
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

# 4. (Optional) add the `mech` extra to generate/execute the gear + threaded
#    + tapped families. This pulls in the build123d stack (build123d,
#    bd_warehouse, py_gearworks). py_gearworks has no PyPI release, so git is required.
uv pip install -e ".[mech]"
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

# Build the ONE base corpus the whole size ladder slices from (-> datasets/base/)
.venv/bin/cadquarry build                       # base, with STEP+STL+renders
.venv/bin/cadquarry build --extend-to 500000    # grow the base later, reusing all prior work
.venv/bin/cadquarry build --no-export           # generate only, skip geometry

# Build the base, then pack each size as a slice + upload to Hugging Face (needs HF_TOKEN)
.venv/bin/cadquarry build && .venv/bin/cadquarry publish

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

| Family | Description | Share |
|---|---|---|
| `plate` | Flat rectangular plates with holes, fillets, chamfers, pockets, bosses | 15% |
| `bracket` | Angle brackets — L (single leg), C/channel (two legs), Z (cranked offset) — with per-leg holes and optional gussets | 12% |
| `revolved` | Shafts, bushings, washers — solid of revolution | 12% |
| `block` | Prismatic blocks/housings with pockets, bosses, and angled / multi-face holes | 9% |
| `compound` | Multi-section assemblies — manifolds (bored side ports), stepped shafts (+ polygon drive heads), polygon standoffs, pedestals, side spigots, mounting tabs, and lofted adapters (reducer + bored neck). Sections use round, polygonal (hex/oct), slot, and rect cross-sections | 9% |
| `sketched` | **Freeform 2D sketches** — closed loops of line / arc / Bézier / spline segments, extruded into organic prisms or revolved into organic turned bodies | 5% |
| `flanged` | Revolved stub + polar bolt-circle pattern | 5% |
| `ribbed` | Base plate + patterned thin ribs | 4% |
| `enclosure` | Shelled box (hollow, open-top) | 4% |
| `profiled` | Long constant cross-section — rod, tube, slot, freeform sketch, or 3/4/5/6/8-sided polygon bar | 3% |
| `lofted` | Smooth lofts through 2–3 stacked cross-sections — square↔round reducers and adapters | 4% |
| `swept` | A section (round or rounded-rect) swept along a smooth spline path — handles, bent tubes, ducts | 3% |
| `tapped` ⚙ | Plates/blocks and cylinders with **real cut ISO internal threads** (tapped holes) | 4% |
| `gear` ⚙ | Real involute/cycloid gears — spur, helical (incl. herringbone), bevel, cycloid, inside-ring; ISO 54 modules, optional profile shift, root fillets, bore, hub | 5% |
| `threaded` ⚙ | Real thread geometry — ISO metric external/internal, plus ACME and metric-trapezoidal lead screws | 5% |

Shares are the approximate sampling fraction of each family under the default
config; the underlying relative weights, tier distributions, and per-family
dimension ranges are all tunable via `configs/default.toml`.

> ⚙ The `gear`, `threaded`, and `tapped` families require the optional **`mech`**
> extra (`uv pip install -e ".[mech]"`) and a compatible CadQuery/build123d
> pairing to execute — their emitted programs are **not** self-contained on
> `cadquery` alone. Consumers of the published dataset who only read the `source`
> text need nothing extra; only re-executing these parts needs the `mech` stack.

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

## Thread & gear standards (the `mech` families)

The `gear`, `threaded`, and `tapped` families build real, standards-based
geometry through the [build123d](https://github.com/gumyr/build123d) stack and
bridge it back into CadQuery via each object's shared OCP `.wrapped` handle — so
a generated gear or thread is, after one line, an ordinary single-solid
`cq.Workplane` that flows through the same execution, dedup, export, and
customizer paths as every other family.

**Gears** ([py_gearworks](https://github.com/GarryBGoode/py_gearworks)):

- Types: spur, helical (including herringbone), bevel, cycloid, and inside-ring.
- ISO 54 preferred module series, 20° pressure angle (py_gearworks default).
- Optional profile shift, root fillets, an axial bore, and a raised hub —
  the bore/hub are added with plain CadQuery ops after the bridge.

**Threads** ([bd_warehouse](https://github.com/gumyr/bd_warehouse)):

- ISO metric, both **external** (thread fused onto a shank sized to the root)
  and **internal** (thread fused into a bored body), drawn from the ISO 261/262
  coarse/fine pitch tables.
- **Lead screws**: ACME and metric-trapezoidal external threads, by standard
  size designation.

**Tapped holes** ([bd_warehouse](https://github.com/gumyr/bd_warehouse)): a
plate/block or cylinder with one or more **real cut ISO internal threads** — for
each hole the body gets a clearance bore at the thread root and an internal
`IsoThread` is unioned in, fused into one valid solid before the bridge. Hole
patterns are centred, corner, or grid; the thread designation (e.g. `M6×1.0`)
is recorded in the part descriptor.

**Reproducibility nuance.** Geometry signatures (quantized volume, area, and
sorted principal moments of inertia) for these solver-built families are stable
to the dedup tolerance (3 significant figures), so dedup and seed-reproducible
*selection* hold. Exact STEP/STL **bytes**, however, come out of a numeric
NURBS/thread solver and are not guaranteed bit-identical across platforms or
library versions the way the primitive families are — pin the `mech` versions
(recorded in [`seeds/v3.toml`](seeds/v3.toml)) if byte-level reproduction
matters.

---

## Performance

Generation validates every part by execution. Importing CadQuery/OCP (~1–1.5s)
is a fixed startup cost, amortized by a pool of **persistent workers** that
import once and then build many parts each, fanning the work across cores. With
the full default config the throughput floor is set by the solver-built
**mechanical families** (`gear` + `threaded` + `tapped`, ~14% of the mix), which
solve real involute/thread geometry and cost a meaningful fraction of a second
each — so end-to-end generation runs at roughly **~8–10 validated parts/sec** rather than
the hundreds/sec the millisecond-scale primitive families alone would sustain.
Sampling stays seeded per-part and the accept/dedup decision stays in strict
attempt order, so output is **bit-for-bit identical regardless of worker count**
— the same seed always yields the same corpus.

Measured on an **AMD Ryzen Threadripper 9960X (24C/48T)** with the default
24-worker pool, full default config (mech families included), `generate` only
(execution-validated, no geometry export):

| Dataset size | Wall-clock time | Throughput |
|---|---|---|
| 1,000   | ~2 m 10 s  | 7.5 part/s |
| 2,000   | ~3 m 30 s  | 9.5 part/s |
| 5,000   | ~9 m 30 s  | 8.7 part/s |
| 10,000  | ~18 m 45 s | 8.9 part/s |
| 20,000  | ~36 m 30 s | 9.1 part/s |
| 50,000  | ~1 h 30 m  | 9.2 part/s |
| 100,000 | ~2 h 50 m  | 9.8 part/s |
| 200,000 | **~6 h** (est.) | ~9 part/s |

Throughput climbs slightly with corpus size as the fixed worker warm-up
amortizes, then settles near **~9–10 parts/s**. The table is the measured
`v0.5.0` benchmark; `v0.6.0` adds the `tapped` family, lifting the solver-paced
`mech` share to ~14% and nudging steady-state throughput marginally lower, so a
full **200k generation lands around ~6 hours** (validated, no export) on this
class of workstation — the bolded row above is that extrapolation, not a fresh
measurement. Because this rate is paced almost entirely by the solver-built
`mech` families, dropping them (or lowering their weight in
`configs/default.toml`) pushes throughput up by more than an order of magnitude,
since the remaining primitive families build in milliseconds.

> Note: meaningful throughput numbers require a corpus large enough to amortize
> per-worker warm-up (each worker imports the OCP kernel once, ~1–1.5 s). Small
> timing runs — especially at high worker counts — are dominated by that startup
> and **understate** steady-state throughput; benchmark at ≥10k parts.

Tune the pool with `--workers N` (default: `min(cores, 24)`). **The best worker
count is system-dependent**: it scales with core count, but the solver-built
families are memory-hungry, so on machines with less RAM you may need *fewer*
workers than you have cores to avoid swapping (very large corpora at high worker
counts can exhaust memory). Times exclude STEP/STL/render export
(`cadquarry export`, which uses the same worker pool).

---

## Parametric format

Every generated `.py` file is self-contained and customizer-compatible:

```python
import cadquery as cq

# Generated by CadQuarry v0.6.0 — seed 42 — CC0-1.0
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
├── manifest.jsonl            one JSON record per part (id, family, tier, signature, descriptor, dimensions, paths)
├── DATASET_CARD.md           scale, distribution, generator version, license
├── parts/{id}.py             parametric CadQuery source
├── params/{id}.params.json   parameter schema sidecar
├── meta/{id}.meta.json       provenance: seed, family, tier, geometry signature, descriptor, dimensions
├── dims/{id}.txt             human-readable, prompt-friendly dimension summary (one line)
├── geometry/{id}.step/.stl   (optional, generated by cadquarry export)
├── renders/{id}/{view}[_{pass}].png  (optional) per-view PNG renders (shaded/normal/depth/edge passes)
└── pointclouds/{id}.ply      (optional) sampled point cloud
```

Each `meta.json` record (mirrored in `manifest.jsonl`) carries a
`geometry_signature` and — for families with categorical design intent — a
machine-readable **`descriptor`** that resolves the part's defining attributes
so you can query them without parsing source. For example a threaded part:

```json
"descriptor": {
  "family": "threaded", "standard": "iso", "external": true,
  "length": 60.0, "major_diameter": 16.0, "pitch": 2.0, "designation": "M16x2"
}
```

and a gear:

```json
"descriptor": {
  "family": "gear", "kind": "helical", "module": 2.0, "teeth": 24,
  "pitch_diameter": 48.0, "pressure_angle_deg": 20.0,
  "helix_angle_deg": 18.0, "herringbone": true, "bore_d": 8.0
}
```

Lead screws record a size designation (e.g. `"ACME 1/2"`, `"Tr 20x4"`). The
descriptor is omitted for families whose intent is fully captured by their
parameters alone.

### Human-readable dimensions

Every part also carries a procedurally-generated, plain-English **dimension
summary** — a short, prompt-friendly description of the part's overall size and
each of its features (holes, fillets, bores, pockets, brackets, gears, threads,
…), resolved against the part's default parameters. There is **no AI** in this
step: it is a pure, deterministic walk of the part's IR, so it is reproducible
bit-for-bit alongside everything else. For example:

```
Rectangular body, 55 × 23.5 × 75.4 mm (W×D×H). Four Ø3.2 mm holes at the corners, 38.5 × 16.45 mm spacing. Fillet radius 1.078 mm on vertical edges.
```

The summary appears in three places: a `dimensions` block in `meta.json` (with
both a `lines` list — one phrase per feature — and a joined `text` blob), a
`dims/{id}.txt` sidecar holding the `text` line, and the `dimensions` column of
the manifest and the published JSONL/Parquet (so it flows straight to the
dataset). It pairs naturally with the renders as a text-image signal for
program-recovery and captioning work.

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
`iso_br`, `iso_bl`). Rendering is done on the GPU through a headless EGL OpenGL
context (a small deferred pipeline with a real depth buffer, screen-space
ambient occlusion and soft hemispherical + key lighting). A single geometry pass
fills a view-space position/normal G-buffer that is reused to emit **four output
passes** per view, so the extra passes are nearly free:

- **`shaded`** — the lit, matte-CAD thumbnail, written to `renders/{id}/{view}.png`.
- **`normal`** — view-space surface normals encoded as an RGB normal map.
- **`depth`** — linearized view depth (near = bright) as grayscale.
- **`edge`** — feature edges (creases + silhouette + depth steps) as black lines
  on a transparent background, i.e. a technical-drawing overlay.

The shaded pass keeps the bare `{view}.png` name; the others land as
`{view}_{pass}.png` siblings. `export` emits all four by default; select a
subset with `--render-passes shaded,normal,…`, choose a view set with
`--render-views`, and tune feature-edge sensitivity (lower it for noisy real
scans) with `--edge-crispness 0..1`. Rendering needs the optional `trimesh` +
`moderngl` + `pillow` deps (covered by the `export` extra, i.e.
`uv pip install -e ".[export]"`) plus an EGL-capable GL driver; if they're
missing, `export` prints one warning and skips renders while still writing
STEP/STL.

---

## Annotate an existing corpus

`cadquarry annotate` supplements an already-generated corpus **in place** with
the procedural dimension metadata (the `dimensions` block, the `dims/{id}.txt`
sidecars, and the manifest column) — without re-running any geometry export:

```bash
.venv/bin/cadquarry annotate dataset/                       # uses configs/default.toml
.venv/bin/cadquarry annotate dataset/ --config configs/default.toml
```

Each part is recomposed deterministically from its stored seed + index
(composition is a pure function of seed + config), so **no CAD execution and no
geometry re-export** are needed — it only re-emits the cheap text artifacts
(`parts/*.py`, `params/*.params.json`, `meta/*.meta.json`, `dims/*.txt`) and
rewrites the manifest, leaving STEP/STL/renders/point clouds untouched. This is
the cheap path to add dimensions to a corpus built before the feature existed,
and it also upgrades the corpus to the current generator version in passing.

Annotate with the **same `--config` the corpus was generated with**: each
recomposition is checked against the stored `ir_hash` (ignoring the
generator-version label, which a version bump is expected to change), so a part
whose geometry-relevant IR diverges is left as-is and reported rather than
overwritten with mismatched dimensions — a wrong `--config` can never corrupt
good source.

---

## Build everything in one pass — nested-prefix ladder

The published size ladder (`[[publish.corpus]]` in
[`seeds/v3.toml`](seeds/v3.toml)) is **one base corpus, sliced**. Generation is a
deterministic prefix stream — each attempt is a pure function of `(seed, index)`
and the accept/dedup decision runs in strict attempt order — so the first *N*
accepted parts of the base **are** the `count=N` corpus. Every size tag
(`1k`…`100k`) is therefore a zero-cost `head -N` slice of the same base.

```bash
# Generate + export STEP/STL/renders for the ONE base corpus -> datasets/base/
.venv/bin/cadquarry build

# Generate only (no geometry)
.venv/bin/cadquarry build --no-export
```

`cadquarry build` generates/exports/renders the base **once**; `cadquarry
publish` then packs each size as a length-N slice of it (no per-size rebuild).
This roughly halves the old per-size-seed pipeline (which rebuilt 388k cumulative
parts) — the base is built one time.

### Growing the dataset later (no duplicated work)

The base records a resume cursor (`state.json`: the high-water *attempt* index +
the dedup signatures), so a long-running machine can extend it without redoing
anything:

```bash
# Build a 100k base now…
.venv/bin/cadquarry build --extend-to 100000

# …then grow it to 500k later. Replays the attempt stream from where it stopped,
# appends only the new accepted parts, and exports/renders only the new tail —
# the existing parts, STEP, STL, and renders are reused byte-for-byte.
.venv/bin/cadquarry build --extend-to 500000
```

Extending is **bit-identical** to a from-scratch `count=500000` run (same seed,
same strict-order dedup), and the smaller published sizes stay byte-identical as
the base grows — only new larger sizes become available. `--workers` is shared
by the generation and export phases; `--force` overrides the version/seed guard.

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
├── dims/{id}.txt        human-readable dimension summary
├── geometry/{id}.step   STEP B-rep solids
├── geometry/{id}.stl    compact binary meshes (for the in-browser preview)
├── renders/{id}/{view}.png  8 perspective PNGs (front/top/right/iso + 4 iso corners)
├── DATASET_CARD.md
└── preview.html         self-contained gallery (three.js, lazy-loaded geometry)
```

It is corpus `demo-1k` (seed `1234`) from the active seed list. Regenerate it —
and everything else the GitHub Pages preview needs — with one command:

```bash
.venv/bin/cadquarry build_sample
```

`build_sample` regenerates the part sources + params + meta deterministically
from the `[[corpus]]` entry, then exports STEP B-reps, the compact binary STL
meshes the in-browser preview loads, and the 8 perspective renders the preview's
"Renders" tab shows — straight into `sample/demo-1k/`. GitHub Pages serves the
repo as-is (`index.html` → `sample/demo-1k/preview.html`), so just commit the
result to publish. Pass `--name <corpus>`, `--out <dir>`, or `--formats …` to
rebuild a different committed sample.

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
print(ds[0]["source"])     # full parametric CadQuery program (the canonical artifact)
print(ds[0]["family"])     # e.g. "plate", "revolved", "compound"
print(ds[0]["params"])     # typed parameter schema (JSON)
print(ds[0]["dimensions"]) # plain-English dimension summary (prompt-friendly)

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

Each corpus size is published as **six content configs**, so you fetch only what
you need. The sizes are **nested prefixes** of one base corpus (the `1k` is the
first 1,000 parts of the `2k`, and so on), so a larger size is always a superset
of every smaller one. Swap the `1k` prefix for any size in the ladder:

| Config | Contents | Format |
|---|---|---|
| `<tag>` | CadQuery source + metadata | JSONL |
| `<tag>-renders` | + 8-view render images | Parquet (`render_*` = image) |
| `<tag>-stl` | + binary STL mesh | Parquet (`stl_bytes` = binary) |
| `<tag>-step` | + binary STEP B-rep | Parquet (`step_bytes` = binary) |
| `<tag>-geo` | + renders + STL | Parquet |
| `<tag>-full` | + renders + STL + STEP | Parquet |

The `render_*` columns hold the shaded view by default; when a corpus was
rendered with the extra passes (the `export` default), each view also carries
`render_{view}_normal`, `render_{view}_depth`, and `render_{view}_edge` image
columns alongside it.

Each content config is also published limited to **complexity tiers 0–2**
(inclusive) under a `-t0-2` suffix — e.g. `<tag>-t0-2`, `<tag>-t0-2-renders`,
`<tag>-t0-2-full` — for consumers who want to exclude the most complex (tier-3)
parts. The unlabeled configs include all tiers (0–3).

Available `<tag>` sizes (from [`seeds/v3.toml`](seeds/v3.toml)): `1k`, `2k`,
`5k`, `10k`, `20k`, `50k`, `100k` — extensible to larger sizes by growing the
base (`cadquarry build --extend-to N`). For example,
`load_dataset("jacobjennings/cadquarry", "50k-stl")` or
`load_dataset("jacobjennings/cadquarry", "50k-t0-2")`.

Every part is reproducible bit-for-bit from its seed, so the published data is a
**convenience artifact** — the generator plus `seeds/v3.toml` is the canonical
source.

---

## Publishing the full corpus to Hugging Face

The whole thing is two commands — **build** the corpora (generate + export), then
**publish** (pack + upload). With `HF_TOKEN` set, the defaults are right:

```bash
uv pip install -e ".[mech,export,publish]"   # mech families, renders, uploader
cadquarry build && cadquarry publish
```

Commands can also be **chained** in a single invocation (each runs with its
defaults), so a full refresh of both the GitHub Pages preview and the dataset is:

```bash
cadquarry build_sample build publish
```

- `cadquarry build` — generates **and** exports the single **base** corpus that
  the size ladder slices from, into `datasets/base/` with STEP + STL + renders.
  It uses the seed list matching the current generator version
  (`seeds/v3.toml`) and regenerates if the existing base was built by an older
  generator (this is the slow part; the base + rendering takes a while).
  `--extend-to N` builds or grows the base to N parts, reusing everything already
  on disk; `--force` rebuilds unconditionally.
- `cadquarry publish` — reads the pre-built `datasets/base/` corpus and uploads
  each ladder size as a **length-N slice** of it, as a set of `load_dataset`
  configs of one HF dataset. It generates and renders **nothing**. Defaults to
  **all** sizes; the repo id comes from
  the seed list (resolving to `<your-hf-user>/cadquarry`) or `CADQUARRY_HF_REPO`.
  The code-only variant is packed into a single `corpus.jsonl` with the
  parametric `source` and `params` inlined; the geometry variants
  (renders/STL/STEP) are packed into Snappy-compressed Parquet with typed binary
  columns. If a size hasn't been built yet it stops and tells you what to build.

```bash
# build the base, then publish a subset of sizes to your own repo
cadquarry build && cadquarry publish --sizes 1k 2k 5k --repo-id <user>/cadquarry

# pack locally without uploading (inspect .hf_build/)
cadquarry publish --sizes 1k --dry-run

# read pre-built corpora from a custom location
cadquarry publish --sizes 1k --out /data/cadquarry
```

**Secrets.** Publishing reads your token from the `HF_TOKEN` environment
variable (or a prior `huggingface-cli login`) and **never prints, logs, or
commits it**. Nothing secret is stored in the repo, so it works as-is for anyone
with their own HF account. The ladder is defined in the active `seeds/v*.toml`:
the shared `base_seed` and `mode = "prefix"` under `[publish]`, and the per-size
prefix lengths under `[[publish.corpus]]`.

---

## Configuration

All distribution knobs live in `configs/default.toml`. Key sections:

```toml
[distribution.families]
plate    = 0.18
revolved = 0.14
bracket  = 0.14
compound = 0.11
gear     = 0.06   # requires the `mech` extra
threaded = 0.06   # requires the `mech` extra
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
| Third-party dataset references under [`sample/compare/`](sample/compare/) | original owners' licenses — see [THIRD_PARTY_LICENSES.md](sample/compare/THIRD_PARTY_LICENSES.md) |

CC0 makes the intent unambiguous: do whatever you want with the data, including commercial use, no attribution required.

The **only** exception is [`sample/compare/`](sample/compare/), which compares CadQuarry against other CAD/geometry datasets. The CadQuarry parts shown there are CC0 like the rest; the third-party dataset names and facts are used for identification only, and **no third-party model files are committed** (their licenses don't permit re-hosting). Details and the per-asset table live in [`sample/compare/THIRD_PARTY_LICENSES.md`](sample/compare/THIRD_PARTY_LICENSES.md).
