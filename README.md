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
- Fallback proxy: [via htmlpreview.github.io](https://htmlpreview.github.io/?https://raw.githubusercontent.com/jacobjennings/CadQuarry/main/sample/demo-1k/preview.html).
- Or locally: `python -m http.server` inside `sample/demo-1k/` and open `preview.html`.

GitHub Pages is the primary link: it serves `preview.html` as real HTML from the
same origin as its data (`manifest.jsonl`, `stl/`), so the page and its lazy-loaded
geometry just work. The site is built by an Actions workflow
([`.github/workflows/pages.yml`](.github/workflows/pages.yml)) on every push to `main`.

> Note: opening `preview.html` from a bare `raw.githubusercontent.com` URL shows
> source, not a rendered page — GitHub serves raw files as `text/plain`. Use the
> Pages link (or the htmlpreview fallback), which serve it as real HTML.

---

## What it does

- Generates large batches of unique, executable [CadQuery](https://cadquery.readthedocs.io) programs across a broad operation vocabulary (plates, shafts, blocks, enclosures, flanged hubs, ribbed structures, profiled extrusions).
- Guarantees validity by execution — every accepted part actually builds to a non-empty solid.
- Every program is **parametric by construction**: it declares a machine-readable `PARAMS` schema (typed, range-bounded, UI-labeled) and is a pure function `build(p)` of those parameters.
- Ships a live **customizer**: open any part, move sliders, the 3D view updates.
- Ships a **gallery**: browse a generated corpus, filter by family/tier, click to open in the customizer.
- Fully **reproducible**: seed + generator version → identical corpus, bit-for-bit.
- **No restrictions on use**: code is Apache-2.0, generated data is CC0-1.0.

---

## Quick start

```bash
# Install (requires Python 3.11+)
pip install -e ".[dev]"

# Install CadQuery (see https://cadquery.readthedocs.io/en/latest/installation.html)
# Easiest via conda:
#   conda install -c conda-forge -c cadquery cadquery

# Generate 100 parts
cadquarry generate --count 100 --seed 42 --out dataset/

# Open the live customizer for a single part
cadquarry serve --part examples/plate_with_holes.py

# Browse a generated corpus
cadquarry serve --dataset dataset/

# Execute a part with parameter overrides
cadquarry run examples/plate_with_holes.py --set plate_w=80 --set thickness=8

# Export to STL
cadquarry run examples/plate_with_holes.py --export stl --out plate.stl

# Print the parameter schema for a part
cadquarry info examples/plate_with_holes.py

# Verify all parts in a corpus re-execute correctly
cadquarry verify dataset/
```

---

## Part families

| Family | Description | Weight |
|---|---|---|
| `plate` | Flat rectangular plates with holes, fillets, pockets | 22% |
| `bracket` | Angle brackets — L (single leg), C/channel (two legs), Z (cranked offset) — with per-leg holes and optional gussets | 18% |
| `revolved` | Shafts, bushings, washers — solid of revolution | 18% |
| `block` | Prismatic blocks/housings with pockets and bosses | 15% |
| `flanged` | Revolved stub + polar bolt-circle pattern | 8% |
| `ribbed` | Base plate + patterned thin ribs | 7% |
| `enclosure` | Shelled box (hollow, open-top) | 7% |
| `profiled` | Long constant cross-section (rod, tube, polygon bar) | 5% |

Family weights, tier distributions, and per-family dimension ranges are all tunable via `configs/default.toml`.

---

## Complexity tiers

| Tier | Description | Default weight |
|---|---|---|
| 0 | Single feature | 25% |
| 1 | 2–3 ops, holes, basic primitives | 40% |
| 2 | Fillets, chamfers, counterbore/countersink | 25% |
| 3 | Pockets, bosses, ribs, deep trees | 10% |

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
├── manifest.jsonl          one JSON record per part (id, family, tier, signature, paths)
├── DATASET_CARD.md         scale, distribution, generator version, license
├── parts/{id}.py           parametric CadQuery source
├── params/{id}.params.json parameter schema sidecar
├── meta/{id}.meta.json     provenance: seed, family, tier, geometry signature
└── geometry/{id}.step/.stl (optional, generated by cadquarry export)
```

---

## Reproducibility

```bash
# Re-generate an exact corpus from its seed
cadquarry generate --seed 42 --count 5000 --config configs/default.toml --out dataset-repro/

# Verify signatures match
cadquarry verify dataset-repro/
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
├── stl/{id}.stl         compact binary meshes (for the in-browser preview)
├── DATASET_CARD.md
└── preview.html         self-contained gallery (three.js, lazy-loaded geometry)
```

It is corpus `demo-1k` from `seeds/v1.toml` (seed `1234`). Regenerate it:

```bash
cadquarry generate --seed 1234 --count 1000 --out sample/demo-1k/
python scripts/export_stl.py sample/demo-1k       # refresh preview meshes
```

---

## Publishing the full corpus to Hugging Face

[`scripts/publish_to_hf.py`](scripts/publish_to_hf.py) builds the reproducible
size ladder (1k, 2k, 5k, 10k, 20k, 50k, 100k, 200k, 500k) and uploads each as a
`load_dataset` config of one HF dataset. Each size is packed into a single
`corpus.jsonl` with the parametric `source` and `params` inlined.

```bash
pip install -e ".[publish]"          # adds huggingface_hub

# Build + upload the small configs to your own repo
python scripts/publish_to_hf.py --sizes 1k 2k 5k --repo-id <user>/cadquarry

# Build everything (large; 500k generation takes a while)
python scripts/publish_to_hf.py --all

# Generate + pack locally without uploading (inspect .hf_build/)
python scripts/publish_to_hf.py --sizes 1k --dry-run
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
plate    = 0.22
revolved = 0.18
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
pip install -e ".[dev]"
pytest                  # runs tests that don't require cadquery
pytest -k "not exec"   # same (explicit filter)
```

Tests in `tests/` that don't touch the executor run without CadQuery. Execution tests require CadQuery.

---

## License

| What | License |
|---|---|
| Generator source code (`cadquarry/`) | [Apache-2.0](LICENSE) |
| Generated data (`.py`, `.step`, `.stl`, `.params.json`, `.meta.json`) | [CC0-1.0](DATA_LICENSE) |

Generated output with no human creative authorship is arguably uncopyrightable in many jurisdictions anyway. CC0 makes the intent unambiguous: do whatever you want with the data, including commercial use, no attribution required.
