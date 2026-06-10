# Third-party licenses — `sample/compare/`

> **This directory is the one place in CadQuarry that references third-party
> datasets.** Everything else CadQuarry ships is original work: generator code
> under **Apache-2.0** and generated data under **CC0-1.0**. The dataset names,
> facts, and the third-party model files under `models/<dataset>/` are **NOT**
> CC0 and are **NOT** covered by CadQuarry's licenses. Each retains the license
> of its origin dataset / original author.

## What is hosted here, and why

The comparison page ([compare.html](compare.html)) renders CadQuarry parts
(CC0, ours to share) beside a sample of ~100 models from other
generated-geometry datasets. To make the comparison concrete and visual, a
**small sample** of third-party meshes is committed under `models/<dataset>/`
**for demonstration / comparison only**. This is a deliberate, eyes-open
choice — these files are not ours to relicense, and some upstream terms are
restrictive (see each row below). If you fork or redistribute this repository,
you are responsible for the third-party content in `models/`.

| Dataset | Hosted here? | Why |
| ------- | ------------ | --- |
| ABC Dataset | ✅ ~100 OBJ→STL | Official OBJ chunk 0 is openly downloadable from the NYU archive |
| Fusion 360 Gallery | ✅ ~100 STL | Public Hugging Face mesh mirror |
| Thingi10K | ✅ ~100 STL | Official Hugging Face dataset exposes per-model meshes |
| DeepCAD | ❌ link-out | Distributed as CAD command-sequences, not meshes; its geometry is a curated subset of ABC |
| ShapeNet | ❌ link-out | Surface meshes are gated behind a signed agreement; ungated mirrors are point clouds / SDFs / renders, not meshes |

## Per-source provenance, mirror, and license

| Dataset | Sample source (mirror) | Upstream license of distribution | Reference |
| ------- | ---------------------- | -------------------------------- | --------- |
| **ABC Dataset** | `archive.nyu.edu` — `abc_0000_obj_v00.7z` (OBJ chunk 0) | No single license: each model retains its original **Onshape author** license | Koch et al., *ABC: A Big CAD Model Dataset*, CVPR 2019 — https://deep-geometry.github.io/abc-dataset/ |
| **Fusion 360 Gallery** | HF `maksimko123/fusion360_test_mesh` | **Autodesk research-only** license (upstream); mirror is public | Willis et al., Autodesk AI Lab — https://github.com/AutodeskAILab/Fusion360GalleryDataset |
| **Thingi10K** | HF `Thingi10K/Thingi10K` (`raw_meshes/`) | **Per-model** Thingiverse licenses (CC0 / BY / BY-SA / BY-NC / BY-ND / all-rights-reserved) | Zhou & Jacobson, *Thingi10K*, 2016 — https://ten-thousand-models.appspot.com/ |
| **DeepCAD** | — (not hosted) | MIT (code); geometry derived from ABC | Wu et al., *DeepCAD*, ICCV 2021 — https://github.com/ChrisWu1997/DeepCAD |
| **ShapeNet** | — (not hosted) | ShapeNet Terms of Use — registration required, **no redistribution** | Chang et al., *ShapeNet*, 2015 — https://shapenet.org/ |

### Tracing an individual model's license

- **Thingi10K**: each committed file is named by its Thingiverse id — find its
  exact license at `https://www.thingiverse.com/thing:<id>`. Some are CC0, some
  forbid derivatives or commercial use; check before any reuse.
- **ABC**: each file is named by its ABC model id; licenses are the original
  Onshape document authors' and are not uniform.
- **Fusion 360 Gallery**: covered by the upstream Autodesk research license.

## Regenerating / extending the samples

The committed samples are produced by
[`scripts/fetch_compare_samples.py`](../../scripts/fetch_compare_samples.py):

```bash
python scripts/fetch_compare_samples.py            # all configured sources
python scripts/fetch_compare_samples.py thingi10k  # one source
```

To add a gated/mesh-less dataset you have obtained yourself (e.g. ShapeNet under
your own account, or DeepCAD reconstructed to meshes):

1. Place the meshes (`.stl`) under `models/<dataset>/` and write a matching
   `models/<dataset>/index.json` (`[{"file": "...", "id": "..."}, …]`).
2. Flip that dataset's `status` to `"hosted"` and set `samples_dir` in
   [manifest.json](manifest.json) — the section becomes a live 3D grid.
3. Record its source and license in the tables above.

CadQuarry is not affiliated with, endorsed by, or sponsored by any of the above
projects or their authors. Dataset names and trademarks belong to their
respective owners and are used here for identification and comparison only.
