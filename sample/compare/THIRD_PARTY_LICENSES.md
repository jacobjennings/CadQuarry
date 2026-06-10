# Third-party licenses — `sample/compare/`

> **This directory is the one place in CadQuarry that references third-party
> datasets.** Everything else CadQuarry ships is original work: generator code
> under **Apache-2.0** and generated data under **CC0-1.0**. The dataset names,
> facts, and any third-party model files referenced below are **NOT** CC0 and
> are **NOT** covered by CadQuarry's licenses. Each retains the license of its
> origin.

## What is and isn't redistributed here

The comparison page ([compare.html](compare.html)) shows CadQuarry parts as live,
interactive 3D (those are CC0 and ours to share) alongside cards for other
generated-geometry datasets. **No third-party model files are committed to this
repository.** For every other dataset the page links out to the dataset's own
distribution. This is deliberate:

- **ShapeNet** and the **Fusion 360 Gallery** are distributed under terms that
  *prohibit redistribution* (ShapeNet requires a signed agreement; Fusion 360
  Gallery is an Autodesk research-only license). Re-hosting even a single model
  would violate those terms — this is stronger than a "non-commercial"
  restriction and is not something CadQuarry can waive.
- **ABC**, **DeepCAD**, and **Thingi10K** have *per-model, inconsistent*
  licensing (Onshape author licenses; or a mix of CC0 / BY / BY-SA / BY-NC /
  BY-ND / all-rights-reserved on Thingiverse). "The dataset is research-use" does
  not establish that any single model may be re-hosted, and some explicitly
  forbid derivatives or reuse.

For a demonstration/comparison, the defensible path is to **link to the source**
(and, if needed, render a single thumbnail under fair-use/commentary) rather than
re-host the asset. That is what this page does.

## Adding a third-party model (only if you have confirmed the right)

If you obtain a specific model whose individual license **clearly permits
redistribution** (e.g. a Thingiverse model released under CC0 or CC BY):

1. Place the mesh under `sample/compare/models/<dataset>/<file>.stl`.
2. Set that dataset's `mesh` field in [manifest.json](manifest.json) to the
   relative path — the card becomes a live 3D viewer automatically.
3. Add a row to the table below recording the **exact** model, its author, its
   license, and the source URL. Attribution is mandatory for CC BY / BY-SA.

| Dataset | File | Original author | License | Source URL |
| ------- | ---- | --------------- | ------- | ---------- |
| _(none committed yet)_ | — | — | — | — |

## Dataset references (for attribution and provenance)

| Dataset | License of distribution | Reference |
| ------- | ----------------------- | --------- |
| ABC Dataset | Per-model (Onshape author licenses) | Koch et al., *ABC: A Big CAD Model Dataset for Geometric Deep Learning*, CVPR 2019 — https://deep-geometry.github.io/abc-dataset/ |
| DeepCAD | MIT (code); data derived from ABC | Wu et al., *DeepCAD: A Deep Generative Network for Computer-Aided Design Models*, ICCV 2021 — https://github.com/ChrisWu1997/DeepCAD |
| Fusion 360 Gallery | Autodesk research-only license | Willis et al., Autodesk AI Lab — https://github.com/AutodeskAILab/Fusion360GalleryDataset |
| Thingi10K | Per-model (Thingiverse licenses) | Zhou & Jacobson, *Thingi10K: A Dataset of 10,000 3D-Printing Models*, 2016 — https://ten-thousand-models.appspot.com/ |
| ShapeNet | ShapeNet Terms of Use (no redistribution) | Chang et al., *ShapeNet: An Information-Rich 3D Model Repository*, 2015 — https://shapenet.org/ |

CadQuarry is not affiliated with, endorsed by, or sponsored by any of the above
projects or their authors. Dataset names and trademarks belong to their
respective owners and are used here for identification and comparison only.
