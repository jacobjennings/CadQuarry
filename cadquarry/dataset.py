"""
Dataset packaging: JSONL manifest, dataset card, corpus statistics.

A generated corpus lives in a directory:

    cadquarry-{version}-seed{seed}/
    ├── manifest.jsonl        one JSON record per part
    ├── parts/{id}.py
    ├── params/{id}.params.json
    ├── meta/{id}.meta.json
    ├── geometry/{id}.step    (optional)
    ├── geometry/{id}.stl     (optional)
    ├── pointclouds/{id}.ply  (optional)
    └── DATASET_CARD.md

Each manifest record carries the same fields as meta.json plus relative
paths to every artifact written for that part.
"""
from __future__ import annotations

import json
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


class DatasetWriter:
    """
    Accumulates part records and writes the dataset directory structure.
    """

    def __init__(self, out_dir: Path, seed: int) -> None:
        self.out_dir = Path(out_dir)
        self.seed = seed
        self._records: list[dict[str, Any]] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add_record(
        self,
        meta: dict[str, Any],
        paths: dict[str, Path],
        geo_signature: dict[str, Any] | None = None,
    ) -> None:
        """
        Register one part.  meta is the .meta.json dict; paths maps artifact
        type ('py', 'params', 'meta', 'step', 'stl', …) to absolute Paths.
        """
        record: dict[str, Any] = dict(meta)
        record["paths"] = {
            k: str(v.relative_to(self.out_dir)) for k, v in paths.items() if v.exists()
        }
        if geo_signature:
            record["geometry_signature"] = geo_signature
        self._records.append(record)

    def finalize(self) -> Path:
        """Write manifest.jsonl and DATASET_CARD.md.  Returns manifest path."""
        manifest_path = self.out_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec) + "\n")

        self._write_dataset_card()
        return manifest_path

    def _write_dataset_card(self) -> None:
        counts = Counter(r.get("family", "unknown") for r in self._records)
        tier_counts = Counter(r.get("tier", -1) for r in self._records)
        total = len(self._records)

        family_lines = "\n".join(
            f"| {fam:<12} | {n:>6} | {n / max(total, 1) * 100:5.1f}% |"
            for fam, n in sorted(counts.items(), key=lambda x: -x[1])
        )
        tier_lines = "\n".join(
            f"| Tier {t} | {n:>6} | {n / max(total, 1) * 100:5.1f}% |"
            for t, n in sorted(tier_counts.items())
        )

        card = textwrap.dedent(f"""\
            # CadQuarry Dataset Card

            | Field            | Value |
            |------------------|-------|
            | Generator        | CadQuarry v{__version__} |
            | Seed             | {self.seed} |
            | Total parts      | {total} |
            | Generated        | {datetime.now(timezone.utc).strftime("%Y-%m-%d")} |
            | Code license     | Apache-2.0 |
            | Data license     | CC0-1.0 |

            ## Part families

            | Family       | Count  | Share |
            |--------------|--------|-------|
            {family_lines}

            ## Complexity tiers

            | Tier   | Count  | Share |
            |--------|--------|-------|
            {tier_lines}

            ## Reproducibility

            Re-generate this exact corpus:

            ```bash
            cadquarry generate --seed {self.seed} --count {total} \\
                --out <output_dir> --config configs/default.toml
            ```

            ## License

            Generated data: **CC0-1.0** (public domain dedication).
            Generator source: **Apache-2.0**.
            See DATA_LICENSE and LICENSE in the repository root.
        """)
        (self.out_dir / "DATASET_CARD.md").write_text(card, encoding="utf-8")


def load_manifest(dataset_dir: Path) -> list[dict[str, Any]]:
    """Load all records from manifest.jsonl."""
    path = dataset_dir / "manifest.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _passes_tier(rec: dict[str, Any], tier_max: int | None) -> bool:
    """Whether a record is within the requested inclusive complexity tier cap."""
    if tier_max is None:
        return True
    tier = rec.get("tier")
    return isinstance(tier, int) and tier <= tier_max


def pack_corpus_jsonl(
    dataset_dir: Path,
    out_path: Path,
    include_source: bool = True,
    tier_max: int | None = None,
) -> int:
    """
    Flatten a corpus directory into a single self-contained JSONL file.

    Each line is one part with the parametric source and parameter schema
    inlined, so the file is directly browsable in the HuggingFace dataset
    viewer and loadable via ``datasets.load_dataset(..., data_files=...)``
    without any sidecar files.  Returns the number of records written.

    Columns: part_id, family, tier, seed, symmetry, op_count, ir_hash,
    generator_version, license, geometry_signature (struct), params (struct),
    and source (the full .py text, when ``include_source``).

    ``tier_max`` (inclusive) drops any record whose complexity ``tier`` exceeds
    it; ``None`` keeps every tier.
    """
    records = load_manifest(dataset_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for rec in records:
            pid = rec.get("part_id")
            if not pid:
                continue
            if not _passes_tier(rec, tier_max):
                continue
            row: dict[str, Any] = {
                "part_id": pid,
                "family": rec.get("family"),
                "tier": rec.get("tier"),
                "seed": rec.get("seed"),
                "symmetry": rec.get("symmetry"),
                "op_count": rec.get("op_count"),
                "ir_hash": rec.get("ir_hash"),
                "generator_version": rec.get("generator_version") or rec.get("cadquarry_version"),
                "license": rec.get("license", "CC0-1.0"),
                "geometry_signature": rec.get("geometry_signature"),
            }
            params_rel = rec.get("paths", {}).get("params")
            if params_rel:
                params_path = dataset_dir / params_rel
                if params_path.exists():
                    try:
                        row["params"] = json.loads(params_path.read_text(encoding="utf-8")).get("params")
                    except (OSError, json.JSONDecodeError):
                        row["params"] = None
            if include_source:
                py_rel = rec.get("paths", {}).get("py")
                if py_rel:
                    py_path = dataset_dir / py_rel
                    if py_path.exists():
                        row["source"] = py_path.read_text(encoding="utf-8")
            out.write(json.dumps(row) + "\n")
            n += 1
    return n


def corpus_stats(dataset_dir: Path) -> dict[str, Any]:
    """Return summary statistics for an existing corpus."""
    records = load_manifest(dataset_dir)
    if not records:
        return {"total": 0, "families": {}, "tiers": {}}
    families = Counter(r.get("family", "unknown") for r in records)
    tiers = Counter(r.get("tier", -1) for r in records)
    return {
        "total": len(records),
        "families": dict(families),
        "tiers": dict(tiers),
    }


# View names produced by export.export_renders / STANDARD_VIEWS (must stay in sync).
RENDER_VIEWS: list[str] = [
    "front", "top", "right", "iso",
    "iso_fr", "iso_fl", "iso_br", "iso_bl",
]


def pack_corpus_parquet(
    dataset_dir: Path,
    out_path: Path,
    include_source: bool = True,
    include_renders: bool = False,
    include_stl: bool = False,
    include_step: bool = False,
    tier_max: int | None = None,
    batch_size: int = 512,
) -> int:
    """
    Pack a corpus into a single Parquet file.

    ``tier_max`` (inclusive) drops any record whose complexity ``tier`` exceeds
    it; ``None`` keeps every tier.

    Binary geometry columns (render_*, stl_bytes, step_bytes) are stored as raw
    bytes so that HuggingFace ``datasets`` can decode them with the correct
    feature types declared in the dataset card (``dtype: image`` for renders,
    ``dtype: binary`` for mesh bytes).

    Renders are read from ``renders/{part_id}/{view}.png``.
    STL / STEP are read from ``geometry/{part_id}.stl`` / ``.step``.
    Missing files produce null values for that column.

    ``batch_size`` rows are buffered and flushed as one Parquet row group, so
    peak memory stays bounded regardless of corpus size (essential for large
    geometry variants whose bytes would otherwise OOM the process).

    Returns the number of rows written.
    Raises ImportError if pyarrow is not installed.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for Parquet export. "
            "Install it with: pip install pyarrow  "
            "(or: pip install -e '.[publish]')"
        ) from exc

    records = load_manifest(dataset_dir)
    renders_base = dataset_dir / "renders"
    geo_base = dataset_dir / "geometry"

    base_cols: list[tuple[str, Any]] = [
        ("part_id", pa.string()),
        ("family", pa.string()),
        ("tier", pa.int32()),
        ("seed", pa.int64()),
        ("symmetry", pa.string()),
        ("op_count", pa.int32()),
        ("ir_hash", pa.string()),
        ("generator_version", pa.string()),
        ("license", pa.string()),
        ("geometry_signature", pa.string()),
    ]
    if include_source:
        base_cols += [
            ("source", pa.string()),
            ("params", pa.string()),
        ]
    if include_renders:
        for v in RENDER_VIEWS:
            base_cols.append((f"render_{v}", pa.binary()))
    if include_stl:
        base_cols.append(("stl_bytes", pa.large_binary()))
    if include_step:
        base_cols.append(("step_bytes", pa.large_binary()))

    col_names = [c[0] for c in base_cols]
    col_types = {c[0]: c[1] for c in base_cols}
    schema = pa.schema([(cn, col_types[cn]) for cn in col_names])
    buffers: dict[str, list] = {n: [] for n in col_names}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream in row-group batches via ParquetWriter. Materializing every row of
    # every column up front holds the whole corpus (renders/STL/STEP bytes) in
    # RAM twice — once as Python bytes, once when copied into Arrow — which OOMs
    # on large geometry variants (a 200k -geo/-full run needs ~190+ GB). Writing
    # one batch at a time bounds peak memory to ~batch_size rows and also keeps
    # each binary column chunk well under Arrow's 2 GB per-chunk offset limit.
    writer = None
    batch_n = 0

    def _flush() -> None:
        nonlocal writer, batch_n
        if batch_n == 0:
            return
        arrays = [pa.array(buffers[cn], type=col_types[cn]) for cn in col_names]
        table = pa.table(dict(zip(col_names, arrays)), schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), schema, compression="snappy")
        writer.write_table(table)
        for cn in col_names:
            buffers[cn].clear()
        batch_n = 0

    n = 0
    for rec in records:
        pid = rec.get("part_id")
        if not pid:
            continue
        if not _passes_tier(rec, tier_max):
            continue

        row: dict[str, Any] = {
            "part_id": pid,
            "family": rec.get("family"),
            "tier": rec.get("tier"),
            "seed": rec.get("seed"),
            "symmetry": rec.get("symmetry"),
            "op_count": rec.get("op_count"),
            "ir_hash": rec.get("ir_hash"),
            "generator_version": rec.get("generator_version") or rec.get("cadquarry_version"),
            "license": rec.get("license", "CC0-1.0"),
            "geometry_signature": json.dumps(rec.get("geometry_signature")),
        }

        if include_source:
            py_rel = rec.get("paths", {}).get("py")
            row["source"] = None
            if py_rel:
                py_p = dataset_dir / py_rel
                if py_p.exists():
                    row["source"] = py_p.read_text(encoding="utf-8")
            params_rel = rec.get("paths", {}).get("params")
            row["params"] = None
            if params_rel:
                p_path = dataset_dir / params_rel
                if p_path.exists():
                    try:
                        raw = json.loads(p_path.read_text(encoding="utf-8"))
                        row["params"] = json.dumps(raw.get("params"))
                    except (OSError, json.JSONDecodeError):
                        pass

        if include_renders:
            rdir = renders_base / pid
            for v in RENDER_VIEWS:
                png = rdir / f"{v}.png"
                row[f"render_{v}"] = png.read_bytes() if png.exists() else None

        if include_stl:
            stl_p = geo_base / f"{pid}.stl"
            row["stl_bytes"] = stl_p.read_bytes() if stl_p.exists() else None

        if include_step:
            step_p = geo_base / f"{pid}.step"
            row["step_bytes"] = step_p.read_bytes() if step_p.exists() else None

        for cn in col_names:
            buffers[cn].append(row.get(cn))
        n += 1
        batch_n += 1
        if batch_n >= batch_size:
            _flush()

    _flush()

    if writer is None:
        # No rows matched (e.g. tier filter excluded everything): write nothing,
        # matching the previous behaviour.
        return 0
    writer.close()
    return n
