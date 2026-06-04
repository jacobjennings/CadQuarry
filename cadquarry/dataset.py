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


def pack_corpus_jsonl(
    dataset_dir: Path,
    out_path: Path,
    include_source: bool = True,
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
    """
    records = load_manifest(dataset_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for rec in records:
            pid = rec.get("part_id")
            if not pid:
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
