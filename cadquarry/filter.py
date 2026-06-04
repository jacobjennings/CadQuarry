"""
Validity checking and geometry-based deduplication.

Geometry signature
==================
A part's signature is a tuple of quantized analytic properties computed
from the B-rep.  Same tuple ⇒ treated as a duplicate.

  (quantized_volume, quantized_surface_area,
   quantized_sorted_principal_moments,
   n_faces, n_edges, n_vertices)

We deliberately exclude point-sampled or mesh-based properties so the
signature is deterministic regardless of sampling seed.

Quantization uses relative rounding to N significant figures so the
same geometry at different scales yields the same signature.

Principal moments of inertia (the three eigenvalues of the inertia tensor
about the centroid) are the strong component: they are invariant to
arbitrary rotation and, taken about the centroid, to translation.  They are
sorted before quantization so orientation never matters.  If the executor
could not compute them (older OCP, degenerate solid), the moment component
is an empty tuple and the signature falls back to volume/area/topology.

The definition is intentionally versioned with the generator: changing it
would alter which parts count as duplicates, breaking seed reproducibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .execute import ExecuteResult

# Signature version — bump this if the definition changes.
# v2: added sorted principal moments of inertia.
SIGNATURE_VERSION = 2

# Relative tolerance: round to this many significant figures.
_SIG_FIGS = 3


def _round_sig(value: float, sig: int = _SIG_FIGS) -> float:
    """Round to sig significant figures."""
    if value == 0:
        return 0.0
    magnitude = math.floor(math.log10(abs(value)))
    factor = 10 ** (sig - 1 - magnitude)
    return round(value * factor) / factor


@dataclass(frozen=True)
class GeometrySignature:
    version: int
    volume: float                       # quantized
    surface_area: float                 # quantized
    principal_moments: tuple[float, ...]  # quantized, sorted ascending
    n_faces: int
    n_edges: int
    n_vertices: int

    def as_tuple(self) -> tuple:
        return (
            self.version,
            self.volume,
            self.surface_area,
            self.principal_moments,
            self.n_faces,
            self.n_edges,
            self.n_vertices,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "principal_moments": list(self.principal_moments),
            "n_faces": self.n_faces,
            "n_edges": self.n_edges,
            "n_vertices": self.n_vertices,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GeometrySignature":
        return cls(
            version=d.get("version", SIGNATURE_VERSION),
            volume=d["volume"],
            surface_area=d["surface_area"],
            principal_moments=tuple(d.get("principal_moments", []) or []),
            n_faces=d["n_faces"],
            n_edges=d["n_edges"],
            n_vertices=d["n_vertices"],
        )


def compute_signature(result: ExecuteResult) -> GeometrySignature:
    moments = tuple(
        _round_sig(m) for m in sorted(result.principal_moments)
    )
    return GeometrySignature(
        version=SIGNATURE_VERSION,
        volume=_round_sig(result.volume),
        surface_area=_round_sig(result.surface_area),
        principal_moments=moments,
        n_faces=result.n_faces,
        n_edges=result.n_edges,
        n_vertices=result.n_vertices,
    )


# ---------------------------------------------------------------------------
# Validity predicate
# ---------------------------------------------------------------------------

@dataclass
class QualityConfig:
    min_volume_mm3: float = 1.0
    max_bbox_ratio: float = 20.0       # max(dims) / min(dims)
    enable_wall_check: bool = False    # opt-in, requires thicker analysis
    min_wall_thickness_mm: float = 0.5


def is_valid(result: ExecuteResult, cfg: QualityConfig | None = None) -> tuple[bool, str]:
    """
    Return (valid, reason).  reason is empty string when valid=True.
    """
    if cfg is None:
        cfg = QualityConfig()

    if not result.success:
        return False, result.error or "execution failed"

    if result.volume < cfg.min_volume_mm3:
        return False, f"volume {result.volume:.4f} < {cfg.min_volume_mm3}"

    dims = result.bbox_dims
    if any(d <= 0 for d in dims):
        return False, f"degenerate bounding box: {dims}"

    max_dim = max(dims)
    min_dim = min(dims)
    ratio = max_dim / min_dim if min_dim > 0 else float("inf")
    if ratio > cfg.max_bbox_ratio:
        return False, f"extreme bbox ratio {ratio:.1f} > {cfg.max_bbox_ratio}"

    return True, ""


# ---------------------------------------------------------------------------
# Deduplication store
# ---------------------------------------------------------------------------


class DedupStore:
    """
    Keeps a set of geometry signatures seen so far.
    Also maintains a fast IR-hash pre-filter for exact duplicates.
    """

    def __init__(self) -> None:
        self._ir_hashes: set[str] = set()
        self._geo_sigs: set[tuple] = set()

    def is_duplicate(
        self,
        ir_hash: str,
        geo_sig: GeometrySignature | None = None,
    ) -> bool:
        """Return True if this part should be dropped as a duplicate."""
        if ir_hash in self._ir_hashes:
            return True
        if geo_sig is not None and geo_sig.as_tuple() in self._geo_sigs:
            return True
        return False

    def register(self, ir_hash: str, geo_sig: GeometrySignature | None = None) -> None:
        self._ir_hashes.add(ir_hash)
        if geo_sig is not None:
            self._geo_sigs.add(geo_sig.as_tuple())

    def __len__(self) -> int:
        return len(self._geo_sigs)
