"""
2D profile samplers.

Each sampler returns a (Profile, params_update) pair.  The params_update dict
contains ParamSpec entries that should be merged into the part's PARAMS schema.
"""
from __future__ import annotations

import math
from random import Random
from typing import Any

from .ir import (
    CircleProfile,
    ParamRef,
    ParamSpec,
    PolygonProfile,
    Profile,
    RectProfile,
    RoundedRectProfile,
    ScaledExpr,
    SketchSeg,
    SketchedProfile,
    SlotProfile,
    lit,
    ref,
    scaled,
)

# Standard metric fastener clearance diameters (M3-M12, close-fit).
FASTENER_CLEARANCE_MM = [3.2, 4.3, 5.3, 6.4, 8.4, 10.5, 13.0]

# Stock plate thicknesses (mm) — common in sheet-metal and machined stock.
STOCK_THICKNESSES_MM = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]

# ---------------------------------------------------------------------------
# Gear + thread standards (used by the `mech` families: gear / threaded)
# ---------------------------------------------------------------------------

# ISO 54 preferred module series (Series I), mm — the standard tooth-size
# increments for metric involute gears.  Restricted to the range that suits
# CadQuarry's part dimensions.
ISO_GEAR_MODULES = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

# ISO 261/262 metric screw threads: major diameter (mm) -> (coarse pitch,
# [fine pitches]).  Used to emit valid IsoThread(major_diameter, pitch) pairs.
METRIC_THREAD_PITCHES: dict[float, tuple[float, list[float]]] = {
    3.0:  (0.5,  [0.35]),
    4.0:  (0.7,  [0.5]),
    5.0:  (0.8,  [0.5]),
    6.0:  (1.0,  [0.75]),
    8.0:  (1.25, [1.0, 0.75]),
    10.0: (1.5,  [1.25, 1.0]),
    12.0: (1.75, [1.5, 1.25]),
    16.0: (2.0,  [1.5]),
    20.0: (2.5,  [1.5]),
    24.0: (3.0,  [2.0]),
    30.0: (3.5,  [2.0]),
    36.0: (4.0,  [3.0]),
}

# bd_warehouse ACME lead-screw size designations (curated to the small/medium
# end of AcmeThread.sizes() so parts stay in CadQuarry's dimension range).
ACME_THREAD_SIZES = [
    "1/4", "5/16", "3/8", "1/2", "5/8", "3/4", "7/8", "1", "1 1/4", "1 1/2",
]

# Thread pitch (mm) for each ACME size, as reported by bd_warehouse's AcmeThread
# (25.4 / standard TPI).  Used to keep a thread's length off integer multiples
# of its pitch — bd_warehouse degenerates the partial end-loop when
# length/pitch is ~integer (a Standard_ConstructionError).
ACME_PITCH_BY_SIZE = {
    "1/4": 1.5875, "5/16": 1.81429, "3/8": 2.11667, "1/2": 2.54,
    "5/8": 3.175, "3/4": 4.23333, "7/8": 4.23333, "1": 5.08,
    "1 1/4": 5.08, "1 1/2": 6.35,
}

# bd_warehouse metric-trapezoidal lead-screw designations ("DxP"), curated
# subset of MetricTrapezoidalThread.sizes().
METRIC_TRAP_THREAD_SIZES = [
    "8x1.5", "10x2", "12x3", "14x3", "16x4", "18x4",
    "20x4", "24x5", "28x5", "30x6", "36x6", "40x7",
]


def snap_to_module(rng: Random, lo: float, hi: float) -> float:
    """
    Pick an ISO 54 preferred gear module within [lo, hi], falling back to a
    plain uniform sample if none of the standard modules fit.  Mirrors
    ``snap_to_stock_thickness``.
    """
    choices = [m for m in ISO_GEAR_MODULES if lo <= m <= hi]
    if choices:
        return rng.choice(choices)
    return round(rng.uniform(lo, hi), 2)


def _nice(rng: Random, lo: float, hi: float, snap_prob: float = 0.5) -> float:
    """Sample a dimension, snapping to a round number with probability snap_prob."""
    v = rng.uniform(lo, hi)
    if rng.random() < snap_prob:
        v = round(v / 5) * 5  # snap to nearest 5
        v = max(lo, min(hi, v))
    return round(v, 2)


def _nice_int(rng: Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def snap_to_stock_thickness(rng: Random, lo: float, hi: float) -> float:
    choices = [t for t in STOCK_THICKNESSES_MM if lo <= t <= hi]
    if choices:
        return rng.choice(choices)
    return _nice(rng, lo, hi)


def snap_fastener_diameter(rng: Random, lo: float, hi: float) -> float:
    choices = [d for d in FASTENER_CLEARANCE_MM if lo <= d <= hi]
    if choices:
        return rng.choice(choices)
    return round(rng.uniform(lo, hi), 1)


# ---------------------------------------------------------------------------
# Rect profile
# ---------------------------------------------------------------------------

def sample_rect_profile(
    rng: Random,
    w_lo: float = 20.0,
    w_hi: float = 120.0,
    aspect_lo: float = 1.0,
    aspect_hi: float = 3.0,
    w_param: str = "width",
    d_param: str = "depth",
    snap_prob: float = 0.5,
) -> tuple[RectProfile, dict[str, ParamSpec]]:
    w_def = _nice(rng, w_lo, w_hi, snap_prob)
    aspect = rng.uniform(aspect_lo, aspect_hi)
    d_def = round(w_def / aspect, 2)
    d_def = max(w_lo * 0.4, min(w_hi, d_def))

    # Ranges: allow ±50% of default within the declared bounds
    w_min = max(w_lo, round(w_def * 0.5, 1))
    w_max = min(w_hi, round(w_def * 1.8, 1))
    d_min = max(w_lo * 0.4, round(d_def * 0.5, 1))
    d_max = min(w_hi, round(d_def * 1.8, 1))

    params: dict[str, ParamSpec] = {
        w_param: ParamSpec(
            type="float", default=w_def, min=w_min, max=w_max, step=1.0,
            group="Body", label="Width",
        ),
        d_param: ParamSpec(
            type="float", default=d_def, min=d_min, max=d_max, step=1.0,
            group="Body", label="Depth",
        ),
    }
    profile = RectProfile(width=ref(w_param), depth=ref(d_param))
    return profile, params


# ---------------------------------------------------------------------------
# Rounded rect profile
# ---------------------------------------------------------------------------

def sample_rounded_rect_profile(
    rng: Random,
    w_lo: float = 20.0,
    w_hi: float = 120.0,
    aspect_lo: float = 1.0,
    aspect_hi: float = 3.0,
    w_param: str = "width",
    d_param: str = "depth",
) -> tuple[RoundedRectProfile, dict[str, ParamSpec]]:
    rect, params = sample_rect_profile(rng, w_lo, w_hi, aspect_lo, aspect_hi, w_param, d_param)
    profile = RoundedRectProfile(width=ref(w_param), depth=ref(d_param))
    return profile, params


# ---------------------------------------------------------------------------
# Circle profile
# ---------------------------------------------------------------------------

def sample_circle_profile(
    rng: Random,
    d_lo: float = 10.0,
    d_hi: float = 80.0,
    d_param: str = "outer_d",
    snap_prob: float = 0.4,
) -> tuple[CircleProfile, dict[str, ParamSpec]]:
    d_def = _nice(rng, d_lo, d_hi, snap_prob)
    d_min = max(d_lo, round(d_def * 0.5, 1))
    d_max = min(d_hi, round(d_def * 1.8, 1))
    params: dict[str, ParamSpec] = {
        d_param: ParamSpec(
            type="float", default=d_def, min=d_min, max=d_max, step=1.0,
            group="Body", label="Outer diameter",
        ),
    }
    return CircleProfile(diameter=ref(d_param)), params


# ---------------------------------------------------------------------------
# Polygon profile
# ---------------------------------------------------------------------------

def sample_polygon_profile(
    rng: Random,
    r_lo: float = 8.0,
    r_hi: float = 40.0,
    sides_choices: list[int] | None = None,
    r_param: str = "circ_r",
    sides_param: str = "n_sides",
) -> tuple[PolygonProfile, dict[str, ParamSpec]]:
    if sides_choices is None:
        sides_choices = [3, 4, 5, 6, 8]
    n_sides = rng.choice(sides_choices)
    r_def = _nice(rng, r_lo, r_hi, 0.4)
    params: dict[str, ParamSpec] = {
        r_param: ParamSpec(
            type="float", default=r_def,
            min=round(r_def * 0.6, 1), max=round(r_def * 1.6, 1), step=1.0,
            group="Body", label="Circumscribed radius",
        ),
        sides_param: ParamSpec(
            type="int", default=n_sides, min=3, max=8, step=1,
            group="Body", label="Sides",
        ),
    }
    return PolygonProfile(sides=ref(sides_param), circumscribed_r=ref(r_param)), params


# ---------------------------------------------------------------------------
# Slot profile
# ---------------------------------------------------------------------------

def sample_slot_profile(
    rng: Random,
    len_lo: float = 20.0,
    len_hi: float = 100.0,
    w_lo: float = 8.0,
    w_hi: float = 30.0,
    len_param: str = "slot_len",
    w_param: str = "slot_w",
) -> tuple[SlotProfile, dict[str, ParamSpec]]:
    l_def = _nice(rng, len_lo, len_hi)
    w_def = _nice(rng, w_lo, min(w_hi, l_def * 0.8))
    params: dict[str, ParamSpec] = {
        len_param: ParamSpec(
            type="float", default=l_def,
            min=round(l_def * 0.5, 1), max=round(l_def * 1.8, 1), step=1.0,
            group="Body", label="Slot length",
        ),
        w_param: ParamSpec(
            type="float", default=w_def,
            min=round(w_def * 0.5, 1), max=round(w_def * 1.6, 1), step=1.0,
            group="Body", label="Slot width",
        ),
    }
    return SlotProfile(length=ref(len_param), width=ref(w_param)), params


# ---------------------------------------------------------------------------
# Sketched profile — freeform closed loops of line / arc / bezier / spline
# ---------------------------------------------------------------------------

# Per-edge segment kinds; line-heavy so loops stay simple and mostly faceted,
# with arcs and curves sprinkled in for organic shapes.
_SKETCH_SEG_KINDS = ["line", "line", "line", "arc", "arc", "bezier", "spline"]


def sample_sketched_profile(
    rng: Random,
    w_lo: float = 40.0,
    w_hi: float = 90.0,
    w_param: str = "sk_w",
    h_param: str = "sk_h",
) -> tuple[SketchedProfile, dict[str, ParamSpec]]:
    """
    A freeform closed cross-section for extrusion.  Builds a star-shaped loop
    (vertices at monotonically increasing angles around the centroid → simple,
    non-self-intersecting by construction) in normalized coords ~[-0.5, 0.5],
    then renders each edge as a line, circular arc, Bézier, or spline.  The
    ``w_param`` / ``h_param`` sliders scale x / y; anisotropic scaling preserves
    simplicity, so validity stays ~100% by construction.
    """
    n = rng.randint(4, 7)
    step = 2 * math.pi / n
    jit = step * 0.35
    angles = sorted(i * step + rng.uniform(-jit, jit) for i in range(n))
    verts = [
        (r * math.cos(a), r * math.sin(a))
        for a, r in ((a, rng.uniform(0.32, 0.5)) for a in angles)
    ]
    cx = sum(v[0] for v in verts) / n
    cy = sum(v[1] for v in verts) / n

    segments: list[SketchSeg] = []
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        edge_len = math.hypot(x1 - x0, y1 - y0)
        kind = rng.choice(_SKETCH_SEG_KINDS)
        if kind == "arc":
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = mx - cx, my - cy
            norm = math.hypot(dx, dy) or 1.0
            bulge = rng.uniform(0.06, 0.18) * edge_len
            segments.append(SketchSeg(
                kind="arc", x=round(x1, 4), y=round(y1, 4),
                mx=round(mx + dx / norm * bulge, 4),
                my=round(my + dy / norm * bulge, 4),
            ))
        elif kind in ("bezier", "spline"):
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ex, ey = x1 - x0, y1 - y0
            norm = math.hypot(ex, ey) or 1.0
            px, py = -ey / norm, ex / norm        # perpendicular
            if (mx - cx) * px + (my - cy) * py < 0:  # point it outward
                px, py = -px, -py
            off = rng.uniform(0.05, 0.15) * edge_len
            segments.append(SketchSeg(
                kind=kind, x=round(x1, 4), y=round(y1, 4),
                ctrl=[(round(mx + px * off, 4), round(my + py * off, 4))],
            ))
        else:
            segments.append(SketchSeg(kind="line", x=round(x1, 4), y=round(y1, 4)))

    w_def = _nice(rng, w_lo, w_hi)
    h_def = _nice(rng, w_lo, w_hi)
    params: dict[str, ParamSpec] = {
        w_param: ParamSpec(
            type="float", default=w_def,
            min=round(w_def * 0.5, 1), max=round(w_def * 1.8, 1), step=1.0,
            group="Sketch", label="Sketch width (mm)",
        ),
        h_param: ParamSpec(
            type="float", default=h_def,
            min=round(h_def * 0.5, 1), max=round(h_def * 1.8, 1), step=1.0,
            group="Sketch", label="Sketch height (mm)",
        ),
    }
    start = (round(verts[0][0], 4), round(verts[0][1], 4))
    return SketchedProfile(
        w_param=w_param, h_param=h_param, start=start, segments=segments,
    ), params


def sample_sketched_revolve_profile(
    rng: Random,
    r_lo: float = 14.0,
    r_hi: float = 45.0,
    h_lo: float = 20.0,
    h_hi: float = 70.0,
    r_param: str = "sk_r",
    h_param: str = "sk_h",
) -> tuple[SketchedProfile, dict[str, ParamSpec]]:
    """
    An axis-safe freeform half-silhouette for revolving around Z.  Normalized x
    is the radius (kept ≥ a small positive floor so the loop never crosses the
    axis), normalized y is the Z height rising 0 → 1 up the outer side, then the
    loop closes across the top and down an inner wall (a centerline solid, or an
    inner bore for a tube).  ``r_param`` scales radius, ``h_param`` scales height.
    """
    x_min = 0.12
    n = rng.randint(3, 5)
    zs = [0.0] + sorted(rng.uniform(0.1, 0.95) for _ in range(n)) + [1.0]
    xs = [rng.uniform(x_min, 1.0) for _ in zs]
    bored = rng.random() < 0.45
    x_in = rng.uniform(x_min * 0.5, min(xs) * 0.7) if bored else 0.0

    segments: list[SketchSeg] = []
    for i in range(1, len(zs)):
        x0, x1 = xs[i - 1], xs[i]
        z0, z1 = zs[i - 1], zs[i]
        kind = rng.choice(_SKETCH_SEG_KINDS)
        if kind == "arc":
            mx = max(x_min * 0.6, (x0 + x1) / 2 + rng.uniform(-0.12, 0.12))
            segments.append(SketchSeg(
                kind="arc", x=round(x1, 4), y=round(z1, 4),
                mx=round(mx, 4), my=round((z0 + z1) / 2, 4),
            ))
        elif kind in ("bezier", "spline"):
            cx = max(x_min * 0.6, (x0 + x1) / 2 + rng.uniform(-0.12, 0.18))
            segments.append(SketchSeg(
                kind=kind, x=round(x1, 4), y=round(z1, 4),
                ctrl=[(round(cx, 4), round((z0 + z1) / 2, 4))],
            ))
        else:
            segments.append(SketchSeg(kind="line", x=round(x1, 4), y=round(z1, 4)))

    # Close across the top, then straight down the inner wall back to the base.
    segments.append(SketchSeg(kind="line", x=round(x_in, 4), y=1.0))
    segments.append(SketchSeg(kind="line", x=round(x_in, 4), y=0.0))

    r_def = _nice(rng, r_lo, r_hi)
    h_def = _nice(rng, h_lo, h_hi)
    params: dict[str, ParamSpec] = {
        r_param: ParamSpec(
            type="float", default=r_def,
            min=round(r_def * 0.5, 1), max=round(r_def * 1.8, 1), step=1.0,
            group="Sketch", label="Max radius (mm)",
        ),
        h_param: ParamSpec(
            type="float", default=h_def,
            min=round(h_def * 0.5, 1), max=round(h_def * 1.8, 1), step=1.0,
            group="Sketch", label="Height (mm)",
        ),
    }
    start = (round(xs[0], 4), 0.0)
    return SketchedProfile(
        w_param=r_param, h_param=h_param, start=start, segments=segments,
    ), params


# ---------------------------------------------------------------------------
# Generic dispatcher used by features.py
# ---------------------------------------------------------------------------

PROFILE_SAMPLERS = ["rect", "rounded_rect", "circle", "polygon", "slot", "sketched"]


def sample_profile(
    rng: Random,
    kind: str | None = None,
    **kwargs: Any,
) -> tuple[Profile, dict[str, ParamSpec]]:
    if kind is None:
        kind = rng.choice(PROFILE_SAMPLERS)
    if kind == "rect":
        return sample_rect_profile(rng, **kwargs)
    if kind == "rounded_rect":
        return sample_rounded_rect_profile(rng, **kwargs)
    if kind == "circle":
        return sample_circle_profile(rng, **kwargs)
    if kind == "polygon":
        return sample_polygon_profile(rng, **kwargs)
    if kind == "slot":
        return sample_slot_profile(rng, **kwargs)
    if kind == "sketched":
        return sample_sketched_profile(rng, **kwargs)
    raise ValueError(f"Unknown profile kind: {kind!r}")
