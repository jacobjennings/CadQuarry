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
    SlotProfile,
    lit,
    ref,
    scaled,
)

# Standard metric fastener clearance diameters (M3-M12, close-fit).
FASTENER_CLEARANCE_MM = [3.2, 4.3, 5.3, 6.4, 8.4, 10.5, 13.0]

# Stock plate thicknesses (mm) — common in sheet-metal and machined stock.
STOCK_THICKNESSES_MM = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]


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
# Generic dispatcher used by features.py
# ---------------------------------------------------------------------------

PROFILE_SAMPLERS = ["rect", "rounded_rect", "circle", "polygon", "slot"]


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
    raise ValueError(f"Unknown profile kind: {kind!r}")
