"""
Part family samplers and distribution control.

Each family function samples a PartIR from scratch using a seeded RNG.
The compose() entry-point picks a family according to the configured
weights, picks a complexity tier, and delegates to the family sampler.

Distribution knobs live in configs/default.toml; this module reads the
relevant sub-tables and falls back to hard-coded defaults when absent.
"""
from __future__ import annotations

import uuid
from random import Random
from typing import Any

from .ir import (
    AttachOp,
    BoxOp,
    CBracketOp,
    ChamferOp,
    CircleProfile,
    ExtrudeOp,
    HolesOp,
    LBracketOp,
    ZBracketOp,
    PartIR,
    PartMetadata,
    PolygonProfile,
    RectProfile,
    RevolveOp,
    Operation,
    ParamSpec,
    ShellOp,
    SlotProfile,
    lit,
    ref,
    scaled,
    min2,
)
from .profiles import (
    snap_to_stock_thickness,
    snap_fastener_diameter,
    sample_rect_profile,
    sample_circle_profile,
    sample_polygon_profile,
    sample_slot_profile,
    _nice,
)
from .features import (
    sample_corner_holes,
    sample_grid_holes,
    sample_bolt_circle,
    sample_staggered_holes,
    sample_fillet,
    sample_chamfer,
    sample_pocket,
    sample_boss,
    sample_ribs,
)

# Cross-cutting symmetry modes (Stage D regularity).
SYMMETRY_MODES = ["mirror_x", "mirror_xy", "radial"]

GENERATOR_VERSION = "0.3.0"

# Default family weights; overridden by config.
DEFAULT_FAMILY_WEIGHTS = {
    "plate":    0.20,
    "bracket":  0.17,
    "revolved": 0.17,
    "block":    0.13,
    "compound": 0.10,
    "flanged":  0.07,
    "ribbed":   0.06,
    "enclosure":0.06,
    "profiled": 0.04,
}

# Default tier weights (before family clamping).
DEFAULT_TIER_WEIGHTS = [0.25, 0.40, 0.25, 0.10]  # tiers 0-3

# Per-family tier span [min_tier, max_tier].
FAMILY_TIER_SPANS = {
    "plate":     (0, 3),
    "bracket":   (1, 3),
    "revolved":  (0, 2),
    "block":     (1, 3),
    "compound":  (1, 3),
    "flanged":   (1, 3),
    "ribbed":    (1, 3),
    "enclosure": (1, 3),
    "profiled":  (0, 2),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_choice(rng: Random, choices: list[str], weights: list[float]) -> str:
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for c, w in zip(choices, weights):
        acc += w
        if r <= acc:
            return c
    return choices[-1]


def _sample_tier(rng: Random, family: str, config: dict) -> int:
    tier_w = config.get("distribution", {}).get("tiers", {})
    weights = [
        tier_w.get("tier0", 0.25),
        tier_w.get("tier1", 0.40),
        tier_w.get("tier2", 0.25),
        tier_w.get("tier3", 0.10),
    ]
    lo, hi = FAMILY_TIER_SPANS.get(family, (0, 3))
    clamped = weights[lo: hi + 1]
    total = sum(clamped)
    if total == 0:
        return lo
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(clamped):
        acc += w
        if r <= acc:
            return lo + i
    return hi


def _make_id(seed: int, family: str, index: int) -> str:
    return f"{family}_{seed:08x}_{index:04d}"


def _sample_symmetry(rng: Random, config: dict, family: str) -> str:
    """
    Sample a cross-cutting symmetry mode for a part.  With probability
    (1 - symmetry_prob) the part is asymmetric ("none").  Radial symmetry only
    makes sense for families with a natural axis, so it is excluded elsewhere.
    """
    reg = config.get("distribution", {}).get("regularities", {})
    prob = reg.get("symmetry_prob", 0.60)
    if rng.random() >= prob:
        return "none"
    modes = list(SYMMETRY_MODES)
    if family not in ("plate", "block", "revolved", "flanged"):
        # No clean rotational axis for these; restrict to mirror symmetry.
        modes = [m for m in modes if m != "radial"]
    return rng.choice(modes)


# ---------------------------------------------------------------------------
# Family: plate
# ---------------------------------------------------------------------------

def sample_plate(rng: Random, tier: int, config: dict, seed: int, index: int) -> PartIR:
    """
    Flat rectangular plate with optional holes, fillets, chamfers.
    Tier 0 → bare plate; Tier 1 → holes; Tier 2 → fillet + hole type variety;
    Tier 3 → pocket or boss added.
    """
    fcfg = config.get("families", {}).get("plate", {})

    w_lo = fcfg.get("width_min", 20.0)
    w_hi = fcfg.get("width_max", 120.0)
    asp_lo = fcfg.get("aspect_min", 1.0)
    asp_hi = fcfg.get("aspect_max", 3.0)
    t_lo = fcfg.get("thickness_min", 2.0)
    t_hi = fcfg.get("thickness_max", 12.0)

    # Base dimensions
    w_def = _nice(rng, w_lo, w_hi)
    asp = rng.uniform(asp_lo, asp_hi)
    d_def = round(max(w_lo * 0.5, min(w_hi, w_def / asp)), 1)
    t_def = snap_to_stock_thickness(rng, t_lo, t_hi)

    params: dict[str, ParamSpec] = {
        "plate_w": ParamSpec(
            type="float", default=w_def,
            min=round(w_lo, 1), max=round(w_hi, 1), step=1.0,
            group="Body", label="Plate width (mm)",
        ),
        "plate_d": ParamSpec(
            type="float", default=d_def,
            min=round(w_lo * 0.4, 1), max=round(w_hi, 1), step=1.0,
            group="Body", label="Plate depth (mm)",
        ),
        "thickness": ParamSpec(
            type="float", default=t_def,
            min=round(t_lo, 1), max=round(t_hi, 1), step=0.5,
            group="Body", label="Thickness (mm)",
        ),
    }

    ops: list[Operation] = [
        BoxOp(width=ref("plate_w"), depth=ref("plate_d"), height=ref("thickness"))
    ]

    if tier >= 1:
        # At tier 1: only round holes. At tier 2+: full variety.
        if tier >= 2:
            hole_type = rng.choice([
                "simple", "simple", "counterbore", "countersink", "square", "slot",
            ])
        else:
            hole_type = "simple"

        sym = config.get("_symmetry", "none")
        if sym == "radial":
            op, p = sample_bolt_circle(rng, "plate_w", w_def)
        elif sym in ("mirror_x", "mirror_xy"):
            op, p = sample_corner_holes(
                rng, "plate_w", "plate_d", w_def, d_def, hole_type=hole_type
            )
        else:
            hole_layout = rng.choice(["corners", "corners", "grid", "staggered"])
            if hole_layout == "corners":
                op, p = sample_corner_holes(
                    rng, "plate_w", "plate_d", w_def, d_def, hole_type=hole_type
                )
            elif hole_layout == "grid":
                grid_shape = rng.choice(["round", "round", "square", "slot"])
                op, p = sample_grid_holes(
                    rng, "plate_w", "plate_d", w_def, d_def, hole_shape=grid_shape
                )
            else:  # staggered
                op, p = sample_staggered_holes(rng, "plate_w", "plate_d", w_def, d_def)
        params.update(p)
        ops.append(op)

    if tier >= 2:
        fillet_op, fp = sample_fillet(rng, "plate_w", "plate_d", edge_selector="|Z")
        params.update(fp)
        ops.append(fillet_op)

    if tier >= 3:
        extra = rng.choice(["pocket", "boss", "chamfer"])
        if extra == "pocket":
            op, p = sample_pocket(
                rng, "plate_w", "plate_d", "thickness",
                w_def, d_def, t_def,
            )
            params.update(p)
            ops.append(op)
        elif extra == "boss":
            op, p = sample_boss(rng, "thickness", t_def)
            params.update(p)
            ops.append(op)
        else:
            op, p = sample_chamfer(rng, edge_selector=">Z")
            params.update(p)
            ops.append(op)

    return PartIR(
        id=_make_id(seed, "plate", index),
        params=params,
        operations=ops,
        metadata=PartMetadata(
            seed=seed,
            generator_version=GENERATOR_VERSION,
            family="plate",
            tier=tier,
            op_count=len(ops),
        ),
    )


# ---------------------------------------------------------------------------
# Family: bracket
# ---------------------------------------------------------------------------

# Bracket shapes and their default sampling weights.  L stays the most common
# (it's the simplest sheet-metal bracket), but channels and offset brackets now
# round out the family so the corpus isn't all right-angle L's.
DEFAULT_BRACKET_SHAPE_WEIGHTS = {"l": 0.5, "c": 0.25, "z": 0.25}


def sample_bracket(rng: Random, tier: int, config: dict, seed: int, index: int) -> PartIR:
    """
    Angle bracket in one of three shapes, with mounting holes on each leg and
    optional corner gussets:

      * ``l`` — flat base leg + a single vertical leg (right-angle L).
      * ``c`` — flat base leg + a vertical leg at *each* end (channel / U).
      * ``z`` — bottom flange, vertical web, and a top flange that extends the
        opposite way (cranked / offset Z).

    Tier 1 → plain shape with holes; Tier 2 → add gussets; Tier 3 → larger hole
    count.  The shape is chosen per part from configurable weights.
    """
    fcfg = config.get("families", {}).get("bracket", {})
    len_lo = fcfg.get("leg_min", 25.0)
    len_hi = fcfg.get("leg_max", 90.0)
    w_lo = fcfg.get("width_min", 15.0)
    w_hi = fcfg.get("width_max", 60.0)
    t_lo = fcfg.get("thickness_min", 3.0)
    t_hi = fcfg.get("thickness_max", 8.0)

    shape_w = fcfg.get("shape_weights", DEFAULT_BRACKET_SHAPE_WEIGHTS)
    shape_names = list(shape_w.keys())
    shape = _weighted_choice(
        rng, shape_names, [shape_w.get(n, 0.0) for n in shape_names]
    )

    base_def = _nice(rng, len_lo, len_hi)
    vert_def = round(base_def * rng.uniform(0.6, 1.2), 1)
    vert_def = max(len_lo, min(len_hi, vert_def))
    w_def = _nice(rng, w_lo, w_hi)
    t_def = snap_to_stock_thickness(rng, t_lo, t_hi)

    params: dict[str, ParamSpec] = {
        "base_len": ParamSpec(
            type="float", default=base_def,
            min=round(len_lo, 1), max=round(len_hi, 1), step=1.0,
            group="Body", label="Base leg length (mm)",
        ),
        "vert_len": ParamSpec(
            type="float", default=vert_def,
            min=round(len_lo, 1), max=round(len_hi, 1), step=1.0,
            group="Body", label="Vertical leg length (mm)",
        ),
        "bracket_w": ParamSpec(
            type="float", default=w_def,
            min=round(w_lo, 1), max=round(w_hi, 1), step=1.0,
            group="Body", label="Width (mm)",
        ),
        "bracket_t": ParamSpec(
            type="float", default=t_def,
            min=round(t_lo, 1), max=round(t_hi, 1), step=0.5,
            group="Body", label="Thickness (mm)",
        ),
    }

    # A Z-bracket needs an independent top-flange length.
    top_len_expr = None
    if shape == "z":
        top_def = round(base_def * rng.uniform(0.6, 1.1), 1)
        top_def = max(len_lo, min(len_hi, top_def))
        params["top_len"] = ParamSpec(
            type="float", default=top_def,
            min=round(len_lo, 1), max=round(len_hi, 1), step=1.0,
            group="Body", label="Top flange length (mm)",
        )
        top_len_expr = ref("top_len")

    # Holes on each leg (standard fastener clearance).
    hole_def = snap_fastener_diameter(rng, 3.0, 7.0)
    n_def = 1 if tier < 3 else rng.choice([1, 2])
    leg_word = "flange" if shape == "z" else "leg"
    params["hole_d"] = ParamSpec(
        type="float", default=hole_def,
        min=3.0, max=round(min(w_def * 0.4, 10.0), 1), step=0.1,
        group="Holes", label="Hole diameter (mm)",
    )
    params["n_holes"] = ParamSpec(
        type="int", default=n_def, min=1, max=3, step=1,
        group="Holes", label=f"Holes per {leg_word}",
    )

    gusset_t_expr = None
    gusset_en_expr = None
    if tier >= 2:
        gt_def = round(t_def * rng.uniform(0.8, 1.5), 1)
        params["gusset_t"] = ParamSpec(
            type="float", default=gt_def,
            min=round(t_def * 0.5, 1), max=round(min(w_def * 0.5, t_def * 3.0), 1), step=0.5,
            group="Gusset", label="Gusset thickness (mm)",
        )
        params["gusset"] = ParamSpec(
            type="bool", default=True, group="Gusset", label="Corner gusset",
        )
        gusset_t_expr = ref("gusset_t")
        gusset_en_expr = ref("gusset")

    if shape == "c":
        bracket_op: Operation = CBracketOp(
            base_len=ref("base_len"),
            vert_len=ref("vert_len"),
            width=ref("bracket_w"),
            thickness=ref("bracket_t"),
            hole_d=ref("hole_d"),
            n_holes=ref("n_holes"),
            gusset_t=gusset_t_expr,
            gusset_enabled=gusset_en_expr,
        )
    elif shape == "z":
        bracket_op = ZBracketOp(
            base_len=ref("base_len"),
            vert_len=ref("vert_len"),
            top_len=top_len_expr,
            width=ref("bracket_w"),
            thickness=ref("bracket_t"),
            hole_d=ref("hole_d"),
            n_holes=ref("n_holes"),
            gusset_t=gusset_t_expr,
            gusset_enabled=gusset_en_expr,
        )
    else:
        bracket_op = LBracketOp(
            base_len=ref("base_len"),
            vert_len=ref("vert_len"),
            width=ref("bracket_w"),
            thickness=ref("bracket_t"),
            hole_d=ref("hole_d"),
            n_holes=ref("n_holes"),
            gusset_t=gusset_t_expr,
            gusset_enabled=gusset_en_expr,
        )

    ops: list[Operation] = [bracket_op]

    return PartIR(
        id=_make_id(seed, "bracket", index),
        params=params,
        operations=ops,
        metadata=PartMetadata(
            seed=seed,
            generator_version=GENERATOR_VERSION,
            family="bracket",
            tier=tier,
            op_count=len(ops),
        ),
    )


# ---------------------------------------------------------------------------
# Family: revolved
# ---------------------------------------------------------------------------

def sample_revolved(rng: Random, tier: int, config: dict, seed: int, index: int) -> PartIR:
    """
    Solid of revolution: shaft, bushing, washer, flange stub.
    """
    fcfg = config.get("families", {}).get("revolved", {})
    od_lo = fcfg.get("outer_d_min", 6.0)
    od_hi = fcfg.get("outer_d_max", 80.0)
    l2d_lo = fcfg.get("length_to_d_min", 0.2)
    l2d_hi = fcfg.get("length_to_d_max", 4.0)

    od_def = _nice(rng, od_lo, od_hi, snap_prob=0.3)
    l2d = rng.uniform(l2d_lo, l2d_hi)
    h_def = round(od_def * l2d, 1)
    h_def = max(2.0, h_def)

    # Optional inner bore
    has_bore = tier >= 1 and rng.random() < 0.6
    id_def = None
    if has_bore:
        id_def = round(rng.uniform(od_def * 0.3, od_def * 0.6), 1)

    params: dict[str, ParamSpec] = {
        "outer_d": ParamSpec(
            type="float", default=od_def,
            min=round(od_lo, 1), max=round(od_hi, 1), step=1.0,
            group="Body", label="Outer diameter (mm)",
        ),
        "length": ParamSpec(
            type="float", default=h_def,
            min=round(max(1.0, h_def * 0.3), 1),
            max=round(min(od_hi * 4.0, h_def * 3.0), 1),
            step=1.0,
            group="Body", label="Length (mm)",
        ),
    }

    inner_expr = None
    if has_bore and id_def is not None:
        params["inner_d"] = ParamSpec(
            type="float", default=id_def,
            min=2.0, max=round(od_def * 0.7, 1), step=0.5,
            group="Body", label="Bore diameter (mm)",
        )
        inner_expr = ref("inner_d")

    # Optional flange
    has_flange = tier >= 1 and rng.random() < 0.35
    flange_d_expr = None
    flange_h_expr = None
    if has_flange:
        fd_def = round(od_def * rng.uniform(1.4, 2.2), 1)
        fh_def = round(h_def * rng.uniform(0.1, 0.3), 1)
        params["flange_d"] = ParamSpec(
            type="float", default=fd_def,
            min=round(od_def * 1.2, 1), max=round(od_def * 3.0, 1), step=1.0,
            group="Body", label="Flange diameter (mm)",
        )
        params["flange_h"] = ParamSpec(
            type="float", default=fh_def,
            min=round(fh_def * 0.3, 1), max=round(h_def * 0.5, 1), step=0.5,
            group="Body", label="Flange height (mm)",
        )
        flange_d_expr = ref("flange_d")
        flange_h_expr = ref("flange_h")

    # Optional groove (tier 2+)
    has_groove = tier >= 2 and rng.random() < 0.4
    groove_depth_expr = None
    groove_w_expr = None
    groove_pos_expr = None
    if has_groove:
        gd_def = round(od_def * rng.uniform(0.03, 0.08), 2)
        gw_def = round(h_def * rng.uniform(0.05, 0.12), 2)
        gp_def = round(h_def * rng.uniform(0.25, 0.75), 1)
        params["groove_depth"] = ParamSpec(
            type="float", default=gd_def,
            min=round(gd_def * 0.4, 2), max=round(gd_def * 3.0, 2), step=0.1,
            group="Body", label="Groove depth (mm)",
        )
        params["groove_w"] = ParamSpec(
            type="float", default=gw_def,
            min=round(gw_def * 0.4, 2), max=round(gw_def * 3.0, 2), step=0.2,
            group="Body", label="Groove width (mm)",
        )
        params["groove_pos"] = ParamSpec(
            type="float", default=gp_def,
            min=round(h_def * 0.15, 1), max=round(h_def * 0.85, 1), step=1.0,
            group="Body", label="Groove position (mm)",
        )
        groove_depth_expr = ref("groove_depth")
        groove_w_expr = ref("groove_w")
        groove_pos_expr = ref("groove_pos")

    ops: list[Operation] = [
        RevolveOp(
            outer_d=ref("outer_d"),
            height=ref("length"),
            inner_d=inner_expr,
            flange_d=flange_d_expr,
            flange_h=flange_h_expr,
            groove_depth=groove_depth_expr,
            groove_w=groove_w_expr,
            groove_pos=groove_pos_expr,
        )
    ]

    # Optional end chamfer (tier 1+)
    if tier >= 1 and rng.random() < 0.5:
        cham_def = round(rng.uniform(0.3, 1.5), 1)
        params["end_chamfer"] = ParamSpec(
            type="float", default=cham_def,
            min=0.1, max=round(cham_def * 3.0, 1), step=0.1,
            group="Body", label="End chamfer (mm)",
        )
        params["end_chamfered"] = ParamSpec(
            type="bool", default=True,
            group="Body", label="End chamfers",
        )
        ops.append(ChamferOp(
            distance=ref("end_chamfer"),
            edge_selector=">Z or <Z",
            enabled=ref("end_chamfered"),
        ))

    return PartIR(
        id=_make_id(seed, "revolved", index),
        params=params,
        operations=ops,
        metadata=PartMetadata(
            seed=seed,
            generator_version=GENERATOR_VERSION,
            family="revolved",
            tier=tier,
            op_count=len(ops),
        ),
    )


# ---------------------------------------------------------------------------
# Family: block
# ---------------------------------------------------------------------------

def sample_block(rng: Random, tier: int, config: dict, seed: int, index: int) -> PartIR:
    """
    Rectangular block / housing with pockets, bosses, and holes.
    """
    fcfg = config.get("families", {}).get("block", {})
    w_lo = fcfg.get("width_min", 15.0)
    w_hi = fcfg.get("width_max", 100.0)

    w_def = _nice(rng, w_lo, w_hi)
    asp = rng.uniform(1.0, 2.5)
    d_def = round(max(w_lo * 0.4, min(w_hi, w_def / asp)), 1)
    h_def = round(rng.uniform(w_def * 0.5, w_def * 2.0), 1)
    h_def = min(h_def, w_hi)

    params: dict[str, ParamSpec] = {
        "block_w": ParamSpec(
            type="float", default=w_def,
            min=round(w_lo, 1), max=round(w_hi, 1), step=1.0,
            group="Body", label="Block width (mm)",
        ),
        "block_d": ParamSpec(
            type="float", default=d_def,
            min=round(w_lo * 0.4, 1), max=round(w_hi, 1), step=1.0,
            group="Body", label="Block depth (mm)",
        ),
        "block_h": ParamSpec(
            type="float", default=h_def,
            min=round(w_lo * 0.3, 1), max=round(w_hi * 2.0, 1), step=1.0,
            group="Body", label="Block height (mm)",
        ),
    }

    ops: list[Operation] = [
        BoxOp(width=ref("block_w"), depth=ref("block_d"), height=ref("block_h"))
    ]

    if tier >= 1:
        sym = config.get("_symmetry", "none")
        if sym == "radial":
            op, p = sample_bolt_circle(rng, "block_w", w_def)
        else:
            hole_type = rng.choice(["simple", "simple", "square", "staggered"])
            if hole_type == "staggered":
                op, p = sample_staggered_holes(rng, "block_w", "block_d", w_def, d_def)
            else:
                op, p = sample_corner_holes(
                    rng, "block_w", "block_d", w_def, d_def,
                    hole_type="square" if hole_type == "square" else "simple",
                )
        params.update(p)
        ops.append(op)

    if tier >= 2:
        extra = rng.choice(["pocket", "fillet", "boss"])
        if extra == "pocket":
            op, p = sample_pocket(rng, "block_w", "block_d", "block_h", w_def, d_def, h_def)
            params.update(p)
            ops.append(op)
        elif extra == "fillet":
            op, p = sample_fillet(rng, "block_w", "block_d", edge_selector="|Z")
            params.update(p)
            ops.append(op)
        else:
            op, p = sample_boss(rng, "block_h", h_def)
            params.update(p)
            ops.append(op)

    if tier >= 3:
        op, p = sample_fillet(
            rng, "block_w", "block_d", edge_selector="|Z",
            r_param="fillet_r2", enabled_param="filleted2",
        )
        if "filleted2" not in params:
            params.update(p)
            ops.append(op)

        # 40% chance: add a side tube attachment for multi-section complexity.
        if rng.random() < 0.40:
            face = rng.choice(_SIDE_FACES)
            fd_a, fd_b = (d_def, h_def) if face in (">X", "<X") else (w_def, h_def)
            min_fd = min(fd_a, fd_b)
            st_d_def = max(4.0, round(min_fd * rng.uniform(0.18, 0.42), 1))
            st_l_def = max(4.0, round(st_d_def * rng.uniform(0.8, 2.0), 1))
            params["side_tube_d"] = ParamSpec(
                type="float", default=st_d_def,
                min=round(st_d_def * 0.4, 1), max=round(min_fd * 0.50, 1),
                step=1.0, group="Side Tube", label="Side tube diameter (mm)",
            )
            params["side_tube_len"] = ParamSpec(
                type="float", default=st_l_def,
                min=round(st_l_def * 0.3, 1), max=round(st_d_def * 3.0, 1),
                step=1.0, group="Side Tube", label="Side tube length (mm)",
            )
            bore_expr = None
            if rng.random() < 0.60:
                sb_def = max(2.0, round(st_d_def * rng.uniform(0.40, 0.65), 1))
                params["side_bore_d"] = ParamSpec(
                    type="float", default=sb_def,
                    min=2.0, max=round(st_d_def * 0.75, 1), step=0.5,
                    group="Side Tube", label="Side bore diameter (mm)",
                )
                bore_expr = ref("side_bore_d")
            ops.append(AttachOp(
                profile=CircleProfile(diameter=ref("side_tube_d")),
                length=ref("side_tube_len"),
                bore_d=bore_expr,
                face=face,
            ))

    return PartIR(
        id=_make_id(seed, "block", index),
        params=params,
        operations=ops,
        metadata=PartMetadata(
            seed=seed,
            generator_version=GENERATOR_VERSION,
            family="block",
            tier=tier,
            op_count=len(ops),
        ),
    )


# ---------------------------------------------------------------------------
# Family: enclosure
# ---------------------------------------------------------------------------

def sample_enclosure(rng: Random, tier: int, config: dict, seed: int, index: int) -> PartIR:
    """
    Hollow shell (box with open top face).
    """
    fcfg = config.get("families", {}).get("enclosure", {})
    w_lo = fcfg.get("width_min", 30.0)
    w_hi = fcfg.get("width_max", 120.0)
    wt_lo = fcfg.get("wall_thickness_min", 2.0)
    wt_hi = fcfg.get("wall_thickness_max", 6.0)

    w_def = _nice(rng, w_lo, w_hi)
    asp = rng.uniform(1.0, 2.5)
    d_def = round(max(w_lo * 0.5, min(w_hi, w_def / asp)), 1)
    h_def = round(rng.uniform(w_def * 0.4, w_def * 1.0), 1)
    wt_def = round(rng.uniform(wt_lo, wt_hi), 1)

    params: dict[str, ParamSpec] = {
        "enc_w": ParamSpec(
            type="float", default=w_def,
            min=round(w_lo, 1), max=round(w_hi, 1), step=1.0,
            group="Body", label="Enclosure width (mm)",
        ),
        "enc_d": ParamSpec(
            type="float", default=d_def,
            min=round(w_lo * 0.4, 1), max=round(w_hi, 1), step=1.0,
            group="Body", label="Enclosure depth (mm)",
        ),
        "enc_h": ParamSpec(
            type="float", default=h_def,
            min=round(h_def * 0.4, 1), max=round(h_def * 2.0, 1), step=1.0,
            group="Body", label="Enclosure height (mm)",
        ),
        "wall_t": ParamSpec(
            type="float", default=wt_def,
            min=round(wt_lo, 1), max=round(wt_hi, 1), step=0.5,
            group="Body", label="Wall thickness (mm)",
        ),
    }

    ops: list[Operation] = [
        BoxOp(width=ref("enc_w"), depth=ref("enc_d"), height=ref("enc_h")),
        ShellOp(thickness=ref("wall_t"), open_face=">Z"),
    ]

    if tier >= 2:
        op, p = sample_corner_holes(rng, "enc_w", "enc_d", w_def, d_def, hole_type="simple")
        params.update(p)
        ops.append(op)

    if tier >= 2 and rng.random() < 0.5:
        op, p = sample_fillet(rng, "enc_w", "enc_d", edge_selector="|Z")
        params.update(p)
        ops.append(op)

    return PartIR(
        id=_make_id(seed, "enclosure", index),
        params=params,
        operations=ops,
        metadata=PartMetadata(
            seed=seed,
            generator_version=GENERATOR_VERSION,
            family="enclosure",
            tier=tier,
            op_count=len(ops),
        ),
    )


# ---------------------------------------------------------------------------
# Family: flanged (flange + bolt circle)
# ---------------------------------------------------------------------------

def sample_flanged(rng: Random, tier: int, config: dict, seed: int, index: int) -> PartIR:
    """
    Flanged part: revolve + polar bolt pattern on the flange face.
    """
    fcfg = config.get("families", {}).get("flanged", {})
    od_lo = fcfg.get("outer_d_min", 20.0)
    od_hi = fcfg.get("outer_d_max", 100.0)

    od_def = _nice(rng, od_lo, od_hi, 0.3)
    fd_def = round(od_def * rng.uniform(1.6, 2.4), 1)
    fh_def = round(od_def * rng.uniform(0.08, 0.18), 1)
    h_def = round(od_def * rng.uniform(0.5, 1.5), 1)
    id_def = round(od_def * rng.uniform(0.3, 0.55), 1)

    params: dict[str, ParamSpec] = {
        "outer_d": ParamSpec(
            type="float", default=od_def,
            min=round(od_lo, 1), max=round(od_hi, 1), step=1.0,
            group="Body", label="Outer diameter (mm)",
        ),
        "length": ParamSpec(
            type="float", default=h_def,
            min=round(h_def * 0.3, 1), max=round(h_def * 2.5, 1), step=1.0,
            group="Body", label="Length (mm)",
        ),
        "inner_d": ParamSpec(
            type="float", default=id_def,
            min=2.0, max=round(od_def * 0.7, 1), step=0.5,
            group="Body", label="Bore diameter (mm)",
        ),
        "flange_d": ParamSpec(
            type="float", default=fd_def,
            min=round(od_def * 1.3, 1), max=round(od_def * 3.0, 1), step=1.0,
            group="Flange", label="Flange diameter (mm)",
        ),
        "flange_h": ParamSpec(
            type="float", default=fh_def,
            min=round(fh_def * 0.3, 1), max=round(h_def * 0.4, 1), step=0.5,
            group="Flange", label="Flange height (mm)",
        ),
    }

    ops: list[Operation] = [
        RevolveOp(
            outer_d=ref("outer_d"),
            height=ref("length"),
            inner_d=ref("inner_d"),
            flange_d=ref("flange_d"),
            flange_h=ref("flange_h"),
        )
    ]

    bolt_op, bp = sample_bolt_circle(rng, "flange_d", fd_def)
    params.update(bp)
    ops.append(bolt_op)

    if tier >= 2 and rng.random() < 0.5:
        cham_def = round(rng.uniform(0.3, 1.0), 1)
        params["end_chamfer"] = ParamSpec(
            type="float", default=cham_def,
            min=0.1, max=2.0, step=0.1,
            group="Body", label="End chamfer (mm)",
        )
        ops.append(ChamferOp(distance=ref("end_chamfer"), edge_selector=">Z or <Z"))

    return PartIR(
        id=_make_id(seed, "flanged", index),
        params=params,
        operations=ops,
        metadata=PartMetadata(
            seed=seed,
            generator_version=GENERATOR_VERSION,
            family="flanged",
            tier=tier,
            op_count=len(ops),
        ),
    )


# ---------------------------------------------------------------------------
# Family: ribbed
# ---------------------------------------------------------------------------

def sample_ribbed(rng: Random, tier: int, config: dict, seed: int, index: int) -> PartIR:
    """
    Base plate with patterned thin ribs on the top face.
    """
    w_def = _nice(rng, 30.0, 100.0)
    asp = rng.uniform(1.0, 2.0)
    d_def = round(max(10.0, min(100.0, w_def / asp)), 1)
    t_def = snap_to_stock_thickness(rng, 3.0, 10.0)

    params: dict[str, ParamSpec] = {
        "base_w": ParamSpec(
            type="float", default=w_def,
            min=20.0, max=120.0, step=1.0,
            group="Base", label="Base width (mm)",
        ),
        "base_d": ParamSpec(
            type="float", default=d_def,
            min=15.0, max=120.0, step=1.0,
            group="Base", label="Base depth (mm)",
        ),
        "base_t": ParamSpec(
            type="float", default=t_def,
            min=2.0, max=15.0, step=0.5,
            group="Base", label="Base thickness (mm)",
        ),
    }

    ops: list[Operation] = [
        BoxOp(width=ref("base_w"), depth=ref("base_d"), height=ref("base_t"))
    ]

    rib_op, rp = sample_ribs(rng, "base_w", w_def)
    params.update(rp)
    ops.append(rib_op)

    if tier >= 2 and rng.random() < 0.4:
        op, p = sample_corner_holes(rng, "base_w", "base_d", w_def, d_def)
        params.update(p)
        ops.append(op)

    return PartIR(
        id=_make_id(seed, "ribbed", index),
        params=params,
        operations=ops,
        metadata=PartMetadata(
            seed=seed,
            generator_version=GENERATOR_VERSION,
            family="ribbed",
            tier=tier,
            op_count=len(ops),
        ),
    )


# ---------------------------------------------------------------------------
# Family: profiled extrusion
# ---------------------------------------------------------------------------

def sample_profiled(rng: Random, tier: int, config: dict, seed: int, index: int) -> PartIR:
    """
    Long constant cross-section extrusion (structural profile, rod, tube).
    """
    profile_kind = rng.choice(["circle", "polygon", "slot"])

    if profile_kind == "circle":
        d_def = _nice(rng, 8.0, 50.0, 0.3)
        has_bore = rng.random() < 0.5
        id_def = round(d_def * rng.uniform(0.4, 0.7), 1) if has_bore else None
        l_def = round(d_def * rng.uniform(3.0, 10.0), 1)
        l_def = min(l_def, 300.0)

        params: dict[str, ParamSpec] = {
            "outer_d": ParamSpec(
                type="float", default=d_def,
                min=round(d_def * 0.5, 1), max=round(d_def * 2.0, 1), step=1.0,
                group="Profile", label="Outer diameter (mm)",
            ),
            "length": ParamSpec(
                type="float", default=l_def,
                min=round(l_def * 0.3, 1), max=round(l_def * 3.0, 1), step=5.0,
                group="Body", label="Length (mm)",
            ),
        }
        inner_expr = None
        if has_bore and id_def:
            params["inner_d"] = ParamSpec(
                type="float", default=id_def,
                min=2.0, max=round(d_def * 0.8, 1), step=0.5,
                group="Profile", label="Bore diameter (mm)",
            )
            inner_expr = ref("inner_d")

        ops: list[Operation] = [
            RevolveOp(outer_d=ref("outer_d"), height=ref("length"), inner_d=inner_expr)
        ]
    elif profile_kind == "polygon":
        r_def = _nice(rng, 6.0, 25.0, 0.3)
        sides = rng.choice([4, 6, 8])
        l_def = round(r_def * rng.uniform(4.0, 12.0), 1)
        l_def = min(l_def, 200.0)

        params = {
            "circ_r": ParamSpec(
                type="float", default=r_def,
                min=round(r_def * 0.5, 1), max=round(r_def * 2.0, 1), step=1.0,
                group="Profile", label="Circumscribed radius (mm)",
            ),
            "n_sides": ParamSpec(
                type="int", default=sides, min=3, max=8, step=1,
                group="Profile", label="Sides",
            ),
            "length": ParamSpec(
                type="float", default=l_def,
                min=round(l_def * 0.3, 1), max=round(l_def * 3.0, 1), step=5.0,
                group="Body", label="Length (mm)",
            ),
        }
        ops = [ExtrudeOp(
            profile=PolygonProfile(sides=ref("n_sides"), circumscribed_r=ref("circ_r")),
            distance=ref("length"),
        )]
    else:  # slot
        len_def = _nice(rng, 20.0, 80.0)
        w_def = _nice(rng, 8.0, 25.0)
        l_def = round(max(len_def, w_def) * rng.uniform(4.0, 10.0), 1)

        params = {
            "slot_len": ParamSpec(
                type="float", default=len_def,
                min=round(len_def * 0.5, 1), max=round(len_def * 2.0, 1), step=1.0,
                group="Profile", label="Slot length (mm)",
            ),
            "slot_w": ParamSpec(
                type="float", default=w_def,
                min=round(w_def * 0.5, 1), max=round(w_def * 2.0, 1), step=0.5,
                group="Profile", label="Slot width (mm)",
            ),
            "length": ParamSpec(
                type="float", default=l_def,
                min=round(l_def * 0.3, 1), max=round(l_def * 3.0, 1), step=5.0,
                group="Body", label="Length (mm)",
            ),
        }
        ops = [ExtrudeOp(
            profile=SlotProfile(length=ref("slot_len"), width=ref("slot_w")),
            distance=ref("length"),
        )]

    return PartIR(
        id=_make_id(seed, "profiled", index),
        params=params,
        operations=ops,
        metadata=PartMetadata(
            seed=seed,
            generator_version=GENERATOR_VERSION,
            family="profiled",
            tier=tier,
            op_count=len(ops),
        ),
    )


# ---------------------------------------------------------------------------
# Family: compound (multi-section parts)
# ---------------------------------------------------------------------------

# Side faces used for attachment operations.
_SIDE_FACES = [">X", "<X", ">Y", "<Y"]


def _sample_block_tube(rng: Random, tier: int, config: dict, seed: int, index: int) -> PartIR:
    """
    Rectangular block with a hollow cylindrical tube projecting from one side
    face — like a hydraulic fitting body, sensor housing, or pipe stub.
    """
    w_def = _nice(rng, 20.0, 80.0)
    d_def = round(max(15.0, w_def * rng.uniform(0.5, 1.4)), 1)
    h_def = round(max(15.0, w_def * rng.uniform(0.5, 1.2)), 1)

    params: dict[str, ParamSpec] = {
        "block_w": ParamSpec(type="float", default=w_def, min=15.0, max=100.0, step=1.0,
                             group="Body", label="Block width (mm)"),
        "block_d": ParamSpec(type="float", default=d_def, min=10.0, max=100.0, step=1.0,
                             group="Body", label="Block depth (mm)"),
        "block_h": ParamSpec(type="float", default=h_def, min=10.0, max=100.0, step=1.0,
                             group="Body", label="Block height (mm)"),
    }
    ops: list[Operation] = [
        BoxOp(width=ref("block_w"), depth=ref("block_d"), height=ref("block_h"))
    ]

    face = rng.choice(_SIDE_FACES)
    face_dim_a, face_dim_b = (d_def, h_def) if face in (">X", "<X") else (w_def, h_def)
    min_face_dim = min(face_dim_a, face_dim_b)

    tube_d_def = round(min_face_dim * rng.uniform(0.20, 0.48), 1)
    tube_d_def = max(5.0, tube_d_def)
    tube_l_def = round(tube_d_def * rng.uniform(0.8, 2.5), 1)
    tube_l_def = max(5.0, tube_l_def)

    params["tube_d"] = ParamSpec(
        type="float", default=tube_d_def,
        min=round(tube_d_def * 0.4, 1),
        max=round(min(min_face_dim * 0.65, tube_d_def * 2.0), 1),
        step=1.0, group="Tube", label="Tube outer diameter (mm)",
    )
    params["tube_len"] = ParamSpec(
        type="float", default=tube_l_def,
        min=round(tube_d_def * 0.3, 1), max=round(tube_d_def * 4.0, 1),
        step=1.0, group="Tube", label="Tube length (mm)",
    )

    bore_expr = None
    if rng.random() < 0.70:
        bore_d_def = max(2.0, round(tube_d_def * rng.uniform(0.40, 0.70), 1))
        params["tube_bore"] = ParamSpec(
            type="float", default=bore_d_def,
            min=2.0, max=round(tube_d_def * 0.80, 1), step=0.5,
            group="Tube", label="Tube bore diameter (mm)",
        )
        bore_expr = ref("tube_bore")

    ops.append(AttachOp(
        profile=CircleProfile(diameter=ref("tube_d")),
        length=ref("tube_len"),
        bore_d=bore_expr,
        face=face,
    ))

    if tier >= 2:
        hole_op, hp = sample_corner_holes(rng, "block_w", "block_d", w_def, d_def)
        params.update(hp)
        ops.append(hole_op)

    if tier >= 3:
        fillet_op, fp = sample_fillet(rng, "block_w", "block_d", edge_selector="|Z")
        params.update(fp)
        ops.append(fillet_op)

    return PartIR(
        id=_make_id(seed, "compound", index),
        params=params, operations=ops,
        metadata=PartMetadata(seed=seed, generator_version=GENERATOR_VERSION,
                              family="compound", tier=tier, op_count=len(ops)),
    )


def _sample_block_tab(rng: Random, tier: int, config: dict, seed: int, index: int) -> PartIR:
    """
    Rectangular block with a flat mounting tab projecting from a side face.
    The tab has corner holes for bolting to a surface — like a motor mount
    flange, wall bracket, or junction plate.
    """
    w_def = _nice(rng, 25.0, 80.0)
    d_def = round(max(15.0, w_def * rng.uniform(0.5, 1.3)), 1)
    h_def = round(max(15.0, w_def * rng.uniform(0.5, 1.3)), 1)

    params: dict[str, ParamSpec] = {
        "block_w": ParamSpec(type="float", default=w_def, min=15.0, max=100.0, step=1.0,
                             group="Body", label="Block width (mm)"),
        "block_d": ParamSpec(type="float", default=d_def, min=10.0, max=100.0, step=1.0,
                             group="Body", label="Block depth (mm)"),
        "block_h": ParamSpec(type="float", default=h_def, min=10.0, max=100.0, step=1.0,
                             group="Body", label="Block height (mm)"),
    }
    ops: list[Operation] = [
        BoxOp(width=ref("block_w"), depth=ref("block_d"), height=ref("block_h"))
    ]

    face = rng.choice(_SIDE_FACES)
    if face in (">X", "<X"):
        tab_a_def = max(10.0, round(d_def * rng.uniform(0.55, 1.0), 1))
        tab_b_def = max(10.0, round(h_def * rng.uniform(0.55, 0.90), 1))
    else:
        tab_a_def = max(10.0, round(w_def * rng.uniform(0.55, 1.0), 1))
        tab_b_def = max(10.0, round(h_def * rng.uniform(0.55, 0.90), 1))
    tab_t_def = snap_to_stock_thickness(rng, 3.0, 10.0)

    params["tab_a"] = ParamSpec(
        type="float", default=tab_a_def,
        min=round(tab_a_def * 0.4, 1), max=round(tab_a_def * 1.6, 1),
        step=1.0, group="Tab", label="Tab width (mm)",
    )
    params["tab_b"] = ParamSpec(
        type="float", default=tab_b_def,
        min=round(tab_b_def * 0.4, 1), max=round(tab_b_def * 1.6, 1),
        step=1.0, group="Tab", label="Tab height (mm)",
    )
    params["tab_t"] = ParamSpec(
        type="float", default=tab_t_def, min=2.0, max=15.0, step=0.5,
        group="Tab", label="Tab thickness (mm)",
    )

    ops.append(AttachOp(
        profile=RectProfile(width=ref("tab_a"), depth=ref("tab_b")),
        length=ref("tab_t"),
        face=face,
    ))

    # Mounting holes drilled from the outer (tab end) face.
    if tier >= 1:
        tab_hole_max = round(min(tab_a_def, tab_b_def) * 0.22, 1)
        tab_hole_max = max(2.5, tab_hole_max)
        tab_hole_d_def = snap_fastener_diameter(rng, 2.5, tab_hole_max)
        params["tab_hole_d"] = ParamSpec(
            type="float", default=tab_hole_d_def,
            min=2.0, max=tab_hole_max, step=0.1,
            group="Tab", label="Tab hole diameter (mm)",
        )
        ops.append(HolesOp(
            diameter=ref("tab_hole_d"),
            placement="corners",
            spacing_x=scaled("tab_a", 0.60),
            spacing_y=scaled("tab_b", 0.60),
            face=face,
        ))

    if tier >= 2:
        if rng.random() < 0.5:
            op, p = sample_pocket(rng, "block_w", "block_d", "block_h", w_def, d_def, h_def)
        else:
            op, p = sample_corner_holes(rng, "block_w", "block_d", w_def, d_def,
                                        d_param="block_hole_d")
        params.update(p)
        ops.append(op)

    if tier >= 3:
        fillet_op, fp = sample_fillet(rng, "block_w", "block_d", edge_selector="|Z")
        params.update(fp)
        ops.append(fillet_op)

    return PartIR(
        id=_make_id(seed, "compound", index),
        params=params, operations=ops,
        metadata=PartMetadata(seed=seed, generator_version=GENERATOR_VERSION,
                              family="compound", tier=tier, op_count=len(ops)),
    )


def _sample_pedestal(rng: Random, tier: int, config: dict, seed: int, index: int) -> PartIR:
    """
    Wide flat base plate with a narrow cylindrical column rising from the top
    center — standoff, pedestal, support post, or spindle base.  The column
    may have a through-bore for a shaft or fastener.
    """
    base_w_def = _nice(rng, 40.0, 120.0)
    base_d_def = round(max(30.0, base_w_def * rng.uniform(0.6, 1.2)), 1)
    base_h_def = snap_to_stock_thickness(rng, 5.0, 20.0)

    min_base = min(base_w_def, base_d_def)
    col_d_def = max(8.0, round(min_base * rng.uniform(0.15, 0.35), 1))
    col_h_def = max(15.0, round(min_base * rng.uniform(0.5, 1.5), 1))

    params: dict[str, ParamSpec] = {
        "base_w": ParamSpec(type="float", default=base_w_def, min=30.0, max=160.0, step=1.0,
                            group="Base", label="Base width (mm)"),
        "base_d": ParamSpec(type="float", default=base_d_def, min=25.0, max=160.0, step=1.0,
                            group="Base", label="Base depth (mm)"),
        "base_h": ParamSpec(type="float", default=base_h_def, min=3.0, max=30.0, step=0.5,
                            group="Base", label="Base thickness (mm)"),
        "col_d": ParamSpec(
            type="float", default=col_d_def,
            min=round(col_d_def * 0.4, 1), max=round(min_base * 0.5, 1),
            step=1.0, group="Column", label="Column diameter (mm)",
        ),
        "col_h": ParamSpec(
            type="float", default=col_h_def,
            min=round(col_h_def * 0.3, 1), max=round(col_h_def * 3.0, 1),
            step=1.0, group="Column", label="Column height (mm)",
        ),
    }
    ops: list[Operation] = [
        BoxOp(width=ref("base_w"), depth=ref("base_d"), height=ref("base_h")),
    ]

    bore_expr = None
    if rng.random() < 0.55:
        bore_d_def = max(2.0, round(col_d_def * rng.uniform(0.30, 0.60), 1))
        params["col_bore"] = ParamSpec(
            type="float", default=bore_d_def,
            min=2.0, max=round(col_d_def * 0.75, 1), step=0.5,
            group="Column", label="Column bore diameter (mm)",
        )
        bore_expr = ref("col_bore")

    ops.append(AttachOp(
        profile=CircleProfile(diameter=ref("col_d")),
        length=ref("col_h"),
        bore_d=bore_expr,
        face=">Z",
    ))

    if tier >= 1:
        hole_op, hp = sample_corner_holes(rng, "base_w", "base_d", base_w_def, base_d_def)
        params.update(hp)
        ops.append(hole_op)

    if tier >= 2 and rng.random() < 0.5:
        fillet_op, fp = sample_fillet(rng, "base_w", "base_d", edge_selector="|Z")
        params.update(fp)
        ops.append(fillet_op)

    if tier >= 3:
        op, p = sample_chamfer(rng, edge_selector=">Z or <Z")
        params.update(p)
        ops.append(op)

    return PartIR(
        id=_make_id(seed, "compound", index),
        params=params, operations=ops,
        metadata=PartMetadata(seed=seed, generator_version=GENERATOR_VERSION,
                              family="compound", tier=tier, op_count=len(ops)),
    )


_COMPOUND_ARCHETYPES = ["block_tube", "block_tab", "pedestal"]


def sample_compound(rng: Random, tier: int, config: dict, seed: int, index: int) -> PartIR:
    """
    Multi-section compound parts — a primary body with one or more distinctly-
    styled secondary sections attached to it.

    Archetypes:
      block_tube  — rectangular block + hollow cylindrical tube off a side face
      block_tab   — rectangular block + flat mounting tab with bolt holes
      pedestal    — wide flat base plate + narrow cylindrical column on top
    """
    archetype = rng.choice(_COMPOUND_ARCHETYPES)
    if archetype == "block_tube":
        return _sample_block_tube(rng, tier, config, seed, index)
    elif archetype == "block_tab":
        return _sample_block_tab(rng, tier, config, seed, index)
    else:
        return _sample_pedestal(rng, tier, config, seed, index)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_FAMILY_SAMPLERS = {
    "plate": sample_plate,
    "bracket": sample_bracket,
    "revolved": sample_revolved,
    "block": sample_block,
    "compound": sample_compound,
    "enclosure": sample_enclosure,
    "flanged": sample_flanged,
    "ribbed": sample_ribbed,
    "profiled": sample_profiled,
}


def compose(
    seed: int,
    index: int = 0,
    config: dict | None = None,
    family: str | None = None,
    tier: int | None = None,
) -> PartIR:
    """
    Sample a single PartIR.

    seed:   Random seed for this part (typically derived from a global seed + index).
    index:  Part index within a batch — used for the part ID.
    config: Parsed TOML config dict; None uses built-in defaults.
    family: Force a specific family; None samples by weight.
    tier:   Force a specific tier; None samples by weight.
    """
    if config is None:
        config = {}

    rng = Random(seed)

    if family is None:
        fw = config.get("distribution", {}).get("families", DEFAULT_FAMILY_WEIGHTS)
        names = list(fw.keys())
        weights = [fw.get(n, 0.0) for n in names]
        # Keep only families we have a sampler for.
        pairs = [(n, w) for n, w in zip(names, weights) if n in _FAMILY_SAMPLERS]
        names, weights = zip(*pairs) if pairs else (list(_FAMILY_SAMPLERS.keys()), [1.0] * len(_FAMILY_SAMPLERS))
        family = _weighted_choice(rng, list(names), list(weights))

    if tier is None:
        tier = _sample_tier(rng, family, config)

    symmetry = _sample_symmetry(rng, config, family)

    # Inject symmetry into a shallow config copy so family samplers can read it
    # without changing every sampler signature.
    cfg = dict(config)
    cfg["_symmetry"] = symmetry

    sampler = _FAMILY_SAMPLERS.get(family, sample_plate)
    part = sampler(rng, tier, cfg, seed, index)
    part.metadata.symmetry = symmetry
    return part
