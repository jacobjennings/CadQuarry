"""
Feature / operation samplers.

Each sampler returns an Operation (or list of Operations) plus a params_update
dict to be merged into the part's PARAMS schema.  The samplers take explicit
arguments for the parent dimensions so they can set valid, proportional ranges.
"""
from __future__ import annotations

from random import Random

from .ir import (
    BossOp,
    ChamferOp,
    CounterboreHolesOp,
    CountersinkHolesOp,
    FilletOp,
    HolesOp,
    MinExpr,
    Operation,
    ParamSpec,
    PocketOp,
    RibsOp,
    lit,
    ref,
    scaled,
    min2,
)
from .profiles import (
    FASTENER_CLEARANCE_MM,
    snap_fastener_diameter,
)


# ---------------------------------------------------------------------------
# Holes
# ---------------------------------------------------------------------------

def sample_corner_holes(
    rng: Random,
    parent_w_param: str,
    parent_d_param: str,
    w_default: float,
    d_default: float,
    margin_factor: float = 0.15,
    d_param: str = "hole_d",
    hole_type: str = "simple",          # simple | counterbore | countersink
    cb_d_param: str = "cb_d",
    cb_depth_param: str = "cb_depth",
    cs_angle_param: str = "cs_angle",
) -> tuple[Operation, dict[str, ParamSpec]]:
    """
    Four corner holes placed at a construction rect = parent * (1 - 2*margin).
    Hole diameter constrained so it fits inside the margin.
    """
    min_dim = min(w_default, d_default)
    hole_d_max = min_dim * (margin_factor * 1.5)
    hole_d_max = min(hole_d_max, 13.0)  # never exceed M12 clearance
    hole_d_min = 2.0

    hole_d_def = snap_fastener_diameter(rng, hole_d_min, hole_d_max)

    params: dict[str, ParamSpec] = {
        d_param: ParamSpec(
            type="float", default=hole_d_def,
            min=round(hole_d_min, 1), max=round(hole_d_max, 1), step=0.1,
            group="Holes", label="Hole diameter (mm)",
        ),
    }

    # Construction rect is 70% of part dims — scales proportionally
    spacing_x = scaled(parent_w_param, 0.70)
    spacing_y = scaled(parent_d_param, 0.70)

    if hole_type == "counterbore":
        cb_d_def = round(hole_d_def * 2.0, 1)
        cb_depth_def = round(hole_d_def * 0.8, 1)
        params[cb_d_param] = ParamSpec(
            type="float", default=cb_d_def,
            min=round(hole_d_def * 1.5, 1), max=round(hole_d_def * 3.0, 1), step=0.5,
            group="Holes", label="Counterbore diameter (mm)",
        )
        params[cb_depth_param] = ParamSpec(
            type="float", default=cb_depth_def,
            min=round(cb_depth_def * 0.5, 1), max=round(cb_depth_def * 2.0, 1), step=0.5,
            group="Holes", label="Counterbore depth (mm)",
        )
        op: Operation = CounterboreHolesOp(
            diameter=ref(d_param),
            cb_diameter=ref(cb_d_param),
            cb_depth=ref(cb_depth_param),
            placement="corners",
            spacing_x=spacing_x,
            spacing_y=spacing_y,
        )
    elif hole_type == "countersink":
        params[cs_angle_param] = ParamSpec(
            type="float", default=82.0, min=60.0, max=120.0, step=15.0,
            group="Holes", label="Countersink angle (°)",
        )
        op = CountersinkHolesOp(
            diameter=ref(d_param),
            cs_angle=ref(cs_angle_param),
            placement="corners",
            spacing_x=spacing_x,
            spacing_y=spacing_y,
        )
    else:
        op = HolesOp(
            diameter=ref(d_param),
            placement="corners",
            spacing_x=spacing_x,
            spacing_y=spacing_y,
        )

    return op, params


def sample_grid_holes(
    rng: Random,
    parent_w_param: str,
    parent_d_param: str,
    w_default: float,
    d_default: float,
    nx_max: int = 4,
    ny_max: int = 4,
    d_param: str = "hole_d",
    nx_param: str = "hole_nx",
    ny_param: str = "hole_ny",
) -> tuple[Operation, dict[str, ParamSpec]]:
    nx_def = rng.randint(2, min(nx_max, 4))
    ny_def = rng.randint(2, min(ny_max, 4))

    min_dim = min(w_default, d_default)
    hole_d_max = min_dim * 0.12
    hole_d_min = 2.0
    hole_d_def = snap_fastener_diameter(rng, hole_d_min, hole_d_max)

    # Spacing so the grid fits within 80% of the part
    sx_def = round((w_default * 0.8) / max(nx_def - 1, 1), 1)
    sy_def = round((d_default * 0.8) / max(ny_def - 1, 1), 1)

    params: dict[str, ParamSpec] = {
        d_param: ParamSpec(
            type="float", default=hole_d_def,
            min=round(hole_d_min, 1), max=round(hole_d_max, 1), step=0.1,
            group="Holes", label="Hole diameter (mm)",
        ),
        nx_param: ParamSpec(
            type="int", default=nx_def, min=1, max=nx_max, step=1,
            group="Holes", label="Holes across",
        ),
        ny_param: ParamSpec(
            type="int", default=ny_def, min=1, max=ny_max, step=1,
            group="Holes", label="Holes deep",
        ),
    }
    op = HolesOp(
        diameter=ref(d_param),
        placement="grid",
        spacing_x=lit(sx_def),
        spacing_y=lit(sy_def),
        nx=ref(nx_param),
        ny=ref(ny_param),
    )
    return op, params


def sample_bolt_circle(
    rng: Random,
    parent_d_param: str,
    d_default: float,
    n_choices: list[int] | None = None,
    d_param: str = "bolt_d",
    r_param: str = "bolt_r",
    n_param: str = "n_bolts",
) -> tuple[Operation, dict[str, ParamSpec]]:
    if n_choices is None:
        n_choices = [4, 6, 8, 12]
    n_def = rng.choice(n_choices)

    r_def = round(d_default * 0.30, 1)
    r_min = round(d_default * 0.20, 1)
    r_max = round(d_default * 0.45, 1)

    hole_d_def = snap_fastener_diameter(rng, 3.0, 8.5)

    params: dict[str, ParamSpec] = {
        d_param: ParamSpec(
            type="float", default=hole_d_def,
            min=3.0, max=13.0, step=0.1,
            group="Holes", label="Bolt hole diameter (mm)",
        ),
        r_param: ParamSpec(
            type="float", default=r_def,
            min=r_min, max=r_max, step=1.0,
            group="Holes", label="Bolt circle radius (mm)",
        ),
        n_param: ParamSpec(
            type="int", default=n_def, min=3, max=12, step=1,
            group="Holes", label="Number of bolts",
        ),
    }
    op = HolesOp(
        diameter=ref(d_param),
        placement="bolt_circle",
        bolt_circle_r=ref(r_param),
        n_bolts=ref(n_param),
    )
    return op, params


# ---------------------------------------------------------------------------
# Fillet / chamfer
# ---------------------------------------------------------------------------

def sample_fillet(
    rng: Random,
    parent_w_param: str,
    parent_d_param: str,
    edge_selector: str = "|Z",
    factor_lo: float = 0.04,
    factor_hi: float = 0.12,
    r_param: str = "fillet_r",
    enabled_param: str = "filleted",
    always_on: bool = False,
) -> tuple[FilletOp, dict[str, ParamSpec]]:
    factor = rng.uniform(factor_lo, factor_hi)
    params: dict[str, ParamSpec] = {}

    if always_on:
        radius_expr = min2(parent_w_param, parent_d_param, factor)
        enabled_expr = None
    else:
        params[enabled_param] = ParamSpec(
            type="bool", default=True,
            group="Body", label="Fillet corners",
        )
        radius_expr = min2(parent_w_param, parent_d_param, factor)
        enabled_expr = ref(enabled_param)

    op = FilletOp(
        radius=radius_expr,
        edge_selector=edge_selector,
        enabled=enabled_expr,
    )
    return op, params


def sample_chamfer(
    rng: Random,
    dist_lo: float = 0.5,
    dist_hi: float = 3.0,
    edge_selector: str = ">Z",
    dist_param: str = "chamfer_d",
    enabled_param: str = "chamfered",
) -> tuple[ChamferOp, dict[str, ParamSpec]]:
    dist_def = round(rng.uniform(dist_lo, dist_hi), 1)
    params: dict[str, ParamSpec] = {
        dist_param: ParamSpec(
            type="float", default=dist_def,
            min=round(dist_lo, 1), max=round(dist_hi, 1), step=0.5,
            group="Body", label="Chamfer distance (mm)",
        ),
        enabled_param: ParamSpec(
            type="bool", default=True,
            group="Body", label="Chamfer edges",
        ),
    }
    return ChamferOp(
        distance=ref(dist_param),
        edge_selector=edge_selector,
        enabled=ref(enabled_param),
    ), params


# ---------------------------------------------------------------------------
# Pocket
# ---------------------------------------------------------------------------

def sample_pocket(
    rng: Random,
    parent_w_param: str,
    parent_d_param: str,
    parent_h_param: str,
    w_default: float,
    d_default: float,
    h_default: float,
    pw_param: str = "pocket_w",
    pd_param: str = "pocket_d",
    ph_param: str = "pocket_h",
) -> tuple[PocketOp, dict[str, ParamSpec]]:
    pw_def = round(w_default * rng.uniform(0.4, 0.7), 1)
    pd_def = round(d_default * rng.uniform(0.4, 0.7), 1)
    ph_def = round(h_default * rng.uniform(0.3, 0.6), 1)

    params: dict[str, ParamSpec] = {
        pw_param: ParamSpec(
            type="float", default=pw_def,
            min=round(pw_def * 0.5, 1), max=round(w_default * 0.8, 1), step=1.0,
            group="Pocket", label="Pocket width (mm)",
        ),
        pd_param: ParamSpec(
            type="float", default=pd_def,
            min=round(pd_def * 0.5, 1), max=round(d_default * 0.8, 1), step=1.0,
            group="Pocket", label="Pocket depth (mm)",
        ),
        ph_param: ParamSpec(
            type="float", default=ph_def,
            min=round(ph_def * 0.4, 1), max=round(h_default * 0.85, 1), step=0.5,
            group="Pocket", label="Pocket height (mm)",
        ),
    }
    return PocketOp(
        width=ref(pw_param),
        depth=ref(pd_param),
        pocket_depth=ref(ph_param),
    ), params


# ---------------------------------------------------------------------------
# Boss
# ---------------------------------------------------------------------------

def sample_boss(
    rng: Random,
    parent_h_param: str,
    h_default: float,
    bd_param: str = "boss_d",
    bh_param: str = "boss_h",
    bore_param: str = "boss_bore_d",
    with_bore: bool | None = None,
) -> tuple[BossOp, dict[str, ParamSpec]]:
    bd_def = round(rng.uniform(8.0, 25.0), 1)
    bh_def = round(rng.uniform(4.0, 12.0), 1)
    if with_bore is None:
        with_bore = rng.random() < 0.6

    params: dict[str, ParamSpec] = {
        bd_param: ParamSpec(
            type="float", default=bd_def,
            min=round(bd_def * 0.5, 1), max=round(bd_def * 2.0, 1), step=1.0,
            group="Boss", label="Boss diameter (mm)",
        ),
        bh_param: ParamSpec(
            type="float", default=bh_def,
            min=round(bh_def * 0.4, 1), max=round(bh_def * 2.5, 1), step=1.0,
            group="Boss", label="Boss height (mm)",
        ),
    }
    bore_expr = None
    if with_bore:
        bore_def = snap_fastener_diameter(rng, 3.0, bd_def * 0.6)
        params[bore_param] = ParamSpec(
            type="float", default=bore_def,
            min=2.0, max=round(bd_def * 0.7, 1), step=0.1,
            group="Boss", label="Bore diameter (mm)",
        )
        bore_expr = ref(bore_param)

    return BossOp(
        diameter=ref(bd_param),
        height=ref(bh_param),
        bore_d=bore_expr,
    ), params


# ---------------------------------------------------------------------------
# Ribs
# ---------------------------------------------------------------------------

def sample_ribs(
    rng: Random,
    parent_w_param: str,
    w_default: float,
    count_param: str = "rib_count",
    height_param: str = "rib_h",
    thickness_param: str = "rib_t",
) -> tuple[RibsOp, dict[str, ParamSpec]]:
    n_def = rng.randint(2, 6)
    rh_def = round(rng.uniform(5.0, 20.0), 1)
    rt_def = round(rng.uniform(2.0, 6.0), 1)

    params: dict[str, ParamSpec] = {
        count_param: ParamSpec(
            type="int", default=n_def, min=2, max=8, step=1,
            group="Ribs", label="Rib count",
        ),
        height_param: ParamSpec(
            type="float", default=rh_def,
            min=round(rh_def * 0.4, 1), max=round(rh_def * 2.0, 1), step=1.0,
            group="Ribs", label="Rib height (mm)",
        ),
        thickness_param: ParamSpec(
            type="float", default=rt_def,
            min=1.0, max=round(rt_def * 2.5, 1), step=0.5,
            group="Ribs", label="Rib thickness (mm)",
        ),
    }
    return RibsOp(
        count=ref(count_param),
        rib_height=ref(height_param),
        rib_thickness=ref(thickness_param),
        span=scaled(parent_w_param, 0.8),
    ), params
