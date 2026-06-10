"""
Procedural, human-readable dimension summaries for a PartIR.

This walks a part's operations and resolves their dimension expressions against
the part's *default* parameters (the canonical instance that gets exported and
rendered), producing a short list of plain-English phrases plus a single
prompt-friendly text blob.  Example:

    Rectangular plate, 80 × 60 × 10 mm (W×D×H). Four Ø5 mm holes at the corners,
    60 × 40 mm spacing. Fillet radius 3 mm on the vertical edges.

There is NO AI in this step — it is pure, deterministic string formatting over
the IR, mirroring the structure of ``emit.emit_source`` / ``ir.*.to_code``.
Nothing here imports cadquery, so it runs without a CAD backend and without
re-executing geometry.
"""
from __future__ import annotations

from typing import Any

from . import ir
from .ir import PartIR

# ---------------------------------------------------------------------------
# Number / token formatting
# ---------------------------------------------------------------------------

def _num(x: Any) -> str:
    """Format a scalar dimension compactly: 80.0 -> '80', 5.50 -> '5.5'."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:  # NaN
        return str(x)
    if abs(f - round(f)) < 1e-6:
        return str(int(round(f)))
    # Trim to a sensible precision, then strip trailing zeros.
    return f"{f:.3f}".rstrip("0").rstrip(".")


def _mm(x: Any) -> str:
    return f"{_num(x)} mm"


def _dia(x: Any) -> str:
    return f"Ø{_num(x)} mm"


def _count_word(n: int) -> str:
    """Small integer counts read better as words at the start of a phrase."""
    words = {
        1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
        7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 12: "Twelve",
    }
    return words.get(n, str(n))


def _ev(expr: ir.Expr | None, params: dict[str, Any]) -> Any:
    """Evaluate an optional IR expression against concrete params (or None)."""
    if expr is None:
        return None
    try:
        return expr.evaluate(params)
    except Exception:
        return None


def _evi(expr: ir.Expr | None, params: dict[str, Any], default: int = 0) -> int:
    v = _ev(expr, params)
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


# Friendly names for CadQuery edge/face selector strings.
_FACE_NAMES = {
    ">Z": "top face", "<Z": "bottom face",
    ">X": "right face", "<X": "left face",
    ">Y": "back face", "<Y": "front face",
}
_EDGE_NAMES = {
    "|Z": "vertical edges", "|X": "edges along X", "|Y": "edges along Y",
    ">Z": "top edges", "<Z": "bottom edges",
}


def _face_name(sel: str) -> str:
    return _FACE_NAMES.get(sel, f"the {sel} face")


def _edge_name(sel: str) -> str:
    return _EDGE_NAMES.get(sel, f"the {sel} edges")


# ---------------------------------------------------------------------------
# Profile cross-sections
# ---------------------------------------------------------------------------

def _describe_profile(profile: ir.Profile, params: dict[str, Any]) -> str:
    if isinstance(profile, (ir.RectProfile, ir.RoundedRectProfile)):
        w = _ev(profile.width, params)
        d = _ev(profile.depth, params)
        return f"{_num(w)} × {_num(d)} mm rectangular section"
    if isinstance(profile, ir.CircleProfile):
        return f"{_dia(_ev(profile.diameter, params))} circular section"
    if isinstance(profile, ir.PolygonProfile):
        sides = _evi(profile.sides, params)
        across = _ev(profile.circumscribed_r, params)
        across = (across * 2) if isinstance(across, (int, float)) else across
        return f"{sides}-sided polygon section, {_mm(across)} across corners"
    if isinstance(profile, ir.SlotProfile):
        return f"{_num(_ev(profile.length, params))} × {_num(_ev(profile.width, params))} mm slot section"
    if isinstance(profile, ir.SketchedProfile):
        return _describe_sketched(profile, params)
    return "profile"


def _describe_sketched(profile: ir.SketchedProfile, params: dict[str, Any]) -> str:
    """Summarize a freeform sketch: segment mix + approximate bounding box."""
    kinds = [s.kind for s in profile.segments]
    straight = kinds.count("line")
    arcs = kinds.count("arc")
    curves = kinds.count("bezier") + kinds.count("spline")
    n = len(kinds)
    bits = []
    if straight:
        bits.append(f"{straight} straight")
    if arcs:
        bits.append(f"{arcs} {_plural(arcs, 'arc')}")
    if curves:
        bits.append(f"{curves} {_plural(curves, 'curve')}")
    mix = ", ".join(bits) if bits else f"{n} segments"
    # Bounding box from normalized extents scaled by the width/height params.
    w = _ev(ir.ParamRef(name=profile.w_param), params)
    h = _ev(ir.ParamRef(name=profile.h_param), params)
    xs = [profile.start[0]] + [s.x for s in profile.segments]
    ys = [profile.start[1]] + [s.y for s in profile.segments]
    bbox = ""
    if isinstance(w, (int, float)) and isinstance(h, (int, float)):
        bw = (max(xs) - min(xs)) * w
        bh = (max(ys) - min(ys)) * h
        bbox = f", ~{_num(bw)} × {_num(bh)} mm"
    return f"freeform sketched section ({n} segments: {mix}){bbox}"


# ---------------------------------------------------------------------------
# Hole counting / placement (shared by the hole op variants)
# ---------------------------------------------------------------------------

def _hole_count(op: Any, params: dict[str, Any]) -> int:
    placement = getattr(op, "placement", "corners")
    if placement == "corners":
        return 4
    if placement == "grid":
        return max(1, _evi(op.nx, params, 2)) * max(1, _evi(op.ny, params, 2))
    if placement == "bolt_circle":
        return max(1, _evi(op.n_bolts, params, 4))
    if placement == "staggered":
        nx = max(2, _evi(op.nx, params, 3))
        return 2 * nx - 1
    return 0


def _placement_phrase(op: Any, params: dict[str, Any]) -> str:
    placement = getattr(op, "placement", "corners")
    if placement == "corners":
        sx, sy = _ev(op.spacing_x, params), _ev(op.spacing_y, params)
        if sx is not None and sy is not None:
            return f"at the corners, {_num(sx)} × {_num(sy)} mm spacing"
        return "at the corners"
    if placement == "grid":
        nx, ny = max(1, _evi(op.nx, params, 2)), max(1, _evi(op.ny, params, 2))
        sx, sy = _ev(op.spacing_x, params), _ev(op.spacing_y, params)
        sp = f", {_num(sx)} × {_num(sy)} mm pitch" if sx is not None and sy is not None else ""
        return f"in a {nx}×{ny} grid{sp}"
    if placement == "bolt_circle":
        r = _ev(op.bolt_circle_r, params)
        return f"on a {_dia(r * 2)} bolt circle" if isinstance(r, (int, float)) else "on a bolt circle"
    if placement == "staggered":
        sx, sy = _ev(op.spacing_x, params), _ev(op.spacing_y, params)
        sp = f", {_num(sx)} × {_num(sy)} mm pitch" if sx is not None and sy is not None else ""
        return f"in two staggered rows{sp}"
    return ""


def _plural(n: int, noun: str) -> str:
    return noun if n == 1 else noun + "s"


# ---------------------------------------------------------------------------
# Per-operation descriptions
# ---------------------------------------------------------------------------

def _describe_op(op: Any, params: dict[str, Any]) -> str | None:  # noqa: C901
    if isinstance(op, ir.BoxOp):
        w, d, h = _ev(op.width, params), _ev(op.depth, params), _ev(op.height, params)
        return f"Rectangular body, {_num(w)} × {_num(d)} × {_num(h)} mm (W×D×H)."

    if isinstance(op, ir.ExtrudeOp):
        return f"{_describe_profile(op.profile, params)}, extruded {_mm(_ev(op.distance, params))} thick."

    if isinstance(op, ir.RevolveOp):
        od, h = _ev(op.outer_d, params), _ev(op.height, params)
        s = f"Revolved body, {_dia(od)} × {_mm(h)} tall"
        extra = []
        if op.inner_d is not None:
            extra.append(f"{_dia(_ev(op.inner_d, params))} bore")
        if op.flange_d is not None and op.flange_h is not None:
            extra.append(f"{_dia(_ev(op.flange_d, params))} × {_mm(_ev(op.flange_h, params))} flange")
        if op.groove_depth is not None and op.groove_w is not None:
            extra.append(f"{_mm(_ev(op.groove_w, params))}-wide groove {_mm(_ev(op.groove_depth, params))} deep")
        if extra:
            s += " with " + ", ".join(extra)
        return s + "."

    if isinstance(op, ir.SketchedRevolveOp):
        return f"Revolved organic body from a {_describe_sketched(op.half_profile, params)} half-silhouette."

    if isinstance(op, ir.HolesOp):
        n = _hole_count(op, params)
        d = _ev(op.diameter, params)
        place = _placement_phrase(op, params)
        if op.shape == "square":
            feature = f"{_num(d)} mm square {_plural(n, 'hole')}"
        elif op.shape == "slot":
            sl = _ev(op.slot_len, params) if op.slot_len is not None else None
            size = f"{_num(sl)} × {_num(d)} mm" if sl is not None else f"{_num(d)} mm"
            feature = f"{size} {_plural(n, 'slot')}"
        else:
            feature = f"{_dia(d)} {_plural(n, 'hole')}"
        return f"{_count_word(n)} {feature} {place}."

    if isinstance(op, ir.CounterboreHolesOp):
        n = _hole_count(op, params)
        d = _ev(op.diameter, params)
        cbd, cbdp = _ev(op.cb_diameter, params), _ev(op.cb_depth, params)
        place = _placement_phrase(op, params)
        return (f"{_count_word(n)} {_dia(d)} counterbored holes "
                f"({_dia(cbd)} × {_mm(cbdp)} counterbore) {place}.")

    if isinstance(op, ir.CountersinkHolesOp):
        n = _hole_count(op, params)
        d = _ev(op.diameter, params)
        ang = _ev(op.cs_angle, params)
        place = _placement_phrase(op, params)
        return f"{_count_word(n)} {_dia(d)} countersunk holes ({_num(ang)}°) {place}."

    if isinstance(op, ir.FilletOp):
        s = f"Fillet radius {_mm(_ev(op.radius, params))} on {_edge_name(op.edge_selector)}"
        if op.enabled is not None and not _ev(op.enabled, params):
            s += " (optional, off by default)"
        return s + "."

    if isinstance(op, ir.ChamferOp):
        s = f"Chamfer {_mm(_ev(op.distance, params))} on {_edge_name(op.edge_selector)}"
        if op.enabled is not None and not _ev(op.enabled, params):
            s += " (optional, off by default)"
        return s + "."

    if isinstance(op, ir.ShellOp):
        return f"Hollowed to {_mm(_ev(op.thickness, params))} wall thickness, open at {_face_name(op.open_face)}."

    if isinstance(op, ir.PocketOp):
        w, d, pd = _ev(op.width, params), _ev(op.depth, params), _ev(op.pocket_depth, params)
        return f"Pocket {_num(w)} × {_num(d)} mm, {_mm(pd)} deep, in {_face_name(op.face)}."

    if isinstance(op, ir.BossOp):
        s = f"Boss {_dia(_ev(op.diameter, params))} × {_mm(_ev(op.height, params))} tall on {_face_name(op.face)}"
        if op.bore_d is not None:
            s += f" with a {_dia(_ev(op.bore_d, params))} bore"
        return s + "."

    if isinstance(op, ir.RibsOp):
        n = _evi(op.count, params, 1)
        return (f"{_count_word(n)} ribs, {_mm(_ev(op.rib_thickness, params))} thick, "
                f"{_mm(_ev(op.rib_height, params))} tall.")

    if isinstance(op, ir.AttachOp):
        s = f"{_describe_profile(op.profile, params)} attachment, {_mm(_ev(op.length, params))} long, from {_face_name(op.face)}"
        if op.bore_d is not None:
            s += f" with a {_dia(_ev(op.bore_d, params))} bore"
        return s + "."

    if isinstance(op, (ir.LBracketOp, ir.CBracketOp, ir.ZBracketOp)):
        return _describe_bracket(op, params)

    if isinstance(op, (ir.GearOp, ir.ThreadedOp)):
        # Reuse the structured descriptor these ops already expose.
        return _describe_from_descriptor(op.describe(params))

    return None


def _describe_bracket(op: Any, params: dict[str, Any]) -> str:
    kind = {"lbracket": "L-bracket", "cbracket": "C-channel bracket",
            "zbracket": "Z-bracket"}[op.type]
    bl, vl = _ev(op.base_len, params), _ev(op.vert_len, params)
    w, t = _ev(op.width, params), _ev(op.thickness, params)
    legs = f"base leg {_mm(bl)}, vertical leg {_mm(vl)}"
    if isinstance(op, ir.ZBracketOp) and op.top_len is not None:
        legs += f", top leg {_mm(_ev(op.top_len, params))}"
    s = f"{kind}: {legs}, {_mm(w)} wide, {_mm(t)} thick"
    if op.hole_d is not None and _ev(op.hole_d, params):
        nh = max(1, _evi(op.n_holes, params, 1))
        s += f"; {nh} × {_dia(_ev(op.hole_d, params))} {_plural(nh, 'hole')} per leg"
    if op.gusset_t is not None:
        enabled = op.gusset_enabled is None or _ev(op.gusset_enabled, params)
        s += f"; {_mm(_ev(op.gusset_t, params))} corner gusset" + ("" if enabled else " (optional)")
    return s + "."


def _describe_from_descriptor(d: dict[str, Any]) -> str:
    """Render the gear / threaded structured descriptor as a phrase."""
    fam = d.get("family")
    if fam == "gear":
        kind = d.get("kind", "spur")
        s = (f"{kind.capitalize()} gear, module {_num(d.get('module'))} mm, "
             f"{d.get('teeth')} teeth (pitch {_dia(d.get('pitch_diameter'))}), "
             f"face width {_mm(d.get('face_width'))}")
        if "helix_angle_deg" in d:
            s += f", {_num(d['helix_angle_deg'])}° helix"
            if d.get("herringbone"):
                s += " (herringbone)"
        if "cone_angle_deg" in d:
            s += f", {_num(d['cone_angle_deg'])}° pitch cone"
        if "bore_d" in d:
            s += f", {_dia(d['bore_d'])} bore"
        if "hub_d" in d and "hub_h" in d:
            s += f", {_dia(d['hub_d'])} × {_mm(d['hub_h'])} hub"
        return s + "."
    if fam == "threaded":
        kind = "internal" if not d.get("external", True) else "external"
        s = f"{d.get('designation')} {kind} thread, {_mm(d.get('length'))} long"
        return s + "."
    return ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def describe_dimensions(part: PartIR) -> dict[str, Any]:
    """
    Build the human-readable dimension summary for a part's canonical (default)
    instance.

    Returns a dict with:
      - ``lines``: list[str], one plain-English phrase per dimensional feature
      - ``text``:  the lines joined into a single prompt-friendly blob
    """
    params = part.default_params()
    lines: list[str] = []
    for op in part.operations:
        try:
            phrase = _describe_op(op, params)
        except Exception:
            phrase = None
        if phrase:
            lines.append(phrase)
    return {"lines": lines, "text": " ".join(lines)}
