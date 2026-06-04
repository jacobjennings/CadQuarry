"""
Intermediate representation for CadQuarry programs.

PartIR is the single source of truth. From it we derive:
  - parametric CadQuery source code (emit.py)
  - PARAMS schema sidecar (emit.py)
  - provenance metadata (emit.py)
  - geometry signature for dedup (filter.py)

Design rule: nothing in this module imports cadquery. The IR is pure
Python / pydantic and can be loaded, inspected, and diffed without a
working CadQuery installation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Expression types
#
# An Expr represents a value that may depend on parameters. The to_code()
# method returns a Python expression string valid inside build(p), and
# evaluate() computes the value given a concrete param dict.
# ---------------------------------------------------------------------------


class LiteralExpr(BaseModel):
    type: Literal["literal"] = "literal"
    value: float | int | bool | str

    def to_code(self) -> str:
        if isinstance(self.value, bool):
            return "True" if self.value else "False"
        if isinstance(self.value, str):
            return repr(self.value)
        return repr(self.value)

    def evaluate(self, p: dict[str, Any]) -> Any:
        return self.value


class ParamRef(BaseModel):
    """Direct reference to a named parameter: p["name"]."""
    type: Literal["param"] = "param"
    name: str

    def to_code(self) -> str:
        return f'p["{self.name}"]'

    def evaluate(self, p: dict[str, Any]) -> Any:
        return p[self.name]


class ScaledExpr(BaseModel):
    """p["param"] * factor — for proportional dimensions."""
    type: Literal["scaled"] = "scaled"
    param: str
    factor: float

    def to_code(self) -> str:
        return f'p["{self.param}"] * {self.factor!r}'

    def evaluate(self, p: dict[str, Any]) -> Any:
        return p[self.param] * self.factor


class MinExpr(BaseModel):
    """min(p["a"], p["b"]) * factor — safe radius/fillet expressions."""
    type: Literal["min2"] = "min2"
    param_a: str
    param_b: str
    factor: float = 1.0

    def to_code(self) -> str:
        inner = f'min(p["{self.param_a}"], p["{self.param_b}"])'
        return f"{inner} * {self.factor!r}" if self.factor != 1.0 else inner

    def evaluate(self, p: dict[str, Any]) -> Any:
        return min(p[self.param_a], p[self.param_b]) * self.factor


class ClampExpr(BaseModel):
    """min(p["param"], limit) — hard upper bound."""
    type: Literal["clamp"] = "clamp"
    param: str
    limit: float

    def to_code(self) -> str:
        return f'min(p["{self.param}"], {self.limit!r})'

    def evaluate(self, p: dict[str, Any]) -> Any:
        return min(p[self.param], self.limit)


class HalfExpr(BaseModel):
    """p["param"] / 2 — diameter → radius conversion."""
    type: Literal["half"] = "half"
    param: str

    def to_code(self) -> str:
        return f'p["{self.param}"] / 2'

    def evaluate(self, p: dict[str, Any]) -> Any:
        return p[self.param] / 2


Expr = Annotated[
    Union[LiteralExpr, ParamRef, ScaledExpr, MinExpr, ClampExpr, HalfExpr],
    Field(discriminator="type"),
]


# Convenience constructors (no need to remember class names in compose.py)
def lit(v: float | int | bool | str) -> LiteralExpr:
    return LiteralExpr(value=v)


def ref(name: str) -> ParamRef:
    return ParamRef(name=name)


def scaled(param: str, factor: float) -> ScaledExpr:
    return ScaledExpr(param=param, factor=factor)


def min2(param_a: str, param_b: str, factor: float = 1.0) -> MinExpr:
    return MinExpr(param_a=param_a, param_b=param_b, factor=factor)


def half(param: str) -> HalfExpr:
    return HalfExpr(param=param)


# ---------------------------------------------------------------------------
# Parameter schema
# ---------------------------------------------------------------------------


class ParamSpec(BaseModel):
    """Schema for one parameter as it appears in the emitted PARAMS dict."""
    type: Literal["float", "int", "bool", "enum"]
    default: float | int | bool | str
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    choices: list[str] | None = None
    group: str = "Body"
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "default": self.default}
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.step is not None:
            d["step"] = self.step
        if self.choices is not None:
            d["choices"] = self.choices
        d["group"] = self.group
        d["label"] = self.label
        return d


# ---------------------------------------------------------------------------
# Profile IR (2D cross-sections)
# ---------------------------------------------------------------------------


class RectProfile(BaseModel):
    type: Literal["rect"] = "rect"
    width: Expr
    depth: Expr

    def to_code(self) -> str:
        return f".rect({self.width.to_code()}, {self.depth.to_code()})"


class RoundedRectProfile(BaseModel):
    type: Literal["rounded_rect"] = "rounded_rect"
    width: Expr
    depth: Expr
    # radius applied after extrude as vertical-edge fillet — emitted separately

    def to_code(self) -> str:
        return f".rect({self.width.to_code()}, {self.depth.to_code()})"


class CircleProfile(BaseModel):
    type: Literal["circle"] = "circle"
    diameter: Expr

    def to_code(self) -> str:
        return f".circle({self.diameter.to_code()} / 2)"


class PolygonProfile(BaseModel):
    """Regular polygon, circumscribed radius given."""
    type: Literal["polygon"] = "polygon"
    sides: Expr
    circumscribed_r: Expr

    def to_code(self) -> str:
        return f".polygon({self.sides.to_code()}, {self.circumscribed_r.to_code()} * 2)"


class SlotProfile(BaseModel):
    """Oblong / slot shape."""
    type: Literal["slot"] = "slot"
    length: Expr
    width: Expr

    def to_code(self) -> str:
        return f".slot2D({self.length.to_code()}, {self.width.to_code()})"


Profile = Annotated[
    Union[RectProfile, RoundedRectProfile, CircleProfile, PolygonProfile, SlotProfile],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Operation IR
#
# Operations are applied sequentially. The first operation creates the base
# solid; subsequent operations modify it. Each has a to_code() method that
# returns a list of Python source lines (strings, no trailing newlines).
# ---------------------------------------------------------------------------


class BoxOp(BaseModel):
    """Simple box — the most common base solid for plates and blocks."""
    type: Literal["box"] = "box"
    width: Expr
    depth: Expr
    height: Expr

    def to_code(self) -> list[str]:
        return [
            f"    result = cq.Workplane('XY').box("
            f"{self.width.to_code()}, {self.depth.to_code()}, {self.height.to_code()})"
        ]


class ExtrudeOp(BaseModel):
    """Extrude a 2D profile."""
    type: Literal["extrude"] = "extrude"
    profile: Profile
    distance: Expr
    plane: str = "XY"

    def to_code(self) -> list[str]:
        return [
            f"    result = ("
            f"cq.Workplane({self.plane!r})"
            f"{self.profile.to_code()}"
            f".extrude({self.distance.to_code()}))"
        ]


class RevolveOp(BaseModel):
    """
    Solid of revolution around the Z axis.
    Draws a half-profile in the XZ plane and revolves it.
    outer_d / inner_d are diameters (converted to radii in codegen).
    """
    type: Literal["revolve"] = "revolve"
    outer_d: Expr
    height: Expr
    inner_d: Expr | None = None        # hollow bore
    flange_d: Expr | None = None       # wider flange at base
    flange_h: Expr | None = None       # height of flange portion
    chamfer_d: Expr | None = None      # end-chamfer diameter offset
    groove_depth: Expr | None = None   # circumferential groove depth
    groove_w: Expr | None = None       # groove width
    groove_pos: Expr | None = None     # groove position along height

    def to_code(self) -> list[str]:  # noqa: C901
        od = f"{self.outer_d.to_code()} / 2"
        h = self.height.to_code()
        lines = ["    result = (", "        cq.Workplane('XZ')"]

        if self.inner_d is not None:
            id_ = f"{self.inner_d.to_code()} / 2"
            lines.append(f"        .moveTo({id_}, 0)")
        else:
            lines.append("        .moveTo(0, 0)")

        if self.flange_d is not None and self.flange_h is not None:
            fd = f"{self.flange_d.to_code()} / 2"
            fh = self.flange_h.to_code()
            lines.append(f"        .lineTo({fd}, 0)")
            lines.append(f"        .lineTo({fd}, {fh})")
            lines.append(f"        .lineTo({od}, {fh})")
        else:
            lines.append(f"        .lineTo({od}, 0)")

        if self.groove_depth is not None and self.groove_w is not None and self.groove_pos is not None:
            gd = self.groove_depth.to_code()
            gw = self.groove_w.to_code()
            gp = self.groove_pos.to_code()
            lines.append(f"        .lineTo({od}, {gp} - {gw} / 2)")
            lines.append(f"        .lineTo({od} - {gd}, {gp} - {gw} / 2)")
            lines.append(f"        .lineTo({od} - {gd}, {gp} + {gw} / 2)")
            lines.append(f"        .lineTo({od}, {gp} + {gw} / 2)")
            lines.append(f"        .lineTo({od}, {h})")
        else:
            lines.append(f"        .lineTo({od}, {h})")

        if self.inner_d is not None:
            id_ = f"{self.inner_d.to_code()} / 2"
            lines.append(f"        .lineTo({id_}, {h})")

        lines.append("        .close()")
        lines.append("        .revolve()")
        lines.append("    )")
        return lines


class HolesOp(BaseModel):
    """
    Drill holes on a face.
    placement: 'corners' (rect grid at 4 corners), 'grid' (rarray), 'bolt_circle' (polarArray).
    """
    type: Literal["holes"] = "holes"
    diameter: Expr
    placement: Literal["corners", "grid", "bolt_circle"] = "corners"
    # corners / grid shared — spacing_x/y define the construction rect
    spacing_x: Expr | None = None
    spacing_y: Expr | None = None
    # grid only
    nx: Expr | None = None
    ny: Expr | None = None
    # bolt_circle only
    bolt_circle_r: Expr | None = None
    n_bolts: Expr | None = None
    face: str = ">Z"

    def to_code(self) -> list[str]:
        d = self.diameter.to_code()
        face = self.face
        if self.placement == "corners":
            sx = self.spacing_x.to_code() if self.spacing_x else "20"
            sy = self.spacing_y.to_code() if self.spacing_y else "20"
            return [
                f"    result = (",
                f"        result.faces({face!r}).workplane()",
                f"        .rect({sx}, {sy}, forConstruction=True)",
                f"        .vertices().hole({d})",
                f"    )",
            ]
        elif self.placement == "grid":
            sx = self.spacing_x.to_code() if self.spacing_x else "10"
            sy = self.spacing_y.to_code() if self.spacing_y else "10"
            nx = self.nx.to_code() if self.nx else "2"
            ny = self.ny.to_code() if self.ny else "2"
            return [
                f"    result = (",
                f"        result.faces({face!r}).workplane()",
                f"        .rarray({sx}, {sy}, {nx}, {ny})",
                f"        .hole({d})",
                f"    )",
            ]
        else:  # bolt_circle
            r = self.bolt_circle_r.to_code() if self.bolt_circle_r else "15"
            n = self.n_bolts.to_code() if self.n_bolts else "4"
            return [
                f"    result = (",
                f"        result.faces({face!r}).workplane()",
                f"        .polarArray({r}, 0, 360, {n})",
                f"        .hole({d})",
                f"    )",
            ]


class CounterboreHolesOp(BaseModel):
    type: Literal["counterbore"] = "counterbore"
    diameter: Expr
    cb_diameter: Expr
    cb_depth: Expr
    placement: Literal["corners", "grid", "bolt_circle"] = "corners"
    spacing_x: Expr | None = None
    spacing_y: Expr | None = None
    nx: Expr | None = None
    ny: Expr | None = None
    bolt_circle_r: Expr | None = None
    n_bolts: Expr | None = None
    face: str = ">Z"

    def to_code(self) -> list[str]:
        d = self.diameter.to_code()
        cbd = self.cb_diameter.to_code()
        cbdp = self.cb_depth.to_code()
        face = self.face
        if self.placement == "corners":
            sx = self.spacing_x.to_code() if self.spacing_x else "20"
            sy = self.spacing_y.to_code() if self.spacing_y else "20"
            return [
                f"    result = (",
                f"        result.faces({face!r}).workplane()",
                f"        .rect({sx}, {sy}, forConstruction=True)",
                f"        .vertices().cboreHole({d}, {cbd}, {cbdp})",
                f"    )",
            ]
        elif self.placement == "grid":
            sx = self.spacing_x.to_code() if self.spacing_x else "10"
            sy = self.spacing_y.to_code() if self.spacing_y else "10"
            nx = self.nx.to_code() if self.nx else "2"
            ny = self.ny.to_code() if self.ny else "2"
            return [
                f"    result = (",
                f"        result.faces({face!r}).workplane()",
                f"        .rarray({sx}, {sy}, {nx}, {ny})",
                f"        .cboreHole({d}, {cbd}, {cbdp})",
                f"    )",
            ]
        else:
            r = self.bolt_circle_r.to_code() if self.bolt_circle_r else "15"
            n = self.n_bolts.to_code() if self.n_bolts else "4"
            return [
                f"    result = (",
                f"        result.faces({face!r}).workplane()",
                f"        .polarArray({r}, 0, 360, {n})",
                f"        .cboreHole({d}, {cbd}, {cbdp})",
                f"    )",
            ]


class CountersinkHolesOp(BaseModel):
    type: Literal["countersink"] = "countersink"
    diameter: Expr
    cs_angle: Expr
    placement: Literal["corners", "grid", "bolt_circle"] = "corners"
    spacing_x: Expr | None = None
    spacing_y: Expr | None = None
    nx: Expr | None = None
    ny: Expr | None = None
    bolt_circle_r: Expr | None = None
    n_bolts: Expr | None = None
    face: str = ">Z"

    def to_code(self) -> list[str]:
        d = self.diameter.to_code()
        # csk outer diameter = 2x the clearance diameter
        outer = f"{self.diameter.to_code()} * 2"
        angle = self.cs_angle.to_code()
        face = self.face
        if self.placement == "corners":
            sx = self.spacing_x.to_code() if self.spacing_x else "20"
            sy = self.spacing_y.to_code() if self.spacing_y else "20"
            return [
                f"    result = (",
                f"        result.faces({face!r}).workplane()",
                f"        .rect({sx}, {sy}, forConstruction=True)",
                f"        .vertices().cskHole({d}, {outer}, {angle})",
                f"    )",
            ]
        elif self.placement == "grid":
            sx = self.spacing_x.to_code() if self.spacing_x else "10"
            sy = self.spacing_y.to_code() if self.spacing_y else "10"
            nx = self.nx.to_code() if self.nx else "2"
            ny = self.ny.to_code() if self.ny else "2"
            return [
                f"    result = (",
                f"        result.faces({face!r}).workplane()",
                f"        .rarray({sx}, {sy}, {nx}, {ny})",
                f"        .cskHole({d}, {outer}, {angle})",
                f"    )",
            ]
        else:
            r = self.bolt_circle_r.to_code() if self.bolt_circle_r else "15"
            n = self.n_bolts.to_code() if self.n_bolts else "4"
            return [
                f"    result = (",
                f"        result.faces({face!r}).workplane()",
                f"        .polarArray({r}, 0, 360, {n})",
                f"        .cskHole({d}, {outer}, {angle})",
                f"    )",
            ]


class FilletOp(BaseModel):
    """Fillet selected edges. If enabled is set it becomes a conditional."""
    type: Literal["fillet"] = "fillet"
    radius: Expr
    edge_selector: str = "|Z"
    enabled: Expr | None = None  # ParamRef to bool param, or None = always

    def to_code(self) -> list[str]:
        body = f"result.edges({self.edge_selector!r}).fillet({self.radius.to_code()})"
        if self.enabled is not None:
            return [
                f"    if {self.enabled.to_code()}:",
                f"        result = {body}",
            ]
        return [f"    result = {body}"]


class ChamferOp(BaseModel):
    type: Literal["chamfer"] = "chamfer"
    distance: Expr
    edge_selector: str = "|Z"
    enabled: Expr | None = None

    def to_code(self) -> list[str]:
        body = f"result.edges({self.edge_selector!r}).chamfer({self.distance.to_code()})"
        if self.enabled is not None:
            return [
                f"    if {self.enabled.to_code()}:",
                f"        result = {body}",
            ]
        return [f"    result = {body}"]


class ShellOp(BaseModel):
    """Shell (hollow out) a solid, leaving one face open."""
    type: Literal["shell"] = "shell"
    thickness: Expr
    open_face: str = ">Z"

    def to_code(self) -> list[str]:
        return [
            f"    result = result.faces({self.open_face!r}).shell(-{self.thickness.to_code()})"
        ]


class PocketOp(BaseModel):
    """Rectangular pocket cut into the top face."""
    type: Literal["pocket"] = "pocket"
    width: Expr
    depth: Expr
    pocket_depth: Expr
    face: str = ">Z"

    def to_code(self) -> list[str]:
        return [
            f"    result = (",
            f"        result.faces({self.face!r}).workplane()",
            f"        .rect({self.width.to_code()}, {self.depth.to_code()})",
            f"        .cutBlind(-{self.pocket_depth.to_code()})",
            f"    )",
        ]


class BossOp(BaseModel):
    """Cylindrical boss (raised pad) with optional through-bore."""
    type: Literal["boss"] = "boss"
    diameter: Expr
    height: Expr
    bore_d: Expr | None = None
    face: str = ">Z"

    def to_code(self) -> list[str]:
        d = self.diameter.to_code()
        h = self.height.to_code()
        face = self.face
        lines = [
            f"    result = (",
            f"        result.faces({face!r}).workplane()",
            f"        .circle({d} / 2).extrude({h})",
            f"    )",
        ]
        if self.bore_d is not None:
            bd = self.bore_d.to_code()
            lines += [
                f"    result = (",
                f"        result.faces('>Z').workplane()",
                f"        .hole({bd})",
                f"    )",
            ]
        return lines


class RibsOp(BaseModel):
    """Evenly-spaced thin ribs on a face."""
    type: Literal["ribs"] = "ribs"
    count: Expr
    rib_height: Expr
    rib_thickness: Expr
    span: Expr          # total span they fill (width of parent face)
    face: str = ">Z"

    def to_code(self) -> list[str]:
        n = self.count.to_code()
        rh = self.rib_height.to_code()
        rt = self.rib_thickness.to_code()
        span = self.span.to_code()
        face = self.face
        spacing = f"({span} / ({n} - 1)) if {n} > 1 else 0"
        return [
            f"    _rib_spacing = {spacing}",
            f"    result = (",
            f"        result.faces({face!r}).workplane()",
            f"        .rarray(_rib_spacing, 1, {n}, 1)",
            f"        .rect({rt}, {span}).extrude({rh})",
            f"    )",
        ]


class LBracketOp(BaseModel):
    """
    L-shaped bracket: a flat base leg and a vertical leg sharing the inner
    corner, built as two boxes and unioned.  Holes are drilled into each leg
    *before* the union so their placement is valid by construction (selecting
    a single face on each box is unambiguous).  An optional triangular gusset
    reinforces the inner corner.

    This is always the base (first) operation for the bracket family.
    """
    type: Literal["lbracket"] = "lbracket"
    base_len: Expr
    vert_len: Expr
    width: Expr
    thickness: Expr
    hole_d: Expr | None = None
    n_holes: Expr | None = None        # holes per leg
    gusset_t: Expr | None = None       # gusset thickness; None = no gusset
    gusset_enabled: Expr | None = None  # ParamRef to bool, or None

    def to_code(self) -> list[str]:  # noqa: C901
        bl = self.base_len.to_code()
        vl = self.vert_len.to_code()
        w = self.width.to_code()
        t = self.thickness.to_code()
        lines = [
            f"    _bl = {bl}",
            f"    _vl = {vl}",
            f"    _bw = {w}",
            f"    _bt = {t}",
            "    base = cq.Workplane('XY').box(_bl, _bw, _bt, centered=(False, True, False))",
            "    vert = cq.Workplane('XY').box(_bt, _bw, _vl, centered=(False, True, False))",
        ]

        if self.hole_d is not None:
            hd = self.hole_d.to_code()
            n = self.n_holes.to_code() if self.n_holes is not None else "1"
            lines += [
                f"    _hd = {hd}",
                f"    _nh = max(1, int({n}))",
                "    if _hd > 0:",
                "        _m = max(_hd, _bt) * 1.2",
                "        _span_b = _bl - _bt - 2 * _m",
                "        if _span_b > 0:",
                "            _xs = ([_bt + _m + _span_b * i / max(_nh - 1, 1) for i in range(_nh)]"
                "                   if _nh > 1 else [(_bt + _bl) / 2])",
                "            base = (base.faces('>Z').workplane(centerOption='CenterOfBoundBox')",
                "                    .pushPoints([(x - _bl / 2, 0) for x in _xs]).hole(_hd))",
                "        _span_v = _vl - _bt - 2 * _m",
                "        if _span_v > 0:",
                "            _zs = ([_bt + _m + _span_v * i / max(_nh - 1, 1) for i in range(_nh)]"
                "                   if _nh > 1 else [(_bt + _vl) / 2])",
                "            vert = (vert.faces('<X').workplane(centerOption='CenterOfBoundBox')",
                "                    .pushPoints([(0, z - _vl / 2) for z in _zs]).hole(_hd))",
            ]

        lines.append("    result = base.union(vert)")

        if self.gusset_t is not None:
            gt = self.gusset_t.to_code()
            indent = "    "
            gusset_body = [
                f"    _gt = {gt}",
                "    _g = min(_bl, _vl) * 0.55 - _bt",
                "    if _g > _bt and _gt < _bw:",
                "        gus = (cq.Workplane('XZ').workplane(offset=_gt / 2)",
                "               .moveTo(_bt, _bt).lineTo(_bt + _g, _bt)",
                "               .lineTo(_bt, _bt + _g).close().extrude(-_gt))",
                "        result = result.union(gus)",
            ]
            if self.gusset_enabled is not None:
                lines.append(f"    if {self.gusset_enabled.to_code()}:")
                lines += [indent + ln for ln in gusset_body]
            else:
                lines += gusset_body

        return lines


class CBracketOp(BaseModel):
    """
    C-shaped (channel / U) bracket: a flat base leg with a vertical leg rising
    at *each* end, built as three boxes and unioned.  Mounting holes are drilled
    into each box *before* the union so their placement is valid by construction.
    Optional triangular gussets reinforce both inner corners.

    Always the base (first) operation when the bracket family samples a channel.
    """
    type: Literal["cbracket"] = "cbracket"
    base_len: Expr
    vert_len: Expr
    width: Expr
    thickness: Expr
    hole_d: Expr | None = None
    n_holes: Expr | None = None         # holes per leg
    gusset_t: Expr | None = None        # gusset thickness; None = no gusset
    gusset_enabled: Expr | None = None  # ParamRef to bool, or None

    def to_code(self) -> list[str]:  # noqa: C901
        bl = self.base_len.to_code()
        vl = self.vert_len.to_code()
        w = self.width.to_code()
        t = self.thickness.to_code()
        lines = [
            f"    _bl = {bl}",
            f"    _vl = {vl}",
            f"    _bw = {w}",
            f"    _bt = {t}",
            "    base = cq.Workplane('XY').box(_bl, _bw, _bt, centered=(False, True, False))",
            "    vert1 = cq.Workplane('XY').box(_bt, _bw, _vl, centered=(False, True, False))",
            "    vert2 = (cq.Workplane('XY').box(_bt, _bw, _vl, centered=(False, True, False))",
            "             .translate((_bl - _bt, 0, 0)))",
        ]

        if self.hole_d is not None:
            hd = self.hole_d.to_code()
            n = self.n_holes.to_code() if self.n_holes is not None else "1"
            lines += [
                f"    _hd = {hd}",
                f"    _nh = max(1, int({n}))",
                "    if _hd > 0:",
                "        _m = max(_hd, _bt) * 1.2",
                "        _span_b = _bl - 2 * _bt - 2 * _m",
                "        if _span_b > 0:",
                "            _xs = ([_bt + _m + _span_b * i / max(_nh - 1, 1) for i in range(_nh)]"
                "                   if _nh > 1 else [_bl / 2])",
                "            base = (base.faces('>Z').workplane(centerOption='CenterOfBoundBox')",
                "                    .pushPoints([(x - _bl / 2, 0) for x in _xs]).hole(_hd))",
                "        _span_v = _vl - _bt - 2 * _m",
                "        if _span_v > 0:",
                "            _zs = ([_bt + _m + _span_v * i / max(_nh - 1, 1) for i in range(_nh)]"
                "                   if _nh > 1 else [(_bt + _vl) / 2])",
                "            vert1 = (vert1.faces('<X').workplane(centerOption='CenterOfBoundBox')",
                "                     .pushPoints([(0, z - _vl / 2) for z in _zs]).hole(_hd))",
                "            vert2 = (vert2.faces('>X').workplane(centerOption='CenterOfBoundBox')",
                "                     .pushPoints([(0, z - _vl / 2) for z in _zs]).hole(_hd))",
            ]

        lines.append("    result = base.union(vert1).union(vert2)")

        if self.gusset_t is not None:
            gt = self.gusset_t.to_code()
            indent = "    "
            gusset_body = [
                f"    _gt = {gt}",
                "    _g = min(_bl / 2, _vl) * 0.55 - _bt",
                "    if _g > _bt and _gt < _bw:",
                "        gus1 = (cq.Workplane('XZ').workplane(offset=_gt / 2)",
                "                .moveTo(_bt, _bt).lineTo(_bt + _g, _bt)",
                "                .lineTo(_bt, _bt + _g).close().extrude(-_gt))",
                "        gus2 = (cq.Workplane('XZ').workplane(offset=_gt / 2)",
                "                .moveTo(_bl - _bt, _bt).lineTo(_bl - _bt - _g, _bt)",
                "                .lineTo(_bl - _bt, _bt + _g).close().extrude(-_gt))",
                "        result = result.union(gus1).union(gus2)",
            ]
            if self.gusset_enabled is not None:
                lines.append(f"    if {self.gusset_enabled.to_code()}:")
                lines += [indent + ln for ln in gusset_body]
            else:
                lines += gusset_body

        return lines


class ZBracketOp(BaseModel):
    """
    Z-shaped (cranked / offset) bracket: a bottom flange, a vertical web, and a
    top flange that extends in the *opposite* direction from the bottom flange —
    the classic offset/joggle bracket.  Built as three boxes and unioned, with
    mounting holes drilled into each flange before the union and optional
    gussets at both bends.

    Always the base (first) operation when the bracket family samples a Z.
    """
    type: Literal["zbracket"] = "zbracket"
    base_len: Expr
    vert_len: Expr
    top_len: Expr
    width: Expr
    thickness: Expr
    hole_d: Expr | None = None
    n_holes: Expr | None = None         # holes per flange
    gusset_t: Expr | None = None        # gusset thickness; None = no gusset
    gusset_enabled: Expr | None = None  # ParamRef to bool, or None

    def to_code(self) -> list[str]:  # noqa: C901
        bl = self.base_len.to_code()
        vl = self.vert_len.to_code()
        tl = self.top_len.to_code()
        w = self.width.to_code()
        t = self.thickness.to_code()
        lines = [
            f"    _bl = {bl}",
            f"    _vl = {vl}",
            f"    _tl = {tl}",
            f"    _bw = {w}",
            f"    _bt = {t}",
            # Bottom flange runs +X from the web; web rises at X in [0, _bt];
            # top flange runs -X from the web's far edge, offset up to the top.
            "    base = cq.Workplane('XY').box(_bl, _bw, _bt, centered=(False, True, False))",
            "    web = cq.Workplane('XY').box(_bt, _bw, _vl, centered=(False, True, False))",
            "    top = (cq.Workplane('XY').box(_tl, _bw, _bt, centered=(False, True, False))",
            "           .translate((_bt - _tl, 0, _vl - _bt)))",
        ]

        if self.hole_d is not None:
            hd = self.hole_d.to_code()
            n = self.n_holes.to_code() if self.n_holes is not None else "1"
            lines += [
                f"    _hd = {hd}",
                f"    _nh = max(1, int({n}))",
                "    if _hd > 0:",
                "        _m = max(_hd, _bt) * 1.2",
                # Bottom flange holes: between the web and the free (+X) end.
                "        _span_b = _bl - _bt - 2 * _m",
                "        if _span_b > 0:",
                "            _xs = ([_bt + _m + _span_b * i / max(_nh - 1, 1) for i in range(_nh)]"
                "                   if _nh > 1 else [(_bt + _bl) / 2])",
                "            base = (base.faces('>Z').workplane(centerOption='CenterOfBoundBox')",
                "                    .pushPoints([(x - _bl / 2, 0) for x in _xs]).hole(_hd))",
                # Top flange holes: between the web and the free (-X) end.
                "        _span_t = _tl - _bt - 2 * _m",
                "        if _span_t > 0:",
                "            _txc = (_bt - _tl) + _tl / 2",
                "            _xt = ([(_bt - _tl) + _m + _span_t * i / max(_nh - 1, 1) for i in range(_nh)]"
                "                   if _nh > 1 else [(_bt - _tl) / 2])",
                "            top = (top.faces('>Z').workplane(centerOption='CenterOfBoundBox')",
                "                   .pushPoints([(x - _txc, 0) for x in _xt]).hole(_hd))",
            ]

        lines.append("    result = base.union(web).union(top)")

        if self.gusset_t is not None:
            gt = self.gusset_t.to_code()
            indent = "    "
            gusset_body = [
                f"    _gt = {gt}",
                "    _g = min(_bl, _tl, _vl) * 0.4 - _bt",
                "    if _g > _bt and _gt < _bw:",
                # Bottom bend: inner corner at (X=_bt, Z=_bt), triangle into +X/+Z.
                "        gus1 = (cq.Workplane('XZ').workplane(offset=_gt / 2)",
                "                .moveTo(_bt, _bt).lineTo(_bt + _g, _bt)",
                "                .lineTo(_bt, _bt + _g).close().extrude(-_gt))",
                # Top bend: inner corner at (X=0, Z=_vl-_bt), triangle into -X/-Z.
                "        gus2 = (cq.Workplane('XZ').workplane(offset=_gt / 2)",
                "                .moveTo(0, _vl - _bt).lineTo(-_g, _vl - _bt)",
                "                .lineTo(0, _vl - _bt - _g).close().extrude(-_gt))",
                "        result = result.union(gus1).union(gus2)",
            ]
            if self.gusset_enabled is not None:
                lines.append(f"    if {self.gusset_enabled.to_code()}:")
                lines += [indent + ln for ln in gusset_body]
            else:
                lines += gusset_body

        return lines


Operation = Annotated[
    Union[
        BoxOp, ExtrudeOp, RevolveOp,
        HolesOp, CounterboreHolesOp, CountersinkHolesOp,
        FilletOp, ChamferOp,
        ShellOp, PocketOp, BossOp, RibsOp,
        LBracketOp, CBracketOp, ZBracketOp,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Part IR — root model
# ---------------------------------------------------------------------------


class PartMetadata(BaseModel):
    seed: int
    generator_version: str
    family: str
    tier: int
    symmetry: str = "none"
    op_count: int = 0


class PartIR(BaseModel):
    id: str
    params: dict[str, ParamSpec]
    operations: list[Operation]
    metadata: PartMetadata

    def params_schema(self) -> dict[str, dict[str, Any]]:
        """PARAMS dict as emitted into the .py file."""
        return {k: v.to_dict() for k, v in self.params.items()}

    def default_params(self) -> dict[str, Any]:
        return {k: v.default for k, v in self.params.items()}

    def ir_hash(self) -> str:
        """Stable hash of this IR for fast exact-duplicate detection."""
        data = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:16]
