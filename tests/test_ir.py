"""Tests for the IR model layer — no cadquery required."""
import unittest
from cadquarry.ir import (
    BoxOp,
    ChamferOp,
    CircleProfile,
    ClampExpr,
    ExtrudeOp,
    FilletOp,
    HalfExpr,
    HolesOp,
    LiteralExpr,
    MinExpr,
    ParamRef,
    ParamSpec,
    PartIR,
    PartMetadata,
    RectProfile,
    RevolveOp,
    ScaledExpr,
    ShellOp,
    lit,
    ref,
    scaled,
    min2,
    half,
)


class TestExpressions(unittest.TestCase):
    def test_literal_float(self):
        e = lit(40.0)
        self.assertEqual(e.to_code(), "40.0")
        self.assertEqual(e.evaluate({}), 40.0)

    def test_literal_bool(self):
        self.assertEqual(lit(True).to_code(), "True")
        self.assertEqual(lit(False).to_code(), "False")

    def test_literal_str(self):
        self.assertEqual(lit("XY").to_code(), "'XY'")

    def test_param_ref(self):
        e = ref("plate_w")
        self.assertEqual(e.to_code(), 'p["plate_w"]')
        self.assertEqual(e.evaluate({"plate_w": 60.0}), 60.0)

    def test_scaled(self):
        e = scaled("plate_w", 0.7)
        self.assertEqual(e.to_code(), 'p["plate_w"] * 0.7')
        self.assertAlmostEqual(e.evaluate({"plate_w": 100.0}), 70.0)

    def test_min2(self):
        e = min2("a", "b", 0.5)
        code = e.to_code()
        self.assertIn('min(p["a"], p["b"])', code)
        self.assertIn("0.5", code)
        self.assertAlmostEqual(e.evaluate({"a": 30.0, "b": 40.0}), 15.0)

    def test_clamp(self):
        e = ClampExpr(param="x", limit=10.0)
        self.assertEqual(e.evaluate({"x": 20.0}), 10.0)
        self.assertEqual(e.evaluate({"x": 5.0}), 5.0)

    def test_half(self):
        e = half("outer_d")
        self.assertEqual(e.to_code(), 'p["outer_d"] / 2')
        self.assertEqual(e.evaluate({"outer_d": 30.0}), 15.0)


class TestParamSpec(unittest.TestCase):
    def test_float_to_dict(self):
        spec = ParamSpec(
            type="float", default=40.0, min=10.0, max=80.0, step=1.0,
            group="Body", label="Width",
        )
        d = spec.to_dict()
        self.assertEqual(d["type"], "float")
        self.assertEqual(d["default"], 40.0)
        self.assertEqual(d["min"], 10.0)
        self.assertEqual(d["group"], "Body")

    def test_bool_to_dict(self):
        spec = ParamSpec(type="bool", default=True, group="Body", label="Filleted")
        d = spec.to_dict()
        self.assertEqual(d["type"], "bool")
        self.assertNotIn("min", d)

    def test_enum_to_dict(self):
        spec = ParamSpec(
            type="enum", default="simple",
            choices=["simple", "counterbore"], group="Holes", label="Style",
        )
        d = spec.to_dict()
        self.assertEqual(d["choices"], ["simple", "counterbore"])


class TestOperationCode(unittest.TestCase):
    def test_box_op(self):
        op = BoxOp(width=ref("w"), depth=ref("d"), height=lit(6.0))
        code = op.to_code()
        self.assertEqual(len(code), 1)
        self.assertIn("cq.Workplane", code[0])
        self.assertIn('p["w"]', code[0])

    def test_extrude_rect(self):
        op = ExtrudeOp(
            profile=RectProfile(width=ref("w"), depth=ref("d")),
            distance=ref("h"),
        )
        code = "\n".join(op.to_code())
        self.assertIn(".rect(", code)
        self.assertIn(".extrude(", code)

    def test_extrude_circle(self):
        op = ExtrudeOp(
            profile=CircleProfile(diameter=ref("od")),
            distance=ref("length"),
        )
        code = "\n".join(op.to_code())
        self.assertIn(".circle(", code)

    def test_revolve_solid(self):
        op = RevolveOp(outer_d=ref("outer_d"), height=ref("length"))
        code = "\n".join(op.to_code())
        self.assertIn(".revolve()", code)
        self.assertIn("moveTo(0, 0)", code)

    def test_revolve_with_bore(self):
        op = RevolveOp(outer_d=ref("od"), height=ref("h"), inner_d=ref("id_"))
        code = "\n".join(op.to_code())
        self.assertIn(".revolve()", code)
        self.assertIn("moveTo", code)

    def test_holes_corners(self):
        op = HolesOp(
            diameter=ref("hole_d"), placement="corners",
            spacing_x=scaled("plate_w", 0.7),
            spacing_y=scaled("plate_d", 0.7),
        )
        code = "\n".join(op.to_code())
        self.assertIn(".vertices().hole(", code)
        self.assertIn("forConstruction=True", code)

    def test_holes_bolt_circle(self):
        op = HolesOp(
            diameter=ref("bolt_d"), placement="bolt_circle",
            bolt_circle_r=ref("bolt_r"), n_bolts=ref("n_bolts"),
        )
        code = "\n".join(op.to_code())
        self.assertIn(".polarArray(", code)

    def test_fillet_conditional(self):
        op = FilletOp(radius=scaled("w", 0.08), edge_selector="|Z", enabled=ref("filleted"))
        lines = op.to_code()
        self.assertTrue(any("if" in l for l in lines))
        self.assertTrue(any("fillet(" in l for l in lines))

    def test_fillet_always(self):
        op = FilletOp(radius=lit(2.0))
        lines = op.to_code()
        self.assertEqual(len(lines), 1)
        self.assertIn("fillet(2.0)", lines[0])

    def test_chamfer_conditional(self):
        op = ChamferOp(distance=ref("ch"), edge_selector=">Z", enabled=ref("use_ch"))
        lines = op.to_code()
        self.assertTrue(any("if" in l for l in lines))

    def test_shell(self):
        op = ShellOp(thickness=ref("wall_t"), open_face=">Z")
        lines = op.to_code()
        self.assertIn(".shell(-", lines[0])


class TestPartIR(unittest.TestCase):
    def _make_part(self):
        return PartIR(
            id="plate_test_0001",
            params={
                "plate_w": ParamSpec(
                    type="float", default=60.0, min=30.0, max=120.0, step=1.0,
                    group="Body", label="Width",
                ),
                "filleted": ParamSpec(type="bool", default=True, group="Body", label="Fillet"),
            },
            operations=[
                BoxOp(width=ref("plate_w"), depth=lit(40.0), height=lit(6.0)),
                FilletOp(radius=scaled("plate_w", 0.08), edge_selector="|Z", enabled=ref("filleted")),
            ],
            metadata=PartMetadata(
                seed=42, generator_version="0.1.0",
                family="plate", tier=1, op_count=2,
            ),
        )

    def test_default_params(self):
        part = self._make_part()
        defaults = part.default_params()
        self.assertEqual(defaults["plate_w"], 60.0)
        self.assertEqual(defaults["filleted"], True)

    def test_params_schema_keys(self):
        part = self._make_part()
        schema = part.params_schema()
        self.assertEqual(set(schema.keys()), {"plate_w", "filleted"})
        self.assertEqual(schema["plate_w"]["type"], "float")

    def test_ir_hash_stable(self):
        part = self._make_part()
        h1 = part.ir_hash()
        h2 = part.ir_hash()
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_ir_hash_changes_with_params(self):
        part = self._make_part()
        h1 = part.ir_hash()
        part2 = part.model_copy(deep=True)
        part2.params["plate_w"] = ParamSpec(
            type="float", default=80.0, min=30.0, max=120.0, step=1.0, group="Body", label="Width"
        )
        h2 = part2.ir_hash()
        self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
