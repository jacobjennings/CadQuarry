"""Tests for code generation — no cadquery required."""
import ast
import json
import sys
import types
import unittest
from cadquarry.ir import (
    BoxOp, FilletOp, GearOp, HolesOp, ThreadedOp, PartIR, PartMetadata, ParamSpec,
    lit, ref, scaled,
)
from cadquarry.emit import emit_source, emit_params_json, emit_meta_json


def _make_plate_part() -> PartIR:
    return PartIR(
        id="plate_emit_test",
        params={
            "plate_w":   ParamSpec(type="float", default=60.0, min=30.0, max=120.0, step=1.0,  group="Body",  label="Width"),
            "plate_d":   ParamSpec(type="float", default=40.0, min=20.0, max=80.0,  step=1.0,  group="Body",  label="Depth"),
            "thickness": ParamSpec(type="float", default=6.0,  min=2.0,  max=12.0,  step=0.5,  group="Body",  label="Thickness"),
            "hole_d":    ParamSpec(type="float", default=5.3,  min=3.2,  max=8.4,   step=0.1,  group="Holes", label="Hole diameter"),
            "filleted":  ParamSpec(type="bool",  default=True,                                  group="Body",  label="Fillet corners"),
        },
        operations=[
            BoxOp(width=ref("plate_w"), depth=ref("plate_d"), height=ref("thickness")),
            FilletOp(radius=scaled("plate_w", 0.08), edge_selector="|Z", enabled=ref("filleted")),
            HolesOp(
                diameter=ref("hole_d"), placement="corners",
                spacing_x=scaled("plate_w", 0.7), spacing_y=scaled("plate_d", 0.7),
            ),
        ],
        metadata=PartMetadata(seed=42, generator_version="0.1.0", family="plate", tier=2, op_count=3),
    )


class TestEmitSource(unittest.TestCase):
    def test_is_valid_python(self):
        code = emit_source(_make_plate_part())
        tree = ast.parse(code)
        self.assertIsNotNone(tree)

    def test_contains_params(self):
        code = emit_source(_make_plate_part())
        self.assertIn("PARAMS", code)
        self.assertIn('"plate_w"', code)
        self.assertIn('"filleted"', code)

    def test_contains_build_function(self):
        code = emit_source(_make_plate_part())
        self.assertIn("def build(p):", code)

    def test_contains_canonical_instance(self):
        code = emit_source(_make_plate_part())
        self.assertIn("result = build(", code)

    def test_contains_cq_import(self):
        code = emit_source(_make_plate_part())
        self.assertIn("import cadquery as cq", code)

    def test_cc0_header(self):
        code = emit_source(_make_plate_part())
        self.assertIn("CC0-1.0", code)

    def test_build_references_params(self):
        code = emit_source(_make_plate_part())
        self.assertIn('p["plate_w"]', code)
        self.assertIn('p["thickness"]', code)
        self.assertIn('p["hole_d"]', code)

    def test_conditional_fillet(self):
        code = emit_source(_make_plate_part())
        self.assertIn('if p["filleted"]:', code)

    def test_no_leaked_format_braces(self):
        # Regression: the canonical-instance template once leaked escaped
        # `{{ }}` into emitted source, producing `build({{...}})` which is a
        # set literal of a dict comprehension and raises at runtime.
        code = emit_source(_make_plate_part())
        self.assertNotIn("{{", code)
        self.assertNotIn("}}", code)

    def test_emitted_module_executes_with_stub(self):
        # Regression: exec the full emitted module (including the canonical
        # `result = build({...})` line) against a chainable cadquery stub so
        # the canonical instance is actually built without real CadQuery.
        code = emit_source(_make_plate_part())

        class _Chain:
            def __getattr__(self, _name):
                def _call(*_args, **_kwargs):
                    return self
                return _call

        stub = types.ModuleType("cadquery")
        stub.Workplane = lambda *a, **k: _Chain()  # type: ignore[attr-defined]
        sys.modules["cadquery"] = stub
        try:
            ns: dict = {}
            exec(compile(code, "<emitted>", "exec"), ns)
            self.assertIn("result", ns)
            self.assertIn("PARAMS", ns)
            self.assertIsInstance(ns["PARAMS"], dict)
        finally:
            sys.modules.pop("cadquery", None)


def _make_gear_part() -> PartIR:
    """A tier-3 helical gear with bore + hub (no cadquery needed to emit)."""
    return PartIR(
        id="gear_emit_test",
        params={
            "module":      ParamSpec(type="float", default=2.0, min=1.0, max=6.0, step=0.25, group="Gear", label="Module"),
            "teeth":       ParamSpec(type="int",   default=24,  min=10,  max=40,  step=1,    group="Gear", label="Teeth"),
            "gear_w":      ParamSpec(type="float", default=8.0, min=3.0, max=20.0, step=1.0, group="Gear", label="Face width"),
            "helix_angle": ParamSpec(type="float", default=18.0, min=5.0, max=40.0, step=1.0, group="Gear", label="Helix angle"),
            "bore_d":      ParamSpec(type="float", default=8.0, min=2.0, max=20.0, step=0.5, group="Bore", label="Bore diameter"),
            "hub_d":       ParamSpec(type="float", default=14.0, min=10.0, max=24.0, step=1.0, group="Hub", label="Hub diameter"),
            "hub_h":       ParamSpec(type="float", default=5.0, min=2.0, max=10.0, step=0.5, group="Hub", label="Hub height"),
        },
        operations=[
            GearOp(
                kind="helical", module=ref("module"), teeth=ref("teeth"), width=ref("gear_w"),
                helix_angle=ref("helix_angle"), herringbone=True,
                bore_d=ref("bore_d"), hub_d=ref("hub_d"), hub_h=ref("hub_h"),
            )
        ],
        metadata=PartMetadata(seed=1, generator_version="0.5.0", family="gear", tier=3, op_count=1),
    )


def _make_threaded_part(standard="iso", external=True) -> PartIR:
    if standard == "iso":
        params = {
            "major_d":    ParamSpec(type="float", default=8.0, min=4.0, max=24.0, step=0.5, group="Thread", label="Major dia"),
            "pitch":      ParamSpec(type="float", default=1.25, min=0.6, max=2.5, step=0.05, group="Thread", label="Pitch"),
            "thread_len": ParamSpec(type="float", default=20.0, min=8.0, max=40.0, step=1.0, group="Thread", label="Length"),
        }
        op = ThreadedOp(standard="iso", external=external, length=ref("thread_len"),
                        major_d=ref("major_d"), pitch=ref("pitch"))
    else:
        params = {
            "thread_size": ParamSpec(type="enum", default="1/2", choices=["3/8", "1/2", "5/8"], group="Thread", label="Size"),
            "thread_len":  ParamSpec(type="float", default=30.0, min=12.0, max=60.0, step=1.0, group="Thread", label="Length"),
        }
        op = ThreadedOp(standard=standard, external=True, length=ref("thread_len"), size=ref("thread_size"))
    return PartIR(
        id=f"threaded_{standard}_emit_test",
        params=params,
        operations=[op],
        metadata=PartMetadata(seed=1, generator_version="0.5.0", family="threaded", tier=2, op_count=1),
    )


class TestGearEmit(unittest.TestCase):
    """Pure-string checks for GearOp — no cadquery / py_gearworks needed."""

    def test_is_valid_python(self):
        ast.parse(emit_source(_make_gear_part()))

    def test_per_part_imports(self):
        code = emit_source(_make_gear_part())
        self.assertIn("import py_gearworks as _gw", code)
        self.assertIn("import math", code)
        # cadquery import always present and first.
        self.assertTrue(code.startswith("import cadquery as cq"))

    def test_emits_gear_class_and_bridge(self):
        code = emit_source(_make_gear_part())
        self.assertIn("_gw.HelicalGear(", code)
        self.assertIn("_gpart = _gear.build_part()", code)
        self.assertIn("cq.Workplane('XY').add(cq.Solid(_gpart.wrapped))", code)

    def test_helix_angle_converted_to_radians(self):
        code = emit_source(_make_gear_part())
        self.assertIn('helix_angle=math.radians(p["helix_angle"])', code)
        self.assertIn("herringbone=True", code)

    def test_bore_and_hub(self):
        code = emit_source(_make_gear_part())
        self.assertIn('.circle(p["hub_d"] / 2).extrude(p["hub_h"])', code)
        self.assertIn('.hole(p["bore_d"])', code)

    def test_teeth_cast_to_int(self):
        code = emit_source(_make_gear_part())
        self.assertIn('number_of_teeth=int(p["teeth"])', code)


class TestThreadedEmit(unittest.TestCase):
    """Pure-string checks for ThreadedOp — no cadquery / bd_warehouse needed."""

    def test_is_valid_python(self):
        ast.parse(emit_source(_make_threaded_part("iso")))
        ast.parse(emit_source(_make_threaded_part("acme")))
        ast.parse(emit_source(_make_threaded_part("trapezoidal")))

    def test_per_part_imports(self):
        code = emit_source(_make_threaded_part("iso"))
        self.assertIn("from bd_warehouse import thread as _bdt", code)
        self.assertIn("import build123d as _bd", code)

    def test_iso_external_shank_fusion(self):
        code = emit_source(_make_threaded_part("iso", external=True))
        self.assertIn("_bdt.IsoThread(major_diameter=", code)
        self.assertIn("external=True", code)
        self.assertIn("_part = _shank + _thr", code)
        self.assertIn("cq.Workplane('XY').add(cq.Solid(_part.wrapped))", code)

    def test_iso_internal_bored_body(self):
        code = emit_source(_make_threaded_part("iso", external=False))
        self.assertIn("external=False", code)
        self.assertIn("_part = (_body - _bore) + _thr", code)

    def test_lead_screw_uses_size_designation(self):
        code = emit_source(_make_threaded_part("acme"))
        self.assertIn("_bdt.AcmeThread(size=", code)
        code = emit_source(_make_threaded_part("trapezoidal"))
        self.assertIn("_bdt.MetricTrapezoidalThread(size=", code)


class TestEmitParamsJson(unittest.TestCase):
    def test_structure(self):
        data = emit_params_json(_make_plate_part())
        self.assertEqual(data["part_id"], "plate_emit_test")
        self.assertIn("params", data)
        self.assertIn("cadquarry_version", data)

    def test_all_params_present(self):
        data = emit_params_json(_make_plate_part())
        self.assertEqual(set(data["params"].keys()), {"plate_w", "plate_d", "thickness", "hole_d", "filleted"})

    def test_is_json_serialisable(self):
        data = emit_params_json(_make_plate_part())
        json.dumps(data)  # should not raise


class TestEmitMetaJson(unittest.TestCase):
    def test_structure(self):
        data = emit_meta_json(_make_plate_part())
        self.assertEqual(data["family"], "plate")
        self.assertEqual(data["tier"], 2)
        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["license"], "CC0-1.0")

    def test_with_signature(self):
        sig = {"version": 1, "volume": 14400.0, "surface_area": 8000.0,
               "n_faces": 8, "n_edges": 12, "n_vertices": 8}
        data = emit_meta_json(_make_plate_part(), geometry_signature=sig)
        self.assertEqual(data["geometry_signature"]["volume"], 14400.0)


if __name__ == "__main__":
    unittest.main()
