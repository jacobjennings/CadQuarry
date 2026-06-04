"""Tests for code generation — no cadquery required."""
import ast
import json
import sys
import types
import unittest
from cadquarry.ir import (
    BoxOp, FilletOp, HolesOp, PartIR, PartMetadata, ParamSpec,
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
