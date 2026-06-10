"""Tests for the procedural dimension summarizer — no cadquery required."""
import json
import tempfile
import unittest
from pathlib import Path

from cadquarry.describe import describe_dimensions
from cadquarry.emit import emit_meta_json, write_part
from cadquarry.ir import (
    BoxOp,
    CircleProfile,
    ExtrudeOp,
    FilletOp,
    GearOp,
    HolesOp,
    ParamSpec,
    PartIR,
    PartMetadata,
    RevolveOp,
    ThreadedOp,
    lit,
    ref,
    scaled,
)


def _plate_part() -> PartIR:
    return PartIR(
        id="plate_desc_test",
        params={
            "plate_w":  ParamSpec(type="float", default=80.0, min=30.0, max=120.0, step=1.0, group="Body", label="Width"),
            "plate_d":  ParamSpec(type="float", default=60.0, min=20.0, max=80.0,  step=1.0, group="Body", label="Depth"),
            "thickness": ParamSpec(type="float", default=10.0, min=2.0, max=12.0, step=0.5, group="Body", label="Thickness"),
            "hole_d":   ParamSpec(type="float", default=5.0, min=3.2, max=8.4, step=0.1, group="Holes", label="Hole diameter"),
            "filleted": ParamSpec(type="bool", default=True, group="Body", label="Fillet corners"),
        },
        operations=[
            BoxOp(width=ref("plate_w"), depth=ref("plate_d"), height=ref("thickness")),
            FilletOp(radius=lit(3.0), edge_selector="|Z", enabled=ref("filleted")),
            HolesOp(
                diameter=ref("hole_d"), placement="corners",
                spacing_x=scaled("plate_w", 0.7), spacing_y=scaled("plate_d", 0.7),
            ),
        ],
        metadata=PartMetadata(seed=42, generator_version="test", family="plate", tier=2, op_count=3),
    )


class TestDescribeDimensions(unittest.TestCase):
    def test_plate_summary(self):
        dims = describe_dimensions(_plate_part())
        self.assertIn("lines", dims)
        self.assertIn("text", dims)
        # Base solid dimensions resolve against defaults.
        self.assertIn("80 × 60 × 10 mm", dims["text"])
        # Hole count + diameter + corner placement.
        self.assertIn("Four", dims["text"])
        self.assertIn("Ø5 mm", dims["text"])
        self.assertIn("corners", dims["text"])
        # Fillet (enabled by default) is reported, not marked optional.
        self.assertIn("Fillet radius 3 mm", dims["text"])
        self.assertNotIn("off by default", dims["text"])

    def test_text_is_lines_joined(self):
        dims = describe_dimensions(_plate_part())
        self.assertEqual(dims["text"], " ".join(dims["lines"]))
        self.assertEqual(len(dims["lines"]), 3)

    def test_disabled_fillet_marked_optional(self):
        part = _plate_part()
        part.params["filleted"].default = False
        dims = describe_dimensions(part)
        self.assertIn("off by default", dims["text"])

    def test_revolved_with_bore_and_flange(self):
        part = PartIR(
            id="rev_desc_test",
            params={
                "od": ParamSpec(type="float", default=50.0, group="Body", label="OD"),
                "h":  ParamSpec(type="float", default=20.0, group="Body", label="Height"),
                "id": ParamSpec(type="float", default=20.0, group="Body", label="Bore"),
            },
            operations=[RevolveOp(outer_d=ref("od"), height=ref("h"), inner_d=ref("id"))],
            metadata=PartMetadata(seed=1, generator_version="test", family="revolved", tier=1, op_count=1),
        )
        text = describe_dimensions(part)["text"]
        self.assertIn("Ø50 mm", text)
        self.assertIn("bore", text)

    def test_extrude_circle_profile(self):
        part = PartIR(
            id="ext_desc_test",
            params={"d": ParamSpec(type="float", default=30.0, group="Body", label="D"),
                    "t": ParamSpec(type="float", default=5.0, group="Body", label="T")},
            operations=[ExtrudeOp(profile=CircleProfile(diameter=ref("d")), distance=ref("t"))],
            metadata=PartMetadata(seed=1, generator_version="test", family="profiled", tier=0, op_count=1),
        )
        text = describe_dimensions(part)["text"]
        self.assertIn("circular section", text)
        self.assertIn("extruded 5 mm", text)

    def test_gear_uses_descriptor(self):
        part = PartIR(
            id="gear_desc_test",
            params={
                "module": ParamSpec(type="float", default=2.0, group="Gear", label="Module"),
                "teeth":  ParamSpec(type="int", default=24, group="Gear", label="Teeth"),
                "gear_w": ParamSpec(type="float", default=10.0, group="Gear", label="Width"),
            },
            operations=[GearOp(kind="spur", module=ref("module"), teeth=ref("teeth"), width=ref("gear_w"))],
            metadata=PartMetadata(seed=1, generator_version="test", family="gear", tier=0, op_count=1),
        )
        text = describe_dimensions(part)["text"]
        self.assertIn("Spur gear", text)
        self.assertIn("24 teeth", text)
        self.assertIn("pitch Ø48 mm", text)  # module * teeth

    def test_threaded_uses_descriptor(self):
        part = PartIR(
            id="thr_desc_test",
            params={
                "major_d": ParamSpec(type="float", default=10.0, group="Thread", label="Major D"),
                "pitch":   ParamSpec(type="float", default=1.5, group="Thread", label="Pitch"),
                "thread_len": ParamSpec(type="float", default=30.0, group="Thread", label="Length"),
            },
            operations=[ThreadedOp(standard="iso", external=True, length=ref("thread_len"),
                                   major_d=ref("major_d"), pitch=ref("pitch"))],
            metadata=PartMetadata(seed=1, generator_version="test", family="threaded", tier=0, op_count=1),
        )
        text = describe_dimensions(part)["text"]
        self.assertIn("M10x1.5", text)
        self.assertIn("external thread", text)

    def test_sketched_extrude_summary(self):
        from cadquarry.ir import SketchedProfile, SketchSeg
        prof = SketchedProfile(
            w_param="sk_w", h_param="sk_h", start=(0.4, 0.0),
            segments=[
                SketchSeg(kind="line", x=0.0, y=0.4),
                SketchSeg(kind="arc", x=-0.4, y=0.0, mx=-0.5, my=0.2),
                SketchSeg(kind="bezier", x=0.0, y=-0.4, ctrl=[(-0.2, -0.3)]),
                SketchSeg(kind="line", x=0.4, y=0.0),
            ],
        )
        part = PartIR(
            id="sk_desc_test",
            params={"sk_w": ParamSpec(type="float", default=80.0, group="Sketch", label="W"),
                    "sk_h": ParamSpec(type="float", default=60.0, group="Sketch", label="H"),
                    "sk_t": ParamSpec(type="float", default=10.0, group="Body", label="T")},
            operations=[ExtrudeOp(profile=prof, distance=ref("sk_t"))],
            metadata=PartMetadata(seed=1, generator_version="test", family="sketched", tier=0, op_count=1),
        )
        text = describe_dimensions(part)["text"]
        self.assertIn("freeform sketched section", text)
        self.assertIn("extruded 10 mm", text)

    def test_loft_summary(self):
        from cadquarry.ir import LoftOp, LoftStation, RectProfile, CircleProfile
        part = PartIR(
            id="loft_desc_test",
            params={"a": ParamSpec(type="float", default=40.0, group="B", label="a"),
                    "d": ParamSpec(type="float", default=16.0, group="B", label="d"),
                    "h": ParamSpec(type="float", default=30.0, group="B", label="h")},
            operations=[LoftOp(stations=[
                LoftStation(profile=RectProfile(width=ref("a"), depth=ref("a"))),
                LoftStation(profile=CircleProfile(diameter=ref("d")), offset=ref("h")),
            ])],
            metadata=PartMetadata(seed=1, generator_version="test", family="lofted", tier=0, op_count=1),
        )
        text = describe_dimensions(part)["text"]
        self.assertIn("Lofted body through 2 sections", text)

    def test_sweep_summary(self):
        from cadquarry.ir import SweepOp, CircleProfile
        part = PartIR(
            id="sweep_desc_test",
            params={"pd": ParamSpec(type="float", default=10.0, group="B", label="pd"),
                    "pw": ParamSpec(type="float", default=30.0, group="B", label="pw"),
                    "ph": ParamSpec(type="float", default=60.0, group="B", label="ph")},
            operations=[SweepOp(profile=CircleProfile(diameter=ref("pd")),
                                path_points=[(0.0, 0.0), (0.5, 0.5), (0.2, 1.0)],
                                path_w_param="pw", path_h_param="ph")],
            metadata=PartMetadata(seed=1, generator_version="test", family="swept", tier=0, op_count=1),
        )
        text = describe_dimensions(part)["text"]
        self.assertIn("swept along a curved path", text)

    def test_tapped_summary(self):
        from cadquarry.ir import TappedHolesOp
        part = PartIR(
            id="tapped_desc_test",
            params={"w": ParamSpec(type="float", default=55.0, group="B", label="w"),
                    "d": ParamSpec(type="float", default=45.0, group="B", label="d"),
                    "h": ParamSpec(type="float", default=12.0, group="B", label="h"),
                    "maj": ParamSpec(type="float", default=6.0, group="T", label="maj"),
                    "pit": ParamSpec(type="float", default=1.0, group="T", label="pit")},
            operations=[TappedHolesOp(body="box", width=ref("w"), depth=ref("d"),
                                      height=ref("h"), major_d=ref("maj"), pitch=ref("pit"),
                                      placement="corners")],
            metadata=PartMetadata(seed=1, generator_version="test", family="tapped", tier=1, op_count=1),
        )
        text = describe_dimensions(part)["text"]
        self.assertIn("M6x1 tapped", text)
        self.assertIn("Four", text)

    def test_sketched_revolve_summary(self):
        from cadquarry.ir import SketchedProfile, SketchSeg, SketchedRevolveOp
        prof = SketchedProfile(
            w_param="sk_r", h_param="sk_h", start=(0.5, 0.0),
            segments=[
                SketchSeg(kind="line", x=0.8, y=0.5),
                SketchSeg(kind="line", x=0.3, y=1.0),
                SketchSeg(kind="line", x=0.0, y=1.0),
                SketchSeg(kind="line", x=0.0, y=0.0),
            ],
        )
        part = PartIR(
            id="skrev_desc_test",
            params={"sk_r": ParamSpec(type="float", default=30.0, group="Sketch", label="R"),
                    "sk_h": ParamSpec(type="float", default=50.0, group="Sketch", label="H")},
            operations=[SketchedRevolveOp(half_profile=prof)],
            metadata=PartMetadata(seed=1, generator_version="test", family="sketched", tier=0, op_count=1),
        )
        text = describe_dimensions(part)["text"]
        self.assertIn("Revolved organic body", text)


class TestEmitIncludesDimensions(unittest.TestCase):
    def test_meta_json_has_dimensions(self):
        meta = emit_meta_json(_plate_part())
        self.assertIn("dimensions", meta)
        self.assertIn("text", meta["dimensions"])
        self.assertTrue(meta["dimensions"]["text"])

    def test_write_part_emits_dims_sidecar(self):
        part = _plate_part()
        with tempfile.TemporaryDirectory() as d:
            paths = write_part(part, Path(d))
            self.assertIn("dims", paths)
            self.assertTrue(paths["dims"].exists())
            sidecar_text = paths["dims"].read_text(encoding="utf-8").strip()
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
            self.assertEqual(sidecar_text, meta["dimensions"]["text"])


if __name__ == "__main__":
    unittest.main()
