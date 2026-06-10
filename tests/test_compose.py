"""Tests for the composition / sampling layer — no cadquery required."""
import ast
import unittest
from cadquarry.compose import (
    compose,
    sample_bracket,
    sample_compound,
    sample_plate,
    sample_revolved,
    sample_block,
    sample_enclosure,
    _sample_block_tube,
    _sample_block_tab,
    _sample_pedestal,
)
from cadquarry.emit import emit_source
from cadquarry.ir import PartIR


def _valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


class TestCompose(unittest.TestCase):
    def test_returns_part_ir(self):
        self.assertIsInstance(compose(seed=42, index=0), PartIR)

    def test_has_params(self):
        self.assertGreater(len(compose(seed=42).params), 0)

    def test_has_operations(self):
        self.assertGreater(len(compose(seed=42).operations), 0)

    def test_metadata_set(self):
        part = compose(seed=42, index=5)
        self.assertEqual(part.metadata.seed, 42)
        self.assertNotEqual(part.metadata.family, "")
        self.assertIn(part.metadata.tier, [0, 1, 2, 3])

    def test_deterministic(self):
        h1 = compose(seed=1234, index=0).ir_hash()
        h2 = compose(seed=1234, index=0).ir_hash()
        self.assertEqual(h1, h2)

    def test_different_seeds_differ(self):
        h1 = compose(seed=100, index=0).ir_hash()
        h2 = compose(seed=101, index=0).ir_hash()
        self.assertNotEqual(h1, h2)

    def test_emits_valid_python(self):
        for seed in range(20):
            part = compose(seed=seed)
            code = emit_source(part)
            self.assertTrue(_valid_python(code), f"seed={seed} produced invalid Python")

    def test_family_override(self):
        self.assertEqual(compose(seed=42, family="plate").metadata.family, "plate")
        self.assertEqual(compose(seed=42, family="revolved").metadata.family, "revolved")

    def test_tier_override(self):
        for tier in [0, 1, 2, 3]:
            part = compose(seed=99, family="plate", tier=tier)
            self.assertEqual(part.metadata.tier, tier)

    def test_emit_contains_build(self):
        code = emit_source(compose(seed=42))
        self.assertIn("def build(p):", code)
        self.assertIn("PARAMS", code)

    def test_emit_source_reproducible(self):
        # Same seed → byte-identical emitted source.
        self.assertEqual(
            emit_source(compose(seed=2024, index=3)),
            emit_source(compose(seed=2024, index=3)),
        )

    def test_symmetry_recorded_and_valid(self):
        seen = set()
        for seed in range(80):
            mode = compose(seed=seed, index=0).metadata.symmetry
            self.assertIn(mode, {"none", "mirror_x", "mirror_xy", "radial"})
            seen.add(mode)
        # Over 80 parts we should see asymmetric and at least one symmetric mode.
        self.assertIn("none", seen)
        self.assertTrue(seen - {"none"}, "no symmetric parts ever sampled")

    def test_bracket_family_available(self):
        from cadquarry.compose import _FAMILY_SAMPLERS
        self.assertIn("bracket", _FAMILY_SAMPLERS)
        part = compose(seed=42, family="bracket")
        self.assertEqual(part.metadata.family, "bracket")

    def test_sketched_family_available(self):
        from cadquarry.compose import _FAMILY_SAMPLERS
        self.assertIn("sketched", _FAMILY_SAMPLERS)
        part = compose(seed=42, family="sketched")
        self.assertEqual(part.metadata.family, "sketched")

    def test_sketched_tiers_emit_valid_python(self):
        from cadquarry.compose import sample_sketched
        from random import Random
        for tier in range(4):
            for seed in (1, 7, 13):
                part = sample_sketched(Random(seed), tier=tier, config={}, seed=seed, index=0)
                self.assertGreaterEqual(len(part.params), 2)
                self.assertIn(part.operations[0].type, ("extrude", "sketched_revolve"))
                code = emit_source(part)
                self.assertTrue(_valid_python(code), f"sketched tier={tier} seed={seed} bad Python")
                # The base op must draw a freeform closed loop.
                self.assertIn(".moveTo(", code)
                self.assertIn(".close()", code)

    def test_sketched_both_modes_appear(self):
        """Over many seeds we sample both extruded and revolved sketches."""
        base_ops = set()
        for seed in range(60):
            part = compose(seed=4000 + seed, index=0, family="sketched")
            base_ops.add(part.operations[0].type)
        self.assertIn("extrude", base_ops)
        self.assertIn("sketched_revolve", base_ops)


class TestFamilySamplers(unittest.TestCase):
    def _check(self, part: PartIR) -> None:
        self.assertGreaterEqual(len(part.params), 2)
        self.assertGreaterEqual(len(part.operations), 1)
        code = emit_source(part)
        self.assertTrue(_valid_python(code))
        self.assertIn('p["', code)

    def test_plate_tiers(self):
        from random import Random
        for tier in range(4):
            self._check(sample_plate(Random(42), tier=tier, config={}, seed=42, index=0))

    def test_revolved_tiers(self):
        from random import Random
        for tier in range(3):
            self._check(sample_revolved(Random(99), tier=tier, config={}, seed=99, index=0))

    def test_block_tiers(self):
        from random import Random
        for tier in [1, 2, 3]:
            self._check(sample_block(Random(7), tier=tier, config={}, seed=7, index=0))

    def test_enclosure(self):
        from random import Random
        part = sample_enclosure(Random(55), tier=2, config={}, seed=55, index=0)
        self._check(part)
        op_types = [op.type for op in part.operations]
        self.assertIn("shell", op_types)

    def test_bracket_tiers(self):
        from random import Random
        bracket_ops = {"lbracket", "cbracket", "zbracket"}
        for tier in [1, 2, 3]:
            part = sample_bracket(Random(11), tier=tier, config={}, seed=11, index=0)
            self._check(part)
            self.assertIn(part.operations[0].type, bracket_ops)

    def test_bracket_shapes_forced(self):
        """Each shape weight, in isolation, yields its dedicated op."""
        from random import Random
        cases = {
            "l": "lbracket",
            "c": "cbracket",
            "z": "zbracket",
        }
        for shape, op_type in cases.items():
            cfg = {"families": {"bracket": {"shape_weights": {shape: 1.0}}}}
            for tier in [1, 2, 3]:
                part = sample_bracket(Random(7), tier=tier, config=cfg, seed=7, index=0)
                self._check(part)
                self.assertEqual(part.operations[0].type, op_type)

    def test_bracket_shapes_all_appear(self):
        """Across many seeds the default weights surface every shape."""
        from random import Random
        seen = set()
        for seed in range(200):
            part = sample_bracket(Random(seed), tier=2, config={}, seed=seed, index=0)
            seen.add(part.operations[0].type)
        self.assertEqual(seen, {"lbracket", "cbracket", "zbracket"})


class TestCompoundFamily(unittest.TestCase):
    def _check(self, part: PartIR) -> None:
        self.assertIsInstance(part, PartIR)
        self.assertGreaterEqual(len(part.params), 4)
        self.assertGreaterEqual(len(part.operations), 2)
        code = emit_source(part)
        self.assertTrue(_valid_python(code), f"invalid Python for {part.id}")
        self.assertIn('p["', code)
        # compound parts must contain at least one AttachOp
        op_types = [op.type for op in part.operations]
        self.assertIn("attach", op_types)

    def test_compound_family_in_registry(self):
        from cadquarry.compose import _FAMILY_SAMPLERS
        self.assertIn("compound", _FAMILY_SAMPLERS)

    def test_compose_compound_selectable(self):
        part = compose(seed=7777, family="compound", tier=1)
        self.assertEqual(part.metadata.family, "compound")
        self.assertGreaterEqual(len(part.operations), 2)

    def test_compound_all_tiers(self):
        from random import Random
        for tier in [1, 2, 3]:
            part = sample_compound(Random(42), tier=tier, config={}, seed=42, index=0)
            self._check(part)
            self.assertEqual(part.metadata.tier, tier)

    def test_block_tube_archetype(self):
        from random import Random
        for seed in range(10):
            part = _sample_block_tube(Random(seed), tier=2, config={}, seed=seed, index=0)
            self._check(part)
            op_types = [op.type for op in part.operations]
            self.assertIn("box", op_types)
            self.assertIn("attach", op_types)

    def test_block_tab_archetype(self):
        from random import Random
        for seed in range(10):
            part = _sample_block_tab(Random(seed), tier=1, config={}, seed=seed, index=0)
            self._check(part)
            op_types = [op.type for op in part.operations]
            self.assertIn("box", op_types)
            self.assertIn("attach", op_types)
            self.assertIn("holes", op_types)

    def test_pedestal_archetype(self):
        from random import Random
        for seed in range(10):
            part = _sample_pedestal(Random(seed), tier=1, config={}, seed=seed, index=0)
            self._check(part)
            op_types = [op.type for op in part.operations]
            self.assertIn("box", op_types)
            self.assertIn("attach", op_types)

    def test_attach_op_face_variety(self):
        """Block tube archetype uses a variety of faces across seeds."""
        from random import Random
        faces_seen = set()
        for seed in range(60):
            part = _sample_block_tube(Random(seed), tier=1, config={}, seed=seed, index=0)
            for op in part.operations:
                if op.type == "attach":
                    faces_seen.add(op.face)
        self.assertGreater(len(faces_seen), 1, "tube always on the same face — no variety")

    def test_block_tier3_may_include_side_tube(self):
        """sample_block tier 3 occasionally includes an attach op."""
        from random import Random
        attach_seen = False
        for seed in range(80):
            part = sample_block(Random(seed), tier=3, config={}, seed=seed, index=0)
            if any(op.type == "attach" for op in part.operations):
                attach_seen = True
                code = emit_source(part)
                self.assertTrue(_valid_python(code))
                break
        self.assertTrue(attach_seen, "block tier 3 never generated a side tube in 80 seeds")

    def test_compound_emits_valid_python_many_seeds(self):
        from random import Random
        for seed in range(30):
            part = sample_compound(Random(seed), tier=rng_tier(seed), config={}, seed=seed, index=0)
            code = emit_source(part)
            self.assertTrue(_valid_python(code), f"seed={seed} produced invalid Python")


def rng_tier(seed: int) -> int:
    from random import Random
    return Random(seed).randint(1, 3)


class TestHoleVariety(unittest.TestCase):
    """Verify that the new hole shapes and placements are reachable through compose."""

    def _collect_hole_shapes(self, seeds: range, family: str = "plate", tier: int = 2) -> set[str]:
        from random import Random
        from cadquarry.compose import sample_plate, sample_block
        sampler = sample_plate if family == "plate" else sample_block
        shapes: set[str] = set()
        for seed in seeds:
            part = sampler(Random(seed), tier=tier, config={}, seed=seed, index=0)
            for op in part.operations:
                if op.type == "holes":
                    shapes.add(op.shape)
        return shapes

    def _collect_hole_placements(self, seeds: range, family: str = "plate", tier: int = 2) -> set[str]:
        from random import Random
        from cadquarry.compose import sample_plate
        placements: set[str] = set()
        for seed in seeds:
            part = sample_plate(Random(seed), tier=tier, config={}, seed=seed, index=0)
            for op in part.operations:
                if op.type == "holes":
                    placements.add(op.placement)
        return placements

    def test_square_holes_appear(self):
        shapes = self._collect_hole_shapes(range(300))
        self.assertIn("square", shapes, "square holes never sampled in 300 seeds")

    def test_slot_holes_appear(self):
        shapes = self._collect_hole_shapes(range(300))
        self.assertIn("slot", shapes, "slot holes never sampled in 300 seeds")

    def test_round_holes_still_appear(self):
        shapes = self._collect_hole_shapes(range(300))
        self.assertIn("round", shapes)

    def test_staggered_placement_appears(self):
        placements = self._collect_hole_placements(range(300))
        self.assertIn("staggered", placements, "staggered placement never sampled in 300 seeds")

    def test_grid_placement_appears(self):
        placements = self._collect_hole_placements(range(300))
        self.assertIn("grid", placements)

    def test_new_hole_shapes_emit_valid_python(self):
        """Every seed that produces square/slot/staggered must emit parseable Python."""
        import ast
        from random import Random
        from cadquarry.compose import sample_plate
        from cadquarry.emit import emit_source
        for seed in range(300):
            part = sample_plate(Random(seed), tier=2, config={}, seed=seed, index=0)
            for op in part.operations:
                if op.type == "holes" and op.shape in ("square", "slot") or (
                        op.type == "holes" and op.placement == "staggered"):
                    code = emit_source(part)
                    try:
                        ast.parse(code)
                    except SyntaxError as e:
                        self.fail(f"seed={seed} produced invalid Python: {e}")
                    break  # one check per part is enough

    def test_block_square_and_staggered_appear(self):
        """Block family also surfaces the new hole types."""
        from random import Random
        from cadquarry.compose import sample_block
        shapes: set[str] = set()
        placements: set[str] = set()
        for seed in range(300):
            part = sample_block(Random(seed), tier=1, config={}, seed=seed, index=0)
            for op in part.operations:
                if op.type == "holes":
                    shapes.add(op.shape)
                    placements.add(op.placement)
        self.assertIn("square", shapes, "block never got square holes in 300 seeds")
        self.assertIn("staggered", placements, "block never got staggered in 300 seeds")

    def test_generator_version_bumped(self):
        from cadquarry.compose import GENERATOR_VERSION
        major, minor, _ = GENERATOR_VERSION.split(".")
        self.assertGreaterEqual(int(minor), 3, "GENERATOR_VERSION should be at least 0.3.x")

    def test_slot_params_present(self):
        """Parts with slot holes must include both width and length params."""
        from random import Random
        from cadquarry.compose import sample_plate
        for seed in range(300):
            part = sample_plate(Random(seed), tier=2, config={}, seed=seed, index=0)
            for op in part.operations:
                if op.type == "holes" and op.shape == "slot":
                    self.assertIn("hole_slot_len", part.params,
                                  f"seed={seed}: slot hole missing hole_slot_len param")
                    break


class TestFilter(unittest.TestCase):
    def test_dedup_store(self):
        from cadquarry.filter import DedupStore, GeometrySignature, SIGNATURE_VERSION
        store = DedupStore()
        sig = GeometrySignature(version=SIGNATURE_VERSION, volume=1234.0, surface_area=567.0,
                                principal_moments=(100.0, 200.0, 300.0),
                                n_faces=8, n_edges=12, n_vertices=8)
        self.assertFalse(store.is_duplicate("abc123", sig))
        store.register("abc123", sig)
        self.assertTrue(store.is_duplicate("abc123"))
        self.assertTrue(store.is_duplicate("xxx", sig))

    def test_ir_hash_dedup(self):
        from cadquarry.filter import DedupStore
        store = DedupStore()
        self.assertFalse(store.is_duplicate("aaa"))
        store.register("aaa")
        self.assertTrue(store.is_duplicate("aaa"))

    def test_signature_moments_sorted_and_quantized(self):
        from cadquarry.filter import compute_signature
        from cadquarry.execute import ExecuteResult
        # Two results with the SAME moments in different order should yield the
        # same signature (sorting makes orientation irrelevant).
        r1 = ExecuteResult(success=True, volume=1000.0, surface_area=600.0,
                           principal_moments=[300.0, 100.0, 200.0],
                           n_faces=6, n_edges=12, n_vertices=8)
        r2 = ExecuteResult(success=True, volume=1000.0, surface_area=600.0,
                           principal_moments=[100.0, 200.0, 300.0],
                           n_faces=6, n_edges=12, n_vertices=8)
        s1 = compute_signature(r1)
        s2 = compute_signature(r2)
        self.assertEqual(s1.as_tuple(), s2.as_tuple())
        # Moments are ascending in the signature.
        self.assertEqual(list(s1.principal_moments), sorted(s1.principal_moments))

    def test_signature_moments_discriminate(self):
        from cadquarry.filter import compute_signature
        from cadquarry.execute import ExecuteResult
        # Same volume/area/topology but different mass distribution → distinct.
        r1 = ExecuteResult(success=True, volume=1000.0, surface_area=600.0,
                           principal_moments=[100.0, 200.0, 300.0],
                           n_faces=6, n_edges=12, n_vertices=8)
        r2 = ExecuteResult(success=True, volume=1000.0, surface_area=600.0,
                           principal_moments=[150.0, 250.0, 400.0],
                           n_faces=6, n_edges=12, n_vertices=8)
        self.assertNotEqual(
            compute_signature(r1).as_tuple(), compute_signature(r2).as_tuple()
        )

    def test_round_sig(self):
        from cadquarry.filter import _round_sig
        self.assertEqual(_round_sig(12345.678, 3), 12300.0)
        self.assertAlmostEqual(_round_sig(0.001234, 3), 0.00123, places=6)
        self.assertEqual(_round_sig(0.0), 0.0)

    def test_signature_as_tuple(self):
        from cadquarry.filter import GeometrySignature, SIGNATURE_VERSION
        sig = GeometrySignature(version=SIGNATURE_VERSION, volume=1000.0, surface_area=800.0,
                                principal_moments=(10.0, 20.0, 30.0),
                                n_faces=6, n_edges=12, n_vertices=8)
        t = sig.as_tuple()
        self.assertEqual(len(t), 7)
        self.assertEqual(t[0], SIGNATURE_VERSION)

    def test_validity_ok(self):
        from cadquarry.filter import is_valid
        from cadquarry.execute import ExecuteResult
        result = ExecuteResult(success=True, volume=1000.0, surface_area=500.0,
                               bbox=[0, 0, 0, 40, 30, 6], n_faces=8, n_edges=12, n_vertices=8)
        valid, reason = is_valid(result)
        self.assertTrue(valid)
        self.assertEqual(reason, "")

    def test_validity_degenerate(self):
        from cadquarry.filter import is_valid
        from cadquarry.execute import ExecuteResult
        result = ExecuteResult(success=True, volume=0.001, surface_area=1.0,
                               bbox=[0, 0, 0, 10, 10, 0.001])
        valid, _ = is_valid(result)
        self.assertFalse(valid)

    def test_validity_failed_exec(self):
        from cadquarry.filter import is_valid
        from cadquarry.execute import ExecuteResult
        result = ExecuteResult(success=False, error="OCC error")
        valid, reason = is_valid(result)
        self.assertFalse(valid)
        self.assertIn("OCC error", reason)


if __name__ == "__main__":
    unittest.main()
