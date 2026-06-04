"""Tests for the composition / sampling layer — no cadquery required."""
import ast
import unittest
from cadquarry.compose import (
    compose,
    sample_bracket,
    sample_plate,
    sample_revolved,
    sample_block,
    sample_enclosure,
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
        for tier in [1, 2, 3]:
            part = sample_bracket(Random(11), tier=tier, config={}, seed=11, index=0)
            self._check(part)
            self.assertEqual(part.operations[0].type, "lbracket")


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
