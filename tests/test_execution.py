"""
Execution-backed tests.

These require a working CadQuery/OCP install and are skipped automatically
when cadquery cannot be imported, so the rest of the suite still runs on a
plain Python without the heavy OCP wheels.

Run with the project's CadQuery environment, e.g.:

    .venv/bin/python -m unittest tests.test_execution
"""
from __future__ import annotations

import unittest

try:
    import cadquery  # noqa: F401
    HAS_CADQUERY = True
except Exception:
    HAS_CADQUERY = False

# The `mech` families (gear / threaded) need the build123d stack on top of
# cadquery; gate those tests so the suite still runs with only cadquery present.
try:
    import build123d  # noqa: F401
    import bd_warehouse.thread  # noqa: F401
    import py_gearworks  # noqa: F401
    HAS_MECH = True
except Exception:
    HAS_MECH = False

# Families that require the `mech` extra to execute.
MECH_FAMILIES = {"gear", "threaded"}

# Generous timeout: gear/thread solving is far slower than primitive families.
MECH_TIMEOUT = 120

from cadquarry.compose import compose
from cadquarry.emit import emit_source
from cadquarry.execute import execute_source
from cadquarry.filter import compute_signature, is_valid


@unittest.skipUnless(HAS_CADQUERY, "cadquery not installed")
class TestGeneratedPartsExecute(unittest.TestCase):
    def test_majority_valid_by_construction(self):
        """A random sample of generated parts should overwhelmingly build."""
        ok = 0
        total = 0
        failures = []
        for i in range(16):
            part = compose(seed=10_000 + i, index=i)
            # Skip mech-family parts when the mech stack isn't installed —
            # they can't execute without it and would unfairly drag the rate.
            if part.metadata.family in MECH_FAMILIES and not HAS_MECH:
                continue
            timeout = MECH_TIMEOUT if part.metadata.family in MECH_FAMILIES else 30
            result = execute_source(emit_source(part), timeout=timeout)
            total += 1
            valid, reason = is_valid(result)
            if valid:
                ok += 1
            else:
                failures.append((part.id, part.metadata.family, reason))
        self.assertGreaterEqual(
            ok / total, 0.8,
            f"only {ok}/{total} parts valid; failures: {failures}",
        )

    def test_every_family_builds(self):
        """Each declared family produces at least one valid part."""
        from cadquarry.compose import _FAMILY_SAMPLERS

        for family in _FAMILY_SAMPLERS:
            if family in MECH_FAMILIES and not HAS_MECH:
                continue
            timeout = MECH_TIMEOUT if family in MECH_FAMILIES else 30
            built = False
            for i in range(6):
                part = compose(seed=2000 + i, index=i, family=family)
                result = execute_source(emit_source(part), timeout=timeout)
                if is_valid(result)[0]:
                    built = True
                    break
            self.assertTrue(built, f"family {family!r} produced no valid part in 6 tries")


@unittest.skipUnless(HAS_CADQUERY, "cadquery not installed")
class TestReproducibility(unittest.TestCase):
    def test_signature_deterministic(self):
        """Executing identical source twice yields an identical signature."""
        part = compose(seed=4242, index=0)
        code = emit_source(part)
        s1 = compute_signature(execute_source(code, timeout=30))
        s2 = compute_signature(execute_source(code, timeout=30))
        self.assertEqual(s1.as_tuple(), s2.as_tuple())

    def test_same_seed_same_signature(self):
        """Same seed → same IR → same emitted code → same geometry signature."""
        a = compose(seed=777, index=0)
        b = compose(seed=777, index=0)
        self.assertEqual(emit_source(a), emit_source(b))
        sa = compute_signature(execute_source(emit_source(a), timeout=30))
        sb = compute_signature(execute_source(emit_source(b), timeout=30))
        self.assertEqual(sa.as_tuple(), sb.as_tuple())


@unittest.skipUnless(HAS_CADQUERY, "cadquery not installed")
class TestRoundTrip(unittest.TestCase):
    def test_canonical_instance_valid(self):
        """build(defaults) — the canonical sample — executes to a valid solid."""
        for seed in (1, 2, 3, 4, 5):
            part = compose(seed=seed, index=0)
            result = execute_source(emit_source(part), timeout=30)
            valid, reason = is_valid(result)
            # Default values are guaranteed valid by Stage C; if a particular
            # seed lands on an invalid default that's a generator bug worth
            # surfacing.
            self.assertTrue(valid, f"{part.id} invalid at defaults: {reason}")


@unittest.skipUnless(HAS_CADQUERY, "cadquery not installed")
class TestParameterSweep(unittest.TestCase):
    def test_single_parameter_moves_stay_mostly_valid(self):
        """
        Sweeping each numeric parameter across its declared range, one at a
        time, should keep the geometry valid for the large majority of moves.
        (Cross-parameter combinations are out of scope — see docs.)
        """
        part = compose(seed=7, index=0, family="plate", tier=2)
        code = emit_source(part)

        valid = 0
        total = 0
        flagged = []
        for name, spec in part.params.items():
            if spec.type not in ("float", "int") or spec.min is None or spec.max is None:
                continue
            for frac in (0.0, 0.5, 1.0):
                val = spec.min + (spec.max - spec.min) * frac
                if spec.type == "int":
                    val = int(round(val))
                result = execute_source(code, params={name: val}, timeout=30)
                total += 1
                if is_valid(result)[0]:
                    valid += 1
                else:
                    flagged.append((name, val))

        self.assertGreater(total, 0, "no numeric parameters to sweep")
        self.assertGreaterEqual(
            valid / total, 0.6,
            f"too many single-parameter moves invalid: {flagged}",
        )


@unittest.skipUnless(HAS_MECH, "mech extra (build123d/bd_warehouse/py_gearworks) not installed")
class TestMechFamiliesExecute(unittest.TestCase):
    """Execution-backed checks for the gear / threaded families.

    Gated on the `mech` extra so they only run where the build123d stack is
    available.  Each must bridge back to a single CadQuery solid and export.
    """

    def _build(self, part):
        result = execute_source(emit_source(part), timeout=MECH_TIMEOUT)
        return result

    def test_spur_gear_builds_single_solid(self):
        part = compose(seed=99, index=0, family="gear", tier=0)
        result = self._build(part)
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.n_solids, 1)
        self.assertGreater(result.volume, 1.0)

    def test_iso_thread_rod_builds_single_solid(self):
        part = compose(seed=77, index=0, family="threaded", tier=0)
        result = self._build(part)
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.n_solids, 1)
        self.assertGreater(result.volume, 1.0)

    def test_gear_exports_step_and_stl(self):
        import tempfile
        from pathlib import Path
        from cadquarry.emit import write_part
        from cadquarry.export import export_part

        part = compose(seed=99, index=0, family="gear", tier=1)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            paths = write_part(part, out)
            exported = export_part(paths["py"], out / "geometry",
                                   formats=["step", "stl"], timeout=MECH_TIMEOUT)
            self.assertIn("step", exported)
            self.assertIn("stl", exported)
            self.assertTrue(exported["step"].exists() and exported["step"].stat().st_size > 0)
            self.assertTrue(exported["stl"].exists() and exported["stl"].stat().st_size > 0)

    def test_thread_exports_step_and_stl(self):
        import tempfile
        from pathlib import Path
        from cadquarry.emit import write_part
        from cadquarry.export import export_part

        part = compose(seed=77, index=0, family="threaded", tier=0)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            paths = write_part(part, out)
            exported = export_part(paths["py"], out / "geometry",
                                   formats=["step", "stl"], timeout=MECH_TIMEOUT)
            self.assertIn("step", exported)
            self.assertIn("stl", exported)
            self.assertTrue(exported["step"].exists() and exported["step"].stat().st_size > 0)

    def test_mech_signature_deterministic(self):
        part = compose(seed=99, index=0, family="gear", tier=2)
        code = emit_source(part)
        s1 = compute_signature(execute_source(code, timeout=MECH_TIMEOUT))
        s2 = compute_signature(execute_source(code, timeout=MECH_TIMEOUT))
        self.assertEqual(s1.as_tuple(), s2.as_tuple())


if __name__ == "__main__":
    unittest.main()
