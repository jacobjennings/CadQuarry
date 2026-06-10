"""
Tests for the nested-prefix size ladder + resumable/extensible generation.

These exercise the deterministic-prefix-stream property (the first N accepted
parts of a count=M≥N run are the count=N corpus), the resume cursor, and the
publish-time prefix slicing — all without a CAD backend (the ``--no-exec``
generation path's ir_hash dedup is deterministic on its own).
"""
import argparse
import contextlib
import io
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from cadquarry.__main__ import (
    _load_publish_ladder,
    _load_publish_meta,
    cmd_generate,
)
from cadquarry.dataset import DatasetWriter, load_state, pack_corpus_jsonl


def _gen(out: Path, count: int, seed: int = 4242, extend: bool = False,
         force: bool = False) -> int:
    args = argparse.Namespace(
        count=count, seed=seed, out=str(out), config=None, timeout=30.0,
        no_exec=True, workers=0, family=None, tier=None, verbose=False,
        extend=extend, force=force,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return cmd_generate(args)


def _ir_hashes(corpus: Path) -> list[str]:
    return [json.loads(line)["ir_hash"]
            for line in (corpus / "manifest.jsonl").read_text().splitlines()]


def _part_ids(jsonl: Path) -> list[str]:
    return [json.loads(line)["part_id"] for line in jsonl.read_text().splitlines()]


class TestResumeCursor(unittest.TestCase):
    def test_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            w = DatasetWriter(Path(d), seed=7)
            w.finalize(attempted=123)
            st = load_state(Path(d))
            self.assertEqual(st["seed"], 7)
            self.assertEqual(st["attempted"], 123)
            self.assertEqual(st["accepted"], 0)

    def test_extend_is_bit_identical_to_fresh(self):
        """20-then-extend-to-40 yields the same corpus as a fresh count=40."""
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "A", Path(d) / "B"
            _gen(a, 40)
            _gen(b, 20)
            self.assertEqual(load_state(b)["accepted"], 20)
            _gen(b, 40, extend=True)
            ha, hb = _ir_hashes(a), _ir_hashes(b)
            self.assertEqual(len(ha), 40)
            self.assertEqual(ha, hb)

    def test_extend_refuses_seed_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d) / "B"
            _gen(b, 20, seed=1)
            self.assertEqual(_gen(b, 40, seed=2, extend=True), 1)  # refuses
            self.assertEqual(_gen(b, 40, seed=2, extend=True, force=True), 0)  # override


class TestPrefixSlice(unittest.TestCase):
    def test_pack_limit_is_a_nested_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            corpus = Path(d) / "c"
            _gen(corpus, 40)
            out = Path(d) / "o.jsonl"
            n10 = pack_corpus_jsonl(corpus, out, limit=10)
            rows10 = _part_ids(out)
            n20 = pack_corpus_jsonl(corpus, out, limit=20)
            rows20 = _part_ids(out)
            nall = pack_corpus_jsonl(corpus, out)
            rowsall = _part_ids(out)
            self.assertEqual((n10, n20, nall), (10, 20, 40))
            self.assertEqual(rows10, rows20[:10])   # nested prefixes
            self.assertEqual(rows20, rowsall[:20])


class TestPrefixLadderLoaders(unittest.TestCase):
    def _write(self, d: Path) -> Path:
        p = d / "v.toml"
        p.write_text(textwrap.dedent("""
            [meta]
            generator_version = "0.6.0"
            [publish]
            mode = "prefix"
            base_seed = 99
            base_tag = "base"
            [[publish.corpus]]
            tag = "1k"
            count = 1000
            [[publish.corpus]]
            tag = "2k"
            count = 2000
        """))
        return p

    def test_prefix_mode_defaults_seed(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(Path(d))
            ladder = _load_publish_ladder(p)
            meta = _load_publish_meta(p)
            self.assertEqual(ladder["1k"]["seed"], 99)
            self.assertEqual(ladder["2k"]["seed"], 99)  # all share base_seed
            self.assertEqual(meta["mode"], "prefix")
            self.assertEqual(meta["base_count"], 2000)
            self.assertEqual(meta["base_tag"], "base")


if __name__ == "__main__":
    unittest.main()
