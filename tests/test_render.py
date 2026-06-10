"""
Tests for the render pipeline's output passes and the view/pass selection that
wires them through export, the dataset packer and the dataset card.

The GPU integration test is skipped unless a headless EGL GL stack is available,
so the pure-Python wiring is always exercised in CI while the actual shading is
verified wherever a GPU is present.
"""
from __future__ import annotations

import numpy as np
import pytest

from cadquarry import dataset, export, publish, render


# ---------------------------------------------------------------------------
# View / pass selection helpers (no GPU needed)
# ---------------------------------------------------------------------------

def test_resolve_views_presets_and_lists():
    assert export.resolve_views(None) == dict(export.STANDARD_VIEWS)
    assert list(export.resolve_views("iso-corners")) == ["iso_fr", "iso_fl", "iso_br", "iso_bl"]
    assert list(export.resolve_views("ortho")) == ["front", "top", "right"]
    # A comma list of explicit names, order preserved.
    assert list(export.resolve_views("top,front")) == ["top", "front"]
    # Case-insensitive preset name.
    assert list(export.resolve_views("ISO-CORNERS")) == ["iso_fr", "iso_fl", "iso_br", "iso_bl"]


def test_resolve_views_rejects_unknown():
    with pytest.raises(ValueError):
        export.resolve_views("nope")


def test_resolve_passes_canonical_order_and_dedup():
    assert export.resolve_passes(None) == ("shaded",)
    # De-duped and returned in canonical PASSES order regardless of input order.
    assert export.resolve_passes("edge,normal,shaded,edge") == ("shaded", "normal", "edge")
    with pytest.raises(ValueError):
        export.resolve_passes("glossy")


def test_render_filename_back_compat():
    # Shaded keeps the historical bare name so existing galleries/columns hold.
    assert export.render_filename("iso_fr", "shaded") == "iso_fr.png"
    assert export.render_filename("iso_fr", "normal") == "iso_fr_normal.png"
    assert export.render_filename("front", "edge") == "front_edge.png"


def test_edge_params_monotonic_with_crispness():
    soft = render._edge_params(0.0)
    crisp = render._edge_params(1.0)
    # Crisper => more sensitive (lower onset), thinner lines, narrower band.
    assert crisp["n_lo"] < soft["n_lo"]
    assert crisp["d_lo"] < soft["d_lo"]
    assert crisp["width"] < soft["width"]
    # Clamped to [0, 1].
    assert render._edge_params(-5.0) == soft
    assert render._edge_params(5.0) == crisp


# ---------------------------------------------------------------------------
# Dataset packing column wiring (no GPU needed)
# ---------------------------------------------------------------------------

def test_render_columns_shaded_only_matches_legacy():
    cols = dataset.render_columns(["front", "iso_fr"], ("shaded",))
    assert cols == [("render_front", "front.png"), ("render_iso_fr", "iso_fr.png")]


def test_render_columns_with_extra_passes():
    cols = dataset.render_columns(["iso_fr"], ("shaded", "normal", "edge"))
    assert cols == [
        ("render_iso_fr", "iso_fr.png"),
        ("render_iso_fr_normal", "iso_fr_normal.png"),
        ("render_iso_fr_edge", "iso_fr_edge.png"),
    ]


def test_available_render_passes_detects_variants(tmp_path):
    renders = tmp_path / "renders"
    part = renders / "p0001"
    part.mkdir(parents=True)
    (part / "iso_fr.png").write_bytes(b"x")
    (part / "iso_fr_normal.png").write_bytes(b"x")
    (part / "iso_fr_edge.png").write_bytes(b"x")
    got = dataset.available_render_passes(renders, ["iso_fr"])
    assert got == ("shaded", "normal", "edge")  # canonical order, no depth


def test_available_render_passes_shaded_only(tmp_path):
    renders = tmp_path / "renders"
    part = renders / "p0001"
    part.mkdir(parents=True)
    (part / "iso_fr.png").write_bytes(b"x")
    assert dataset.available_render_passes(renders, ["iso_fr"]) == ("shaded",)
    # Missing dir degrades gracefully.
    assert dataset.available_render_passes(tmp_path / "none", ["iso_fr"]) == ("shaded",)


def test_card_feature_schema_matches_render_columns():
    # The HF card's image features must name exactly the parquet render columns.
    passes = ("shaded", "normal", "edge")
    yaml = publish._features_yaml(True, False, False, render_passes=passes)
    for col, _fname in dataset.render_columns(dataset.RENDER_VIEWS, passes):
        assert f"- name: {col}" in yaml


# ---------------------------------------------------------------------------
# GPU integration: actually render the passes off the shared G-buffer
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not render.render_available(), reason="no headless EGL GL stack")
def test_render_mesh_emits_all_passes():
    # A unit cube centered at the origin.
    v = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float)
    f = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
    ])
    r = render.get_renderer(size=64, ssaa=1)
    out = r.render_mesh(v, f, views={"iso_fr": render.STANDARD_VIEWS["iso_fr"]},
                        passes=("shaded", "normal", "depth", "edge"))
    imgs = out["iso_fr"]
    assert set(imgs) == {"shaded", "normal", "depth", "edge"}
    for arr in imgs.values():
        assert arr.shape == (64, 64, 4) and arr.dtype == np.uint8

    cov = imgs["normal"][..., 3] > 127  # covered pixels
    assert cov.any(), "the cube should cover some pixels"

    # Normal map encodes a unit vector -> ||rgb*2-1|| ~= 1 on covered pixels.
    n = imgs["normal"][..., :3].astype(np.float32) / 255.0 * 2.0 - 1.0
    mag = np.linalg.norm(n[cov], axis=-1)
    assert np.abs(mag - 1.0).mean() < 0.1

    # Edge pass is a transparent overlay: black where drawn, alpha = strength,
    # and crisper than the shaded pass (it has at least some strong-edge pixels).
    edge_a = imgs["edge"][..., 3]
    assert (edge_a > 200).any(), "a cube's silhouette/creases should be crisp"
    assert np.all(imgs["edge"][edge_a > 200][..., :3] < 10), "edge lines are black"


@pytest.mark.skipif(not render.render_available(), reason="no headless EGL GL stack")
def test_render_mesh_rejects_unknown_pass():
    v = np.zeros((3, 3)); v[1, 0] = 1; v[2, 1] = 1
    f = np.array([[0, 1, 2]])
    r = render.get_renderer(size=32, ssaa=1)
    with pytest.raises(ValueError):
        r.render_mesh(v, f, passes=("shaded", "bogus"))
