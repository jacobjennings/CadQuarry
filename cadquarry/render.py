"""
Headless GPU renderer for shaded part thumbnails.

Replaces the old matplotlib ``Poly3DCollection`` path, which had no real
z-buffer (painter's-algorithm depth errors bled back/internal triangles
through front faces) and flat per-face Lambert shading (a single dead-flat
color per face).

This renderer uses a standalone EGL OpenGL context (moderngl) and a small
deferred pipeline:

  1. Geometry pass  -> view-space position + flat geometric normal G-buffer
                       (true depth buffer, so no see-through artifacts).
  2. SSAO pass      -> screen-space ambient occlusion (depth in crevices and
                       along edges — the main "CAD" depth cue), then blurred.
  3. Lighting pass  -> soft hemispherical ambient + a single wrapped, positional
                       key light. The light is positional, so N·L varies across
                       a *flat* face, giving a gentle gradient instead of a flat
                       fill. No shadow maps, so lighting is inherently soft with
                       no harsh shadows.
  4. Supersample    -> everything renders at ``ssaa``× resolution and is box-
                       downsampled with premultiplied alpha for clean, anti-
                       aliased edges over a transparent background.

The context and all FBOs/programs are created once per process and reused
across parts and views (cheap on a GPU; the per-part cost is just a VBO
upload and a handful of draw calls).
"""
from __future__ import annotations

import colorsys
import hashlib
import math
from pathlib import Path

import numpy as np

# Matches the previous renderer's viewpoints so existing galleries/manifests
# keep working unchanged. (elevation_deg, azimuth_deg), Z-up.
STANDARD_VIEWS: dict[str, tuple[float, float]] = {
    "front":  (0.0,   -90.0),
    "top":    (90.0,  -90.0),
    "right":  (0.0,     0.0),
    "iso":    (28.0,  -55.0),
    "iso_fr": (30.0,  -45.0),
    "iso_fl": (30.0, -135.0),
    "iso_br": (30.0,   45.0),
    "iso_bl": (30.0,  135.0),
}

# Default matte CAD look (display-space colors).  The lit faces are kept below
# clamp so the color stays saturated rather than washing out toward white, and
# there is a clear range between the brightest (top) and shaded faces.
BASE_COLOR = (0.42, 0.56, 0.86)
SKY_COLOR = (0.72, 0.74, 0.82)
GROUND_COLOR = (0.26, 0.28, 0.36)
KEY_STRENGTH = 0.60
KEY_WRAP = 0.50  # 0 = hard terminator, 1 = very soft

# Subtle procedural surface texture (world-space fbm) so flat faces aren't dead
# flat.  Amplitude is a small brightness modulation; frequency is in cycles per
# bounding radius so the grain looks the same on parts of any size.
TEX_AMP = 0.10
TEX_CYCLES = 11.0

_KERNEL_SIZE = 24


# ---------------------------------------------------------------------------
# Deterministic per-part color
# ---------------------------------------------------------------------------

def color_for(name: str) -> tuple[float, float, float]:
    """
    Deterministic, well-spread, mostly-saturated base color for a part.

    Hashing the part id gives a stable color regardless of render order or
    parallelism.  Saturation stays high and value stays high so colors read as
    vivid and never so dark that shading crushes them out of dynamic range.
    """
    h = hashlib.md5(name.encode("utf-8")).digest()
    hue = int.from_bytes(h[0:4], "big") / 2**32
    sat = 0.62 + (h[4] / 255.0) * 0.23   # 0.62 .. 0.85
    val = 0.82 + (h[5] / 255.0) * 0.13   # 0.82 .. 0.95
    return colorsys.hsv_to_rgb(hue, sat, val)


# ---------------------------------------------------------------------------
# Small matrix helpers (no extra deps; GL wants column-major float32).
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = _normalize(target - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _ortho(half: float, near: float, far: float) -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = 1.0 / half
    m[1, 1] = 1.0 / half
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m


def _gl_bytes(m: np.ndarray) -> bytes:
    # numpy is row-major / math convention (M @ v); GL expects column-major.
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()


def _view_for(elev_deg: float, azim_deg: float, center: np.ndarray, dist: float):
    el = math.radians(elev_deg)
    az = math.radians(azim_deg)
    direction = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float64,
    )
    eye = center + direction * dist
    up = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    return _look_at(eye, center, up)


# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------

_GEOM_VERT = """
#version 330
in vec3 in_pos;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_viewpos;
void main() {
    vec4 vp = u_view * vec4(in_pos, 1.0);
    v_viewpos = vp.xyz;
    gl_Position = u_proj * vp;
}
"""

_GEOM_FRAG = """
#version 330
in vec3 v_viewpos;
layout(location = 0) out vec4 o_pos;
layout(location = 1) out vec4 o_nrm;
void main() {
    // Flat geometric normal straight from screen-space derivatives -> exact
    // per-triangle normal with no need to duplicate/precompute normals.
    vec3 n = normalize(cross(dFdx(v_viewpos), dFdy(v_viewpos)));
    // Visible surfaces face the camera (origin, looking -Z), so orient toward
    // it; makes shading robust to inconsistent STL winding.
    if (n.z < 0.0) n = -n;
    o_pos = vec4(v_viewpos, 1.0);   // w = coverage
    o_nrm = vec4(n, 1.0);
}
"""

_FS_VERT = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_SSAO_FRAG = """
#version 330
in vec2 v_uv;
out vec4 o_ao;
uniform sampler2D u_pos;
uniform sampler2D u_nrm;
uniform sampler2D u_noise;
uniform mat4 u_proj;
uniform vec2 u_noise_scale;
uniform vec3 u_kernel[KSIZE];
uniform float u_radius;
uniform float u_bias;
void main() {
    vec4 P = texture(u_pos, v_uv);
    if (P.w < 0.5) { o_ao = vec4(1.0); return; }
    vec3 pos = P.xyz;
    vec3 N = normalize(texture(u_nrm, v_uv).xyz);
    vec3 rv = texture(u_noise, v_uv * u_noise_scale).xyz;
    vec3 T = normalize(rv - N * dot(rv, N));
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    float occ = 0.0;
    for (int i = 0; i < KSIZE; i++) {
        vec3 sp = pos + TBN * u_kernel[i] * u_radius;
        vec4 off = u_proj * vec4(sp, 1.0);
        off.xyz /= off.w;
        off.xyz = off.xyz * 0.5 + 0.5;
        if (off.x < 0.0 || off.x > 1.0 || off.y < 0.0 || off.y > 1.0) continue;
        vec4 sP = texture(u_pos, off.xy);
        if (sP.w < 0.5) continue;
        float sample_z = sP.z;
        float range = smoothstep(0.0, 1.0, u_radius / max(1e-4, abs(pos.z - sample_z)));
        occ += (sample_z >= sp.z + u_bias ? 1.0 : 0.0) * range;
    }
    o_ao = vec4(1.0 - occ / float(KSIZE));
}
""".replace("KSIZE", str(_KERNEL_SIZE))

_BLUR_FRAG = """
#version 330
in vec2 v_uv;
out vec4 o_ao;
uniform sampler2D u_ao;
uniform vec2 u_texel;
void main() {
    float s = 0.0;
    for (int x = -2; x <= 1; x++)
        for (int y = -2; y <= 1; y++)
            s += texture(u_ao, v_uv + vec2(x, y) * u_texel).r;
    o_ao = vec4(s / 16.0);
}
"""

# Supersample downsample done on the GPU: each output pixel box-averages its
# s*s source texels with premultiplied alpha (so the transparent background
# doesn't darken edges).  Doing this on-GPU and reading back only the final
# (small) image avoids a very expensive CPU/numpy reduction of the big buffer.
_DOWN_FRAG = """
#version 330
out vec4 o_col;
uniform sampler2D u_src;
uniform int u_ssaa;
void main() {
    ivec2 base = ivec2(gl_FragCoord.xy) * u_ssaa;
    vec3 acc = vec3(0.0);
    float aacc = 0.0;
    for (int j = 0; j < u_ssaa; j++)
        for (int i = 0; i < u_ssaa; i++) {
            vec4 c = texelFetch(u_src, base + ivec2(i, j), 0);
            acc += c.rgb * c.a;
            aacc += c.a;
        }
    float n = float(u_ssaa * u_ssaa);
    vec3 rgb = aacc > 1e-4 ? acc / aacc : vec3(0.0);
    o_col = vec4(rgb, aacc / n);
}
"""

_LIGHT_FRAG = """
#version 330
in vec2 v_uv;
out vec4 o_col;
uniform sampler2D u_pos;
uniform sampler2D u_nrm;
uniform sampler2D u_ao;
uniform mat3 u_view_rot_t;   // view-space normal/dir -> world-space
uniform vec3 u_cam_t;        // view-space translation (for world reconstruct)
uniform vec3 u_light_pos;    // key light, view space
uniform vec3 u_base;
uniform vec3 u_sky;
uniform vec3 u_ground;
uniform float u_key;
uniform float u_wrap;
uniform float u_tex_freq;    // cycles per world unit
uniform float u_tex_amp;

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash13(i + vec3(0,0,0)), hash13(i + vec3(1,0,0)), f.x),
            mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash13(i + vec3(0,0,1)), hash13(i + vec3(1,0,1)), f.x),
            mix(hash13(i + vec3(0,1,1)), hash13(i + vec3(1,1,1)), f.x), f.y),
        f.z);
}

void main() {
    vec4 P = texture(u_pos, v_uv);
    if (P.w < 0.5) { o_col = vec4(0.0); return; }
    vec3 pos = P.xyz;
    vec3 N = normalize(texture(u_nrm, v_uv).xyz);
    float ao = texture(u_ao, v_uv).r;

    // World-space normal & position (view is rigid: world = R^T (view - t)).
    vec3 Nw = normalize(u_view_rot_t * N);
    vec3 wpos = u_view_rot_t * (pos - u_cam_t);

    // Hemispherical ambient about world up (+Z): soft, no hard shadows.
    float hemi = Nw.z * 0.5 + 0.5;
    vec3 ambient = mix(u_ground, u_sky, hemi);

    // Positional key light -> N.L varies across a flat face => gradient.
    vec3 L = normalize(u_light_pos - pos);
    float ndl = dot(N, L);
    float diff = max(0.0, (ndl + u_wrap) / (1.0 + u_wrap));

    vec3 color = u_base * (ambient * ao + u_key * diff * (0.5 + 0.5 * ao));

    // Subtle two-octave matte texture so flat faces have a little life.
    vec3 wp = wpos * u_tex_freq;
    float t = 0.62 * vnoise(wp) + 0.38 * vnoise(wp * 2.9);
    color *= 1.0 + u_tex_amp * (t - 0.5);

    o_col = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class GLRenderer:
    """Reusable headless renderer. One per process; see :func:`get_renderer`."""

    def __init__(self, size: int = 512, ssaa: int = 2):
        import moderngl

        self.size = size
        self.ssaa = ssaa
        self.rs = size * ssaa  # internal (supersampled) resolution

        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.geom_prog = self.ctx.program(vertex_shader=_GEOM_VERT, fragment_shader=_GEOM_FRAG)
        self.ssao_prog = self.ctx.program(vertex_shader=_FS_VERT, fragment_shader=_SSAO_FRAG)
        self.blur_prog = self.ctx.program(vertex_shader=_FS_VERT, fragment_shader=_BLUR_FRAG)
        self.light_prog = self.ctx.program(vertex_shader=_FS_VERT, fragment_shader=_LIGHT_FRAG)
        self.down_prog = self.ctx.program(vertex_shader=_FS_VERT, fragment_shader=_DOWN_FRAG)

        rs = self.rs
        # G-buffer: position + normal (float) + depth.
        self.g_pos = self.ctx.texture((rs, rs), 4, dtype="f4")
        self.g_nrm = self.ctx.texture((rs, rs), 4, dtype="f4")
        self.g_depth = self.ctx.depth_renderbuffer((rs, rs))
        self.g_fbo = self.ctx.framebuffer([self.g_pos, self.g_nrm], self.g_depth)

        self.ao_tex = self.ctx.texture((rs, rs), 1, dtype="f2")
        self.ao_fbo = self.ctx.framebuffer([self.ao_tex])
        self.aob_tex = self.ctx.texture((rs, rs), 1, dtype="f2")
        self.aob_fbo = self.ctx.framebuffer([self.aob_tex])

        self.col_tex = self.ctx.texture((rs, rs), 4, dtype="f1")
        self.col_fbo = self.ctx.framebuffer([self.col_tex])

        # Final, downsampled output (read back instead of the big SSAA buffer).
        self.out_tex = self.ctx.texture((size, size), 4, dtype="f1")
        self.out_fbo = self.ctx.framebuffer([self.out_tex])

        for t in (self.g_pos, self.g_nrm, self.ao_tex, self.aob_tex, self.col_tex):
            t.repeat_x = t.repeat_y = False

        # Fullscreen triangle.
        quad = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        self._quad_vbo = self.ctx.buffer(quad.tobytes())
        self._ssao_vao = self.ctx.simple_vertex_array(self.ssao_prog, self._quad_vbo, "in_pos")
        self._blur_vao = self.ctx.simple_vertex_array(self.blur_prog, self._quad_vbo, "in_pos")
        self._light_vao = self.ctx.simple_vertex_array(self.light_prog, self._quad_vbo, "in_pos")
        self._down_vao = self.ctx.simple_vertex_array(self.down_prog, self._quad_vbo, "in_pos")
        self.down_prog["u_ssaa"].value = ssaa

        self._init_ssao_inputs()

    def _init_ssao_inputs(self) -> None:
        rng = np.random.default_rng(0)
        kernel = []
        for i in range(_KERNEL_SIZE):
            v = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 1)])
            v = _normalize(v)
            scale = i / _KERNEL_SIZE
            v *= 0.1 + 0.9 * scale * scale  # cluster samples near the origin
            kernel.append(v)
        flat = np.array(kernel, dtype="f4").reshape(-1)
        self.ssao_prog["u_kernel"].write(flat.tobytes())
        self.ssao_prog["u_radius"].value = 0.0  # set per-part (model scale)
        self.ssao_prog["u_bias"].value = 0.0

        noise = np.zeros((4, 4, 4), dtype="f4")
        noise[..., 0] = rng.uniform(-1, 1, (4, 4))
        noise[..., 1] = rng.uniform(-1, 1, (4, 4))
        self.noise_tex = self.ctx.texture((4, 4), 4, noise.tobytes(), dtype="f4")
        self.noise_tex.repeat_x = self.noise_tex.repeat_y = True
        self.ssao_prog["u_noise_scale"].value = (self.rs / 4.0, self.rs / 4.0)

    def render_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        views: dict[str, tuple[float, float]] | None = None,
        base_color: tuple[float, float, float] = BASE_COLOR,
    ) -> dict[str, np.ndarray]:
        """Render a mesh from each view; returns {view: HxWx4 uint8 RGBA}."""
        import moderngl

        if views is None:
            views = STANDARD_VIEWS

        verts = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces)
        soup = verts[faces].reshape(-1, 3).astype("f4")
        vbo = self.ctx.buffer(soup.tobytes())
        geom_vao = self.ctx.simple_vertex_array(self.geom_prog, vbo, "in_pos")

        center = (verts.min(0) + verts.max(0)) * 0.5
        radius = float(np.linalg.norm(verts - center, axis=1).max()) or 1.0
        half = radius * 1.08
        dist = radius * 4.0
        near = dist - radius * 2.0
        far = dist + radius * 2.0
        proj = _ortho(half, near, far)

        self.ssao_prog["u_radius"].value = radius * 0.18
        self.ssao_prog["u_bias"].value = radius * 0.01
        self.geom_prog["u_proj"].write(_gl_bytes(proj))
        self.ssao_prog["u_proj"].write(_gl_bytes(proj))

        self.light_prog["u_base"].value = tuple(base_color)
        self.light_prog["u_sky"].value = SKY_COLOR
        self.light_prog["u_ground"].value = GROUND_COLOR
        self.light_prog["u_key"].value = KEY_STRENGTH
        self.light_prog["u_wrap"].value = KEY_WRAP
        self.light_prog["u_tex_amp"].value = TEX_AMP
        self.light_prog["u_tex_freq"].value = TEX_CYCLES / radius
        # Key light rigged to the camera (view space): upper-front-left, finite
        # distance so its direction sweeps across the part for a soft gradient.
        light_dir = _normalize(np.array([-0.45, 0.65, 0.75]))
        self.light_prog["u_light_pos"].value = tuple((light_dir * dist * 1.0).astype(float))

        texel = (1.0 / self.rs, 1.0 / self.rs)
        out: dict[str, np.ndarray] = {}
        try:
            for name, (elev, azim) in views.items():
                view = _view_for(elev, azim, center, dist)
                self.geom_prog["u_view"].write(_gl_bytes(view))
                # view-space normal -> world normal = R^T n  (R = view rotation).
                # numpy is row-major so view[:3,:3] sent as-is is already R^T in
                # GL's column-major reading.
                self.light_prog["u_view_rot_t"].write(
                    np.ascontiguousarray(view[:3, :3], dtype="f4").tobytes()
                )
                self.light_prog["u_cam_t"].value = tuple(view[:3, 3].astype(float))

                # 1. Geometry pass.
                self.g_fbo.use()
                self.ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
                geom_vao.render(moderngl.TRIANGLES)

                # 2. SSAO + blur.
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.ao_fbo.use()
                self.g_pos.use(0); self.ssao_prog["u_pos"].value = 0
                self.g_nrm.use(1); self.ssao_prog["u_nrm"].value = 1
                self.noise_tex.use(2); self.ssao_prog["u_noise"].value = 2
                self._ssao_vao.render(moderngl.TRIANGLES)

                self.aob_fbo.use()
                self.ao_tex.use(0); self.blur_prog["u_ao"].value = 0
                self.blur_prog["u_texel"].value = texel
                self._blur_vao.render(moderngl.TRIANGLES)

                # 3. Lighting / composite (at supersampled resolution).
                self.col_fbo.use()
                self.ctx.clear(0.0, 0.0, 0.0, 0.0)
                self.g_pos.use(0); self.light_prog["u_pos"].value = 0
                self.g_nrm.use(1); self.light_prog["u_nrm"].value = 1
                self.aob_tex.use(2); self.light_prog["u_ao"].value = 2
                self._light_vao.render(moderngl.TRIANGLES)

                # 4. GPU box-downsample to output resolution, then read back the
                # small image (premultiplied alpha, done on-GPU — far cheaper
                # than reducing the big SSAA buffer in numpy).
                self.out_fbo.use()
                self.col_tex.use(0); self.down_prog["u_src"].value = 0
                self._down_vao.render(moderngl.TRIANGLES)
                self.ctx.enable(moderngl.DEPTH_TEST)

                raw = self.out_fbo.read(components=4, dtype="f1")
                img = np.frombuffer(raw, dtype=np.uint8).reshape(self.size, self.size, 4)
                out[name] = np.ascontiguousarray(np.flipud(img))  # GL origin bottom-left
        finally:
            geom_vao.release()
            vbo.release()
        return out


_RENDERER: GLRenderer | None = None


def get_renderer(size: int = 512, ssaa: int = 2) -> GLRenderer:
    """Process-wide cached renderer (creating an EGL context is not free)."""
    global _RENDERER
    if _RENDERER is None or _RENDERER.size != size or _RENDERER.ssaa != ssaa:
        _RENDERER = GLRenderer(size=size, ssaa=ssaa)
    return _RENDERER


def render_available() -> bool:
    """True if the GPU render stack (moderngl + EGL + trimesh) is usable.

    Uses a throwaway context (released immediately) rather than the cached
    process singleton, so callers can probe availability and then safely spawn
    render workers without a live GL context lingering in the parent.
    """
    try:
        import moderngl
        import trimesh  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    try:
        ctx = moderngl.create_standalone_context(backend="egl")
        ctx.release()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Parallel corpus rendering
# ---------------------------------------------------------------------------
#
# The per-view cost is dominated by CPU work (STL load, supersample downsample,
# PNG zlib encode), not the GPU, so throughput scales with processes.  Each
# worker owns one EGL context (created lazily via the module singleton and kept
# warm for the worker's lifetime), and they share the one physical GPU.  We use
# the "spawn" start method so no GL state is inherited across the fork.

def _render_task(args: tuple[str, str, tuple | None, int, int]) -> tuple[str, bool, str]:
    stl_path, out_dir, base_color, size, ssaa = args
    from .export import export_renders

    stem = Path(stl_path).stem
    try:
        export_renders(Path(stl_path), Path(out_dir), size=size, ssaa=ssaa, base_color=base_color)
        return stem, True, ""
    except Exception as exc:  # pragma: no cover - reported to caller
        return stem, False, str(exc)


def _render_worker_init(size: int, ssaa: int) -> None:
    # Warm the EGL context once per worker so the first task isn't penalised.
    try:
        get_renderer(size=size, ssaa=ssaa)
    except Exception:
        pass


def render_stls(
    tasks: list[tuple[str, str, tuple | None]],
    n_workers: int = 1,
    size: int = 512,
    ssaa: int = 2,
    progress_cb=None,
) -> dict[str, tuple[bool, str]]:
    """
    Render many STLs to per-part render dirs, optionally across processes.

    ``tasks`` is a list of ``(stl_path, out_dir, base_color_or_None)``.  Returns
    ``{stem: (ok, error)}``.  ``progress_cb`` (if given) is called once per
    completed part.
    """
    results: dict[str, tuple[bool, str]] = {}
    full = [(s, o, c, size, ssaa) for (s, o, c) in tasks]

    if n_workers <= 1 or len(full) <= 1:
        for t in full:
            stem, ok, err = _render_task(t)
            results[stem] = (ok, err)
            if progress_cb:
                progress_cb()
        return results

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_render_worker_init,
        initargs=(size, ssaa),
    ) as ex:
        futs = [ex.submit(_render_task, t) for t in full]
        for fut in as_completed(futs):
            stem, ok, err = fut.result()
            results[stem] = (ok, err)
            if progress_cb:
                progress_cb()
    return results


def default_render_workers(requested: int | None = None) -> int:
    """
    Worker count for rendering.

    The heavy per-part cost is now CPU-side PNG encoding (the supersample
    downsample moved onto the GPU), so throughput scales with processes only up
    to a point: benchmarking the full demo-1k corpus shows it peaks around
    2x a 16-core base and *regresses* past it (process-startup + CPU/GPU-context
    contention outweigh the gain — e.g. 128 workers is ~5x slower than 32).  So
    we mildly oversubscribe to ~32 by default rather than chasing huge counts.
    """
    import os
    if requested and requested > 0:
        return requested
    cpu = os.cpu_count() or 4
    return max(1, min(cpu, 16) * 2)
