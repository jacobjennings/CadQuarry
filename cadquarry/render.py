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
# clamp so the blue stays saturated rather than washing out toward white, and
# there is a clear range between the brightest (top) and shaded faces.
BASE_COLOR = (0.42, 0.56, 0.86)
SKY_COLOR = (0.72, 0.74, 0.82)
GROUND_COLOR = (0.26, 0.28, 0.36)
KEY_STRENGTH = 0.60
KEY_WRAP = 0.50  # 0 = hard terminator, 1 = very soft

_KERNEL_SIZE = 24


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

_LIGHT_FRAG = """
#version 330
in vec2 v_uv;
out vec4 o_col;
uniform sampler2D u_pos;
uniform sampler2D u_nrm;
uniform sampler2D u_ao;
uniform mat3 u_view_rot_t;   // view-space normal -> world-space normal
uniform vec3 u_light_pos;    // key light, view space
uniform vec3 u_base;
uniform vec3 u_sky;
uniform vec3 u_ground;
uniform float u_key;
uniform float u_wrap;
void main() {
    vec4 P = texture(u_pos, v_uv);
    if (P.w < 0.5) { o_col = vec4(0.0); return; }
    vec3 pos = P.xyz;
    vec3 N = normalize(texture(u_nrm, v_uv).xyz);
    float ao = texture(u_ao, v_uv).r;

    // Hemispherical ambient about world up (+Z): soft, no hard shadows.
    vec3 Nw = normalize(u_view_rot_t * N);
    float hemi = Nw.z * 0.5 + 0.5;
    vec3 ambient = mix(u_ground, u_sky, hemi);

    // Positional key light -> N.L varies across a flat face => gradient.
    vec3 L = normalize(u_light_pos - pos);
    float ndl = dot(N, L);
    float diff = max(0.0, (ndl + u_wrap) / (1.0 + u_wrap));

    vec3 color = u_base * (ambient * ao + u_key * diff * (0.5 + 0.5 * ao));
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

        for t in (self.g_pos, self.g_nrm, self.ao_tex, self.aob_tex):
            t.repeat_x = t.repeat_y = False

        # Fullscreen triangle.
        quad = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        self._quad_vbo = self.ctx.buffer(quad.tobytes())
        self._ssao_vao = self.ctx.simple_vertex_array(self.ssao_prog, self._quad_vbo, "in_pos")
        self._blur_vao = self.ctx.simple_vertex_array(self.blur_prog, self._quad_vbo, "in_pos")
        self._light_vao = self.ctx.simple_vertex_array(self.light_prog, self._quad_vbo, "in_pos")

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
                # view-space normal -> world normal = R^T n  (R = view rotation)
                self.light_prog["u_view_rot_t"].write(
                    np.ascontiguousarray(view[:3, :3], dtype="f4").tobytes()
                )

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

                # 3. Lighting / composite.
                self.col_fbo.use()
                self.ctx.clear(0.0, 0.0, 0.0, 0.0)
                self.g_pos.use(0); self.light_prog["u_pos"].value = 0
                self.g_nrm.use(1); self.light_prog["u_nrm"].value = 1
                self.aob_tex.use(2); self.light_prog["u_ao"].value = 2
                self._light_vao.render(moderngl.TRIANGLES)
                self.ctx.enable(moderngl.DEPTH_TEST)

                # 4. Read back + supersample downsample (premultiplied alpha).
                raw = self.col_fbo.read(components=4, dtype="f1")
                img = np.frombuffer(raw, dtype=np.uint8).reshape(self.rs, self.rs, 4)
                img = np.flipud(img)  # GL origin is bottom-left
                out[name] = self._downsample(img)
        finally:
            geom_vao.release()
            vbo.release()
        return out

    def _downsample(self, img: np.ndarray) -> np.ndarray:
        s = self.ssaa
        if s == 1:
            return img.copy()
        h = w = self.size
        f = img.astype(np.float32) / 255.0
        f = f.reshape(h, s, w, s, 4)
        rgb = f[..., :3]
        a = f[..., 3:4]
        # Premultiply so transparent background doesn't darken edges.
        prem = (rgb * a).mean(axis=(1, 3))
        am = a.mean(axis=(1, 3))
        rgb_out = np.divide(prem, am, out=np.zeros_like(prem), where=am > 1e-5)
        out = np.concatenate([rgb_out, am], axis=-1)
        return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


_RENDERER: GLRenderer | None = None


def get_renderer(size: int = 512, ssaa: int = 2) -> GLRenderer:
    """Process-wide cached renderer (creating an EGL context is not free)."""
    global _RENDERER
    if _RENDERER is None or _RENDERER.size != size or _RENDERER.ssaa != ssaa:
        _RENDERER = GLRenderer(size=size, ssaa=ssaa)
    return _RENDERER


def render_available() -> bool:
    """True if the GPU render stack (moderngl + EGL + trimesh) is usable."""
    try:
        import moderngl  # noqa: F401
        import trimesh  # noqa: F401
    except ImportError:
        return False
    try:
        get_renderer()
        return True
    except Exception:
        return False
