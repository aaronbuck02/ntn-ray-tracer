"""Random scene generator for Sionna RT.

Two primitives: "block" (PPP building footprints on a grid, ITU-R P.1410
Rayleigh heights, built-up fraction alpha = lam * a_p) and "wall" (the Chiu &
Roy process: PPP wall segments in a disc, orientation U[0,pi), fixed length).

Emits a Mitsuba 3 / Sionna RT scene with walls, roofs and ground as separate
meshes so each carries its own material. See docs/pipelines.md.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass, field

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Scene specification container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SceneSpec:
    """Pure-geometry description of a generated scene (no file I/O)."""

    # --- scene extent -------------------------------------------------------
    Lx: float
    Ly: float
    cell_res: float
    x_edges: np.ndarray                      # [nx+1]
    y_edges: np.ndarray                      # [ny+1]

    # --- PPP realisation ----------------------------------------------------
    lam: float                               # intensity [pts / m^2]
    a_p: float                               # unit area per point [m^2]
    points: np.ndarray                       # [N, 2] raw PPP samples
    counts: np.ndarray                       # [nx, ny] integer k_ij

    # --- emitted buildings --------------------------------------------------
    centers: np.ndarray                      # [B, 2] cell centres
    sides: np.ndarray                        # [B]    footprint side [m]
    heights: np.ndarray                      # [B]    building height [m]
    cell_idx: np.ndarray                     # [B, 2] (i, j) of each building

    # --- bookkeeping --------------------------------------------------------
    meta: dict = field(default_factory=dict)

    # --- wall primitive only (see sample_ppp_walls) -------------------------
    # ``primitive`` selects the mesh emitter in write_sionna_scene: "block"
    # (default, sample_ppp_scene) or "wall" (sample_ppp_walls). In wall mode
    # ``sides`` holds the wall LENGTH l_f rather than a footprint side, and
    # ``orientations``/``thickness`` carry the rest of the slab geometry.
    primitive: str = "block"
    orientations: np.ndarray | None = None   # [B] wall azimuth xi [rad]
    thickness: float = 0.0                   # [m] slab thickness (wall mode)
    radius: float = 0.0                      # [m] disc radius (wall mode)

    @property
    def n_buildings(self) -> int:
        return len(self.sides)

    @property
    def area(self) -> float:
        if self.primitive == "wall":
            return float(np.pi * self.radius ** 2)
        return self.Lx * self.Ly

    @property
    def built_area(self) -> float:
        if self.primitive == "wall":
            return float(np.sum(self.sides) * self.thickness)
        return float(np.sum(self.sides ** 2))

    def bbox(self) -> tuple:
        return (-self.Lx / 2, self.Lx / 2, -self.Ly / 2, self.Ly / 2)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1-4 : sample the scene
# ══════════════════════════════════════════════════════════════════════════════

def sample_ppp_scene(
    Lx: float = 1000.0,
    Ly: float = 1000.0,
    cell_res: float = 50.0,
    *,
    # --- density: give EITHER (alpha, points_per_cell) OR (lam, a_p) --------
    alpha: float | None = None,          # target built-up area fraction [0, 1]
    points_per_cell: float | None = None,  # mean PPP points per cell
    lam: float | None = None,            # PPP intensity [pts / m^2]
    a_p: float | None = None,            # unit area carried by one point [m^2]
    # --- geometry -----------------------------------------------------------
    max_fill: float = 0.98,              # cap on cell fill fraction
    min_side: float = 3.0,               # drop slivers smaller than this [m]
    # --- ITU-R P.1410 heights ----------------------------------------------
    gamma: float = 20.0,                 # Rayleigh scale [m]
    h_min: float = 4.0,
    h_max: float = 120.0,
    # --- rng ----------------------------------------------------------------
    seed: int | None = 0,
    rng: np.random.Generator | None = None,
    random_grid_shift: bool = True,
) -> SceneSpec:
    """Realise one random urban scene."""
    if rng is None:
        rng = np.random.default_rng(seed)

    A_c = cell_res ** 2

    # ---- resolve density parameterisation ---------------------------------
    if lam is None or a_p is None:
        if alpha is None or points_per_cell is None:
            raise ValueError(
                "Specify either (lam, a_p) or (alpha, points_per_cell)."
            )
        lam = float(points_per_cell) / A_c
        a_p = float(alpha) / lam
    lam = float(lam)
    a_p = float(a_p)
    alpha_target = lam * a_p
    if alpha_target > 1.0:
        raise ValueError(
            f"lam * a_p = {alpha_target:.3f} > 1 — every cell would saturate. "
            "Lower the density or the unit area."
        )

    # ---- grid --------------------------------------------------------------
    nx = int(round(Lx / cell_res))
    ny = int(round(Ly / cell_res))
    if nx < 1 or ny < 1:
        raise ValueError("cell_res is larger than the scene.")
    if random_grid_shift:
        grid_shift_x = float(rng.uniform(0.0, cell_res))
        grid_shift_y = float(rng.uniform(0.0, cell_res))
    else:
        grid_shift_x = grid_shift_y = 0.0
    x_edges = -Lx / 2 + grid_shift_x + cell_res * np.arange(nx + 1)
    y_edges = -Ly / 2 + grid_shift_y + cell_res * np.arange(ny + 1)

    # ---- homogeneous PPP over the rectangle --------------------------------
    N = rng.poisson(lam * Lx * Ly)
    pts = np.column_stack([
        rng.uniform(x_edges[0], x_edges[-1], N),
        rng.uniform(y_edges[0], y_edges[-1], N),
    ])

    # ---- bin into cells ----------------------------------------------------
    counts, _, _ = np.histogram2d(
        pts[:, 0], pts[:, 1], bins=[x_edges, y_edges]
    )
    counts = counts.astype(int)

    # ---- saturating area accumulation -> footprint side --------------------
    A_build = np.minimum(counts * a_p, max_fill * A_c)
    side = np.sqrt(A_build)
    occ = side >= min_side

    ii, jj = np.nonzero(occ)
    cx = 0.5 * (x_edges[ii] + x_edges[ii + 1])
    cy = 0.5 * (y_edges[jj] + y_edges[jj + 1])
    centers = np.column_stack([cx, cy])
    sides = side[ii, jj]

    # ---- ITU-R P.1410 Rayleigh heights, one draw per building --------------
    heights = rayleigh_heights(len(sides), gamma, h_min, h_max, rng)

    spec = SceneSpec(
        Lx=float(Lx), Ly=float(Ly), cell_res=float(cell_res),
        x_edges=x_edges, y_edges=y_edges,
        lam=lam, a_p=a_p, points=pts, counts=counts,
        centers=centers, sides=sides, heights=heights,
        cell_idx=np.column_stack([ii, jj]),
    )
    spec.meta = dict(
        seed=seed,
        n_points=int(N),
        n_cells=int(nx * ny),
        nx=nx, ny=ny,
        grid_shift_x=grid_shift_x, grid_shift_y=grid_shift_y,
        alpha_target=float(alpha_target),
        alpha_realised=float(spec.built_area / spec.area),
        saturated_cells=int(np.sum(counts * a_p >= max_fill * A_c)),
        max_fill=float(max_fill),
        min_side=float(min_side),
        gamma=float(gamma), h_min=float(h_min), h_max=float(h_max),
        n_buildings=int(spec.n_buildings),
        mean_height=float(np.mean(heights)) if len(heights) else 0.0,
    )
    spec.meta.update(itu_parameters(spec))
    return spec


def rayleigh_heights(n, gamma, h_min=0.0, h_max=np.inf, rng=None):
    """
    ITU-R P.1410 building heights: Rayleigh(gamma), truncated to [h_min, h_max]
    by inverse-CDF sampling (no rejection loop, no bias at the clip points).
    """
    if rng is None:
        rng = np.random.default_rng()
    if n == 0:
        return np.zeros(0)

    def cdf(h):
        return 1.0 - np.exp(-(h ** 2) / (2.0 * gamma ** 2))

    u_lo, u_hi = cdf(h_min), cdf(min(h_max, 1e6))
    u = rng.uniform(u_lo, u_hi, n)
    return gamma * np.sqrt(-2.0 * np.log1p(-u))


# ══════════════════════════════════════════════════════════════════════════════
#  Wall primitive: the scene process of Chiu & Roy eq. (4)
# ══════════════════════════════════════════════════════════════════════════════

# (gamma [m], lambda_b [walls/m^2], l_f [m]) -- Chiu & Roy Sec. VI. The
# densities are the paper's 1.96e-3 / 3.14e-3 written the way the reference
# implementation writes them, as lambda_1D = 2 lambda_b l_f / pi.
WALL_SCENARIOS = {
    "urban": dict(gamma=8.0, lam_b=0.025 * np.pi / 40.0, l_f=20.0),
    "rural": dict(gamma=3.0, lam_b=0.020 * np.pi / 20.0, l_f=10.0),
}


def sample_ppp_walls(
    radius: float = 2000.0,
    *,
    scenario: str | None = None,
    lam_b: float | None = None,
    l_f: float | None = None,
    gamma: float | None = None,
    thickness: float = 0.3,
    h_min: float = 0.0,
    h_max: float = np.inf,
    seed: int | None = 0,
    rng: np.random.Generator | None = None,
) -> SceneSpec:
    """Realise one wall scene: the spatial process Phi of Chiu & Roy eq. (4)."""
    if rng is None:
        rng = np.random.default_rng(seed)

    if scenario is not None:
        if scenario not in WALL_SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; "
                             f"expected one of {sorted(WALL_SCENARIOS)}")
        defaults = WALL_SCENARIOS[scenario]
        lam_b = defaults["lam_b"] if lam_b is None else lam_b
        l_f   = defaults["l_f"]   if l_f   is None else l_f
        gamma = defaults["gamma"] if gamma is None else gamma
    if lam_b is None or l_f is None or gamma is None:
        raise ValueError("give scenario=, or all of lam_b/l_f/gamma.")
    if not 0.0 < thickness < l_f:
        raise ValueError(f"thickness {thickness} must be in (0, l_f={l_f}).")

    n = int(rng.poisson(lam_b * np.pi * radius ** 2))
    r   = radius * np.sqrt(rng.random(n))          # uniform in the disc
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    centers = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    orientations = rng.uniform(0.0, np.pi, n)
    heights = rayleigh_heights(n, gamma, h_min, h_max, rng)

    # Lx/Ly are the square that circumscribes the disc -- bbox() and the ground
    # quad are written against them, and the ground must reach past the walls.
    side = 2.0 * radius
    spec = SceneSpec(
        Lx=side, Ly=side, cell_res=float(l_f),
        x_edges=np.array([-radius, radius]), y_edges=np.array([-radius, radius]),
        lam=float(lam_b), a_p=float(l_f * thickness),
        points=centers.copy(), counts=np.zeros((1, 1), dtype=int),
        centers=centers, sides=np.full(n, float(l_f)), heights=heights,
        cell_idx=np.zeros((n, 2), dtype=int),
        primitive="wall", orientations=orientations,
        thickness=float(thickness), radius=float(radius),
    )
    spec.meta = dict(
        primitive="wall", seed=seed, scenario=scenario,
        radius=float(radius), lam_b=float(lam_b), l_f=float(l_f),
        thickness=float(thickness),
        lambda_1d=float(2.0 * lam_b * l_f / np.pi),
        gamma=float(gamma), h_min=float(h_min), h_max=float(h_max),
        n_walls=n, n_buildings=n,
        mean_height=float(np.mean(heights)) if n else 0.0,
        density_realised=float(n / (np.pi * radius ** 2)),
    )
    spec.meta.update(itu_parameters(spec))
    return spec


def itu_parameters(spec: SceneSpec) -> dict:
    """
    Report the scene in ITU-R P.1410 terms:
      alpha – fraction of land area covered by buildings
      beta  – mean number of buildings per km^2
      gamma – Rayleigh height scale (as configured / as fitted)
    """
    area_km2 = spec.area / 1e6
    h = spec.heights
    gamma_fit = float(np.sqrt(np.mean(h ** 2) / 2.0)) if len(h) else 0.0
    return dict(
        itu_alpha=float(spec.built_area / spec.area),
        itu_beta=float(spec.n_buildings / area_km2) if area_km2 > 0 else 0.0,
        itu_gamma_fit=gamma_fit,
    )


def _box_wall_mesh(cx, cy, side, h, z0=0.0):
    """Bottom + 4 walls of an axis-aligned box (10 tris)."""
    s = side / 2.0
    v = np.array([
        [cx - s, cy - s, z0],
        [cx + s, cy - s, z0],
        [cx + s, cy + s, z0],
        [cx - s, cy + s, z0],
        [cx - s, cy - s, z0 + h],
        [cx + s, cy - s, z0 + h],
        [cx + s, cy + s, z0 + h],
        [cx - s, cy + s, z0 + h],
    ], dtype=float)
    f = np.array([
        [0, 2, 1], [0, 3, 2],          # bottom  (CW seen from above -> -z)
        [0, 1, 5], [0, 5, 4],          # -y wall
        [1, 2, 6], [1, 6, 5],          # +x wall
        [2, 3, 7], [2, 7, 6],          # +y wall
        [3, 0, 4], [3, 4, 7],          # -x wall
    ], dtype=int)
    return v, f


def _box_roof_mesh(cx, cy, side, h, z0=0.0):
    """Roof cap of an axis-aligned box (2 tris)."""
    s = side / 2.0
    v = np.array([
        [cx - s, cy - s, z0 + h],
        [cx + s, cy - s, z0 + h],
        [cx + s, cy + s, z0 + h],
        [cx - s, cy + s, z0 + h],
    ], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return v, f


def _slab_corners(cx, cy, length, thickness, angle):
    """[4, 2] footprint corners of an oriented slab, CCW in xy (as _box_*_mesh)."""
    u = np.array([np.cos(angle), np.sin(angle)])        # along the wall
    v = np.array([-np.sin(angle), np.cos(angle)])       # across it (facade normal)
    c = np.array([cx, cy])
    a, b = length / 2.0, thickness / 2.0
    return np.array([c - a * u - b * v, c + a * u - b * v,
                     c + a * u + b * v, c - a * u + b * v])


def _slab_wall_mesh(cx, cy, length, thickness, h, angle, z0=0.0):
    """Bottom + 4 faces of an oriented thin slab (10 tris)."""
    p = _slab_corners(cx, cy, length, thickness, angle)
    v = np.vstack([np.column_stack([p, np.full(4, z0)]),
                   np.column_stack([p, np.full(4, z0 + h)])])
    f = np.array([
        [0, 2, 1], [0, 3, 2],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=int)
    return v, f


def _slab_roof_mesh(cx, cy, length, thickness, h, angle, z0=0.0):
    """Roof cap of an oriented thin slab (2 tris) -- the n=3/2 wedge's top face."""
    p = _slab_corners(cx, cy, length, thickness, angle)
    v = np.column_stack([p, np.full(4, z0 + h)])
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return v, f


def _primitive_meshes(spec: SceneSpec, b: int):
    """(wall_verts, wall_faces, roof_verts, roof_faces) for building/wall ``b``."""
    if spec.primitive == "wall":
        cx, cy = spec.centers[b]
        args = (cx, cy, spec.sides[b], spec.thickness, spec.heights[b],
                float(spec.orientations[b]))
        return (*_slab_wall_mesh(*args), *_slab_roof_mesh(*args))
    args = (spec.centers[b, 0], spec.centers[b, 1], spec.sides[b], spec.heights[b])
    return (*_box_wall_mesh(*args), *_box_roof_mesh(*args))


def _quad_mesh(xmin, xmax, ymin, ymax, z=0.0):
    v = np.array([
        [xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z]
    ], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return v, f


def write_ply(path, verts, faces, binary: bool = False):
    """Minimal PLY writer (triangles only) — matches Mitsuba's loader."""
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header = (
        "ply\n"
        f"format {fmt}\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    if not binary:
        with open(path, "w") as fh:
            fh.write(header)
            for x, y, z in verts:
                fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
            for a, b, c in faces:
                fh.write(f"3 {a} {b} {c}\n")
        return

    # Face records are packed, not aligned: uchar count + 3 int32 = 13 bytes.
    face_rec = np.empty(len(faces), dtype=np.dtype(
        [("n", "u1"), ("v", "<i4", 3)], align=False))
    face_rec["n"] = 3
    face_rec["v"] = faces
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(np.ascontiguousarray(verts, dtype="<f4").tobytes())
        fh.write(face_rec.tobytes())


# Default per-role EM parameters (ITU-R P.2040 frequency law:
# relative_permittivity = a * f_ghz**b, conductivity = c * f_ghz**d), applied
# by scene_utils.build_scene_batch() at simulation time -- see
# generate_scene()/generate_scene_ensemble()'s wall_params/roof_params/
# ground_params kwargs. These reproduce Sionna's own built-in ITU table
# entries for concrete/brick/wet_ground exactly, so leaving the kwargs at
# their defaults changes nothing numerically versus today.
DEFAULT_WALL_PARAMS = dict(a=5.24, b=0.0, c=0.0462, d=0.7822,   # ITU concrete
                           scattering_coefficient=0.3, xpd_coefficient=0.0)
DEFAULT_ROOF_PARAMS = dict(a=3.91, b=0.0, c=0.0238, d=0.16,     # ITU brick
                           scattering_coefficient=0.3, xpd_coefficient=0.0)
DEFAULT_GROUND_PARAMS = dict(a=30.0, b=-0.4, c=0.15, d=1.30,    # ITU wet_ground
                             scattering_coefficient=0.3, xpd_coefficient=0.0)


# ITU radio materials -> a plausible diffuse RGB for the Mitsuba preview.
# Sionna keys the *radio* material off the bsdf id prefix "mat-", so the id
# string is what matters; the rgb only affects visual renders.
_MAT_RGB = {
    "itu_concrete":   (0.539, 0.539, 0.539),
    "itu_brick":      (0.472, 0.258, 0.191),
    "itu_glass":      (0.271, 0.442, 0.510),
    "itu_wood":       (0.436, 0.311, 0.181),
    "itu_metal":      (0.700, 0.700, 0.700),
    "itu_wet_ground": (0.220, 0.250, 0.180),
    "itu_medium_dry_ground": (0.372, 0.334, 0.253),
    "itu_plasterboard": (0.700, 0.690, 0.650),
    "itu_marble":        (0.780, 0.760, 0.720),
}


def _bsdf_xml(mat_name: str) -> str:
    r, g, b = _MAT_RGB.get(mat_name, (0.5, 0.5, 0.5))
    return (
        f'    <bsdf type="twosided" id="mat-{mat_name}">\n'
        f'        <bsdf type="diffuse">\n'
        f'            <rgb value="{r} {g} {b}" name="reflectance"/>\n'
        f'        </bsdf>\n'
        f'    </bsdf>\n'
    )


def write_sionna_scene(
    spec: SceneSpec,
    out_dir: str,
    *,
    scene_name: str = "ppp_scene",
    building_material: str = "itu_concrete",
    roof_material: str = "itu_brick",
    ground_material: str = "itu_wet_ground",
    wall_params: dict | None = None,
    roof_params: dict | None = None,
    ground_params: dict | None = None,
    ground_margin: float = 0.0,
    one_ply_per_building: bool = True,
    add_sensor: bool = True,
    binary_ply: bool = False,
) -> str:
    """Emit ``<out_dir>/<scene_name>.xml`` + ``<out_dir>/meshes/*.ply``."""
    wall_params   = dict(DEFAULT_WALL_PARAMS)   if wall_params   is None else dict(wall_params)
    roof_params   = dict(DEFAULT_ROOF_PARAMS)   if roof_params   is None else dict(roof_params)
    ground_params = dict(DEFAULT_GROUND_PARAMS) if ground_params is None else dict(ground_params)

    _roles = [("wall", building_material, wall_params),
              ("roof", roof_material, roof_params),
              ("ground", ground_material, ground_params)]
    _seen = {}
    for _role, _name, _params in _roles:
        if _name in _seen and _seen[_name][1] != _params:
            print(f"[scene_gen_ppp] WARNING: '{_seen[_name][0]}' and '{_role}' "
                  f"both resolve to Sionna BSDF id 'mat-{_name}' with "
                  f"DIFFERENT params -- only one set can apply (they share "
                  f"one RadioMaterial object). Use a distinct dummy ITU "
                  f"identity per role to parameterise them independently.")
        _seen.setdefault(_name, (_role, _params))

    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    xmin, xmax, ymin, ymax = spec.bbox()
    shapes = []

    # ---- ground ------------------------------------------------------------
    gv, gf = _quad_mesh(xmin - ground_margin, xmax + ground_margin,
                        ymin - ground_margin, ymax + ground_margin, z=0.0)
    write_ply(os.path.join(mesh_dir, "ground.ply"), gv, gf, binary=binary_ply)
    shapes.append(("ground", "meshes/ground.ply", ground_material))

    # ---- buildings (walls+bottom and roof are separate meshes/materials) --
    if one_ply_per_building:
        for b in range(spec.n_buildings):
            wv, wf, rv, rf = _primitive_meshes(spec, b)
            wfn = f"building_{b:05d}_walls.ply"
            rfn = f"building_{b:05d}_roof.ply"
            write_ply(os.path.join(mesh_dir, wfn), wv, wf, binary=binary_ply)
            write_ply(os.path.join(mesh_dir, rfn), rv, rf, binary=binary_ply)
            shapes.append((f"building_{b:05d}_walls", f"meshes/{wfn}",
                           building_material))
            shapes.append((f"building_{b:05d}_roof", f"meshes/{rfn}",
                           roof_material))
    else:
        WV, WF, woff = [], [], 0
        RV, RF, roff = [], [], 0
        for b in range(spec.n_buildings):
            wv, wf, rv, rf = _primitive_meshes(spec, b)
            WV.append(wv); WF.append(wf + woff); woff += len(wv)
            RV.append(rv); RF.append(rf + roff); roff += len(rv)
        if WV:
            write_ply(os.path.join(mesh_dir, "building_walls.ply"),
                      np.vstack(WV), np.vstack(WF), binary=binary_ply)
            shapes.append(("building_walls", "meshes/building_walls.ply",
                           building_material))
        if RV:
            write_ply(os.path.join(mesh_dir, "building_roofs.ply"),
                      np.vstack(RV), np.vstack(RF), binary=binary_ply)
            shapes.append(("building_roofs", "meshes/building_roofs.ply",
                           roof_material))

    # ---- xml ---------------------------------------------------------------
    mats = sorted({ground_material, building_material, roof_material})
    L = []
    L.append('<scene version="2.1.0">\n')
    L.append('    <integrator type="path"/>\n\n')
    for m in mats:
        L.append(_bsdf_xml(m))
    L.append("\n")

    if add_sensor:
        d = max(spec.Lx, spec.Ly)
        L.append(
            '    <sensor type="perspective">\n'
            '        <string name="fov_axis" value="x"/>\n'
            '        <float name="fov" value="45"/>\n'
            '        <transform name="to_world">\n'
            f'            <lookat origin="{-0.9*d:.1f} {-0.9*d:.1f} {0.7*d:.1f}"'
            f' target="0 0 0" up="0 0 1"/>\n'
            '        </transform>\n'
            '        <sampler type="independent">\n'
            '            <integer name="sample_count" value="128"/>\n'
            '        </sampler>\n'
            '        <film type="hdrfilm">\n'
            '            <integer name="width" value="1024"/>\n'
            '            <integer name="height" value="768"/>\n'
            '        </film>\n'
            '    </sensor>\n\n'
        )
        L.append(
            '    <emitter type="constant">\n'
            '        <rgb value="1.0 1.0 1.0" name="radiance"/>\n'
            '    </emitter>\n\n'
        )

    for sid, fn, mat in shapes:
        L.append(
            f'    <shape type="ply" id="{sid}">\n'
            f'        <string name="filename" value="{fn}"/>\n'
            f'        <boolean name="face_normals" value="true"/>\n'
            f'        <ref id="mat-{mat}" name="bsdf"/>\n'
            f'    </shape>\n'
        )
    L.append('</scene>\n')

    xml_path = os.path.join(out_dir, f"{scene_name}.xml")
    with open(xml_path, "w") as fh:
        fh.writelines(L)

    # ---- sidecar metadata (reproducibility) --------------------------------
    meta = dict(spec.meta)
    meta.update(dict(
        Lx=spec.Lx, Ly=spec.Ly, cell_res=spec.cell_res,
        lam=spec.lam, a_p=spec.a_p,
        building_material=building_material,
        roof_material=roof_material,
        ground_material=ground_material,
        wall_params=wall_params,
        roof_params=roof_params,
        ground_params=ground_params,
        xml=os.path.basename(xml_path),
        n_shapes=len(shapes),
    ))
    with open(os.path.join(out_dir, f"{scene_name}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    _arrays = dict(
        points=spec.points, counts=spec.counts, centers=spec.centers,
        sides=spec.sides, heights=spec.heights, cell_idx=spec.cell_idx,
        x_edges=spec.x_edges, y_edges=spec.y_edges,
    )
    if spec.orientations is not None:
        _arrays["orientations"] = spec.orientations
    np.savez_compressed(os.path.join(out_dir, f"{scene_name}_spec.npz"), **_arrays)

    print(f"[scene_gen_ppp] {spec.n_buildings} buildings, "
          f"alpha={meta['itu_alpha']:.3f}, beta={meta['itu_beta']:.0f}/km^2, "
          f"gamma_fit={meta['itu_gamma_fit']:.1f} m  ->  {xml_path}")
    return xml_path


def _sample(primitive: str, **sample_kwargs) -> SceneSpec:
    """Dispatch to the sampler for ``primitive`` ("block" or "wall")."""
    if primitive == "wall":
        return sample_ppp_walls(**sample_kwargs)
    if primitive == "block":
        return sample_ppp_scene(**sample_kwargs)
    raise ValueError(f"unknown primitive {primitive!r}; expected 'block' or 'wall'.")


def generate_scene(out_dir: str, *, scene_name: str = "ppp_scene",
                   building_material: str = "itu_concrete",
                   roof_material: str = "itu_brick",
                   ground_material: str = "itu_wet_ground",
                   wall_params: dict | None = None,
                   roof_params: dict | None = None,
                   ground_params: dict | None = None,
                   one_ply_per_building: bool = True,
                   primitive: str = "block",
                   **sample_kwargs) -> tuple:
    """Sample + write in one call.  Returns ``(xml_path, spec)``."""
    spec = _sample(primitive, **sample_kwargs)
    xml = write_sionna_scene(
        spec, out_dir, scene_name=scene_name,
        building_material=building_material, roof_material=roof_material,
        ground_material=ground_material,
        wall_params=wall_params, roof_params=roof_params,
        ground_params=ground_params,
        one_ply_per_building=one_ply_per_building,
    )
    return xml, spec


def _point_in_any_footprint(x: float, y: float, spec: SceneSpec) -> bool:
    """True if (x, y) falls inside any building's square footprint."""
    if spec.n_buildings == 0:
        return False
    if spec.primitive == "wall":
        # Oriented slabs: test in each wall's own frame. Walls carry almost no
        # footprint area, so this practically never fires -- the wall ensemble
        # is meant to run with rx_check_xy=None anyway (the reference model has
        # no exclusion zone about the terminal).
        d = spec.centers - np.array([x, y])
        ca, sa = np.cos(spec.orientations), np.sin(spec.orientations)
        along = np.abs(d[:, 0] * ca + d[:, 1] * sa)
        across = np.abs(-d[:, 0] * sa + d[:, 1] * ca)
        return bool(np.any((along <= spec.sides / 2.0) &
                           (across <= spec.thickness / 2.0)))
    half = spec.sides / 2.0
    return bool(np.any((np.abs(spec.centers[:, 0] - x) <= half) &
                       (np.abs(spec.centers[:, 1] - y) <= half)))


def generate_scene_streaming(out_dir: str, seed: int, *,
                             scene_name: str = "scene",
                             primitive: str = "wall",
                             building_material: str = "itu_concrete",
                             roof_material: str = "itu_brick",
                             ground_material: str = "itu_wet_ground",
                             wall_params: dict | None = None,
                             roof_params: dict | None = None,
                             ground_params: dict | None = None,
                             one_ply_per_building: bool = False,
                             binary_ply: bool = True,
                             verify: bool = False,
                             **sample_kwargs) -> tuple:
    """Generate ONE scene for immediate use, then discard: the unit of work for a
scene-outer sweep (run_chiu_sweep.py).
    """
    spec = _sample(primitive, seed=seed, **sample_kwargs)
    xml_path = write_sionna_scene(
        spec, out_dir, scene_name=scene_name,
        building_material=building_material, roof_material=roof_material,
        ground_material=ground_material,
        wall_params=wall_params, roof_params=roof_params,
        ground_params=ground_params,
        one_ply_per_building=one_ply_per_building,
        binary_ply=binary_ply,
        add_sensor=False,          # nothing renders a streamed scene
    )
    if verify:
        verify_scene(xml_path, spec)
    return xml_path, spec


def generate_scene_ensemble(
    out_dir: str, n_scenes: int, base_seed: int = 0, *,
    scene_name: str = "ppp_scene",
    building_material: str = "itu_concrete",
    roof_material: str = "itu_brick",
    ground_material: str = "itu_wet_ground",
    wall_params: dict | None = None,
    roof_params: dict | None = None,
    ground_params: dict | None = None,
    one_ply_per_building: bool = True,
    rx_check_xy: tuple = (0.0, 0.0),
    max_seed_attempts: int | None = None,
    primitive: str = "block",
    binary_ply: bool = False,
    verify_every: int = 1,
    **sample_kwargs,
) -> list:
    """Generate ``n_scenes`` independent PPP scene realisations, one subdirectory
per accepted scene: ``out_dir/scene_seed{seed:04d}/{scene_name}.xml``
(+ ``meshes/``, ``{scene_name}_meta.json``, ``{scene_name}_spec.npz``).
    """
    wall_params   = dict(DEFAULT_WALL_PARAMS)   if wall_params   is None else dict(wall_params)
    roof_params   = dict(DEFAULT_ROOF_PARAMS)   if roof_params   is None else dict(roof_params)
    ground_params = dict(DEFAULT_GROUND_PARAMS) if ground_params is None else dict(ground_params)
    if max_seed_attempts is None:
        max_seed_attempts = max(100, 20 * n_scenes)

    os.makedirs(out_dir, exist_ok=True)
    scenes = []
    skipped_seeds = []
    seed = base_seed
    attempts = 0
    while len(scenes) < n_scenes:
        if attempts >= max_seed_attempts:
            raise RuntimeError(
                f"[scene_gen_ppp] gave up after {attempts} candidate seed(s) "
                f"starting at base_seed={base_seed}: only {len(scenes)}/{n_scenes} "
                f"accepted, {len(skipped_seeds)} skipped for a receiver-in-footprint "
                f"collision at rx_check_xy={rx_check_xy}. alpha="
                f"{sample_kwargs.get('alpha')} is likely too high for a receiver "
                f"fixed at that position -- lower alpha, or pass a larger "
                f"max_seed_attempts if you expect this many collisions."
            )
        attempts += 1

        # Sample in-memory first (cheap, no disk I/O) so a colliding candidate
        # never gets written -- only accepted seeds leave files behind.
        spec = _sample(primitive, seed=seed, **sample_kwargs)
        if rx_check_xy is not None and \
                _point_in_any_footprint(rx_check_xy[0], rx_check_xy[1], spec):
            skipped_seeds.append(seed)
            seed += 1
            continue

        scene_dir = os.path.join(out_dir, f"scene_seed{seed:04d}")
        xml_path = write_sionna_scene(
            spec, scene_dir, scene_name=scene_name,
            building_material=building_material, roof_material=roof_material,
            ground_material=ground_material,
            wall_params=wall_params, roof_params=roof_params,
            ground_params=ground_params,
            one_ply_per_building=one_ply_per_building,
            binary_ply=binary_ply,
        )
        # verify_scene() is a Mitsuba load plus one downward ray per building.
        # At a 2 km wall radius that is ~25k rays on a 16 MB scene, per scene --
        # real money over a 2000-scene ensemble, for a geometric check that does
        # not vary between draws. verify_every=N checks every Nth scene; 0 skips
        # entirely. Default 1 keeps the original every-scene behaviour.
        _do_verify = verify_every > 0 and (len(scenes) % verify_every == 0)
        verify_report = verify_scene(xml_path, spec) if _do_verify else None

        scenes.append({
            "seed": seed,
            "xml_path": xml_path,
            "meta_path": os.path.join(scene_dir, f"{scene_name}_meta.json"),
            "spec_path": os.path.join(scene_dir, f"{scene_name}_spec.npz"),
            "n_buildings": spec.n_buildings,
            "verify_report": verify_report,
        })
        print(f"[scene_gen_ppp] ensemble {len(scenes)}/{n_scenes}: seed={seed}  "
              f"n_buildings={spec.n_buildings}  "
              f"{'verify OK' if _do_verify else '(verify skipped)'}")
        seed += 1

    if skipped_seeds:
        print(f"[scene_gen_ppp] skipped {len(skipped_seeds)} seed(s) for "
              f"receiver-in-footprint collisions: {skipped_seeds}")

    manifest = {
        "base_seed": base_seed,
        "n_scenes": n_scenes,
        "skipped_seeds": skipped_seeds,
        "primitive": primitive,
        "sample_kwargs": dict(sample_kwargs),
        "building_material": building_material,
        "roof_material": roof_material,
        "ground_material": ground_material,
        "wall_params": wall_params,
        "roof_params": roof_params,
        "ground_params": ground_params,
        "rx_check_xy": tuple(rx_check_xy) if rx_check_xy is not None else None,
        "scenes": scenes,
    }
    manifest_path = os.path.join(out_dir, "scenes_manifest.pkl")
    with open(manifest_path, "wb") as f:
        pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[scene_gen_ppp] wrote ensemble manifest ({n_scenes} scenes) -> {manifest_path}")

    return scenes


# ══════════════════════════════════════════════════════════════════════════════
#  Analytic free-space grid (no Mitsuba needed)
# ══════════════════════════════════════════════════════════════════════════════

def load_scene_spec(spec_path: str, meta_path: str) -> SceneSpec:
    """Rebuild a :class:`SceneSpec` from the sidecar files written alongside a
generated scene.
    """
    npz = np.load(spec_path, allow_pickle=True)
    with open(meta_path, "r") as fh:
        meta = json.load(fh)
    return SceneSpec(
        Lx=float(meta["Lx"]), Ly=float(meta["Ly"]),
        cell_res=float(meta["cell_res"]),
        lam=float(meta["lam"]), a_p=float(meta["a_p"]),
        meta=meta,
        primitive=str(meta.get("primitive", "block")),
        thickness=float(meta.get("thickness", 0.0)),
        radius=float(meta.get("radius", 0.0)),
        **{k: npz[k] for k in npz.files},
    )


def free_space_grid(spec: SceneSpec, grid_res: float = 2.0,
                    margin: float = 20.0, clearance: float = 0.0):
    """
    Exact outdoor mask, computed from the box footprints instead of ray-casting.

    Drop-in replacement for ``scene_utils.build_occupancy_grid``: returns
    ``(free_xy, grid_meta)`` with the same keys, so
    ``scene_utils.sample_rx_positions`` works unchanged.  ``clearance`` inflates
    each footprint so receivers keep a stand-off from walls.
    """
    xmin, xmax, ymin, ymax = spec.bbox()
    xs = np.arange(xmin, xmax + grid_res, grid_res)
    ys = np.arange(ymin, ymax + grid_res, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    xf, yf = XX.ravel(), YY.ravel()

    occ = np.zeros(xf.shape, dtype=bool)
    for (cx, cy), s in zip(spec.centers, spec.sides):
        h = s / 2.0 + clearance
        occ |= (np.abs(xf - cx) <= h) & (np.abs(yf - cy) <= h)

    in_zone = ((xf >= xmin + margin) & (xf <= xmax - margin) &
               (yf >= ymin + margin) & (yf <= ymax - margin))
    free_mask = (~occ) & in_zone

    free_xy = np.column_stack([xf[free_mask], yf[free_mask]])
    grid_meta = {
        "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
        "scene_xmin": xmin, "scene_xmax": xmax,
        "scene_ymin": ymin, "scene_ymax": ymax,
        "valid_xmin": xmin + margin, "valid_xmax": xmax - margin,
        "valid_ymin": ymin + margin, "valid_ymax": ymax - margin,
        "grid_res": grid_res, "margin": margin,
        "n_total": int(xf.size),
        "n_free": int(free_mask.sum()),
        "n_free_full": int((~occ).sum()),
        "n_occupied": int(occ.sum()),
        "occupied_xy": np.column_stack([xf[occ], yf[occ]]),
    }
    print(f"[scene_gen_ppp] free cells: {grid_meta['n_free']} / "
          f"{grid_meta['n_total']} (res={grid_res} m, margin={margin} m)")
    return free_xy, grid_meta


# ══════════════════════════════════════════════════════════════════════════════
#  Quick-look plot
# ══════════════════════════════════════════════════════════════════════════════

def render_preview(xml_path: str, out_png: str | None = None,
                   spp: int = 64, variant: str = "scalar_rgb",
                   origin=None, target=(0., 0., 0.), fov: float = 45.0,
                   width: int = 1024, height: int = 768):
    """
    Headless render of the written scene using Mitsuba alone (no Sionna, no GPU).

    Useful as a CI/smoke check that the emitted XML+PLY actually parse and that
    the geometry is where you think it is.  Returns the linear RGB image as a
    numpy array; also writes ``out_png`` (tone-mapped) if given.
    """
    import mitsuba as mi
    # See build_occupancy_grid() in scene_utils.py for why this save/restore
    # matters: mi.set_variant() is global interpreter state, and leaving it on
    # a plain visual variant (this function never needs Sionna's radio-aware
    # variants, just geometry) breaks any later sionna.rt.load_scene() call in
    # the same process with "Plugin itu-radio-material not found!".
    try:
        _prev_variant = mi.variant()
    except Exception:
        _prev_variant = None

    try:
        mi.set_variant(variant)
    except Exception:
        mi.set_variant("llvm_ad_rgb")

    try:
        scene = mi.load_file(xml_path)
        bbox = scene.bbox()

        if origin is None:
            d = float(max(bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1]))
            origin = (-0.9 * d, -0.9 * d, 0.7 * d)

        sensor = mi.load_dict({
            "type": "perspective",
            "fov": fov, "fov_axis": "x",
            "to_world": mi.ScalarTransform4f().look_at(
                origin=list(origin), target=list(target), up=[0, 0, 1]),
            "sampler": {"type": "independent", "sample_count": spp},
            "film": {"type": "hdrfilm", "width": width, "height": height,
                     "pixel_format": "rgb"},
        })

        img = mi.render(scene, sensor=sensor, spp=spp)
        arr = np.array(img)
        if out_png:
            # write_async=True (mitsuba's default) returns before the file is
            # actually on disk -- every caller here immediately re-opens out_png
            # (this cell, the Sionna preview cell, the Monte-Carlo contact sheet),
            # racing the async write. write_async=False blocks until it's done.
            mi.util.write_bitmap(out_png, img, write_async=False)
            print(f"[scene_gen_ppp] preview render -> {out_png}")
        print(f"[scene_gen_ppp] mitsuba bbox: "
              f"x[{bbox.min[0]:.1f},{bbox.max[0]:.1f}] "
              f"y[{bbox.min[1]:.1f},{bbox.max[1]:.1f}] "
              f"z[{bbox.min[2]:.1f},{bbox.max[2]:.1f}]")
        return arr
    finally:
        if _prev_variant is not None:
            mi.set_variant(_prev_variant)


def _tallest_covering(spec: SceneSpec, pts: np.ndarray,
                      chunk: int = 4096) -> np.ndarray:
    """Height of the tallest wall slab whose footprint covers each of ``pts``."""
    ca, sa = np.cos(spec.orientations), np.sin(spec.orientations)
    half_l, half_t = spec.sides / 2.0, spec.thickness / 2.0
    out = np.zeros(len(pts))
    for lo in range(0, len(pts), chunk):
        q = pts[lo:lo + chunk]
        dx = q[:, 0, None] - spec.centers[None, :, 0]
        dy = q[:, 1, None] - spec.centers[None, :, 1]
        cover = (np.abs(dx * ca + dy * sa) <= half_l) & \
                (np.abs(-dx * sa + dy * ca) <= half_t)
        out[lo:lo + chunk] = np.where(cover, spec.heights, -np.inf).max(axis=1)
    return out


def verify_scene(xml_path: str, spec: SceneSpec, variant: str = "llvm_ad_rgb",
                 tol: float = 1e-3) -> dict:
    """
    Hard correctness check: cast a downward ray at every building centre and at
    a point known to be street, and compare the hit heights against ``spec``.

    Returns a report dict; raises AssertionError if any building is missing or
    at the wrong height.  This is the check that says "the scene really loaded",
    as opposed to "the file parsed".
    """
    import mitsuba as mi
    # Same save/restore reasoning as render_preview()/build_occupancy_grid():
    # this function's variant choice must not leak past its return (including
    # on the assert failures below -- try/finally covers those too), or the
    # next sionna.rt.load_scene() call in the process breaks.
    try:
        _prev_variant = mi.variant()
    except Exception:
        _prev_variant = None

    for v in (variant, "llvm_ad_rgb", "scalar_rgb"):
        try:
            mi.set_variant(v)
            break
        except Exception:
            continue

    try:
        vectorised = not mi.variant().startswith("scalar")

        scene = mi.load_file(xml_path)
        bbox = scene.bbox()
        z_test = float(bbox.max[2]) + 100.0

        n = spec.n_buildings
        ox = spec.centers[:, 0].astype(np.float32)
        oy = spec.centers[:, 1].astype(np.float32)

        if vectorised:
            rays = mi.Ray3f(
                mi.Point3f(ox, oy, np.full(n, z_test, np.float32)),
                mi.Vector3f(np.zeros(n, np.float32), np.zeros(n, np.float32),
                            np.full(n, -1.0, np.float32)),
            )
            si = scene.ray_intersect(rays)
            hit = np.array(si.is_valid()).astype(bool)
            hz = np.array(si.p[2], dtype=float)
        else:                                   # scalar variant: loop
            hit = np.zeros(n, dtype=bool)
            hz = np.zeros(n, dtype=float)
            for b in range(n):
                r = mi.Ray3f(mi.Point3f(float(ox[b]), float(oy[b]), z_test),
                             mi.Vector3f(0.0, 0.0, -1.0))
                s = scene.ray_intersect(r)
                hit[b] = bool(s.is_valid())
                hz[b] = float(s.p[2])

        # Wall slabs overlap freely (random orientations, no exclusion rule), so
        # a downward ray at wall i's centre legitimately lands on a TALLER wall
        # crossing it. Compare against the highest slab actually covering that
        # point rather than against the wall's own height -- a stricter test,
        # since it has to get the occlusion order right too.
        expected = spec.heights if spec.primitive != "wall" \
            else _tallest_covering(spec, spec.centers)
        err = np.abs(hz - expected)
        report = dict(
            n_buildings=n,
            n_hit=int(hit.sum()),
            max_height_error=float(err.max()) if n else 0.0,
            bbox_z_max=float(bbox.max[2]),
            spec_z_max=float(spec.heights.max()) if n else 0.0,
            n_shapes_xml=None,
        )
        assert hit.all(), f"{n - hit.sum()} building centres had no ray hit"
        assert err.max() < max(tol, 1e-3), \
            f"roof height mismatch, max error {err.max():.4g} m"
        print(f"[scene_gen_ppp] verify OK: {n}/{n} roofs hit, "
              f"max height error {err.max():.2e} m, "
              f"bbox z_max {report['bbox_z_max']:.2f} m")
        return report
    finally:
        if _prev_variant is not None:
            mi.set_variant(_prev_variant)


def _plot_wall_scene(spec: SceneSpec, figsize=(13, 4.4)):
    """
    Wall-primitive diagnostic: segments coloured by height, the orientation
    histogram, and the height PDF.

    Panel 2 is the orientation histogram rather than the block scene's per-cell
    counts (which is a 1x1 dummy here): xi ~ U[0, pi) is an assumption the
    analytics leans on directly, so it is worth being able to see.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    R = spec.radius
    half = spec.sides / 2.0
    ca, sa = np.cos(spec.orientations), np.sin(spec.orientations)

    a = ax[0]
    segs = np.stack([
        np.column_stack([spec.centers[:, 0] - half * ca, spec.centers[:, 1] - half * sa]),
        np.column_stack([spec.centers[:, 0] + half * ca, spec.centers[:, 1] + half * sa]),
    ], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidths=1.0)
    lc.set_array(spec.heights)
    a.add_collection(lc)
    a.add_patch(plt.Circle((0, 0), R, fill=False, ls="--", lw=0.8, color="k"))
    a.plot(0, 0, marker="*", ms=13, color="crimson", zorder=5, label="GT")
    a.set_xlim(-R, R); a.set_ylim(-R, R); a.set_aspect("equal")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.legend(fontsize=8, loc="upper right")
    a.set_title(f"{spec.n_buildings} walls  "
                f"$\\ell_f$={spec.meta['l_f']:g} m  $R$={R:g} m")
    fig.colorbar(lc, ax=a, label="height [m]", fraction=0.046)

    a = ax[1]
    if spec.n_buildings:
        a.hist(spec.orientations, bins=36, range=(0, np.pi), density=True,
               color="darkseagreen", edgecolor="k")
    a.axhline(1.0 / np.pi, color="r", lw=2, label=r"$\mathcal{U}[0,\pi)$")
    a.set_xlim(0, np.pi); a.set_xlabel(r"wall orientation $\xi$ [rad]")
    a.set_ylabel("pdf"); a.set_title("orientation"); a.legend(fontsize=8)

    a = ax[2]
    if spec.n_buildings:
        a.hist(spec.heights, bins=30, density=True, alpha=0.65,
               color="steelblue", edgecolor="k", label="sampled")
        g = spec.meta["gamma"]
        hh = np.linspace(0, spec.heights.max() * 1.05, 300)
        a.plot(hh, hh / g ** 2 * np.exp(-hh ** 2 / (2 * g ** 2)), 'r-', lw=2,
               label=f"Rayleigh $\\gamma$={g:g} m")
    a.set_xlabel("wall height [m]"); a.set_ylabel("pdf")
    a.set_title("ITU-R P.1410 heights"); a.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_scene(spec: SceneSpec, show_points: bool = True, figsize=(13, 4.4)):
    """Three-panel diagnostic: PPP + footprints, cell counts, height PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    if spec.primitive == "wall":
        return _plot_wall_scene(spec, figsize)

    xmin, xmax, ymin, ymax = spec.bbox()
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # -- footprints coloured by height --------------------------------------
    a = ax[0]
    patches = [Rectangle((c[0] - s / 2, c[1] - s / 2), s, s)
               for c, s in zip(spec.centers, spec.sides)]
    pc = PatchCollection(patches, cmap="viridis", edgecolor="k", linewidths=0.2)
    pc.set_array(spec.heights)
    a.add_collection(pc)
    if show_points and len(spec.points):
        a.plot(spec.points[:, 0], spec.points[:, 1], '.', ms=1.0,
               color="crimson", alpha=0.35, zorder=3)
    a.set_xlim(xmin, xmax); a.set_ylim(ymin, ymax); a.set_aspect("equal")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    a.set_title(f"{spec.n_buildings} buildings  "
                f"$\\alpha$={spec.meta['itu_alpha']:.2f}")
    fig.colorbar(pc, ax=a, label="height [m]", fraction=0.046)

    # -- cell counts ---------------------------------------------------------
    a = ax[1]
    im = a.imshow(spec.counts.T, origin="lower", cmap="magma",
                  extent=[xmin, xmax, ymin, ymax])
    a.set_title("PPP points per cell $k_{ij}$")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    fig.colorbar(im, ax=a, fraction=0.046)

    # -- height distribution -------------------------------------------------
    a = ax[2]
    if spec.n_buildings:
        a.hist(spec.heights, bins=30, density=True, alpha=0.65,
               color="steelblue", edgecolor="k", label="sampled")
        g = spec.meta["gamma"]
        hh = np.linspace(0, spec.heights.max() * 1.05, 300)
        a.plot(hh, hh / g ** 2 * np.exp(-hh ** 2 / (2 * g ** 2)), 'r-', lw=2,
               label=f"ITU Rayleigh $\\gamma$={g:g} m")
    a.set_xlabel("building height [m]"); a.set_ylabel("pdf")
    a.set_title("ITU-R P.1410 heights"); a.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--name", default="ppp_scene")
    p.add_argument("--size", nargs=2, type=float, default=[1000., 1000.],
                   metavar=("LX", "LY"))
    p.add_argument("--cell-res", type=float, default=50.0)
    p.add_argument("--alpha", type=float, default=0.35,
                   help="target built-up area fraction")
    p.add_argument("--points-per-cell", type=float, default=8.0)
    p.add_argument("--lam", type=float, default=None,
                   help="PPP intensity [pts/m^2] (overrides --alpha/--points-per-cell)")
    p.add_argument("--a-p", type=float, default=None,
                   help="unit area per point [m^2]")
    p.add_argument("--gamma", type=float, default=20.0)
    p.add_argument("--h-min", type=float, default=4.0)
    p.add_argument("--h-max", type=float, default=120.0)
    p.add_argument("--max-fill", type=float, default=0.98)
    p.add_argument("--min-side", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--merge", action="store_true",
                   help="write a single merged buildings.ply")
    p.add_argument("--plot", default=None, help="save a diagnostic PNG here")
    a = p.parse_args()

    xml, spec = generate_scene(
        a.out, scene_name=a.name,
        one_ply_per_building=not a.merge,
        Lx=a.size[0], Ly=a.size[1], cell_res=a.cell_res,
        alpha=a.alpha, points_per_cell=a.points_per_cell,
        lam=a.lam, a_p=a.a_p,
        gamma=a.gamma, h_min=a.h_min, h_max=a.h_max,
        max_fill=a.max_fill, min_side=a.min_side, seed=a.seed,
    )
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_scene(spec)
        fig.savefig(a.plot, dpi=140)
        print(f"[scene_gen_ppp] plot -> {a.plot}")
    return xml


if __name__ == "__main__":
    _cli()
