"""Scene loading, receiver placement and antenna configuration.

build_scene_batch is the chokepoint every path-solving entry point passes
through, so the ITU range widening and the Sionna patch verification both live
there. Its Sionna imports stay deferred so this module's import graph is
GPU-free and the launcher can set CUDA_VISIBLE_DEVICES first.

See docs/pipelines.md and docs/sionna-patches.md.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import pickle

_CP_PATTERNS_REGISTERED = False   # set True after first register_cp_patterns() call
_ITU_RANGES_WIDENED     = False   # set True after first itu_range_patch.widen() call
_SIONNA_PATCH_VERIFIED  = False   # set True after first sionna_patch_check.require() call


def _resolve_scene_path(path):
    """Expand ~ and $VARS, falling back to a repo-relative lookup.

    Lets a committed config carry a portable path. Paths that already resolve
    are returned untouched, so existing absolute configs are unaffected.
    """
    path = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isabs(path) and not os.path.exists(path):
        repo_rel = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.exists(repo_rel):
            return repo_rel
    return path


# ─── Occupancy Grid via Mitsuba Ray Intersection ──────────────────────────────

def build_occupancy_grid(
    xml_path: str,
    rx_height: float = 1.5,
    grid_res: float  = 2.0,
    z_test: float    = 500.0,
    margin: float    = 20.0,
    building_thresh: float = 0.5,
    variant: str = "cuda_ad_rgb",
) -> tuple:
    """Flat-ground occupancy grid -> (free_xy [M,2], grid_meta)."""
    import mitsuba as mi

    # mi.set_variant() changes GLOBAL interpreter state, not something scoped
    # to this call. Sionna RT's own scene loading (sionna.rt.load_scene(),
    # used elsewhere in this same notebook/pipeline) needs one of Sionna's
    # radio-aware variants (e.g. "cuda_ad_mono_polarized") because the custom
    # "itu-radio-material" BSDF plugin it relies on is only registered there
    # -- a plain visual variant like "llvm_ad_rgb" (used below, since this
    # function only ever does geometry-only ray casting) does not have it.
    # Leaving the variant on "llvm_ad_rgb" after this function returns was
    # silently breaking any *later* sionna.rt.load_scene() call in the same
    # process with "Plugin itu-radio-material not found!" -- save/restore so
    # this function's own variant choice never leaks past its return.
    try:
        _prev_variant = mi.variant()
    except Exception:
        _prev_variant = None

    try:
        mi.set_variant(variant)
    except Exception:
        mi.set_variant("llvm_ad_rgb")   # CPU fallback

    try:
        scene_mi = mi.load_file(xml_path)
        bbox = scene_mi.bbox()

        # Grid covers the full scene bbox (no outward expansion)
        xmin = float(bbox.min[0])
        xmax = float(bbox.max[0])
        ymin = float(bbox.min[1])
        ymax = float(bbox.max[1])

        # Inner valid-RX zone: inset by margin on every side
        valid_xmin = xmin + margin
        valid_xmax = xmax - margin
        valid_ymin = ymin + margin
        valid_ymax = ymax - margin

        xs = np.arange(xmin, xmax + grid_res, grid_res)
        ys = np.arange(ymin, ymax + grid_res, grid_res)
        XX, YY = np.meshgrid(xs, ys)
        x_flat = XX.flatten().astype(np.float32)
        y_flat = YY.flatten().astype(np.float32)
        n_pts  = len(x_flat)

        # Cast rays straight down
        ray_o = mi.Point3f(x_flat, y_flat, np.full(n_pts, z_test, dtype=np.float32))
        ray_d = mi.Vector3f(
            np.zeros(n_pts, dtype=np.float32),
            np.zeros(n_pts, dtype=np.float32),
            np.full(n_pts, -1.0, dtype=np.float32),
        )
        rays = mi.Ray3f(ray_o, ray_d)

        si = scene_mi.ray_intersect(rays)

        hit_valid = np.array(si.is_valid()).astype(bool)
        hit_z     = np.array(si.p[2], dtype=float)

        # Cells that are physically clear of buildings
        ground_hit = hit_valid & (hit_z <= rx_height + building_thresh)
        # Cells inside the inset valid zone
        in_valid_zone = (
            (x_flat >= valid_xmin) & (x_flat <= valid_xmax) &
            (y_flat >= valid_ymin) & (y_flat <= valid_ymax)
        )

        free_mask_full     = ground_hit                    # all ground-level cells in scene
        free_mask_valid    = ground_hit & in_valid_zone    # eligible RX cells (margin applied)
        occupied_mask_full = hit_valid & ~ground_hit       # building rooftop hits

        free_xy     = np.column_stack([x_flat[free_mask_valid].astype(float),
                                       y_flat[free_mask_valid].astype(float)])
        occupied_xy = np.column_stack([x_flat[occupied_mask_full].astype(float),
                                       y_flat[occupied_mask_full].astype(float)])

        grid_meta = {
            # Full scene extent (no extra padding)
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            # Scene bbox — same as full extent here, kept for API compatibility
            "scene_xmin": xmin, "scene_xmax": xmax,
            "scene_ymin": ymin, "scene_ymax": ymax,
            # Inner RX-valid zone (margin applied)
            "valid_xmin": valid_xmin, "valid_xmax": valid_xmax,
            "valid_ymin": valid_ymin, "valid_ymax": valid_ymax,
            "grid_res": grid_res,
            "margin": margin,
            # Counts (n_free reflects the margin-restricted valid cells)
            "n_total":    n_pts,
            "n_free":     int(np.sum(free_mask_valid)),
            "n_free_full": int(np.sum(free_mask_full)),   # without margin restriction
            "n_occupied": int(np.sum(occupied_mask_full)),
            # Occupied cell coordinates (for full-scene visualisation)
            "occupied_xy": occupied_xy,
        }
        print(
            f"[scene_utils] Occupancy grid: {grid_meta['n_free']} valid free cells "
            f"(margin={margin} m) / {grid_meta['n_free_full']} total ground cells / "
            f"{grid_meta['n_total']} grid cells  (res={grid_res} m)"
        )
        return free_xy, grid_meta
    finally:
        if _prev_variant is not None:
            mi.set_variant(_prev_variant)


# ─── Terrain-Aware Occupancy Grid (scenes with ground relief) ─────────────────

def build_occupancy_grid_terrain(
    xml_path: str,
    ground_xml_path: str,
    rx_height: float = 1.5,
    grid_res: float  = 1.0,
    z_test: float    = 500.0,
    margin: float    = 15.0,
    building_thresh: float = 1.5,
    exclude_shape_ids: tuple = (),
    variant: str = "cuda_ad_rgb",
) -> tuple:
    """Occupancy grid for scenes whose ground is NOT a flat plane."""
    import mitsuba as mi

    # Same global-variant save/restore rationale as build_occupancy_grid().
    try:
        _prev_variant = mi.variant()
    except Exception:
        _prev_variant = None

    try:
        mi.set_variant(variant)
    except Exception:
        mi.set_variant("llvm_ad_rgb")   # CPU fallback

    try:
        scene_full   = mi.load_file(xml_path)
        scene_ground = mi.load_file(ground_xml_path)
        bbox = scene_full.bbox()

        xmin, xmax = float(bbox.min[0]), float(bbox.max[0])
        ymin, ymax = float(bbox.min[1]), float(bbox.max[1])

        valid_xmin, valid_xmax = xmin + margin, xmax - margin
        valid_ymin, valid_ymax = ymin + margin, ymax - margin
        if valid_xmin >= valid_xmax or valid_ymin >= valid_ymax:
            raise ValueError(
                f"margin={margin} m leaves an empty valid zone for a "
                f"{xmax - xmin:.1f} x {ymax - ymin:.1f} m scene -- lower it."
            )

        xs = np.arange(xmin, xmax + grid_res, grid_res)
        ys = np.arange(ymin, ymax + grid_res, grid_res)
        XX, YY = np.meshgrid(xs, ys)
        x_flat = XX.flatten().astype(np.float32)
        y_flat = YY.flatten().astype(np.float32)
        n_pts  = len(x_flat)

        def _cast(scene_mi):
            rays = mi.Ray3f(
                mi.Point3f(x_flat, y_flat, np.full(n_pts, z_test, dtype=np.float32)),
                mi.Vector3f(np.zeros(n_pts, dtype=np.float32),
                            np.zeros(n_pts, dtype=np.float32),
                            np.full(n_pts, -1.0, dtype=np.float32)),
            )
            si = scene_mi.ray_intersect(rays)
            return si, np.array(si.is_valid()).astype(bool), np.array(si.p[2], dtype=float)

        _,       top_valid,    top_z    = _cast(scene_full)
        si_gnd,  ground_valid, ground_z = _cast(scene_ground)

        # Relief-invariant occupancy: how far the full scene stands above earth.
        both      = top_valid & ground_valid
        ground_hit = both & ((top_z - ground_z) <= building_thresh)

        # Optionally reject surfaces you may not stand on (e.g. open water).
        n_excluded = 0
        if exclude_shape_ids:
            g_root = ET.parse(ground_xml_path).getroot()
            g_ids  = [s.get("id") for s in g_root.findall("shape")]
            bad_idx = {i for i, sid in enumerate(g_ids) if sid in exclude_shape_ids}
            if bad_idx:
                # si.shape is a pointer array into the scene's shape list; compare
                # against each excluded shape's own pointer rather than an index.
                shapes = scene_ground.shapes()
                on_bad = np.zeros(n_pts, dtype=bool)
                for i in bad_idx:
                    on_bad |= np.array(
                        mi.Bool(si_gnd.shape == mi.ShapePtr(shapes[i]))
                    ).astype(bool)
                n_excluded = int(np.sum(ground_hit & on_bad))
                ground_hit = ground_hit & ~on_bad

        in_valid_zone = (
            (x_flat >= valid_xmin) & (x_flat <= valid_xmax) &
            (y_flat >= valid_ymin) & (y_flat <= valid_ymax)
        )

        free_mask_full     = ground_hit
        free_mask_valid    = ground_hit & in_valid_zone
        occupied_mask_full = both & ~ground_hit

        free_z  = ground_z[free_mask_valid] + rx_height
        free_xyz = np.column_stack([
            x_flat[free_mask_valid].astype(float),
            y_flat[free_mask_valid].astype(float),
            free_z,
        ])
        occupied_xy = np.column_stack([x_flat[occupied_mask_full].astype(float),
                                       y_flat[occupied_mask_full].astype(float)])

        grid_meta = {
            "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
            "scene_xmin": xmin, "scene_xmax": xmax,
            "scene_ymin": ymin, "scene_ymax": ymax,
            "valid_xmin": valid_xmin, "valid_xmax": valid_xmax,
            "valid_ymin": valid_ymin, "valid_ymax": valid_ymax,
            "grid_res": grid_res,
            "margin": margin,
            "n_total":     n_pts,
            "n_free":      int(np.sum(free_mask_valid)),
            "n_free_full": int(np.sum(free_mask_full)),
            "n_occupied":  int(np.sum(occupied_mask_full)),
            "occupied_xy": occupied_xy,
            # terrain-aware extras
            "free_z":          free_z,
            "ground_z_range":  (float(ground_z[ground_valid].min()),
                                float(ground_z[ground_valid].max())) if ground_valid.any()
                               else (0.0, 0.0),
            "ground_xml_path": ground_xml_path,
            "n_excluded_surface": n_excluded,
            "excluded_shape_ids": list(exclude_shape_ids),
        }
        print(
            f"[scene_utils] Terrain grid: {grid_meta['n_free']} valid free cells "
            f"(margin={margin} m) / {grid_meta['n_free_full']} total ground cells / "
            f"{n_pts} grid cells  (res={grid_res} m)"
        )
        print(
            f"[scene_utils]   ground z range {grid_meta['ground_z_range'][0]:.1f}"
            f"..{grid_meta['ground_z_range'][1]:.1f} m; "
            f"RX z {free_z.min():.1f}..{free_z.max():.1f} m"
            + (f"; {n_excluded} cells dropped on {list(exclude_shape_ids)}"
               if n_excluded else "")
        )
        return free_xyz, grid_meta
    finally:
        if _prev_variant is not None:
            mi.set_variant(_prev_variant)


# ─── XML Bounding-Box Fallback ────────────────────────────────────────────────

def _parse_matrix(mat_str):
    """Parse a Mitsuba <matrix value='...'> string into a 4×4 numpy array."""
    vals = [float(v) for v in mat_str.strip().split()]
    if len(vals) == 16:
        return np.array(vals).reshape(4, 4)
    return np.eye(4)


def build_occupancy_grid_xml(
    xml_path: str,
    rx_height: float = 1.5,
    grid_res: float  = 2.0,
    margin: float    = 20.0,
    scene_bbox: tuple = None,   # (xmin, xmax, ymin, ymax) override
) -> tuple:
    """Fallback occupancy grid from Mitsuba XML transforms, no ray tracer needed.

    Applies each <transform><matrix> to a unit cube and projects the resulting
    AABBs onto the ground plane to mark blocked cells. `scene_bbox` overrides the
    discovered extent. Returns (free_xy [M,2], grid_meta).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Collect footprints from all shapes with transforms
    footprints = []
    # Unit-cube corners in XY (z ignored for footprint)
    cube_corners = np.array([
        [-0.5, -0.5, 0, 1],
        [ 0.5, -0.5, 0, 1],
        [ 0.5,  0.5, 0, 1],
        [-0.5,  0.5, 0, 1],
        [-0.5, -0.5, 1, 1],
        [ 0.5, -0.5, 1, 1],
        [ 0.5,  0.5, 1, 1],
        [-0.5,  0.5, 1, 1],
    ])

    for shape in root.iter("shape"):
        mat_elem = shape.find(".//transform/matrix")
        if mat_elem is None:
            continue
        mat = _parse_matrix(mat_elem.get("value", ""))
        world = (mat @ cube_corners.T).T
        xs_w  = world[:, 0]
        ys_w  = world[:, 1]
        footprints.append((xs_w.min(), ys_w.min(), xs_w.max(), ys_w.max()))

    if not footprints:
        raise RuntimeError(
            "No shapes with <matrix> transforms found in XML. "
            "Use build_occupancy_grid() (Mitsuba ray-cast) instead."
        )

    all_x = [v for fp in footprints for v in [fp[0], fp[2]]]
    all_y = [v for fp in footprints for v in [fp[1], fp[3]]]

    if scene_bbox is not None:
        xmin, xmax, ymin, ymax = scene_bbox
    else:
        xmin = min(all_x)
        xmax = max(all_x)
        ymin = min(all_y)
        ymax = max(all_y)

    # Inner valid-RX zone: inset by margin on every side
    valid_xmin = xmin + margin
    valid_xmax = xmax - margin
    valid_ymin = ymin + margin
    valid_ymax = ymax - margin

    xs = np.arange(xmin, xmax + grid_res, grid_res)
    ys = np.arange(ymin, ymax + grid_res, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    x_flat = XX.flatten()
    y_flat = YY.flatten()

    occupied = np.zeros(len(x_flat), dtype=bool)
    for (fxmin, fymin, fxmax, fymax) in footprints:
        in_fp = (
            (x_flat >= fxmin) & (x_flat <= fxmax) &
            (y_flat >= fymin) & (y_flat <= fymax)
        )
        occupied |= in_fp

    ground_clear = ~occupied
    in_valid_zone = (
        (x_flat >= valid_xmin) & (x_flat <= valid_xmax) &
        (y_flat >= valid_ymin) & (y_flat <= valid_ymax)
    )

    free_mask  = ground_clear & in_valid_zone   # eligible RX cells (margin applied)
    free_xy    = np.column_stack([x_flat[free_mask], y_flat[free_mask]])
    occupied_xy = np.column_stack([x_flat[occupied], y_flat[occupied]])

    grid_meta = {
        "xmin": xmin, "xmax": xmax,
        "ymin": ymin, "ymax": ymax,
        "scene_xmin": xmin, "scene_xmax": xmax,
        "scene_ymin": ymin, "scene_ymax": ymax,
        "valid_xmin": valid_xmin, "valid_xmax": valid_xmax,
        "valid_ymin": valid_ymin, "valid_ymax": valid_ymax,
        "grid_res": grid_res,
        "margin": margin,
        "n_total": len(x_flat),
        "n_free": int(np.sum(free_mask)),
        "n_free_full": int(np.sum(ground_clear)),
        "n_occupied": int(np.sum(occupied)),
        "occupied_xy": occupied_xy,
        "footprints": footprints,
    }
    print(
        f"[scene_utils/xml] Occupancy grid: {grid_meta['n_free']} valid free cells "
        f"(margin={margin} m) / {grid_meta['n_free_full']} total ground cells / "
        f"{grid_meta['n_total']} grid cells  (res={grid_res} m)"
    )
    return free_xy, grid_meta


# ─── Sample Receiver Positions from Free Grid ─────────────────────────────────

def sample_rx_positions(
    free_xy:   np.ndarray,
    n:         int,
    rx_height: float,
    rng:       np.random.Generator = None,
    jitter:    float = None,         # random offset within each cell (defaults to grid_res/2)
    grid_res:  float = 2.0,
    free_z:    np.ndarray = None,
) -> np.ndarray:
    """Draw N receiver positions [N,3] from the free-cell catalogue.

    Each cell is jittered uniformly in XY by +/-`jitter` (default grid_res/2) so
    receivers are not grid-locked. Height comes from the per-cell `free_z`, or
    from a 3-column `free_xy` as returned by `build_occupancy_grid_terrain`;
    with neither, every receiver sits at the flat-ground `rx_height`.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if jitter is None:
        jitter = grid_res / 2.0

    free_xy = np.asarray(free_xy)
    if free_z is None and free_xy.ndim == 2 and free_xy.shape[1] >= 3:
        free_z = free_xy[:, 2]
    free_xy = free_xy[:, :2]

    M = len(free_xy)
    if M == 0:
        raise ValueError("free_xy is empty — no valid receiver positions available.")
    if n > M:
        print(
            f"[scene_utils] Warning: requested {n} positions but only {M} free cells. "
            "Sampling with replacement."
        )
        replace = True
    else:
        replace = False

    idx = rng.choice(M, size=n, replace=replace)
    chosen = free_xy[idx].copy()

    # Add uniform jitter within each cell (XY only -- jittering Z would lift
    # receivers off / bury them under the terrain surface they belong to)
    chosen += rng.uniform(-jitter, jitter, size=chosen.shape)

    if free_z is None:
        z_col = np.full((n, 1), rx_height)
    else:
        z_col = np.asarray(free_z)[idx].reshape(-1, 1)
    rx_pos = np.hstack([chosen, z_col])
    return rx_pos


# ─── Antenna array factory ────────────────────────────────────────────────────

def _make_antenna_array(pol_type: str, base_pattern: str = "iso"):
    """Build a 1×1 PlanarArray for the requested polarization type."""
    from sionna.rt import PlanarArray

    _LINEAR = {"V", "H", "VH", "CROSS"}
    pol_up = pol_type.upper()

    if pol_up in _LINEAR:
        sionna_pol = "cross" if pol_up == "CROSS" else pol_type
        return PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="iso", polarization=sionna_pol,
        )

    if pol_up in ("RHCP", "LHCP"):
        from Ray_Tracing.src.utils.circular_polarization import register_cp_patterns

        # Register CP named patterns once per process. Sionna silently allows
        # re-registration so we guard with a module-level flag to avoid noise.
        global _CP_PATTERNS_REGISTERED
        if not _CP_PATTERNS_REGISTERED:
            register_cp_patterns()
            _CP_PATTERNS_REGISTERED = True

        # Pattern string format: "{rhcp|lhcp}_{base}", e.g. "lhcp_iso"
        _VALID_BASES = {"iso", "dipole", "hw_dipole", "tr38901"}
        pat_base = base_pattern if base_pattern in _VALID_BASES else "iso"
        pattern_name = f"{pol_up.lower()}_{pat_base}"
        return PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern=pattern_name, polarization="V",
        )

    raise ValueError(
        f"Unknown polarization type {pol_type!r}. "
        "Valid options: 'V', 'H', 'VH', 'cross', 'RHCP', 'LHCP'."
    )


# ─── In-place scene retargeting ───────────────────────────────────────────────
# A loaded Sionna Scene can be re-aimed without re-parsing its geometry, which
# is what makes a scene-outer sweep (run_chiu_sweep.py: one scene, many
# elevations and polarizations) worth doing -- at a 2 km scene radius the mesh
# parse dominates a single solve. Verified bit-exact: retargeting a loaded scene
# and rebuilding it from scratch give identical delays and path powers across
# elevations and both polarizations.
#
# build_scene_batch() below calls these too, so there is one code path.

def set_polarization(scene, tx_pol: str, rx_pol: str, base_pattern: str = "iso"):
    """Reassign the TX/RX antenna arrays on a (possibly already loaded) scene."""
    # Sionna RT's jones_vec_dot computes u^H·v where the RX pattern u is
    # first passed through jones_matrix_rotator_flip_forward, which negates
    # the phi-component.  For CP this flips handedness: LHCP→RHCP, RHCP→LHCP.
    # To preserve the standard co-pol convention (LHCP TX → LHCP RX = max
    # coupling), we use the conjugate handedness internally for the RX array.
    _CP_CONJ = {"RHCP": "LHCP", "LHCP": "RHCP"}
    rx_pol_internal = _CP_CONJ.get(rx_pol.upper(), rx_pol)
    scene.tx_array = _make_antenna_array(tx_pol, base_pattern)
    scene.rx_array = _make_antenna_array(rx_pol_internal, base_pattern)


def set_tx_pose(scene, tx_pos, scene_center, tx_velocity=None,
                omega_F: float = 0.0):
    """
    Move the transmitter to ``tx_pos`` and re-aim it at the scene centre.

    Use this to step a loaded scene through an elevation sweep. The TX must
    already exist (build_scene_batch adds it).
    """
    tx = scene.get("tx")
    if tx is None:
        raise KeyError("scene has no transmitter named 'tx'")
    tx.position = [float(tx_pos[0]), float(tx_pos[1]), float(tx_pos[2])]
    if tx_velocity is not None:
        tx.velocity = [float(tx_velocity[0]), float(tx_velocity[1]),
                       float(tx_velocity[2])]
    tx.look_at([float(scene_center[0]), float(scene_center[1]), 0.0])

    # Apply Faraday rotation: add omega_F as a roll offset around the TX
    # boresight (the axis pointing at the scene after look_at).  This rotates
    # the V-pol E-field toward H-pol by the sampled Faraday angle, so Sionna
    # computes Fresnel coefficients for the correct mixed polarisation.
    # look_at() resets orientation, so this must follow it.
    if omega_F != 0.0:
        orient = tx.orientation          # [alpha, beta, gamma] — yaw/pitch/roll
        # tx.orientation returns a drjit tensor; extract numpy scalars before
        # calling float() — plain float() cannot handle drjit types.
        orient_np = np.array(orient).ravel()
        tx.orientation = [float(orient_np[0]),
                          float(orient_np[1]),
                          float(orient_np[2]) + float(omega_F)]
    return tx


def set_rx_positions(scene, rx_positions_batch):
    """
    Move the existing receivers. Cheaper than rebuilding the scene for a few
    centimetres of jitter (the ground-path dropout retry does exactly that).
    """
    for i, rx_pos in enumerate(rx_positions_batch):
        rx = scene.get(f"rx_{i}")
        if rx is None:
            raise KeyError(f"scene has no receiver named 'rx_{i}'")
        rx.position = [float(rx_pos[0]), float(rx_pos[1]), float(rx_pos[2])]


# ─── Sionna Scene Builder (batched) ───────────────────────────────────────────

def build_scene_batch(rx_positions_batch, tx_pos, cfg: dict,
                      omega_F: float = 0.0, tx_velocity=None):
    """Load the urban scene and configure one TX and a batch of RXs."""
    from sionna.rt import load_scene, Transmitter, Receiver, DirectivePattern

    # Must run before `scene.frequency = ...` below, which fires every ITU
    # material's frequency callback and raises for out-of-band materials.
    # Deferred to here, not module import, to keep sionna out of scene_utils'
    # import graph -- the launcher sets CUDA_VISIBLE_DEVICES before any GPU import.
    global _ITU_RANGES_WIDENED
    if not _ITU_RANGES_WIDENED:
        from Ray_Tracing.src.utils.itu_range_patch import widen
        widen()
        _ITU_RANGES_WIDENED = True

    # Verify the patched Sionna is the one loaded. Checked here, not in a
    # runner, because build_scene_batch is the chokepoint every path-solving
    # entry point passes through. Deferred import for the same reason as the
    # ITU patch above: scene_utils' import graph stays GPU-free so the launcher
    # can set CUDA_VISIBLE_DEVICES first. See docs/sionna-patches.md.
    global _SIONNA_PATCH_VERIFIED
    if not _SIONNA_PATCH_VERIFIED:
        import Ray_Tracing.src.sionna_patch_check as sionna_patch_check
        sionna_patch_check.require("build_scene_batch")
        _SIONNA_PATCH_VERIFIED = True

    # merge_shapes groups meshes by radio material. Default False keeps the
    # per-object identity the Sieg/Wharf/PPP scenes were run with. Scenes with
    # thousands of shapes need True: Sionna's per-shape solver dispatch does not
    # complete on San Francisco's 4735 shapes even at 0.1% of the production
    # sample budget, against ~1 s merged. Mitsuba's merge concatenates without
    # welding, so half-edge adjacency -- and the diffracting-wedge set -- is
    # unchanged; paths.interactions carries type codes, not object ids, so
    # split_by_interaction_type is unaffected either way.
    scene = load_scene(_resolve_scene_path(cfg["SCENE_XML"]),
                       merge_shapes=bool(cfg.get("MERGE_SHAPES", False)))
    scene.frequency = cfg["FREQUENCY"]

    tx_pol      = cfg.get("TX_POL_TYPE",     "V")
    rx_pol      = cfg.get("RX_POL_TYPE",     "V")
    base_pat    = cfg.get("POL_BASE_PATTERN", "iso")

    set_polarization(scene, tx_pol, rx_pol, base_pat)

    # mitsuba.Point3f/Vector3f only accept native Python float — cast explicitly
    tx_pos_f = [float(tx_pos[0]), float(tx_pos[1]), float(tx_pos[2])]
    tx_vel_f = [0.0, 0.0, 0.0] if tx_velocity is None else \
               [float(tx_velocity[0]), float(tx_velocity[1]), float(tx_velocity[2])]
    tx = Transmitter(name="tx", position=tx_pos_f, velocity=tx_vel_f)
    scene.add(tx)
    tx.power_dbm = float(cfg["TX_POWER_DBM"])
    set_tx_pose(scene, tx_pos, cfg["SCENE_CENTER"],
                tx_velocity=tx_velocity, omega_F=omega_F)

    for i, rx_pos in enumerate(rx_positions_batch):
        rx_pos_f = [float(rx_pos[0]), float(rx_pos[1]), float(rx_pos[2])]
        rx = Receiver(name=f"rx_{i}", position=rx_pos_f, orientation=[0., 0., 0.])
        scene.add(rx)

    # Apply material properties to all concrete objects
    for mat_name in cfg.get("MATERIAL_NAMES", ["itu_concrete"]):
        mat = scene.get(mat_name)
        if mat is not None:
            mat.scattering_coefficient = cfg["SCATTERING_COEFF"]
            mat.scattering_pattern     = DirectivePattern()
            # mat.relative_permittivity  = cfg["ETA_R"]
            # mat.conductivity           = cfg["SIGMA"]

    # Per-role material parameterisation (additive, optional). Overrides
    # relative_permittivity/conductivity/scattering_coefficient per named
    # material using each one's own ITU-R P.2040 a,b,c,d coefficients,
    # independent of whatever built-in ITU table values the dummy identity
    # string in the XML implied. Runs AFTER `scene.frequency = ...` above
    # already fired every ITU material's one-time frequency_update_callback
    # (which set the built-in defaults), so this unconditionally wins and is
    # never re-clobbered later (frequency is set exactly once in this
    # function). Callers that never set MATERIAL_PARAMS (e.g. the site
    # pipeline) are completely unaffected -- this block is then a no-op.
    material_params = cfg.get("MATERIAL_PARAMS")
    if material_params:
        f_ghz = cfg["FREQUENCY"] / 1e9
        for mat_name, p in material_params.items():
            mat = scene.get(mat_name)
            if mat is None:
                continue
            mat.relative_permittivity  = float(p["a"]) * f_ghz ** float(p["b"])
            mat.conductivity           = float(p["c"]) * f_ghz ** float(p["d"])
            mat.scattering_coefficient = float(p.get("scattering_coefficient", 0.0))
            if "xpd_coefficient" in p:
                mat.xpd_coefficient = float(p["xpd_coefficient"])
            # Sionna models a material as a SLAB of finite `thickness` (default
            # 0.1 m), so a wall/ground reflection carries thin-film interference
            # with its own back face rather than the semi-infinite half-space
            # Fresnel coefficient. At 2 GHz that is a multi-dB, erratically
            # permittivity-dependent error. Setting thickness >= ~1 m on a lossy
            # material attenuates the internal round trip and recovers the
            # half-space coefficient (verified to <0.15 dB). A LOSSLESS slab
            # never converges at any thickness -- give it a small conductivity.
            if "thickness" in p:
                import mitsuba as mi
                mat.thickness = mi.Float(float(p["thickness"]))
            mat.scattering_pattern = DirectivePattern()

    return scene, len(rx_positions_batch)


# ─── Save / Load Valid RX Positions ───────────────────────────────────────────

def save_rx_positions(rx_pos: np.ndarray, path: str, meta: dict = None):
    """Save generated RX positions (and optional metadata) to a pickle file."""
    payload = {"rx_positions": rx_pos}
    if meta:
        payload["meta"] = meta
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[scene_utils] Saved {len(rx_pos)} RX positions → {path}")


def load_rx_positions(path: str) -> tuple:
    """Load RX positions from a pickle file saved by save_rx_positions."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    rx_pos = payload["rx_positions"]
    meta   = payload.get("meta", None)
    print(f"[scene_utils] Loaded {len(rx_pos)} RX positions ← {path}")
    return rx_pos, meta