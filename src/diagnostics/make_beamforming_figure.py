"""Before/after scene renders for the ray-beamforming Sionna patch.

Traces one NTN link -- one scene, one receiver, one satellite geometry, one ray
budget -- twice: once the way stock ``sionna-rt 1.2.1`` launches rays
(isotropically over 4*pi) and once the way the ``beamforming`` patch does
(confined to a cone of half-angle arctan(scene_radius / slant_range) about TX
boresight). Both solves are drawn by Sionna's own path renderer, giving a
web-ready before/after pair.

The "before" panel legitimately comes out near-empty. At a 500 km slant range a
230 m scene subtends about (230/5e5)^2/4 ~ 5e-8 of the sphere, so an isotropic
launch puts no shoot-and-bounce rays on it at any tractable budget. Only the
deterministic line-of-sight test -- which the patch does not touch -- survives.
That is the defect the patch exists to fix, so the panel is annotated to say so
rather than being made to look less broken.

The ray budget is deliberately small (~50k, against 10M in production). The
point here is legibility, not convergence: Sionna draws every path segment as
an emissive cylinder, and a few hundred of them read as distinct rays where a
few thousand bloom into a white wash.

Run from the parent of the repository so ``Ray_Tracing.src.*`` resolves::

    cd /path/to
    CUDA_VISIBLE_DEVICES=0 python -m Ray_Tracing.src.diagnostics.make_beamforming_figure
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SCENE_XML = os.path.join(REPO_ROOT, "RT Scenarios", "sieg_scene", "sieg_notrees.xml")
DEFAULT_CONFIG    = os.path.join(REPO_ROOT, "src", "simulation_configs", "sim_config.json")
DEFAULT_OUTDIR    = os.path.join(REPO_ROOT, "figures", "beamforming_demo")
# Built with build_occupancy_grid_terrain against sieg_ground_only.xml; the
# flat-ground grid does not work on this scene (see pick_receiver).
DEFAULT_RX_FILE   = os.path.join(REPO_ROOT, "src", "simulation_configs",
                                 "rx_positions_sieg.pkl")

TYPE_NAMES = {0: "LoS", 1: "Specular", 2: "Diffuse", 4: "Refraction", 8: "Diffraction"}
# Order the legend rarest-last so the bulk mechanisms read first.
TYPE_ORDER = [2, 1, 4, 8, 0]

CAMERA_D    = 340.0     # standoff basis, as in make_report_figures.fig_sieg_scene
# Higher and tighter than fig_sieg_scene: the ground then fills the frame, so
# there is no blank sky for rays to vanish into and the scene-edge geometry
# stays out of shot. Receivers must sit centrally for this to hold -- see
# pick_receiver's radius_frac.
CAMERA_OFF  = 0.66
CAMERA_H    = 0.90
CAMERA_FOV  = 34.0
RESOLUTION  = (1600, 1000)


def tune_path_colors(first_leg_scale: float = 1.0):
    """Darken the first-leg colour used by Sionna's path renderer.

    Sionna hardcodes segment colour inside get_overlay_scene -- there is no
    argument for it -- but the values come from module-level constants, so they
    can be re-pointed for this process. The first leg is the one worth dimming:
    every path has one, they all run on the same bearing, and at LOS_COLOR's
    default mid-grey a few hundred emissive cylinders bloom into a curtain that
    buries the scene and the mechanism colours alike.
    """
    from sionna.rt.constants import LOS_COLOR
    from sionna.rt.utils import render as render_mod

    render_mod.LOS_COLOR = tuple(float(c) * float(first_leg_scale)
                                 for c in LOS_COLOR)
    return render_mod.LOS_COLOR


def sionna_segment_colors(first_leg_scale: float = 1.0):
    """Sionna's own segment colours, after `tune_path_colors` has run.

    Sionna colours each SEGMENT by the interaction preceding it, not each path
    by its primary type: the leg leaving the source is always LOS_COLOR, and
    the colour changes exactly at the bend that caused it.
    """
    from sionna.rt.constants import (SPECULAR_COLOR, DIFFUSE_COLOR,
                                     REFRACTION_COLOR, DIFFRACTION_COLOR)
    from sionna.rt.utils import render as render_mod
    return {0: tuple(render_mod.LOS_COLOR), 1: SPECULAR_COLOR,
            2: DIFFUSE_COLOR, 4: REFRACTION_COLOR, 8: DIFFRACTION_COLOR}


# ─── Path geometry ────────────────────────────────────────────────────────────

def extract_geometry(paths):
    """(vertices [D,P,3], interactions [D,P], valid [P]) for the first antenna pair.

    Paths carries [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths,
    3] with synthetic_array=False. Ray geometry is identical across antenna
    elements -- only the field differs -- so index 0 of the collapsed middle
    axis is representative.
    """
    verts = paths.vertices.numpy()
    inter = paths.interactions.numpy()
    valid = paths.valid.numpy()

    depth, n_paths = verts.shape[0], verts.shape[-2]
    if n_paths == 0:
        # A solve can legitimately find nothing -- a blocked direct ray with
        # scattering disabled, say. numpy cannot infer the collapsed axis from
        # a zero-size array, so short-circuit.
        return (np.zeros((depth, 0, 3)), np.zeros((depth, 0), dtype=int),
                np.zeros(0, dtype=bool))
    verts = verts.reshape(depth, -1, n_paths, 3)[:, 0]
    inter = inter.reshape(depth, -1, n_paths)[:, 0]
    valid = valid.reshape(-1, n_paths)[0]
    return verts, inter, valid.astype(bool)


def primary_types(inter):
    """First non-zero interaction per path; 0 (LoS) if a path has none.

    Same rule as channel_utils.split_by_interaction_type, so a path that
    diffracts then reflects counts as diffraction here too.
    """
    nonzero = inter != 0
    types = np.zeros(inter.shape[1], dtype=int)
    active = np.any(nonzero, axis=0)
    if np.any(active):
        idx = np.where(active)[0]
        first = np.argmax(nonzero[:, idx], axis=0)
        types[idx] = inter[first, idx]
    return types

# ─── Preparing a Paths object for rendering ──────────────────────────────────

def clip_source(paths, tx_pos, anchor, standoff_m):
    """Pull the path source in to a proxy standoff, and return its position.

    Sionna builds every path's first segment from ``paths.sources``
    (sionna/rt/utils/render.py:150), so at a true 500 km slant range each one
    becomes a 500 km emissive cylinder. Rendered as-is the frame washes to
    white; cropping cannot help, because it is the IN-frame portion of those
    cylinders that covers everything.

    Rewriting the private backing attribute is the only hook Sionna offers --
    segment length, radius and colour are all fixed inside get_overlay_scene.

    This is a real distortion, not just a crop: a point source at finite range
    cannot reproduce a beam that is parallel to within 0.02 deg, so the drawn
    rays visibly diverge from the proxy. Standoff is the trade -- closer gives
    shorter, less cluttered legs but more divergence, and a bounce r metres
    off-centre is misaimed by arctan(r / standoff). `anchor` is the point the
    bearing is taken from, and so the one point whose incoming ray is drawn
    exactly right: the scene centre when bounces are spread over the whole
    scene, the receiver when one particular ray is the subject.
    """
    import mitsuba as mi

    anchor = np.asarray(anchor, dtype=float)
    bearing = np.asarray(tx_pos, dtype=float) - anchor
    bearing /= np.linalg.norm(bearing)
    proxy = anchor + bearing * float(standoff_m)
    paths._src_positions = mi.Point3f(*[float(v) for v in proxy])
    return proxy


def thin_paths(paths, keep_n, rng):
    """Keep at most `keep_n` valid paths, by rewriting the validity mask.

    A fallback, not the primary control: the ray budget is what keeps the panel
    legible. Returns the number of paths left live.
    """
    import drjit as dr
    import mitsuba as mi

    full = paths.valid.numpy()
    shape = full.shape
    flat = full.reshape(-1, shape[-1]).copy()
    live = np.where(flat.any(axis=0))[0]
    if keep_n is None or len(live) <= keep_n:
        return int(len(live))

    drop = rng.choice(live, size=len(live) - int(keep_n), replace=False)
    flat[:, drop] = False
    paths._valid = mi.TensorXb(dr.cuda.ad.Bool(flat.reshape(-1).tolist()), shape)
    return int(keep_n)


def segment_report(paths):
    """(n_cylinders, longest segment [m]) that Sionna is about to draw.

    The guard against a silently failed clip: a maximum on the order of the
    slant range instead of the standoff means the render will come out white.
    """
    from sionna.rt.utils.render import paths_to_segments

    out = paths_to_segments(paths)
    if not out:
        return 0, 0.0
    starts, ends, _ = out
    if not starts:
        return 0, 0.0
    lengths = np.linalg.norm(np.asarray(ends) - np.asarray(starts), axis=1)
    return len(starts), float(lengths.max())


# ─── Rendering ────────────────────────────────────────────────────────────────

def make_camera(scene_center, resolution, fov):
    from sionna.rt import Camera
    cx, cy = float(scene_center[0]), float(scene_center[1])
    d = CAMERA_D
    return Camera(position=[cx - CAMERA_OFF * d, cy - CAMERA_OFF * d,
                            CAMERA_H * d],
                  look_at=[cx, cy, 0.0]), fov


def render_case(scene, cam, fov, paths, out_png, spp, resolution,
                show_devices=False):
    """One native Sionna render with the paths drawn in.

    The scene XML has no emitter -- Sionna supplies the lighting -- so this has
    to go through Sionna's renderer rather than a bare Mitsuba one, or it comes
    out black.

    `show_devices` draws a sphere at every radio device. Off by default: at
    whole-scene framing the transmitter marker would sit at the real slant
    range and make a visual claim about where the satellite is. A close-up
    where the transmitter is hundreds of kilometres out of frame can turn it
    on and get a free receiver marker.
    """
    scene.render_to_file(camera=cam, filename=out_png,
                         resolution=list(resolution), num_samples=spp,
                         fov=fov, paths=paths, show_devices=show_devices)
    print(f"[figures] {out_png}")
    return plt.imread(out_png)


# ─── Solving ──────────────────────────────────────────────────────────────────

def solve_case(solver, scene, cfg, samples, max_paths, seed, beamforming, beam_angle):
    """One PathSolver call.

    `beam_angle` must be None whenever `beamforming` is False. path_solver.py
    forwards beam_angle to the field calculator unconditionally, outside the
    `if beamforming` branch, so an isotropic launch carrying the cone angle
    would weight every ray tube with 2*pi*(1-cos psi)/N instead of 4*pi/N --
    a silent ~1e-6 attenuation of diffuse power rather than stock behaviour.
    """
    import drjit as dr

    if not beamforming and beam_angle is not None:
        raise ValueError("beam_angle must be None when beamforming is False")

    return solver(
        scene=scene,
        max_depth=int(cfg["MAX_DEPTH"]),
        los=bool(cfg.get("LOS", True)),
        specular_reflection=bool(cfg.get("SPECULAR_REFLECTION", True)),
        diffuse_reflection=bool(cfg.get("DIFFUSE_REFLECTION", True)),
        refraction=bool(cfg.get("REFRACTION", False)),
        diffraction=bool(cfg.get("DIFFRACTION", True)),
        edge_diffraction=bool(cfg.get("EDGE_DIFFRACTION", False)),
        synthetic_array=False,
        beamforming=beamforming,
        beam_angle=beam_angle,
        samples_per_src=int(samples),
        max_num_paths_per_src=int(max_paths),
        seed=dr.cuda.ad.UInt(int(seed)),
    )


def case_stats(paths, cfg):
    """Path counts per mechanism plus LoS/NLoS powers, via channel_utils."""
    from Ray_Tracing.src.utils.channel_utils import split_by_interaction_type

    verts, inter, valid = extract_geometry(paths)
    types = primary_types(inter)[valid]
    counts = {TYPE_NAMES[t]: int(np.sum(types == t)) for t in TYPE_NAMES}

    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    a = np.asarray(a).ravel()[: inter.shape[1]][valid]
    tau = np.asarray(tau).ravel()[: inter.shape[1]][valid]
    split = split_by_interaction_type(tau, a, inter[:, valid],
                                      float(cfg["FREQUENCY"]), incoherent=True)

    return {
        "n_paths": int(valid.sum()),
        "n_sbr": int(np.sum(types != 0)),
        "counts": counts,
        "has_los": bool(split["has_los"]),
        "P_los": float(split["P_los"]),
        "P_nlos": float(split["P_nlos"]),
    }


def pick_receiver(cfg, solver, tx_pos, beam_angle, rng, rx_file,
                  n_candidates=16, radius_frac=0.35, probe_samples=200_000):
    """Choose one receiver that is both visible to the satellite and well scattered.

    Candidates come from the pipeline's own receiver catalogue rather than a
    freshly built occupancy grid: sieg_notrees has ground relief spanning ~15 m,
    so the flat-ground build_occupancy_grid finds almost no free cells on it.
    rx_positions_sieg.pkl was built with build_occupancy_grid_terrain against
    sieg_ground_only.xml and already carries per-cell heights.

    Candidates are probed with a cheap beamformed solve; the winner has LoS and
    the most non-LoS paths. A receiver without LoS would leave the stock panel
    completely empty, and one without scatterers would make the patched panel
    look no different.
    """
    from Ray_Tracing.src.utils.scene_utils import load_rx_positions, build_scene_batch

    rx_all, _ = load_rx_positions(rx_file)
    rx_all = np.asarray(rx_all, dtype=float)

    cx, cy = float(cfg["SCENE_CENTER"][0]), float(cfg["SCENE_CENTER"][1])
    near = np.linalg.norm(rx_all[:, :2] - [cx, cy], axis=1) \
        <= radius_frac * float(cfg["SCENE_RADIUS"])
    pool = rx_all[near]
    if len(pool) == 0:
        raise SystemExit("no receivers within the central zone of the scene")
    cands = pool[rng.choice(len(pool), size=min(n_candidates, len(pool)),
                            replace=False)]

    scene, _ = build_scene_batch(cands, tx_pos, cfg)
    paths = solve_case(solver, scene, cfg, probe_samples, probe_samples,
                       cfg["RNG_SEED"], True, beam_angle)

    # extract_geometry collapses the RX axis, so score from the full tensors.
    inter = paths.interactions.numpy()
    valid = paths.valid.numpy()
    n_rx = paths.num_rx
    depth, n_paths = inter.shape[0], inter.shape[-1]
    inter = inter.reshape(depth, n_rx, -1, n_paths)[:, :, 0]
    valid = valid.reshape(n_rx, -1, n_paths)[:, 0].astype(bool)

    scored = []
    for i in range(n_rx):
        t = primary_types(inter[:, i])[valid[i]]
        has_los = bool(np.any(t == 0))
        n_scat = int(np.sum(t != 0))
        dist = float(np.linalg.norm(cands[i][:2] - [cx, cy]))
        scored.append((has_los, n_scat, dist, cands[i]))
        print(f"[figures]   candidate {i}: {cands[i].round(1)} "
              f"los={has_los} scattered={n_scat} r={dist:.0f} m")

    with_los = [c for c in scored if c[0]]
    if not with_los:
        print("[figures] WARNING: no candidate has line of sight; "
              "the stock panel will be empty")
        with_los = scored

    # Among the well-scattered candidates, take the most central one. Ranking on
    # scattering alone drifts to the edge of the built-up area, where half the
    # rays leave the frame; the tighter camera makes centrality matter.
    cutoff = 0.6 * max(c[1] for c in with_los)
    pool = [c for c in with_los if c[1] >= cutoff] or with_los
    best = min(pool, key=lambda c: c[2])
    return np.asarray(best[3], dtype=float)

# ─── Figure assembly ──────────────────────────────────────────────────────────

def _stat_lines(stats, samples):
    counts = stats["counts"]
    lines = [f"ray budget      {samples:,} rays",
             f"paths found     {stats['n_paths']:,}",
             f"  shoot-&-bounce {stats['n_sbr']:,}"]
    for t in TYPE_ORDER:
        name = TYPE_NAMES[t]
        if counts.get(name, 0):
            lines.append(f"  {name:<13} {counts[name]:,}")
    return lines


def legend_handles(present, colors):
    labels = {**TYPE_NAMES, 0: "Direct / first leg"}
    return [Line2D([0], [0], color=colors[t], lw=2.4, label=labels[t])
            for t in TYPE_ORDER if t in present]


def draw_panel(ax, image, title, note, stats, resolution):
    """Place a rendered panel and annotate it. No ray geometry is drawn here."""
    ax.imshow(image, extent=[0, resolution[0], resolution[1], 0], zorder=0)
    ax.set_xlim(0, resolution[0])
    ax.set_ylim(resolution[1], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3f4b")
        spine.set_linewidth(1.2)

    ax.set_title(title, color="#e8eaed", fontsize=15, fontweight="semibold",
                 pad=10, loc="left")
    ax.text(0.982, 0.975, "\n".join(stats), transform=ax.transAxes, va="top",
            ha="right", multialignment="left", fontsize=10.5, color="#e8eaed",
            family="DejaVu Sans Mono", linespacing=1.5, zorder=7,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#11141ad0",
                      edgecolor="#3a3f4b", linewidth=0.8))
    ax.text(0.018, 0.025, note, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=10.5, color="#f0f2f5", zorder=7,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#11141ad8",
                      edgecolor="none"))


def build_figure(cases, resolution, outdir, meta, dpi):
    """Composite the two native renders side by side with title and legend."""
    os.makedirs(outdir, exist_ok=True)
    colors = sionna_segment_colors()

    present = set()
    for case in cases:
        for t in TYPE_ORDER:
            if case["stats"]["counts"].get(TYPE_NAMES[t], 0):
                present.add(t)
    present.add(0)   # every path has a first leg

    fig, axes = plt.subplots(1, 2,
                             figsize=(2 * resolution[0] / dpi,
                                      resolution[1] / dpi + 1.6), dpi=dpi)
    fig.patch.set_facecolor("#0c0e13")
    for ax, case in zip(axes, cases):
        draw_panel(ax, case["image"], case["title"], case["note"],
                   case["lines"], resolution)

    handles = legend_handles(present, colors)
    if handles:
        legend = fig.legend(handles=handles, loc="lower center",
                            ncol=len(handles), frameon=False,
                            labelcolor="#c9cdd4", fontsize=11,
                            title="each segment is coloured by the interaction it leaves",
                            title_fontsize=10, bbox_to_anchor=(0.5, 0.03))
        legend.get_title().set_color("#8b919c")

    fig.suptitle("Ray beamforming for non-terrestrial links in Sionna RT",
                 color="#f2f4f7", fontsize=18, fontweight="semibold", y=0.985)
    fig.text(0.5, 0.008, meta, ha="center", va="bottom", color="#8b919c",
             fontsize=10.5)
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    path = os.path.join(outdir, "ntn_rays_comparison.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[figures] {path}")
    return path



# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--scene_xml", default=DEFAULT_SCENE_XML,
                    help="overrides the config; its absolute SCENE_XML is stale")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--elevation", type=float, default=40.0)
    ap.add_argument("--samples", type=int, default=20_000,
                    help="samples_per_src, identical for both panels. Small on\n                         purpose: Sionna draws every segment as an emissive\n                         cylinder, so a few hundred paths read as distinct rays\n                         where a few thousand bloom into a white wash.")
    ap.add_argument("--max_paths", type=int, default=1_000_000)
    ap.add_argument("--first_leg_dim", type=float, default=0.55,
                    help="scale on Sionna's first-leg colour; 1.0 leaves it alone")
    ap.add_argument("--standoff_m", type=float, default=220.0,
                    help="proxy source distance from scene centre; see clip_source")
    ap.add_argument("--max_drawn", type=int, default=None,
                    help="fallback cap on drawn paths; off by default")
    ap.add_argument("--rx", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--rx_file", default=DEFAULT_RX_FILE,
                    help="receiver catalogue to draw candidates from")
    ap.add_argument("--spp", type=int, default=256, help="render samples/pixel")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(args.scene_xml):
        raise SystemExit(f"scene not found: {args.scene_xml}")

    with open(args.config) as fh:
        cfg = json.load(fh)
    cfg["SCENE_XML"] = args.scene_xml
    seed = args.seed if args.seed is not None else int(cfg["RNG_SEED"])
    cfg["RNG_SEED"] = seed
    rng = np.random.default_rng(seed)

    from Ray_Tracing.src.utils.channel_utils import tx_position_from_elevation
    from Ray_Tracing.src.utils.scene_utils import build_scene_batch
    from sionna.rt import PathSolver

    slant = float(cfg["SLANT_DIST_M"])
    tx_pos, beam_angle = tx_position_from_elevation(
        args.elevation, slant, cfg["SCENE_CENTER"], float(cfg["SCENE_RADIUS"]),
        azimuth_deg=float(cfg["TX_AZIMUTH_DEG"]))
    print(f"[figures] elev={args.elevation:.0f} deg  slant={slant/1e3:.0f} km  "
          f"tx={np.round(np.array(tx_pos)/1e3, 2)} km  "
          f"beam half-angle={np.degrees(beam_angle):.4f} deg")

    tune_path_colors(args.first_leg_dim)
    solver = PathSolver()

    if args.rx is not None:
        rx_pos = np.asarray(args.rx, dtype=float)
    else:
        print("[figures] probing receiver candidates ...")
        rx_pos = pick_receiver(cfg, solver, tx_pos, beam_angle, rng, args.rx_file)
    print(f"[figures] receiver at {rx_pos.round(2)}  "
          f"(reproduce with --rx {rx_pos[0]:.2f} {rx_pos[1]:.2f} {rx_pos[2]:.2f})")

    scene, _ = build_scene_batch(np.asarray([rx_pos]), tx_pos, cfg)
    cam, fov = make_camera(cfg["SCENE_CENTER"], RESOLUTION, CAMERA_FOV)
    os.makedirs(args.outdir, exist_ok=True)

    panels = [
        ("stock", False, None, "ntn_rays_stock.png",
         "Stock Sionna RT — isotropic launch over 4π"),
        ("beamformed", True, beam_angle, "ntn_rays_beamformed.png",
         "Patched — launch confined to the beam cone"),
    ]

    cases, stats_all = [], {}
    for label, beamforming, angle, fname, title in panels:
        mode = "cone" if beamforming else "isotropic 4pi"
        print(f"[figures] solving {label} ({mode} launch), {args.samples:,} rays ...")
        paths = solve_case(solver, scene, cfg, args.samples, args.max_paths,
                           seed, beamforming, angle)
        stats = case_stats(paths, cfg)
        stats_all[label] = stats

        # Everything below mutates `paths` for drawing only; the statistics are
        # already taken from the untouched solve above.
        proxy = clip_source(paths, tx_pos,
                            [cfg["SCENE_CENTER"][0], cfg["SCENE_CENTER"][1], 0.0],
                            args.standoff_m)
        n_live = thin_paths(paths, args.max_drawn, np.random.default_rng(seed))
        n_cyl, longest = segment_report(paths)
        print(f"[figures]   {stats['n_paths']} paths "
              f"({stats['n_sbr']} shoot-&-bounce), drawing {n_live} as "
              f"{n_cyl} cylinders, longest segment {longest:.1f} m")
        if longest > 10.0 * args.standoff_m:
            raise SystemExit(
                f"source clip did not take: longest segment is {longest:.0f} m "
                f"against a {args.standoff_m:.0f} m standoff. The render would "
                "come out white.")

        image = render_case(scene, cam, fov, paths,
                            os.path.join(args.outdir, fname), args.spp, RESOLUTION)
        print(f"[figures]   mean luminance {float(np.asarray(image)[..., :3].mean()):.3f}")

        if beamforming:
            note = (f"Same ray budget, aimed into a "
                    f"{np.degrees(beam_angle)*1000:.1f} mdeg cone about boresight,\n"
                    "with the ray-tube solid angle rescaled to match.")
        else:
            note = (f"{stats['n_sbr']} of {args.samples:,} shoot-and-bounce rays "
                    f"reach the scene.\nThe "
                    f"{'single path' if stats['n_paths'] == 1 else 'paths'} shown "
                    "here came from the deterministic\nline-of-sight test, which "
                    "the patch does not touch.")

        cases.append({"label": label, "image": image, "title": title,
                      "note": note, "stats": stats,
                      "lines": _stat_lines(stats, args.samples)})

    meta = (f"UW campus scene · {float(cfg['FREQUENCY'])/1e9:.0f} GHz · "
            f"{args.elevation:.0f}° elevation · {slant/1e3:.0f} km slant range · "
            f"max depth {cfg['MAX_DEPTH']} · {args.samples:,} rays per panel · "
            f"source drawn at a {args.standoff_m:.0f} m proxy, so rays are shown "
            f"diverging where they in fact arrive parallel to 0.02°")

    build_figure(cases, RESOLUTION, args.outdir, meta, args.dpi)

    stats_path = os.path.join(args.outdir, "beamforming_demo_stats.json")
    with open(stats_path, "w") as fh:
        json.dump({
            "elevation_deg": args.elevation,
            "slant_range_m": slant,
            "frequency_hz": float(cfg["FREQUENCY"]),
            "beam_half_angle_rad": float(beam_angle),
            "beam_half_angle_deg": float(np.degrees(beam_angle)),
            "scene_xml": args.scene_xml,
            "scene_radius_m": float(cfg["SCENE_RADIUS"]),
            "tx_position_m": [float(v) for v in tx_pos],
            "rx_position_m": [float(v) for v in rx_pos],
            "render_proxy_source_m": [float(v) for v in proxy],
            "render_standoff_m": args.standoff_m,
            "samples_per_src": args.samples,
            "max_depth": int(cfg["MAX_DEPTH"]),
            "seed": seed,
            "stock": stats_all["stock"],
            "beamformed": stats_all["beamformed"],
        }, fh, indent=2)
    print(f"[figures] {stats_path}")


if __name__ == "__main__":
    main()
