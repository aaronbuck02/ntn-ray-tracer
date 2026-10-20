"""Before/after renders for the ray-beamforming patch's sibling: `epsilon_guard`.

Stock ``sionna-rt 1.2.1`` shortens every occlusion ray by a RELATIVE epsilon,
``maxt *= (1 - EPSILON_FLOAT)`` with ``EPSILON_FLOAT = 100*2**-24 = 5.96e-6``.
As a self-intersection guard in units of float32 precision that is well judged,
and at terrestrial range it is sub-millimetre. Over a LEO link it is not: the
direct-path test is spawned TX -> RX, so the shortening lands at the RECEIVER,
and at a 1816 km slant it leaves a **10.82 m** stretch in front of the receiver
that the ray never searches. A wall standing there is invisible and the blocked
link is reported as line-of-sight. The patch caps the guard at 1 mm.

Both panels come from the real solver. Stock behaviour is reproduced in-process
by raising ``MAX_RAY_GUARD_M`` (see `stock_guard`), not mocked and not by
swapping the installed package.

Only the direct ray is solved for -- every scattering mechanism is switched off
-- so the panels differ by exactly one thing: the ray is there, or it is not.

Run from the parent of the repository so ``Ray_Tracing.src.*`` resolves::

    cd /path/to
    CUDA_VISIBLE_DEVICES=0 python -m Ray_Tracing.src.diagnostics.make_occlusion_figure
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Ray_Tracing.src.diagnostics.make_beamforming_figure import (
    DEFAULT_CONFIG, DEFAULT_SCENE_XML, REPO_ROOT, RESOLUTION,
    clip_source, draw_panel, extract_geometry, primary_types, render_case,
    segment_report, solve_case, tune_path_colors,
)

DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "figures", "occlusion_demo")

# Sionna's relative guard, sionna/rt/utils/ray_tracing.py.
EPSILON_FLOAT = 100 * 2 ** -24

# The blocked receiver and the pass geometry it is blocked under. Found by
# sweeping rx_positions_sieg.pkl for receivers whose nearest blocker along the
# satellite bearing falls inside the stock dead zone -- 87 of 10,000 do at this
# elevation. This one has its blocker 10.30 m out against a 10.82 m dead zone,
# i.e. only just inside, which makes the gap large enough to see.
DEFAULT_RX      = (31.17, 12.26, 9.44)
DEFAULT_ELEV    = 10.0
DEFAULT_ALT_KM  = 550.0
DEFAULT_AZIMUTH = 90.0


@contextlib.contextmanager
def stock_guard(enabled: bool):
    """Run the block with stock Sionna's occlusion guard, then restore.

    The patch reads ``MAX_RAY_GUARD_M`` as a module global at call time inside
    ``spawn_ray_to``, and applies

        maxt - min(maxt * EPSILON_FLOAT, MAX_RAY_GUARD_M)

    so raising the cap out of reach collapses it to ``maxt - maxt*EPSILON_FLOAT``
    -- stock's relative guard. That makes the "before" panel a real solve
    through the real code path rather than a mock, without touching the
    installed package or tripping sionna_patch_check (which asserts the guard
    EXPRESSION is present, not its constant).
    """
    import sionna.rt.utils.ray_tracing as rtu

    previous = rtu.MAX_RAY_GUARD_M
    if enabled:
        rtu.MAX_RAY_GUARD_M = 1e30
    try:
        yield
    finally:
        rtu.MAX_RAY_GUARD_M = previous


def blocker_distance(scene_xml, tx_pos, rx_pos):
    """How far in front of the receiver the first occluder stands [m].

    Measured from the receiver back along the TX -> RX ray, the same direction
    and orientation Sionna spawns the direct-path test in, so it is directly
    comparable with the dead zone.
    """
    import mitsuba as mi

    previous = None
    with contextlib.suppress(Exception):
        previous = mi.variant()
    mi.set_variant("llvm_ad_rgb")
    try:
        scene = mi.load_file(scene_xml)
        tx = np.asarray(tx_pos, dtype=float)
        rx = np.asarray(rx_pos, dtype=float)
        delta = rx - tx
        length = float(np.linalg.norm(delta))
        direction = delta / length
        ray = mi.Ray3f(mi.Point3f(*[float(v) for v in tx]),
                       mi.Vector3f(*[float(v) for v in direction]))
        ray.maxt = mi.Float(float(length))
        si = scene.ray_intersect(ray)
        if not bool(np.asarray(si.is_valid()).ravel()[0]):
            return None, None
        t = float(np.asarray(si.t).ravel()[0])
        return length - t, tx + direction * t
    finally:
        if previous is not None:
            mi.set_variant(previous)


def make_closeup_camera(rx_pos, blocker_pos, fov, frame_m, side=1.0,
                        pull=3.6, lift=1.0):
    """Frame the receiver and the blocker that shadows it.

    Views across the incoming bearing rather than along it, so the ray's slope
    and the gap between wall and receiver both read.

    `frame_m` is the approximate width of the view at the subject, which is the
    knob worth tuning: it sets how much surrounding scene comes in alongside
    the ~10 m gap the figure is about. `pull` and `lift` only set the direction
    the camera looks from, not how far away it is.
    """
    from sionna.rt import Camera

    rx = np.asarray(rx_pos, dtype=float)
    blocker = np.asarray(blocker_pos, dtype=float)
    look = 0.5 * (rx + blocker)

    along = blocker - rx
    along /= np.linalg.norm(along)
    across = np.cross(along, [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across)

    offset = side * across * float(pull) + np.array([0.0, 0.0, float(lift)])
    offset /= np.linalg.norm(offset)
    distance = 0.5 * float(frame_m) / np.tan(np.radians(float(fov)) / 2.0)

    position = look + offset * distance
    return Camera(position=[float(v) for v in position],
                  look_at=[float(v) for v in look]), fov


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--scene_xml", default=DEFAULT_SCENE_XML)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--rx", type=float, nargs=3, default=list(DEFAULT_RX),
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--elevation", type=float, default=DEFAULT_ELEV)
    ap.add_argument("--altitude_km", type=float, default=DEFAULT_ALT_KM)
    ap.add_argument("--azimuth", type=float, default=DEFAULT_AZIMUTH)
    ap.add_argument("--samples", type=int, default=20_000)
    ap.add_argument("--standoff_m", type=float, default=90.0,
                    help="proxy source distance, anchored on the RECEIVER so the "
                         "direct ray is drawn along the true bearing")
    ap.add_argument("--rx_radius", type=float, default=1.1,
                    help="drawn radius of the receiver marker [m]")
    ap.add_argument("--frame_m", type=float, default=90.0,
                    help="approximate width of the view at the subject [m]")
    ap.add_argument("--fov", type=float, default=36.0)
    ap.add_argument("--side", type=float, default=1.0,
                    help="which side of the ray to view from: +1 or -1")
    ap.add_argument("--pull", type=float, default=3.6)
    ap.add_argument("--lift", type=float, default=1.8)
    ap.add_argument("--first_leg_dim", type=float, default=1.0,
                    help="no dimming by default: there is only one ray to draw")
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    if not os.path.exists(args.scene_xml):
        raise SystemExit(f"scene not found: {args.scene_xml}")

    with open(args.config) as fh:
        cfg = json.load(fh)
    cfg["SCENE_XML"] = args.scene_xml
    # Direct ray only: every scattering mechanism off, so the two panels differ
    # by exactly one thing -- whether the occlusion test found the wall.
    cfg.update(LOS=True, SPECULAR_REFLECTION=False, DIFFUSE_REFLECTION=False,
               DIFFRACTION=False, EDGE_DIFFRACTION=False, REFRACTION=False)
    seed = int(cfg["RNG_SEED"])

    from Ray_Tracing.src.utils.channel_utils import tx_position_from_elevation
    from Ray_Tracing.src.utils.orbital_utils import slant_range_m
    from Ray_Tracing.src.utils.scene_utils import build_scene_batch
    from sionna.rt import PathSolver

    rx_pos = np.asarray(args.rx, dtype=float)
    slant = float(slant_range_m(args.altitude_km, args.elevation))
    tx_pos, beam_angle = tx_position_from_elevation(
        args.elevation, slant, cfg["SCENE_CENTER"], float(cfg["SCENE_RADIUS"]),
        azimuth_deg=args.azimuth)

    stock_dz = slant * EPSILON_FLOAT
    import sionna.rt.utils.ray_tracing as rtu
    patched_dz = min(stock_dz, float(rtu.MAX_RAY_GUARD_M))
    gap, blocker_pos = blocker_distance(args.scene_xml, tx_pos, rx_pos)

    print(f"[figures] elev={args.elevation:.0f} deg  alt={args.altitude_km:.0f} km  "
          f"slant={slant/1e3:.1f} km")
    print(f"[figures] dead zone: stock {stock_dz:.2f} m -> patched "
          f"{patched_dz*1e3:.2f} mm")
    if gap is None:
        raise SystemExit("this receiver is not blocked at all; nothing to show")
    print(f"[figures] receiver {rx_pos.round(2)}  nearest blocker {gap:.2f} m away "
          f"at {np.asarray(blocker_pos).round(2)}")
    if gap >= stock_dz:
        raise SystemExit(
            f"blocker at {gap:.2f} m is outside the {stock_dz:.2f} m dead zone, so "
            "stock would report it correctly and there is no bug to show")

    scene, _ = build_scene_batch(np.asarray([rx_pos]), tx_pos, cfg)
    # Sionna sizes device markers off the whole scene (0.005 * scene_scale,
    # here 2.3 m). In a 35 m close-up that is a balloon wider than the gap
    # under discussion, so set it explicitly.
    receiver = scene.get("rx_0")
    if receiver is not None:
        receiver.display_radius = args.rx_radius
    cam, fov = make_closeup_camera(rx_pos, blocker_pos, args.fov, args.frame_m,
                                   side=args.side, pull=args.pull, lift=args.lift)
    solver = PathSolver()
    tune_path_colors(args.first_leg_dim)
    os.makedirs(args.outdir, exist_ok=True)

    cases, record = [], {}
    for label, emulate_stock, fname, title in (
        ("stock", True, "ntn_occlusion_stock.png",
         "Stock Sionna RT — occlusion guard scales with range"),
        ("patched", False, "ntn_occlusion_patched.png",
         "Patched — guard capped at 1 mm"),
    ):
        guard_m = stock_dz if emulate_stock else patched_dz
        with stock_guard(emulate_stock):
            paths = solve_case(solver, scene, cfg, args.samples, 200_000,
                               seed, True, beam_angle)
            verts, inter, valid = extract_geometry(paths)
            n_los = int(np.sum(primary_types(inter)[valid] == 0))
            clip_source(paths, tx_pos, rx_pos, args.standoff_m)
            n_cyl, longest = segment_report(paths)
            print(f"[figures] {label}: guard {guard_m:.3f} m -> {n_los} direct "
                  f"path(s), {n_cyl} cylinder(s), longest {longest:.1f} m")
            if longest > 10.0 * args.standoff_m:
                raise SystemExit("source clip did not take; the render would be white")
            # With no valid paths Sionna is handed an empty overlay; skip it.
            # show_devices marks the receiver -- the point the ray does or
            # does not reach. The transmitter marker is 1816 km away and never
            # enters the frame.
            image = render_case(scene, cam, fov, paths if n_cyl else None,
                                os.path.join(args.outdir, fname), args.spp,
                                RESOLUTION, show_devices=True)

        verdict = "CLEAR — line of sight" if n_los else "BLOCKED — no line of sight"
        correct = "wrong" if n_los else "correct"
        note = (f"Verdict: {verdict}  ({correct})\n"
                + (f"The ray passes through the building and still reports clear:\n"
                   f"the wall is {gap:.1f} m from the receiver, inside the {guard_m:.1f} m\n"
                   f"at the end of the ray that is never searched."
                   if n_los else
                   f"The guard is {guard_m*1e3:.1f} mm, so the occlusion ray\n"
                   f"reaches the wall {gap:.1f} m ahead of the receiver."))
        lines = [f"slant range     {slant/1e3:,.0f} km",
                 f"dead zone       {guard_m:.3f} m"
                 if guard_m >= 0.01 else
                 f"dead zone       {guard_m*1e3:.1f} mm",
                 f"blocker at      {gap:.2f} m",
                 f"direct paths    {n_los}"]
        cases.append({"label": label, "image": image, "title": title,
                      "note": note, "lines": lines})
        record[label] = {"guard_m": guard_m, "n_direct_paths": n_los,
                         "reports_los": bool(n_los)}

    meta = (f"UW campus scene · {float(cfg['FREQUENCY'])/1e9:.0f} GHz · "
            f"{args.elevation:.0f}° elevation · {slant/1e3:,.0f} km slant range · "
            f"scattering disabled, direct ray only · "
            f"source drawn at a {args.standoff_m:.0f} m proxy on the receiver's bearing")

    dpi = args.dpi
    fig, axes = plt.subplots(1, 2, figsize=(2 * RESOLUTION[0] / dpi,
                                            RESOLUTION[1] / dpi + 1.2), dpi=dpi)
    fig.patch.set_facecolor("#0c0e13")
    for ax, case in zip(axes, cases):
        draw_panel(ax, case["image"], case["title"], case["note"],
                   case["lines"], RESOLUTION)
    fig.suptitle("Occlusion-ray dead zone over a non-terrestrial link",
                 color="#f2f4f7", fontsize=18, fontweight="semibold", y=0.985)
    fig.text(0.5, 0.008, meta, ha="center", va="bottom", color="#8b919c",
             fontsize=10.5)
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    out = os.path.join(args.outdir, "ntn_occlusion_comparison.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[figures] {out}")

    stats_path = os.path.join(args.outdir, "occlusion_demo_stats.json")
    with open(stats_path, "w") as fh:
        json.dump({
            "elevation_deg": args.elevation,
            "altitude_km": args.altitude_km,
            "slant_range_m": slant,
            "epsilon_float": EPSILON_FLOAT,
            "stock_dead_zone_m": stock_dz,
            "patched_dead_zone_m": patched_dz,
            "rx_position_m": [float(v) for v in rx_pos],
            "tx_position_m": [float(v) for v in tx_pos],
            "blocker_position_m": [float(v) for v in np.asarray(blocker_pos)],
            "blocker_distance_from_rx_m": float(gap),
            "samples_per_src": args.samples,
            "seed": seed,
            **record,
        }, fh, indent=2)
    print(f"[figures] {stats_path}")


if __name__ == "__main__":
    main()
