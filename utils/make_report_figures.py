"""Scene views, example PDP, PPP ensemble and UE-spread figures for the report.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_FIG_DIR = os.environ.get("REPORT_FIG_DIR", "./report_figures")
SIEG_XML = os.environ.get("SIEG_XML", "")
SCENE_CENTER = (13.2, -1.3)
SIEG_RESULTS = "./results_sieg_2GHz"
PDP_ELEV = 30

# Table tab:interaction, same names/colours as plot_utils.plot_cir.
TYPE_NAMES = {0: "LoS", 1: "Specular", 2: "Diffuse", 4: "Refraction", 8: "Diffraction"}
TYPE_COLORS = {0: "steelblue", 1: "darkorange", 2: "seagreen",
               4: "purple", 8: "crimson"}


def _out(name):
    os.makedirs(REPORT_FIG_DIR, exist_ok=True)
    return os.path.join(REPORT_FIG_DIR, name)


# ─── 1. UW campus scene ───────────────────────────────────────────────────────

def fig_sieg_scene(spp=256, resolution=(1600, 1000)):
    """Sionna renders of sieg_notrees. Sionna supplies the lighting; the XML has
    no emitter, so a bare Mitsuba render of it comes out black."""
    from sionna.rt import load_scene, Camera

    if not SIEG_XML:
        raise RuntimeError("SIEG_XML env var must point to the scene XML.")
    scene = load_scene(SIEG_XML)
    cx, cy = SCENE_CENTER
    d = 340.0
    # fov/height chosen so the +-170 m scene fills the frame without clipping:
    # vertical half-extent seen is H*tan(fov/2)/aspect.
    views = {
        "sieg_scene_preview.png": ([cx - 0.85 * d, cy - 0.85 * d, 0.5 * d], [cx, cy, 0.0], 35.0),
        "sieg_scene_top.png":     ([cx, cy - 14.0, 700.0],                  [cx, cy, 0.0], 45.0),
    }
    written = []
    for name, (pos, look, fov) in views.items():
        cam = Camera(position=pos, look_at=look)
        path = _out(name)
        scene.render_to_file(camera=cam, filename=path, resolution=list(resolution),
                             num_samples=spp, fov=fov, show_devices=False)
        print(f"[figures] {path}")
        written.append(path)
    return written


# ─── 2. Example PDP: one UE's CIR beside the ensemble ─────────────────────────

def fig_example_pdp(elev=PDP_ELEV, results_dir=SIEG_RESULTS, xmax_ns=300.0):
    import Ray_Tracing.src.utils.cluster_stats as cs
    from Ray_Tracing.src.utils.checkpoint_utils import load_results

    res = load_results(results_dir, elev)
    raw = [r for r in res["raw_results"] if r["has_los"]]
    if not raw:
        raise RuntimeError(f"no LoS raw_results at {elev} deg in {results_dir}")

    # Median-K UE, so the panel is representative rather than the best case.
    k_db = np.array([10 * np.log10(r["P_los"] / max(r["P_scat"], 1e-300)) for r in raw])
    idx = int(np.argsort(k_db)[len(k_db) // 2])
    rec, k_rec = raw[idx], float(k_db[idx])

    t, p, prim, _, _ = cs._normalize(rec)
    p_db = 10 * np.log10(np.maximum(p / p.max(), 1e-300))
    floor = -60.0

    ctr, pdp, n_ue = cs.ensemble_pdp(res, state="LOS")
    pdp_db = np.where(pdp > 0, 10 * np.log10(np.maximum(pdp, 1e-300)), np.nan)
    ds_ens = cs.pdp_delay_spread_ns(ctr, pdp)
    clusters, meta = cs.cluster_rx(rec)

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.6), sharex=True, sharey=True)

    a = ax[0]
    for c in clusters:
        a.axvline(c["mean_delay_ns"], color="crimson", linestyle=":", linewidth=0.9,
                  zorder=1)
    a.plot([], [], color="crimson", linestyle=":", linewidth=0.9,
           label=f"cluster centres ({meta['n_clusters']})")
    for itype in np.unique(prim):
        m = prim == itype
        lw = 2.0 if itype == 0 else 0.7
        a.vlines(t[m], floor, p_db[m], color=TYPE_COLORS.get(int(itype), "grey"),
                 linewidth=lw, alpha=0.85, zorder=3 if itype == 0 else 2,
                 label=f"{TYPE_NAMES.get(int(itype), itype)} ({int(m.sum())})")
    a.plot(t[prim == 0], p_db[prim == 0], "o", ms=5, color=TYPE_COLORS[0], zorder=4)
    a.set_title(f"(a) One UE, {len(t):,} paths", fontsize=11)
    a.set_ylabel("Path power [dB rel. strongest]")
    a.legend(fontsize=8, loc="upper right")
    a.text(0.98, 0.55,
           f"$K$ = {k_rec:.1f} dB\n"
           f"$\\tau_{{\\mathrm{{RMS}}}}$ = {rec['tau_rms_s']*1e9:.1f} ns",
           transform=a.transAxes, ha="right", va="top", fontsize=9,
           bbox=dict(fc="white", ec="0.7", alpha=0.9))

    a = ax[1]
    a.plot(ctr, pdp_db, color="0.25", linewidth=1.0)
    a.fill_between(ctr, floor, pdp_db, color="0.25", alpha=0.18)
    a.set_title(f"(b) Ensemble PDP, {n_ue} LoS UEs", fontsize=11)
    a.text(0.98, 0.55,
           f"$\\tau_{{\\mathrm{{RMS}}}}$ = {ds_ens:.1f} ns\n"
           f"1 ns bins, per-UE unit power",
           transform=a.transAxes, ha="right", va="top", fontsize=9,
           bbox=dict(fc="white", ec="0.7", alpha=0.9))

    for a in ax:
        a.set_xlim(-8, xmax_ns)
        a.set_ylim(floor, 3)
        a.set_xlabel("Excess delay [ns]")
        a.grid(alpha=0.3)
    fig.suptitle(f"UW campus, {res['frequency']/1e9:.0f} GHz, {elev}$^\\circ$ elevation, LoS state",
                 fontsize=12)
    fig.tight_layout()
    path = _out(f"example_pdp_2GHz_{elev}deg.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {path}  (ensemble DS {ds_ens:.2f} ns over {n_ue} UEs)")
    return path


# ─── 3. PPP ensemble scenes ───────────────────────────────────────────────────

def _load_spec(seed_dir):
    from Ray_Tracing.src.utils.scene_gen_ppp import load_scene_spec
    return load_scene_spec(os.path.join(seed_dir, "ppp_scene_spec.npz"),
                           os.path.join(seed_dir, "ppp_scene_meta.json"))


def fig_ppp_scenes(seeds=(0, 1, 2), scenes_root="./ppp_scenes"):
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    from Ray_Tracing.src.utils.scene_gen_ppp import plot_scene

    specs = [_load_spec(os.path.join(scenes_root, f"scene_seed{s:04d}")) for s in seeds]

    # Anatomy of one realization: footprints, cell counts, height PDF.
    fig = plot_scene(specs[0])
    path_one = _out("ppp_scene_anatomy.png")
    fig.savefig(path_one, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {path_one}")

    # Ensemble variety: footprints only, one panel per seed.
    fig, axes = plt.subplots(1, len(specs), figsize=(4.4 * len(specs), 4.6))
    hmax = max(s.heights.max() for s in specs)
    for a, s, seed in zip(np.atleast_1d(axes), specs, seeds):
        xmin, xmax, ymin, ymax = s.bbox()
        pc = PatchCollection([Rectangle((c[0] - w / 2, c[1] - w / 2), w, w)
                              for c, w in zip(s.centers, s.sides)],
                             cmap="viridis", edgecolor="k", linewidths=0.2)
        pc.set_array(s.heights)
        pc.set_clim(0, hmax)
        a.add_collection(pc)
        a.set_xlim(xmin, xmax); a.set_ylim(ymin, ymax); a.set_aspect("equal")
        a.set_xlabel("x [m]")
        a.set_title(f"seed {seed}: {s.n_buildings} buildings, "
                    f"$\\alpha$={s.meta['itu_alpha']:.2f}, "
                    f"$\\beta$={s.meta['itu_beta']:.0f}/km$^2$", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("y [m]")
    fig.colorbar(pc, ax=np.atleast_1d(axes).tolist(), label="height [m]", fraction=0.03)
    path_row = _out("ppp_example_scenes.png")
    fig.savefig(path_row, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {path_row}")

    # Mitsuba render -- these XMLs carry a constant emitter, unlike the UW scene.
    from Ray_Tracing.src.utils.scene_gen_ppp import render_preview
    path_render = _out("ppp_scene_render.png")
    render_preview(os.path.join(scenes_root, f"scene_seed{seeds[0]:04d}", "ppp_scene.xml"),
                   out_png=path_render, spp=128, width=1200, height=800)
    return [path_one, path_row, path_render]


# ─── 4. UE spread ─────────────────────────────────────────────────────────────

def fig_ue_spread(src=os.path.join(SIEG_RESULTS, "rx_positions_overview.png")):
    path = _out("ue_spread_sieg.png")
    shutil.copyfile(src, path)
    print(f"[figures] {path}  (copied from {src})")
    return path


# ─── 5. LaTeX snippets ────────────────────────────────────────────────────────

SNIPPETS = r"""% Generated by make_report_figures.py -- paste over the \PlaceholderFig stubs.

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.49\linewidth]{figures/sieg_scene_top.png}
  \hfill
  \includegraphics[width=0.49\linewidth]{figures/sieg_scene_preview.png}
  \caption{The \code{sieg\_notrees} scene --- the University of Washington
  campus centred on Sieg Hall --- rendered by Sionna~RT from the same geometry
  the path solver runs against, plan view (left) and oblique (right).
  Vegetation has been removed; grey roofs are \code{itu\_metal}, facades
  \code{itu\_concrete}, terrain \code{itu\_medium\_dry\_ground} and the
  fountain \code{itu\_wet\_ground} (Table~\ref{tab:sieg-materials}). The
  footprints are the same ones that appear as excluded cells in
  Figure~\ref{fig:uw-rx}.}
  \label{fig:sieg-scene}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=\linewidth]{figures/example_pdp_2GHz_30deg.png}
  \caption{Per-UE CIR versus ensemble PDP, UW campus at $2$\,GHz, $30^\circ$
  elevation, LoS state. (a) One representative UE (median $K$ of the LoS
  population): every ray-traced path as a stem, coloured by first interaction
  per Table~\ref{tab:interaction}. (b) The ensemble PDP of
  Eq.~\eqref{eq:ensemble-pdp} over the LoS UEs at the same elevation, in
  $1$\,ns bins with each UE normalised to unit power first. Dotted lines mark
  the cluster centres extracted from the panel-(a) UE by
  Eq.~\eqref{eq:peakset}. Both panels are truncated at $300$\,ns; the profile
  extends to ${\approx}1930$\,ns.}
  \label{fig:example-pdp}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=\linewidth]{figures/ppp_example_scenes.png}\\[4pt]
  \includegraphics[width=\linewidth]{figures/ppp_scene_anatomy.png}
  \caption{Members of the PPP scene ensemble. Top: building footprints coloured
  by height for three seeds, with the realized ITU-R P.1410 parameters
  $(\alpha,\beta)$ of Eq.~\eqref{eq:ppp_alpha} annotated --- $157$ to $190$
  buildings and $\alpha = 0.25$ to $0.31$ against the $0.30$ target, the
  shortfall being the cell saturation of Eq.~\eqref{eq:ppp_build}. Bottom:
  anatomy of one realization (seed $0$) --- footprints with the underlying PPP
  points, the per-cell point counts $k_{ij}$, and the sampled building heights
  against the Rayleigh density of Eq.~\eqref{eq:rayleigh_h}.}
  \label{fig:ppp-scenes}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.8\linewidth]{figures/ppp_scene_render.png}
  \caption{Mitsuba render of the emitted scene for seed $0$ of the ensemble of
  Figure~\ref{fig:ppp-scenes}: $157$ blocks on a $500\times500$\,m ground
  plane, walls, roofs and ground written as separate meshes so each can carry
  its own ITU material (Table~\ref{tab:ppp-materials}).}
  \label{fig:ppp-render}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.85\linewidth]{figures/ue_spread_sieg.png}
  \caption{Sampled UE positions over the UW campus scene, drawn from the
  terrain-following occupancy grid of Section~\ref{sec:site-ue}: $10{,}000$ UEs
  over $59{,}539$ free cells, with building cells and the $15$\,m
  \code{GRID\_MARGIN} exclusion ring shown.}
  \label{fig:uw-rx}
\end{figure}
"""


def write_snippets():
    path = _out("snippets.tex")
    with open(path, "w") as f:
        f.write(SNIPPETS)
    print(f"[figures] {path}")
    return path


if __name__ == "__main__":
    fig_ue_spread()
    fig_example_pdp()
    fig_ppp_scenes()
    fig_sieg_scene()
    write_snippets()
