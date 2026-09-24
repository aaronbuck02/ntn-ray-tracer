"""Plotting helpers for the results notebooks.

K-factor, delay and angular spreads, CDFs, coverage maps and elevation trends.
Notebook-only: nothing in the pipeline imports this. See docs/analysis.md.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ─── RX Position Overview ─────────────────────────────────────────────────────

def plot_rx_positions(nominal_rx_positions, tx_pos=None,
                      free_xy=None, occupied_xy=None, grid_meta=None,
                      title="Nominal RX Positions (top view)",
                      figsize=(10, 8)):
    """Top-view scatter of nominal receiver positions overlaid on the complete
scene grid, including buffer margin cells.
    """
    rx = np.asarray(nominal_rx_positions)
    fig, ax = plt.subplots(figsize=figsize)

    # ── Reconstruct full grid background from grid_meta ───────────────────────
    if grid_meta is not None:
        xmin     = grid_meta["xmin"]
        xmax     = grid_meta["xmax"]
        ymin     = grid_meta["ymin"]
        ymax     = grid_meta["ymax"]
        grid_res = grid_meta.get("grid_res", 2.0)
        margin   = grid_meta.get("margin", None)

        xs = np.arange(xmin, xmax + grid_res, grid_res)
        ys = np.arange(ymin, ymax + grid_res, grid_res)
        XX, YY = np.meshgrid(xs, ys)
        all_x  = XX.flatten()
        all_y  = YY.flatten()

        # Determine which cells fall in the margin ring vs the valid inner zone
        has_valid_zone = all(k in grid_meta for k in (
            "valid_xmin", "valid_xmax", "valid_ymin", "valid_ymax"
        ))
        if has_valid_zone:
            vx0 = grid_meta["valid_xmin"]; vx1 = grid_meta["valid_xmax"]
            vy0 = grid_meta["valid_ymin"]; vy1 = grid_meta["valid_ymax"]
            in_valid = (
                (all_x >= vx0) & (all_x <= vx1) &
                (all_y >= vy0) & (all_y <= vy1)
            )
            # Margin ring cells (inside scene but outside valid zone)
            ring_x = all_x[~in_valid]
            ring_y = all_y[~in_valid]
            inner_x = all_x[in_valid]
            inner_y = all_y[in_valid]

            ax.scatter(ring_x, ring_y, c='#f5e6c8', s=2, zorder=1,
                       linewidths=0, label=f'Margin ring ({len(ring_x):,} cells)')
            ax.scatter(inner_x, inner_y, c='#dce9f5', s=2, zorder=1,
                       linewidths=0, label=f'Valid zone ({len(inner_x):,} cells)')
        else:
            # No valid-zone info — show all cells with a single colour
            ax.scatter(all_x, all_y, c='#dce9f5', s=2, zorder=1,
                       linewidths=0, label=f'Full grid ({len(all_x):,} cells)')

        # Occupied cells (building footprints) — drawn over the background
        occ = (occupied_xy if occupied_xy is not None
               else grid_meta.get("occupied_xy", None))
        if occ is not None and len(occ):
            ax.scatter(occ[:, 0], occ[:, 1],
                       c='tomato', s=3, zorder=2, linewidths=0, alpha=0.7,
                       label=f'Buildings ({len(occ):,})')

        # Valid-zone boundary rectangle (inner margin boundary)
        if has_valid_zone:
            rect = Rectangle((vx0, vy0), vx1 - vx0, vy1 - vy0,
                              linewidth=1.5, edgecolor='black',
                              facecolor='none', linestyle='--', zorder=6,
                              label=f'Valid RX zone ({margin:.0f} m inset)' if margin else 'Valid RX zone')
            ax.add_patch(rect)

            # Scene outer boundary (solid thin line for reference)
            outer = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               linewidth=0.8, edgecolor='dimgray',
                               facecolor='none', linestyle='-', zorder=6,
                               label='Scene boundary')
            ax.add_patch(outer)

    # ── Free cells ────────────────────────────────────────────────────────────
    if free_xy is not None and len(free_xy):
        ax.scatter(free_xy[:, 0], free_xy[:, 1],
                   c='limegreen', s=4, zorder=3, linewidths=0, alpha=0.8,
                   label=f'Free cells ({len(free_xy):,})')

    # ── Selected RX positions ─────────────────────────────────────────────────
    ax.scatter(rx[:, 0], rx[:, 1],
               c='royalblue', s=25, zorder=4, alpha=0.75,
               label=f'Nominal RX ({len(rx):,})')

    # ── TX ground projection ──────────────────────────────────────────────────
    if tx_pos is not None:
        ax.scatter([tx_pos[0]], [tx_pos[1]],
                   marker='*', c='red', s=280, zorder=7,
                   label=f'TX ground proj. (z={tx_pos[2]/1e3:.1f} km)')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=2, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', 'datalim')
    plt.tight_layout()
    return fig


# ─── Figure 1 : Spatial Heat-Maps ─────────────────────────────────────────────

def plot_spatial_stats(nominal_rx_positions, K_power_ratio, tau_rms_mean_s,
                       elevation_deg, frequency, slant_dist_m,
                       los_probability=None,
                       cfg_label="", figsize=(19, 8)):
    """Two-panel figure:
  (a) K-Factor [dB] spatial heat-map — LoS receivers only (NLOS shown in grey)
  (b) RMS Delay Spread [ns] spatial heat-map — all receivers
    """
    rx  = np.asarray(nominal_rx_positions)
    K   = np.asarray(K_power_ratio)
    tau = np.asarray(tau_rms_mean_s)

    K_dB   = 10 * np.log10(K + 1e-12)
    tau_ns = tau * 1e9

    has_los = (np.asarray(los_probability, dtype=bool)
               if los_probability is not None
               else np.ones(len(rx), dtype=bool))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        f"Rician Channel Statistics — Urban Scenario\n"
        f"f = {frequency/1e9:.1f} GHz  |  Elev = {elevation_deg:.0f}°  |  "
        f"Slant = {slant_dist_m/1e3:.0f} km"
        + (f"  |  {cfg_label}" if cfg_label else ""),
        fontsize=12, fontweight='bold'
    )

    # ── (a) K-Factor — LoS receivers only ─────────────────────────────────────
    ax = axes[0]
    nlos_mask = ~has_los
    los_mask  = has_los

    # Grey background markers for NLOS positions
    if np.any(nlos_mask):
        ax.scatter(rx[nlos_mask, 0], rx[nlos_mask, 1],
                   c='lightgrey', s=20, edgecolors='none', alpha=0.5,
                   zorder=2, label=f'NLOS ({nlos_mask.sum():,})')

    # Coloured K-factor markers for LoS positions
    if np.any(los_mask):
        K_los = K_dB[los_mask]
        sc = ax.scatter(rx[los_mask, 0], rx[los_mask, 1],
                        c=K_los, cmap='plasma',
                        s=40, edgecolors='none', alpha=0.90, zorder=3)
        plt.colorbar(sc, ax=ax, label='K-Factor [dB]  (LoS only)')
        ax.legend(loc='upper right', fontsize=8, markerscale=1.2)
    else:
        ax.text(0.5, 0.5, 'No LoS receivers', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='grey')

    n_los  = int(np.sum(los_mask))
    n_nlos = int(np.sum(nlos_mask))
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title(
        f'(a) Rician K-Factor [dB]  —  LoS only\n'
        f'LoS: {n_los:,} ({n_los/len(rx)*100:.1f}%)  |  '
        f'NLOS: {n_nlos:,} ({n_nlos/len(rx)*100:.1f}%)'
    )
    ax.grid(True, alpha=0.25)
    ax.set_aspect('equal', 'datalim')

    # ── (b) RMS Delay Spread — all receivers ──────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(rx[:, 0], rx[:, 1], c=tau_ns, cmap='viridis',
                    s=40, edgecolors='none', alpha=0.85, zorder=3)
    plt.colorbar(sc, ax=ax, label='τ_rms [ns]')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title('(b) RMS Delay Spread [ns]  (all receivers)')
    ax.grid(True, alpha=0.25)
    ax.set_aspect('equal', 'datalim')

    plt.tight_layout()
    return fig


# ─── Figure 2 : LoS / NLOS Map ────────────────────────────────────────────────

def plot_los_map(nominal_rx_positions, los_probability, elevation_deg,
                 figsize=(9, 7)):
    """
    Binary LoS / NLOS spatial map (green = LoS, red = NLOS).
    """
    rx      = np.asarray(nominal_rx_positions)
    has_los = np.asarray(los_probability, dtype=bool)

    fig, ax = plt.subplots(figsize=figsize)
    colors = np.where(has_los, 1.0, 0.0)
    sc = ax.scatter(rx[:, 0], rx[:, 1], c=colors, cmap='RdYlGn',
                    vmin=0, vmax=1, s=40, edgecolors='none', alpha=0.85, zorder=3)
    cbar = plt.colorbar(sc, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['NLOS', 'LoS'])

    n_los  = int(np.sum(has_los))
    n_nlos = len(has_los) - n_los
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title(
        f'LoS / NLOS Map — Elev = {elevation_deg:.0f}°\n'
        f'LoS: {n_los} ({n_los/len(has_los)*100:.1f}%)  '
        f'NLOS: {n_nlos} ({n_nlos/len(has_los)*100:.1f}%)'
    )
    ax.grid(True, alpha=0.25)
    ax.set_aspect('equal', 'datalim')
    plt.tight_layout()
    return fig


# ─── Figure 3 : CDFs ──────────────────────────────────────────────────────────

def plot_cdfs(K_power_ratio, tau_rms_mean_s, los_probability,
              elevation_deg, figsize=(14, 5)):
    """
    Two-panel CDF figure:
      (a) K-Factor [dB] — LoS receivers ONLY
          NLOS positions are excluded because K = P_LoS/P_NLoS is not a
          meaningful Rician parameter without a dominant component.
      (b) RMS Delay Spread [ns] — LoS and NLOS shown separately for comparison
    """
    K       = np.asarray(K_power_ratio)
    tau_ns  = np.asarray(tau_rms_mean_s) * 1e9
    has_los = np.asarray(los_probability, dtype=bool)
    K_dB    = 10 * np.log10(K + 1e-12)

    n_los  = int(np.sum(has_los))
    n_nlos = int(np.sum(~has_los))
    n_all  = len(K)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        f"Channel Statistics CDFs — Elev = {elevation_deg:.0f}°  |  "
        f"LoS: {n_los:,} ({n_los/n_all*100:.1f}%)  NLOS: {n_nlos:,} ({n_nlos/n_all*100:.1f}%)",
        fontsize=12, fontweight='bold'
    )

    # ── (a) K-Factor CDF — LoS only ───────────────────────────────────────────
    ax = axes[0]
    vals_los = K_dB[has_los & np.isfinite(K_dB)]

    if len(vals_los) > 0:
        sv  = np.sort(vals_los)
        cdf = np.arange(1, len(sv) + 1) / len(sv)
        ax.plot(sv, cdf, color='seagreen', linewidth=2.0,
                label=f'LoS  (n={len(vals_los):,})')

        # Annotate key percentiles
        for pct in [10, 50, 90]:
            pval = np.percentile(sv, pct)
            ax.axvline(pval, color='seagreen', linestyle='--', linewidth=0.9, alpha=0.7)
            ax.text(pval, pct / 100, f' P{pct}={pval:.1f}', fontsize=7,
                    color='seagreen', va='bottom')
    else:
        ax.text(0.5, 0.5, 'No LoS receivers', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='grey')

    ax.set_xlabel('K-Factor [dB]')
    ax.set_ylabel('CDF')
    ax.set_title('(a) Rician K-Factor CDF  —  LoS receivers only')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    # ── (b) Delay Spread CDF — LoS and NLOS separately ────────────────────────
    ax = axes[1]
    for mask, label, color in [
        (np.ones(n_all, dtype=bool), 'All',  'steelblue'),
        (has_los,                    'LoS',  'seagreen'),
        (~has_los,                   'NLOS', 'tomato'),
    ]:
        vals = tau_ns[mask & np.isfinite(tau_ns)]
        if len(vals) == 0:
            continue
        sv  = np.sort(vals)
        cdf = np.arange(1, len(sv) + 1) / len(sv)
        ax.plot(sv, cdf, label=f'{label} (n={len(vals):,})', linewidth=1.8,
                color=color)

    ax.set_xlabel('τ_rms [ns]')
    ax.set_ylabel('CDF')
    ax.set_title('(b) RMS Delay Spread CDF')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    return fig


# ─── Figure 4 : Elevation-Angle Comparison ────────────────────────────────────

def plot_angle_comparison(all_results: dict, figsize=(16, 10)):
    """Four-panel comparison across elevation angles.
K-Factor panels (a) and (d) use LoS receivers ONLY.
    """
    angles  = sorted(all_results.keys())
    labels  = [f"{int(a)}°" for a in angles]
    colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(angles)))

    K_mean                   = []
    tau_mean                 = []
    plos_pct                 = []

    for angle in angles:
        res     = all_results[angle]
        
        P_los_arr = np.array(res["P_los"])
        P_nlos_arr = np.array(res["P_nlos"])
        mean_Plos  = float(np.mean(P_los_arr))
        mean_Pnlos = float(np.mean(P_nlos_arr))
        
        K_mean_dB   = 10.0 * np.log10(mean_Plos / (mean_Pnlos + 1e-30) + 1e-30)
        tau_ns  = np.array(res["tau_rms_mean_s"]) * 1e9
        p_los   = np.array(res["los_probability"])
        has_los = p_los.astype(bool)
        
        
        K_mean.append(K_mean_dB)

        

        # Delay spread — all receivers
        fin_t = np.isfinite(tau_ns)
        tau_mean.append(np.mean(tau_ns[fin_t]))

        plos_pct.append(float(np.mean(p_los) * 100))

    
    K_mean = np.array(K_mean, dtype=float)
    tau_mean = np.array(tau_mean, dtype=float)
    x      = np.arange(len(angles))

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Channel Statistics vs. Elevation Angle — Urban Scenario\n",
        fontsize=13, fontweight='bold'
    )

    # ── (a) K-Factor vs elevation ──────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, K_mean, color=colors[0])
    
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel('Elevation Angle'); ax.set_ylabel('K-Factor [dB]')
    ax.set_title('(a) Mean K-Factor')
    ax.grid(True, alpha=0.3, axis='y'); ax.set_axisbelow(True)

    # ── (b) LoS Probability vs elevation ─────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(x, plos_pct, color=colors[1])
    for xi, pct in zip(x, plos_pct):
        ax.text(xi, pct + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel('Elevation Angle'); ax.set_ylabel('P(LoS) [%]')
    ax.set_ylim(0, 115)
    ax.set_title('(b) LoS Probability')
    ax.grid(True, alpha=0.3, axis='y'); ax.set_axisbelow(True)

    # ── (c) Delay Spread vs elevation — all receivers ─────────────────────────
    ax = axes[1, 0]
    ax.plot(x, tau_mean, color=colors[2])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel('Elevation Angle'); ax.set_ylabel('τ_rms [ns]')
    ax.set_title('(c) Mean RMS Delay Spread')
    ax.grid(True, alpha=0.3, axis='y'); ax.set_axisbelow(True)

    # ── (d) K-Factor CDFs overlaid  ─────────────────────────────────
    ax = axes[1, 1]
    for i, (angle, color) in enumerate(zip(angles, colors)):
        res     = all_results[angle]
        K       = np.array(res["K_power_ratio"])
        p_los   = np.array(res["los_probability"])
        has_los = p_los.astype(bool)
        K_dB    = 10 * np.log10(K + 1e-12)
        los_fin = has_los & np.isfinite(K_dB)
        if not np.any(los_fin):
            continue
        sv  = np.sort(K_dB[los_fin])
        cdf = np.arange(1, len(sv) + 1) / len(sv)
        ax.plot(sv, cdf, label=f"{int(angle)}°  (n={len(sv):,})",
                color=color, linewidth=2)

    ax.set_xlabel('K-Factor [dB]')
    ax.set_ylabel('CDF')
    ax.set_title('(d) K-Factor CDF')
    ax.legend(fontsize=9, title='Elevation')
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    return fig


# ─── Figure 5 : CIR Stem Plot ─────────────────────────────────────────────────

def plot_cir(raw_result: dict, rx_idx: int, K_dB: float, tau_rms_ns: float,
             figsize=(12, 8)):
    """Channel Impulse Response stem plot for a single receiver (scattered paths only).

  Top panel   : normalised path power (linear) coloured by interaction type
  Bottom panel: cumulative scattered power fraction vs delay
    """
    # Interaction-type integer → human-readable label and colour
    TYPE_NAMES  = {0: 'LoS', 1: 'Specular', 2: 'Diffuse', 4: 'Refraction', 8: 'Diffraction'}
    TYPE_COLORS = {0: 'steelblue', 1: 'darkorange', 2: 'seagreen',
                   4: 'purple',    8: 'crimson'}

    taus_all = np.asarray(raw_result['taus']) * 1e9          # → ns
    pwr_all  = np.abs(np.asarray(raw_result['amps'])) ** 2   # linear power
    id0_all  = np.asarray(raw_result['inter_d0'], dtype=int)
    rx_nom   = raw_result['rx_nominal']

    # Remove LoS paths (inter_d0 == 0)
    scat_mask = id0_all != 0
    taus_plot = taus_all[scat_mask]
    pwr_plot  = pwr_all[scat_mask]
    id0_plot  = id0_all[scat_mask]

    # Normalise to the strongest scattered path (linear)
    pwr_norm = pwr_plot / (pwr_plot.max() + 1e-300)

    unique_types = np.unique(id0_plot)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        f"CIR (scattered paths) — RX {rx_idx + 1}\n"
        f"Position: ({rx_nom[0]:.2f}, {rx_nom[1]:.2f}, {rx_nom[2]:.2f}) m  |  "
        f"K = {K_dB:.1f} dB  |  "
        f"τ_rms = {tau_rms_ns:.3f} ns  |  "
        f"{'LoS' if raw_result['has_los'] else 'NLOS'}",
        fontsize=11, fontweight='bold'
    )

    # ── Top: normalised power stem (scattered only) ────────────────────────────
    ax = axes[0]
    for itype in unique_types:
        mask = id0_plot == itype
        lbl  = TYPE_NAMES.get(itype, f'Type {itype}')
        clr  = TYPE_COLORS.get(itype, 'grey')
        markerline, stemlines, baseline = ax.stem(
            taus_plot[mask], pwr_norm[mask],
            linefmt=clr, markerfmt='o', basefmt=' ',
            label=lbl
        )
        markerline.set_color(clr); markerline.set_markersize(6)
        plt.setp(stemlines, linewidth=1.5, color=clr)

    ax.set_ylabel('Normalised power  (linear)')
    ax.set_title('(a) Scattered path power (LoS removed), coloured by interaction type')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.35)

    # ── Bottom: cumulative scattered power ────────────────────────────────────
    ax = axes[1]
    sort_idx = np.argsort(taus_plot)
    t_sorted = taus_plot[sort_idx]
    p_sorted = pwr_plot[sort_idx]
    p_cum    = np.cumsum(p_sorted) / (np.sum(p_sorted) + 1e-300)

    ax.plot(t_sorted, p_cum, color='dimgrey', linewidth=2)
    tau_rms_s = raw_result['tau_rms_s']
    ax.axvline(tau_rms_s * 1e9 + t_sorted[0],
               color='red', linestyle='--', linewidth=1.5,
               label=f'τ_mean + τ_rms = {tau_rms_s*1e9:.3f} ns')
    ax.set_xlabel('Delay [ns]')
    ax.set_ylabel('Cumulative scattered power fraction')
    ax.set_title('(b) Cumulative scattered power vs. delay')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.35)

    plt.tight_layout()
    return fig


# ─── Figure 6 : Multi-Config Metrics vs Elevation Angle ──────────────────────

# Visual style tables — edit here to change the look of all six configs at once.
_FREQ_STYLES = {
    "1.9GHz":   "-",
    "6GHz":  "--",
    "11GHz": "-.",
}
_POL_COLORS = {
    "VV": "steelblue",
    "HH": "darkorange",
    "LHCP": "seagreen",
}
_POL_MARKERS = {
    "VV": "o",
    "HH": "s",
    "LHCP": "^",
}


def _parse_config_label(label: str):
    """
    Extract polarisation and frequency strings from a config label.

    Expected label formats (case-insensitive):
      "VV 1.9GHz", "HH_6GHz", "lhcp-11ghz", "6 GHz VV", etc.

    Returns (pol, freq_key) e.g. ("VV", "6GHz"), or (None, None).
    """
    label_up = label.upper().replace("_", " ").replace("-", " ")
    if "LHCP" in label_up:
        pol = "LHCP"
    elif "VV" in label_up:
        pol = "VV"
    elif "HH" in label_up:
        pol = "HH"
    else:
        pol = None
    freq = None
    label_compact = label_up.replace(" ", "")
    for key in ("1.9GHZ", "6GHZ", "11GHZ"):   # longest first to avoid "1ghz" matching "11ghz"
        if key in label_compact:
            freq = key.replace("GHZ", "GHz")
            break
    return pol, freq


def _config_style(label: str):
    """Return (color, linestyle, marker) for a config label."""
    pol, freq = _parse_config_label(label)
    color  = _POL_COLORS.get(pol,   "gray")
    ls     = _FREQ_STYLES.get(freq, "-")
    marker = _POL_MARKERS.get(pol,  "o")
    return color, ls, marker


def _extract_metric_vs_elevation(angle_results: dict, metric: str,
                                  los_only: bool = False):
    """Compute mean for ``metric`` at each elevation angle."""
    angles  = sorted(angle_results.keys())
    K_means, tau_means, plos_pct = [], [],[]

    for ang in angles:
        res     = angle_results[ang]
        P_los_arr = np.array(res["P_los"])
        P_nlos_arr = np.array(res["P_nlos"])
        mean_Plos  = float(np.mean(P_los_arr))
        mean_Pnlos = float(np.mean(P_nlos_arr))
        
        K_mean_dB   = 10.0 * np.log10(mean_Plos / (mean_Pnlos + 1e-30) + 1e-30)
        
        tau_ns  = np.array(res["tau_rms_mean_s"]) * 1e9
        p_los   = np.array(res["los_probability"])
        
        
        K_means.append(K_mean_dB)
        
        fin_t = np.isfinite(tau_ns)
        tau_means.append(np.mean(tau_ns[fin_t]))
        plos_pct.append(float(np.mean(p_los) * 100))
        
    if metric == "K_dB":
        return (np.array(angles), np.array(K_means))
    elif metric == "tau_rms_ns":
        return (np.array(angles), np.array(tau_means))
    elif metric == "P_los":
        return (np.array(angles), np.array(plos_pct))
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    


def plot_metrics_vs_elevation_multi_config(
    configs_results: dict,
    marker_every: int = 1,
    figsize=(14, 14),
):
    """Three-panel figure: K-factor, RMS delay spread, and LoS delay spread
as a function of elevation angle, with all six simulation configurations
overlaid on each panel.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        "Channel Metrics vs. Elevation Angle for Satellite Pass\n"
        "Color -> Polarization (VV/HH/RHCP)  | Frequency: 11 GHz",
        fontsize=13, fontweight='bold'
    )

    panel_specs = [
        # (ax,       metric,       los_only, ylabel,                    panel_label)
        (axes[0], "K_dB",       False,  "Rician K-Factor [dB]",    "(a) K-Factor "),
        (axes[1], "tau_rms_ns", False, "RMS Delay Spread [ns]",   "(b) RMS Delay Spread"),
        # (axes[2], "P_los", False, "LoS Probability(%)",   "(c) Line of Sight Probability")
    ]

    for ax, metric, los_only, ylabel, panel_label in panel_specs:
        handles = []
        for label, all_results in configs_results.items():
            color, ls, marker = _config_style(label)
            (angles, mean) = _extract_metric_vs_elevation(all_results, metric, los_only=False)

            # Plot
            line, = ax.plot(
                angles, mean,
                color=color, linestyle=ls, linewidth=2.0,
                marker=marker, markersize=5,
                markevery=marker_every,
                zorder=3, label=label
            )
            handles.append(line)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(panel_label, fontsize=10, loc='left')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        if handles:
            ax.legend(
                handles=handles,
                fontsize=8,
                ncol=min(3, len(handles)),
                loc='best',
                title='Config',
                title_fontsize=7,
            )

    axes[-1].set_xlabel('Elevation Angle [°]', fontsize=10)

    # Shared x-axis: nice tick spacing
    all_angles = sorted({
        ang
        for angle_results in configs_results.values()
        for ang in angle_results.keys()
    })
    if all_angles:
        axes[-1].set_xlim(min(all_angles) - 2, max(all_angles) + 2)
        axes[-1].set_xticks(all_angles)
        axes[-1].set_xticklabels([f"{int(a)}°" for a in all_angles], fontsize=8)

    plt.tight_layout()
    return fig


def export_metrics_vs_elevation_multi_config_csv(
    configs_results: dict,
    csv_path: str,
) -> None:
    """Save the Fig 6 plotted summary values to a CSV file."""
    fieldnames = ["config", "elevation_deg", "K_dB", "tau_rms_ns", "P_los_pct"]
    metric_columns = [
        ("K_dB", "K_dB"),
        ("tau_rms_ns", "tau_rms_ns"),
        ("P_los", "P_los_pct"),
    ]

    rows = []
    for label, all_results in configs_results.items():
        series_by_angle = {}
        for metric, column in metric_columns:
            angles, values = _extract_metric_vs_elevation(
                all_results,
                metric,
                los_only=False,
            )
            for angle, value in zip(angles, values):
                row = series_by_angle.setdefault(
                    float(angle),
                    {"config": label, "elevation_deg": float(angle)},
                )
                row[column] = float(value)

        rows.extend(
            series_by_angle[angle]
            for angle in sorted(series_by_angle)
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─── Summary Stats Printer ────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    """Print formatted summary statistics for one elevation angle.

    K-Factor statistics are reported for LoS receivers ONLY.
    Delay spread and LoS probability use all receivers.
    """
    tau_ns  = np.array(results["tau_rms_mean_s"]) * 1e9
    has_los = np.array(results["los_probability"], dtype=bool)
    N       = results["n_rx_positions"]
    elev    = results["elevation_deg"]

    n_los  = int(np.sum(has_los))
    n_nlos = N - n_los

    # Ensemble K —
    # E[P_los] / E[P_nlos]  (ratio-of-means, numerically stable)
    P_nlos_arr    = np.array(results.get("P_nlos", [0.0] * N))
    P_los_arr  = np.array(results.get("P_los",  None) or ([0.0] * N))

    K_mean     = (float(np.mean(P_los_arr))
                    / (float(np.mean(P_nlos_arr)) + 1e-30))
    K_mean_dB  = 10.0 * np.log10(K_mean + 1e-12)
    

    SEP = "═" * 70
    print(SEP)
    print(f"  CHANNEL STATISTICS  —  Elev = {elev:.0f}°  |  N_RX = {N:,}")
    print(SEP)

    def _row(label, arr, fmt=".3f", unit=""):
        v = arr[np.isfinite(arr)]
        if len(v) == 0:
            print(f"  {label:<32}  (no valid samples)")
            return
        print(
            f"  {label:<32}  "
            f"mean={np.mean(v):{fmt}}  "
            f"median={np.median(v):{fmt}}  "
            f"std={np.std(v):{fmt}}  {unit}"
        )

    print(f"\n  LoS Statistics:")
    print(f"  {'LoS positions':<32}  {n_los:,}  ({n_los/N*100:.1f} %)")
    print(f"  {'NLOS positions':<32}  {n_nlos:,}  ({n_nlos/N*100:.1f} %)")

    print(f"\n  Rician K-Factor:")
    print(f"  {'  K_mean [dB]':<32}  {K_mean_dB:+.2f} dB  (linear {K_mean:.4f})")
    

    print(f"\n  RMS Delay Spread:")
    _row("  τ_rms LoS [ns]",   tau_ns[has_los],  ".4f", "ns")
    print(f"\n  RMS Delay Spread  (NLOS receivers, n={n_nlos:,}):")
    if n_nlos > 0:
        _row("  τ_rms NLOS [ns]",  tau_ns[~has_los], ".4f", "ns")
    else:
        print("  (no NLOS receivers)")

    print(SEP)