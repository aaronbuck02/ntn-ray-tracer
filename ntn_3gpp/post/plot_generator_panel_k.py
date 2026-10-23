"""K at the UE panel, RT against the TR 38.811 channel generator (post/generator_panel_k.py output).
Usage: python post/plot_generator_panel_k.py RUNS [BAND=Ku]   ->  RUNS/figures/generator_panel_k_BAND.png"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = sys.argv[1]; band = sys.argv[2] if len(sys.argv) > 2 else "Ku"
MEAN = "--mean" in sys.argv          # TR 38.811 definition: mean over UEs of each UE's K in dB
C = lambda k: k.replace("_dB", "_mean_dB") if MEAN else k
N = {"Ku": 9, "Ka": 16}[band]
LINES = [("K_panel_RT_dB", "RT with non-isotropic antenna, with diffuse", "k", "-", 2.2),
         ("K_panel_proposed_dB", "Generation with non-isotropic antenna, proposed: K/ASA/ZSA obtained in isotropic RT"
          " + mean zenith AoA offset from LoS", "tab:orange", "-", 2.0),
         ("K_panel_generator_dB", "Generation with non-isotropic antenna, K/ASA/ZSA obtained in isotropic RT", "tab:red", "--", 1.8)]
if band == "Ka":                                   # 38.811 has its own Ka table: what an evaluator gets today
    LINES.append(("K_panel_generator_38811_dB", "Generation with non-isotropic antenna, TR 38.811 Ka K/ASA/ZSA", "tab:gray", ":", 1.8))
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
for ax, scen, t in zip(axes, ("urban", "rural"), "ab"):
    try:
        rows = list(csv.DictReader(open(f"{runs}/figures/generator_panel_k_{scen}_{band}.csv")))
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"{scen} {band} RT not here yet", transform=ax.transAxes, ha="center", color="gray"); continue
    x = [int(r["elevation_deg"]) for r in rows]
    for key, label, c, ls, lw in LINES:
        ax.plot(x, [float(r[C(key)]) for r in rows], color=c, ls=ls, lw=lw, marker="o", ms=3.5, label=label)
    ax.set_title(f"({t}) {scen.capitalize()}, LoS, {band}", loc="left", fontsize=11)
    ax.set_xlabel("Elevation angle [°]"); ax.set_xticks(x); ax.grid(True, alpha=0.3)
axes[0].set_ylabel("mean over UEs of K [dB] at the UE antenna output" if MEAN else "K-factor at the UE antenna output [dB]")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=1, fontsize=9, title_fontsize=9, alignment="left",
           title=f"Non-isotropic antenna: {N}x{N} panel pointed at the satellite. Generation: TR 38.811 cluster-and-ray procedure (TR 38.901 Sec. 7.5)")
out = f"{runs}/figures/generator_panel_k_{band}{'_meandB' if MEAN else ''}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
