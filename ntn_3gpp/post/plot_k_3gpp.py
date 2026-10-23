"""K in the TR 38.811 definition: LoS K = mean over LoS UEs of each UE's K in dB. RT with diffuse (post/k_3gpp.py),
the model with one drawn city per UE (post/model_panel_k.py, K_iso_mean_dB) and TR 38.811 (its mu_K).
Usage: python post/plot_k_3gpp.py RUNS [BAND=S]   ->  RUNS/figures/K_3gpp_def_BAND.png"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = sys.argv[1]; band = sys.argv[2] if len(sys.argv) > 2 else "S"
SUF = "" if band == "S" else f"_{band}"
TAG = {"S": "S 2 GHz, V+H", "Ku": "Ku 12 GHz, RHCP", "Ka": "Ka 20 GHz, RHCP"}[band]
tr = {}
for row in csv.DictReader(r for r in open(f"{KIT}/post/tr38811_{'ka' if band == 'Ka' else 'sband'}.csv") if not r.startswith("#")):
    if row["metric"] == "K_LoS_dB":
        tr.setdefault(row["scenario"], []).append((int(float(row["elevation_deg"])), float(row["value"])))
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
for ax, scen, t in zip(axes, ("urban", "rural"), "ab"):
    f = f"{runs}/figures/k_3gpp_{scen}{SUF}.csv"
    if os.path.exists(f):
        rows = list(csv.DictReader(open(f)))
        x = [int(r["elevation_deg"]) for r in rows]
        ax.plot(x, [float(r["mu_K_dB"]) for r in rows], color="tab:purple", marker="P", lw=2, label=f"RT {TAG}, with diffuse (bar: 25-75% of scenes)")
        ax.vlines(x, [float(r["q25"]) for r in rows], [float(r["q75"]) for r in rows], color="tab:purple", lw=1.3, alpha=0.8)
    else:
        ax.text(0.5, 0.06, f"{scen} {band} RT not here yet", transform=ax.transAxes, ha="center", color="gray")
    mf = f"{runs}/figures/model_panel_k_{band}.csv"
    if os.path.exists(mf):
        md = [r for r in csv.DictReader(open(mf)) if r["scenario"] == scen]
        ax.plot([int(r["elevation_deg"]) for r in md], [float(r["K_iso_mean_dB"]) for r in md], "k--", lw=2, label="geometry-stochastic model")
    x, y = zip(*sorted(tr[scen]))
    ax.plot(x, y, color="gray", ls=":", marker="o", mfc="none", lw=1.5, label=f"TR 38.811 {'Ka' if band == 'Ka' else 'S'} band")
    ax.set_title(f"({t}) {scen.capitalize()}", loc="left", fontsize=11)
    ax.set_xlabel("Elevation angle [°]"); ax.set_xticks(range(10, 100, 10)); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
axes[0].set_ylabel("K, LoS state: mean over UEs of K [dB]")
fig.savefig(f"{runs}/figures/K_3gpp_def_{band}.png", dpi=150, bbox_inches="tight")
print(f"saved {runs}/figures/K_3gpp_def_{band}.png")
