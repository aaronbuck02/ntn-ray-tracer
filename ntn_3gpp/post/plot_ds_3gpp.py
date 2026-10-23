"""S-band DS in the TR 38.811 definition (mean over UEs of log10 DS, per state): RT with and without diffuse
(post/ds_3gpp.py output) against TR 38.811. Per-UE DS over all paths (no floor). Bars: 25th-75th percentile over scenes. Model: one drawn city per UE, same rules (post/model_ds_3gpp.py).
Usage: python post/plot_ds_3gpp.py RUNS NODIFF_RUNS [BAND=S]   ->  RUNS/figures/DS_3gpp_def_BAND.png
(Ku/Ka: the RHCP member, model at that carrier, TR 38.811 Ka table at Ka.)"""
import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs, nodiff = sys.argv[1:3]; band = sys.argv[3] if len(sys.argv) > 3 else "S"
TAG = {"S": "S 2 GHz, V+H", "Ku": "Ku 12 GHz, RHCP", "Ka": "Ka 20 GHz, RHCP"}[band]; SUF = "" if band == "S" else f"_{band}"
SRC = [(runs, f"RT {TAG}, with diffuse", "tab:purple", "-", -0.6), (nodiff, f"RT {TAG}, without diffuse", "tab:cyan", "--", 0.6)]
tr = {}
for row in csv.DictReader(r for r in open(f"{KIT}/post/tr38811_{'ka' if band == 'Ka' else 'sband'}.csv") if not r.startswith("#")):
    tr.setdefault((row["scenario"], row["metric"]), []).append((float(row["elevation_deg"]), float(row["value"])))
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True, layout="constrained")
for i, state in enumerate(("LoS", "NLoS")):
    for j, scen in enumerate(("urban", "rural")):
        ax = axes[i, j]
        for R, label, c, ls, dx in SRC:
            if not os.path.exists(f"{R}/figures/ds_3gpp_{scen}{SUF}.csv"):
                continue
            rows = [r for r in csv.DictReader(open(f"{R}/figures/ds_3gpp_{scen}{SUF}.csv")) if r["state"] == state]
            for version, lw, alpha, lab in (("all paths", 2, 1.0, f"{label} (bar: 25-75% of scenes)"),):
                d = [r for r in rows if r["version"] == version]
                x = np.array([int(r["elevation_deg"]) for r in d], float) + dx
                ax.plot(x, [float(r["mu_lgDS"]) for r in d], color=c, ls=ls, lw=lw, alpha=alpha, marker="P" if lw == 2 else None,
                        ms=5, mfc=None if ls == "-" else "none", label=lab)
                if version == "all paths":
                    ax.vlines(x, [float(r["q25"]) for r in d], [float(r["q75"]) for r in d], color=c, lw=1.3, alpha=0.8)
        md = sorted((r for f in glob.glob(f"{runs}/figures/model_ds_3gpp_{band}*.csv") for r in csv.DictReader(open(f))
                     if r["scenario"] == scen and r["state"] == state), key=lambda r: int(r["elevation_deg"]))   # 89 deg in its own file
        for version, lw, alpha, lab in (("all paths", 2, 1.0, "geometry-stochastic model"),):
            d = [r for r in md if r["version"] == version]
            ax.plot([int(r["elevation_deg"]) for r in d], [float(r["mu_lgDS"]) for r in d], "k--", lw=lw, alpha=alpha, label=lab)
        x, y = zip(*tr[(scen, f"lgDS_{state}")])
        ax.plot(x, y, color="gray", ls=":", marker="o", mfc="none", lw=1.5, label=f"TR 38.811 {'Ka' if band == 'Ka' else 'S'} band (mean of log10 DS)")
        ax.set_title(f"({'abcd'[2 * i + j]}) {scen.capitalize()}, {state} state", loc="left", fontsize=11)
        ax.set_xticks(range(10, 100, 10)); ax.grid(True, alpha=0.3)
        if i == 1: ax.set_xlabel("Elevation angle [°]")
        if j == 0: ax.set_ylabel("mean over UEs of log10(DS / 1 s)")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=9,
           title="Delay spread as TR 38.811 tabulates it: each UE's own DS, then the mean of log10(DS) over the UEs of that state",
           title_fontsize=9)
fig.savefig(f"{runs}/figures/DS_3gpp_def_{band}.png", dpi=150, bbox_inches="tight")
print(f"saved {runs}/figures/DS_3gpp_def_{band}.png")
