"""S-band P_LOS, delay spread (LoS and NLoS state) and LoS-state K vs elevation, urban and rural: RT (post/metrics.py
output) against the geometry-stochastic model (Chiu & Roy, matched to our scenes: post/chiu_roy_model.csv) and TR 38.811
(post/tr38811_sband.csv). Run post/metrics.py RUNS SCENARIO first (rural: on the PC that holds its results).
RT bars: 25th-75th percentile over the 10 scenes (each scene's own value of the plotted quantity).
Usage: python post/plot_vs_elevation.py RUNS [NODIFF_RUNS]   ->  RUNS/figures/{P_LoS,DS,K}_vs_elevation.png
NODIFF_RUNS: a run with diffuse scattering off (scattering coefficient 0); its RT line replaces the restored "without
diffuse" line of the scenarios it holds.
       python post/plot_vs_elevation.py RUNS [NODIFF_RUNS] --band Ku|Ka   ->  RUNS/figures/{DS,K}_vs_elevation_BAND.png:
RT at that band, isotropic UE (post/metrics_band.py output), with and without diffuse, against the model at that band
(at 2 GHz if it has not been run there) and TR 38.811 S band."""
import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELEV = list(range(10, 100, 10))
SCENS = ("urban", "rural")
MODEL = {"urban": "urban_ours", "rural": "rural_ours_earth"}
LOW_SAMPLE = 20                    # pooled DS points from fewer raw-sample UEs in that state are labeled with n
BAND = sys.argv.pop(sys.argv.index("--band") + 1) if "--band" in sys.argv else "S"
sys.argv = [a for a in sys.argv if a != "--band"]
LINES = [("S_VH", "RT S 2 GHz, V+H, with diffuse", "-", "tab:purple", -0.6),          # x offset keeps the two bars apart
         ("S_VH_nodiff", "RT S 2 GHz, V+H, without diffuse", "--", "tab:cyan", 0.6)]
ANNOT = "S_VH"                     # the line whose DS points carry the n= label (the one that may use the raw sample)
if BAND != "S":
    GHZ, N = {"Ku": (12, 9), "Ka": (20, 16)}[BAND]
    LINES = [(BAND, f"RT {BAND} {GHZ} GHz, RHCP, isotropic UE, with diffuse", "-", "tab:purple", -0.6),
             (f"{BAND}_nodiff", f"RT {BAND} {GHZ} GHz, RHCP, isotropic UE, without diffuse", "--", "tab:cyan", 0.6)]
    ANNOT = BAND
lg = lambda v: np.log10(np.asarray(v, float) * 1e-9)

runs = sys.argv[1]
rt = {}            # (scenario, line) -> {elev: row}
for f in glob.glob(f"{runs}/figures/metrics_*.csv"):
    for row in csv.DictReader(open(f)):
        rt.setdefault((row["scenario"], row["line"]), {})[int(row["elevation_deg"])] = {k: float(v) for k, v in row.items()
                                                                                     if k not in ("scenario", "line")}
for f in glob.glob(f"{sys.argv[2]}/figures/metrics_*.csv") if len(sys.argv) > 2 else []:
    rows = [r for r in csv.DictReader(open(f)) if r["line"] == ("S_VH" if BAND == "S" else BAND)]
    if rows:
        rt[(rows[0]["scenario"], "S_VH_nodiff" if BAND == "S" else f"{BAND}_nodiff")] = {int(r["elevation_deg"]): {k: float(v) for k, v in r.items()
                                                                           if k not in ("scenario", "line")} for r in rows}
ref = {}           # (source, scene, metric) -> [(elev, value)]
for row in csv.DictReader(open(f"{KIT}/post/chiu_roy_model.csv")):
    ref.setdefault(("model", row["scene"], row["metric"]), []).append((float(row["beta_deg"]), float(row["value"])))
for row in csv.DictReader(r for r in open(f"{KIT}/post/tr38811_{'ka' if BAND == 'Ka' else 'sband'}.csv") if not r.startswith("#")):
    ref.setdefault(("tr", row["scenario"], row["metric"]), []).append((float(row["elevation_deg"]), float(row["value"])))


def panel(ax, scen, metric, f, model_metric, tr_metric, state=None, title=""):
    drawn = False
    for key, label, ls, color, dx in LINES:
        # P_LoS is the same with or without diffuse: drawn once, from the diffuse-off run only when that is all there is
        if (scen, key) not in rt or (metric == "P_LoS" and key != "S_VH" and (scen, "S_VH") in rt):
            continue
        if metric == "K_LoS_dB" and key.endswith("_nodiff"):          # K figure: with diffuse only
            continue
        d = rt[(scen, key)]
        x = [e for e in ELEV if e in d and np.isfinite(d[e][metric])]
        if x:
            xs = np.array(x, float) + (0 if metric == "P_LoS" else dx)
            ax.plot(xs, f([d[e][metric] for e in x]), color=color, marker="P", ls=ls, lw=2 if ls == "-" else 1.5,
                    ms=5, mfc=None if ls == "-" else "none",
                    label=("RT S 2 GHz, with and without diffuse" if metric == "P_LoS" else label) + " (bar: 25-75%)")
            # metrics files from an older metrics.py have no quartiles: no bars
            lo, hi = (f([d[e].get(f"{metric}_{q}", np.nan) for e in x]) for q in ("q25", "q75"))
            ax.vlines(xs, lo, hi, color=color, lw=1.3, alpha=0.8)              # 25th-75th percentile over scenes
            ax.plot(np.r_[xs, xs], np.r_[lo, hi], ls="none", marker="_", ms=6, color=color, alpha=0.8)
            drawn = True
            for e in x if state and key == ANNOT else []:           # same n for both S lines: label once
                n = int(d[e][f"n_{state}_sample"])
                if n < LOW_SAMPLE:
                    ax.annotate(f"n={n}", (e, f([d[e][metric]])[0]), textcoords="offset points", xytext=(4, 6),
                                fontsize=8, color=color)
    ms = f"{MODEL[scen]}_{BAND}" if ("model", f"{MODEL[scen]}_{BAND}", model_metric) in ref else MODEL[scen]   # band run if made
    if ("model", ms, model_metric) in ref:
        x, y = zip(*ref[("model", ms, model_metric)])
        ax.plot(x, f(y), "k--", lw=2, label="geometry-stochastic model" + ("" if BAND == "S" else
                                                                           f" ({GHZ if ms != MODEL[scen] else 2} GHz)"))
    x, y = zip(*ref[("tr", scen, tr_metric)])
    ax.plot(x, y if tr_metric.startswith("lgDS") else f(y), color="gray", ls=":", marker="o", mfc="none", lw=1.5,
            label=f"TR 38.811 {'Ka' if BAND == 'Ka' else 'S'} band" + (f", {state}" if state else ""))
    if not drawn:
        ax.text(0.5, 0.06, f"{scen} RT not on this PC yet", transform=ax.transAxes, ha="center", color="gray")
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xticks(ELEV)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


SUF = "" if BAND == "S" else f"_{BAND}"
for name, metric, ylabel, f, tr_metric in (("P_LoS", "P_LoS", "LoS probability", np.asarray, "P_LoS"),
                                           ("K", "K_LoS_dB", "K-factor, LoS state [dB]", np.asarray, "K_LoS_dB")):
    if name == "P_LoS" and BAND != "S":                       # the LoS state does not depend on the band
        continue
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    for ax, scen, t in zip(axes, SCENS, "ab"):
        panel(ax, scen, metric, f, metric, tr_metric, title=f"({t}) {scen.capitalize()}")
        ax.set_xlabel("Elevation angle [°]")
    axes[0].set_ylabel(ylabel)
    fig.savefig(f"{runs}/figures/{name}_vs_elevation{SUF}.png", dpi=150, bbox_inches="tight")

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True, layout="constrained")
for i, state in enumerate(("LoS", "NLoS")):
    for j, scen in enumerate(SCENS):
        panel(axes[i, j], scen, f"DS_{state}_ns", lg, f"DS_{state}_ns", f"lgDS_{state}", state=state,
              title=f"({'abcd'[2 * i + j]}) {scen.capitalize()}, {state} state")
    axes[i, 0].set_ylabel("lgDS = log10(DS / 1 s)")
for ax in axes[1]:
    ax.set_xlabel("Elevation angle [°]")
fig.savefig(f"{runs}/figures/DS_vs_elevation{SUF}.png", dpi=150, bbox_inches="tight")
print(f"saved {runs}/figures/{{P_LoS,DS,K}}_vs_elevation{SUF}.png")
