"""Normalized non-direct PDP per state, urban and rural at 10/20/40/80 deg:
geometry-stochastic model (eval29 PDPs of the matched scenes, chiu_roy_model/) as lines, RT (post/pdp.py output)
as markers, with and without diffuse.
Usage: python post/plot_pdp.py RUNS [NODIFF_RUNS]   ->  RUNS/figures/PDP_vs_delay.png
NODIFF_RUNS: a run with diffuse scattering off; its RT PDPs replace the restored "without diffuse" ones."""
import csv, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = {"urban": f"{KIT}/../chiu_roy_model/data/model/eval29_urban_ours_n6000_s3.json",
         "rural": f"{KIT}/../chiu_roy_model/data/model/eval29_rural_ours_earth_n6000_s4.json"}
ELEV = (10, 20, 40, 80)
RT = {"with": ("tab:purple", "o", "RT S 2 GHz, V+H, with diffuse"), "without": ("tab:cyan", "^", "RT S 2 GHz, V+H, without diffuse")}

runs = sys.argv[1]
rt = {}
for f in glob.glob(f"{runs}/figures/pdp_*.csv"):
    for r in csv.DictReader(open(f)):
        d = rt.setdefault((r["scenario"], r["variant"], r["state"], int(r["elevation_deg"])), {"x": [], "y": [], "n": int(r["n_UEs"])})
        d["x"].append(float(r["delay_ns"])); d["y"].append(float(r["p"]))
for f in glob.glob(f"{sys.argv[2]}/figures/pdp_*.csv") if len(sys.argv) > 2 else []:
    rows = [r for r in csv.DictReader(open(f)) if r["variant"] == "with"]
    for k in [k for k in rt if rows and k[0] == rows[0]["scenario"] and k[1] == "without"]:
        del rt[k]
    for r in rows:
        d = rt.setdefault((r["scenario"], "without", r["state"], int(r["elevation_deg"])), {"x": [], "y": [], "n": int(r["n_UEs"])})
        d["x"].append(float(r["delay_ns"])); d["y"].append(float(r["p"]))

fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True, sharey=True, layout="constrained")
for i, scen in enumerate(("urban", "rural")):
    pdp = json.load(open(MODEL[scen]))["pdp"]
    name = json.load(open(MODEL[scen]))["meta"]["scene"]
    for j, e in enumerate(ELEV):
        ax = axes[i, j]
        notes = []
        for st, ls in (("LoS", "-"), ("NLoS", "--")):
            m = pdp.get(f"{name}_{st.lower()}_{e:g}")
            if m:
                c, p = np.array(m["centers_ns"]), np.array(m["p"])
                ax.plot(c[p > 0], p[p > 0], "k", ls=ls, lw=1.5, label=f"geometry-stochastic model, {st}")
            for v, (col, mk, lab) in RT.items():
                d = rt.get((scen, v, st, e))
                if d:
                    ax.plot(d["x"], d["y"], ls="none", marker=mk, ms=4, color=col, mfc=col if st == "LoS" else "none",
                            label=f"{lab}, {st}")
                    if v == "with":
                        notes.append(f"RT {st}: {d['n']} UEs")
        ax.text(0.97, 0.97, "\n".join(notes), transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(10, 1e4); ax.set_ylim(1e-7, 1.5)
        ax.grid(True, which="major", alpha=0.3)
        ax.set_title(f"({'abcdefgh'[4 * i + j]}) {scen.capitalize()}, {e}°", loc="left", fontsize=10)
        if j == 0:
            ax.set_ylabel("Normalized non-direct PDP")
        if i == 1:
            ax.set_xlabel("Excess delay [ns]")
h, l = axes[0, 0].get_legend_handles_labels()
src = lambda lab: 0 if lab.startswith("geometry") else 1 if "with diffuse" in lab else 2
order = sorted(range(len(l)), key=lambda i: (src(l[i]), l[i].endswith("NLoS")))   # one column per source: LoS, then NLoS
fig.legend([h[i] for i in order], [l[i] for i in order], loc="outside lower center", ncol=3, fontsize=8)
fig.savefig(f"{runs}/figures/PDP_vs_delay.png", dpi=150, bbox_inches="tight")
print(f"saved {runs}/figures/PDP_vs_delay.png")
