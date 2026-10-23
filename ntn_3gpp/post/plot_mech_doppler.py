"""LoS-state mechanism shares (non-direct power, path count) and the RMS
Doppler spread / coherence time per state, urban and rural, S band. RT: post/mech_doppler.py output (with and without
diffuse); geometry-stochastic model: matched scenes (chiu_roy_model/, run_v2_all and eval29).
The model's pass is at 550 km, ours at 600 km. The model's Doppler is nu = -f_c * beta_dot * d(excess delay)/d(beta), so
it is rescaled exactly by beta_dot(600 km) / beta_dot(550 km) at each elevation.
Usage: python post/plot_mech_doppler.py RUNS [NODIFF_RUNS]   ->  RUNS/figures/{mechanism_shares,doppler_coherence}.png
NODIFF_RUNS: a run with diffuse scattering off; its RT values replace the restored "without diffuse" ones."""
import csv, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{KIT}/../chiu_roy_model/data/model"
MODEL = {"urban": ("urban_ours", 3, 5), "rural": ("rural_ours_earth", 4, 6)}          # scene, eval29 seed, v2all seed
MECH = (("reflection", "tab:blue"), ("diffraction", "tab:green"))      # ground + wall, roof edge + corner
TC_MAX = 30.0                                  # coherence-time axis top [s]; RT points above it sit on the edge, labeled
LOW = 20                                       # RT points from fewer sample UEs in that state get an n= label
R_E, MU = 6371e3, 3.986004418e14


def beta_dot(beta_deg, h):
    """Elevation rate [rad/s] of a circular overhead pass at altitude h, at elevation beta."""
    r = R_E + h
    g = np.linspace(1e-6, np.arccos(R_E / r), 200001)                  # central angle, horizon to zenith
    b = np.arctan2(r * np.cos(g) - R_E, r * np.sin(g))
    db = np.gradient(b, g)
    return np.abs(np.interp(np.radians(beta_deg), b[::-1], db[::-1])) * np.sqrt(MU / r ** 3)


runs = sys.argv[1]
rt = {}
for f in glob.glob(f"{runs}/figures/mech_doppler_*.csv"):
    for r in csv.DictReader(open(f)):
        d = {k: float(v) for k, v in r.items() if k not in ("scenario", "variant", "state")}
        for q in ("P", "N"):                       # older files: one combined edge column, read as roof, no corner split;
            if f"{q}_edge" in d:                   # diffuse inside ground and wall
                d[f"{q}_roof"], d[f"{q}_corner"] = d.pop(f"{q}_edge"), np.nan
            d.setdefault(f"{q}_diffuse", np.nan)
        rt[(r["scenario"], r["variant"], r["state"], int(r["elevation_deg"]))] = d
for f in glob.glob(f"{sys.argv[2]}/figures/mech_doppler_*.csv") if len(sys.argv) > 2 else []:
    rows = [r for r in csv.DictReader(open(f)) if r["variant"] == "with"]
    for k in [k for k in rt if rows and k[0] == rows[0]["scenario"] and k[1] == "without"]:
        del rt[k]
    for r in rows:
        rt[(r["scenario"], "without", r["state"], int(r["elevation_deg"]))] = {k: float(v) for k, v in r.items()
                                                                             if k not in ("scenario", "variant", "state")}
mod = {}
for scen, (name, s29, s2) in MODEL.items():
    v2 = json.load(open(f"{M}/v2all_{name}_n3000_s{s2}.json"))["res"]
    mod[(scen, "mech")] = v2
    ev = [r for f in sorted(glob.glob(f"{M}/eval29_{name}_n6000_s{s29}*.json")) for r in json.load(open(f))["res"]]
    mod[(scen, "dop")] = sorted(ev, key=lambda r: r["beta"])

# ---- mechanism shares, LoS state. RT as simulated: reflection + diffraction = 100 %; diffuse (the model has
# none) as a percentage of all non-direct power (of reflection + diffraction it passes 10^4 % in rural, where the rough
# ground leaves almost no specular reflection) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
for j, scen in enumerate(("urban", "rural")):
    ax = axes[j]
    v2 = mod[(scen, "mech")]
    X = np.array([[r["Pg"] + r["Pw"], r["Pe"]] for r in v2])
    b = np.array([r["beta"] for r in v2])
    pts = [(e, rt[(scen, "with", "LoS", e)]) for e in range(10, 100, 10) if (scen, "with", "LoS", e) in rt]
    pts = [(e, d) for e, d in pts if np.isfinite(d["P_diffuse"])]       # older files keep diffuse inside ground and wall
    Y = np.array([[d["P_ground"] + d["P_wall"], d["P_roof"] + d["P_corner"], d["P_diffuse"]] for _, d in pts]).reshape(-1, 3)
    tot = Y.sum(1); Y[:, :2] *= 100 / Y[:, :2].sum(1, keepdims=True); Y[:, 2] *= 100 / tot
    for c, (m, col) in enumerate(MECH):
        ax.plot(b, 100 * X[:, c] / X.sum(1), color=col, lw=2)
    for c, col in enumerate(("tab:blue", "tab:green", "tab:purple")):
        ax.plot([e for e, _ in pts], Y[:, c], ls="none", marker="o", ms=6, color=col)
    ax.axhline(100, color="gray", lw=0.8, ls=":")
    ax.set_ylim(bottom=-3); ax.grid(True, alpha=0.3)
    ax.set_title(f"({'ab'[j]}) {scen.capitalize()}: share of reflection + diffraction power, LoS state", loc="left", fontsize=10)
    ax.set_xlabel("Elevation angle [°]")
    if j == 0:
        ax.set_ylabel("Share [%]")
    if not pts:
        ax.text(0.5, 0.5, "RT: needs the rerun mech_doppler CSV (diffuse split)", transform=ax.transAxes, ha="center", color="gray")
# color = mechanism; line = model, circle = RT
h = [Patch(color=c, label=m) for m, c in MECH]
h += [Patch(color="tab:purple", label="diffuse scattering (RT only), % of all non-direct power"),
      Line2D([], [], color="gray", lw=2, label="geometry-stochastic model"),
      Line2D([], [], color="gray", ls="none", marker="o", label="RT S 2 GHz, V+H")]
fig.legend(handles=h, loc="outside lower center", ncol=3, fontsize=9)
fig.savefig(f"{runs}/figures/mechanism_shares.png", dpi=150, bbox_inches="tight")

# ---- Doppler spread and coherence time ----
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, layout="constrained")
for j, scen in enumerate(("urban", "rural")):
    ev = mod[(scen, "dop")]
    b = np.array([r["beta"] for r in ev])
    scale = beta_dot(b, 600e3) / beta_dot(b, 550e3)
    for st, ls in (("LoS", "-"), ("NLoS", "--")):
        s = np.array([r[f"dsp_{st.lower()}_Hz"] for r in ev]) * scale
        axes[0, j].plot(b, s, "k", ls=ls, lw=1.8, label=f"geometry-stochastic model, {st}")
        axes[1, j].plot(b, 1 / (2 * np.pi * s), "k", ls=ls, lw=1.8)
        for var, col in (("with", "tab:purple"), ("without", "tab:cyan")):
            pts = [(e, rt[(scen, var, st, e)]) for e in range(10, 100, 10) if (scen, var, st, e) in rt]
            x, y, n = [], [], []
            for e, d in pts:
                m0, m1, m2 = d["dop_m0"], d["dop_m1"], d["dop_m2"]
                if np.isfinite(m0) and m0 > 0:
                    x.append(e); y.append(np.sqrt(max(m2 / m0 - (m1 / m0) ** 2, 0))); n.append(int(d["n_UEs"]))
            if x:
                y = np.array(y)
                lab = f"RT S 2 GHz, V+H, {var} diffuse, {st}"
                axes[0, j].plot(x, y, ls="none", marker="o" if var == "with" else "^", ms=6, color=col,
                                mfc=col if st == "LoS" else "none", label=lab)
                tc = 1 / (2 * np.pi * np.maximum(y, 1e-12))
                axes[1, j].plot(x, np.minimum(tc, TC_MAX), ls="none", marker="o" if var == "with" else "^", ms=6, color=col,
                                mfc=col if st == "LoS" else "none")
                for xe, t in zip(x, tc):
                    if t > TC_MAX:
                        axes[1, j].annotate(f"{t:.0f} s", (xe, TC_MAX), textcoords="offset points", xytext=(5, -3),
                                            fontsize=7, color=col)
                for xe, ye, ne in zip(x, y, n):
                    if ne < LOW and var == "with":
                        axes[0, j].annotate(f"n={ne}", (xe, ye), textcoords="offset points", xytext=(4, 5), fontsize=7, color=col)
    for i, lab in enumerate(("RMS Doppler spread σν [Hz]", "Coherence time 1/(2πσν) [s]")):
        ax = axes[i, j]
        ax.set_yscale("linear" if i == 0 else "log")                    # Doppler spread linear, coherence time log
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(bottom=0) if i == 0 else ax.set_ylim(top=1.4 * TC_MAX)
        ax.set_title(f"({'abcd'[2 * i + j]}) {scen.capitalize()}: {'RMS Doppler spread' if i == 0 else 'coherence time'}",
                     loc="left", fontsize=10)
        if j == 0:
            ax.set_ylabel(lab)
        if i == 1:
            ax.set_xlabel("Elevation angle [°]")
        if not any(k[0] == scen for k in rt):
            ax.text(0.5, 0.5, f"{scen} RT not on this PC yet", transform=ax.transAxes, ha="center", color="gray")
h, l = axes[0, 0].get_legend_handles_labels()
src = lambda s: 0 if s.startswith("geometry") else 1 if "with diffuse" in s else 2
order = sorted(range(len(l)), key=lambda i: (src(l[i]), l[i].endswith("NLoS")))
fig.legend([h[i] for i in order], [l[i] for i in order], loc="outside lower center", ncol=3, fontsize=9)
fig.savefig(f"{runs}/figures/doppler_coherence.png", dpi=150, bbox_inches="tight")
print(f"saved {runs}/figures/mechanism_shares.png and doppler_coherence.png")
