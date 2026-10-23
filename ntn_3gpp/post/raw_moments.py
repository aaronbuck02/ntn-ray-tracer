"""Per-UE path-power moments of the raw 1-in-50 sample, split by mechanism, so K and delay spread can be recomputed
with a mechanism reweighted (e.g. diffuse scattering removed) without keeping the paths.
As a script: python post/raw_moments.py RUNS OUT [GLOB]   (GLOB default "*", e.g. "rural_*")
  -> OUT/<folder>/results_elevXXXdeg.pkl, the results file with raw_results replaced by raw_moments (~5 MB)."""
import glob, os, pickle, sys
import numpy as np

# direct path, specular reflection off the ground (arrives from below), other specular reflections, diffuse, diffraction, rest
CLASSES = ("direct", "spec_ground", "spec_other", "diffuse", "diffraction", "other")


def raw_moments(x):
    """[sum p, sum p*tau, sum p*tau^2] per class, tau measured from the UE's first arrival t0 (sum over paths)."""
    t = np.asarray(x["taus"], float); p = np.abs(np.asarray(x["amps"])) ** 2
    ty = np.asarray(x["inter_d0"]); below = np.asarray(x["theta_r"]) > np.pi / 2
    t0 = float(t.min()) if len(t) else 0.0
    d = t - t0
    # with every bounce's type (runner: inter_all), a path with a diffuse bounce at any depth counts as diffuse; the raw
    # sample keeps only the first bounce, so there a specular-first path with a later diffuse bounce counts as specular
    dif = (np.asarray(x["inter_all"]) == 2).any(0) if "inter_all" in x else ty == 2
    masks = [ty == 0, (ty == 1) & below & ~dif, (ty == 1) & ~below & ~dif, dif, (ty == 8) & ~dif]
    masks.append(~np.any(masks, axis=0))
    mom = np.array([[p[m].sum(), (p[m] * d[m]).sum(), (p[m] * d[m] ** 2).sum()] for m in masks])
    return dict(seed=x["seed"], rx_nominal=x["rx_nominal"], has_los=bool(x["has_los"]), t0=t0, mom=mom)


if __name__ == "__main__":
    runs, out = sys.argv[1:3]
    pat = sys.argv[3] if len(sys.argv) > 3 else "*"
    for f in sorted(glob.glob(f"{runs}/{pat}/results_elev*.pkl")):
        r = pickle.load(open(f, "rb"))
        r["raw_moments"] = [raw_moments(x) for x in r.pop("raw_results", [])]
        g = os.path.join(out, os.path.relpath(f, runs))
        os.makedirs(os.path.dirname(g), exist_ok=True)
        pickle.dump(r, open(g, "wb"))
        print(g, len(r["P_los"]), r["cfg_snapshot"]["_SIONNA_PATCH"]["archive_rev"])
