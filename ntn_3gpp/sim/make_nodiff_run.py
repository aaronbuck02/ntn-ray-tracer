"""Diffuse-off copy of a run for the "without diffuse" curves: scattering coefficient 0 on walls, roofs and
ground (ground roughness 0), diffuse reflection off, everything else as the source run (up to 3 bounces, same scenes
and UEs). Specular reflections keep their full power, so nothing has to be reweighted afterwards.

    python sim/make_nodiff_run.py SRC_RUNS OUT_RUNS SCENARIO [BAND]      (e.g. runs runs_nodiff rural Ku; BAND default S)

Writes OUT_RUNS/<member>/{config.json, manifest.pkl} for each member of SRC_RUNS/groups/<scenario>_<band>.json and
OUT_RUNS/groups/<scenario>_<band>.json. Run with
    RUNS=OUT_RUNS ./run.sh GPU SCENARIO BAND [ELEVATION...]
"""
import json, os, pickle, sys

src, out, scen = sys.argv[1:4]
band = sys.argv[4] if len(sys.argv) > 4 else "S"
members = []
for name in [m["name"] for m in json.load(open(os.path.join(src, "groups", f"{scen}_{band}.json")))["members"]]:
    d = os.path.abspath(os.path.join(out, name))
    os.makedirs(d, exist_ok=True)
    cfg = json.load(open(os.path.join(src, name, "config.json")))
    cfg.update(DIFFUSE_REFLECTION=False, SCATTERING_COEFF=0.0)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=2)
    m = pickle.load(open(os.path.join(src, name, "manifest.pkl"), "rb"))
    for k in ("wall_params", "roof_params", "ground_params"):
        m[k]["scattering_coefficient"] = 0.0
    for s in m["scenes"]:
        s["cfg_overrides"]["GROUND_ROUGHNESS_M"] = 0.0      # the runner sets the ground's S from the roughness
    pickle.dump(m, open(os.path.join(d, "manifest.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    members.append(dict(name=name, dir=d, config=os.path.join(d, "config.json"), manifest=os.path.join(d, "manifest.pkl")))
os.makedirs(os.path.join(out, "groups"), exist_ok=True)
json.dump({"members": members}, open(os.path.join(out, "groups", f"{scen}_{band}.json"), "w"), indent=2)

# check: what the runner will read
m = pickle.load(open(members[0]["manifest"], "rb"))
assert all(m[k]["scattering_coefficient"] == 0 for k in ("wall_params", "roof_params", "ground_params"))
assert all(s["cfg_overrides"]["GROUND_ROUGHNESS_M"] == 0 for s in m["scenes"])
assert json.load(open(members[0]["config"]))["DIFFUSE_REFLECTION"] is False
print(f"wrote {', '.join(x['name'] for x in members)} and groups/{scen}_{band}.json: {len(m['scenes'])} scenes, diffuse off")
