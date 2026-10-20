"""Write and launch stochastic-sweep variants (driven by custom_drivers/scene_class_dataset.ipynb).

A variant is a dict: name, elevations, cfg_overrides, and one of
material_overrides / lhs / oat / manifest; runner='site' variants take
base_config and rx_file instead. lhs/oat draws are layered on the base
manifest's own per-scene draws and orbits. Each gets Results/<sweep>/<name>/ holding its
config, manifest, variant.json and the runner's per-elevation files. Variants
that differ only in UE-side keys (polarization, height, orientation) run as one
group job sharing each scene's ray trace; results still land per variant.
See docs/pipelines.md (Sensitivity sweeps).
"""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

import numpy as np

from Ray_Tracing.src.utils import variant_manifest as vm
from Ray_Tracing.src.utils.checkpoint_utils import results_path, simulation_status

_SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "simulations")
RUNNERS = {"stochastic": os.path.join(_SIM_DIR, "run_stochastic_sim.py"),
           "site":       os.path.join(_SIM_DIR, "run_elevation_sim.py")}


def variant_dir(sweep_dir, v):
    return os.path.join(sweep_dir, v["name"])


def variant_inputs(v, base_cfg, base_manifest=None):
    """(cfg, manifest) the variant runs with, built in memory; manifest is None for site runs."""
    if v.get("base_config"):   # e.g. a site config for runner='site'
        with open(v["base_config"]) as f:
            base_cfg = json.load(f)
    cfg = {**base_cfg, **v.get("cfg_overrides", {})}
    if v.get("runner", "stochastic") != "stochastic":
        return cfg, None
    if v.get("n_scenes") and base_manifest is not None:   # first n scenes of the base ensemble
        base_manifest = vm.first_scenes(base_manifest, v["n_scenes"])
    if v.get("manifest"):
        m = vm.load_manifest(v["manifest"])
    elif v.get("lhs"):
        m = vm.lhs_manifest(base_manifest, v["lhs"]["factors"], seed=v["lhs"].get("seed", 0))
    elif v.get("oat"):
        m = vm.oat_uniform_manifest(base_manifest, v["oat"]["factor"], v["oat"]["spec"],
                                    seed=v["oat"].get("seed", 0))
    else:
        m = vm.variant_manifest(base_manifest, v.get("material_overrides"), name=v["name"])
    return cfg, m


def write_variant(sweep_dir, v, base_cfg, base_manifest=None):
    """Write config.json, scenes_manifest.pkl (stochastic) and variant.json; return paths."""
    d = variant_dir(sweep_dir, v)
    os.makedirs(d, exist_ok=True)
    cfg, m = variant_inputs(v, base_cfg, base_manifest)
    cfg_path = os.path.join(d, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    paths = {"dir": d, "config": cfg_path}
    if m is not None:
        paths["manifest"] = vm.save_manifest(m, os.path.join(d, "scenes_manifest.pkl"))
    with open(os.path.join(d, "variant.json"), "w") as f:
        json.dump({k: v[k] for k in v if k != "manifest"}, f, indent=2, default=str)
    return paths


# UE-side keys: variants differing only in these share one solve (same rays,
# polarization read out from a dual-pol solve, atmospheric loss applied after the
# trace). Everything else is in the key, including VTEC (Faraday rolls the TX).
UE_KEYS = {"TX_POL_TYPE", "RX_POL_TYPE", "RX_HEIGHT", "RX_PITCH_DEG", "RX_YAW_DEG",
           "RX_ORIENTATION_DEG", "N_RX_PER_SCENE", "N_RX_KEEP_BY_ELEV",
           "RAIN_R_MM_H", "ATM_RHO_G_M3", "UE_TRAJ_STEPS", "UE_TRAJ_DT_S", "UE_SPEED_RANGE_MPS"}
_META_KEYS = {"design", "variant"}   # manifest bookkeeping, not inputs


def _canon(x):
    if isinstance(x, dict):
        return {str(k): _canon(x[k]) for k in sorted(x, key=str)}
    if isinstance(x, (list, tuple)):
        return [_canon(i) for i in x]
    if isinstance(x, np.ndarray):
        return _canon(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def solve_key(cfg, manifest):
    """Hash of everything a solve depends on except the UE-side keys."""
    c = {k: v for k, v in cfg.items() if k not in UE_KEYS and not k.startswith("_")}
    m = {k: v for k, v in manifest.items() if k not in _META_KEYS and k != "scenes"}
    scenes = []
    for e in manifest["scenes"]:
        e = dict(e)
        ov = {k: v for k, v in e.pop("cfg_overrides", {}).items() if k not in UE_KEYS}
        if ov:
            e["cfg_overrides"] = ov
        scenes.append(e)
    blob = json.dumps(_canon({"cfg": c, "manifest": m, "scenes": scenes}),
                      sort_keys=True, default=repr)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def pending_elevations(sweep_dir, v):
    """Elevations of v with no results file yet."""
    d = variant_dir(sweep_dir, v)
    return [e for e in v["elevations"] if not os.path.exists(results_path(d, e))]


def _cmd(v, paths, elev, gpu, resume):
    c = [sys.executable, RUNNERS[v.get("runner", "stochastic")],
         "--elevation", str(elev), "--gpu_id", str(gpu),
         "--config", paths["config"], "--output", paths["dir"],
         "--resume" if resume else "--no-resume"]
    if "manifest" in paths:
        c += ["--scenes_manifest", paths["manifest"]]
    else:
        c += ["--rx_file", v["rx_file"]]
    return c


def group_jobs(sweep_dir, variants, base_cfg, base_manifest=None):
    """[(members, elev)]: pending stochastic jobs merged by (solve_key, elev); site jobs stay single."""
    groups, singles = {}, []
    for v in variants:
        paths = write_variant(sweep_dir, v, base_cfg, base_manifest)
        pend = pending_elevations(sweep_dir, v)
        if "manifest" not in paths:
            singles += [([(v, paths)], e) for e in pend]
            continue
        with open(paths["config"]) as f:
            cfg = json.load(f)
        key = solve_key(cfg, vm.load_manifest(paths["manifest"]))
        for e in pend:
            groups.setdefault((key, e), []).append((v, paths))
    return [(m, e) for (_, e), m in groups.items()] + singles


def _group_cmd(sweep_dir, members, elev, gpu, resume):
    if len(members) == 1:
        v, paths = members[0]
        return _cmd(v, paths, elev, gpu, resume), os.path.join(
            paths["dir"], f"log_elev{int(elev):03d}deg.txt")
    with open(members[0][1]["config"]) as f:
        key = solve_key(json.load(f), vm.load_manifest(members[0][1]["manifest"]))
    gdir = os.path.join(sweep_dir, "_groups")
    os.makedirs(gdir, exist_ok=True)
    gpath = os.path.join(gdir, f"{key}_elev{int(elev):03d}.json")
    with open(gpath, "w") as f:
        json.dump({"members": [{"name": v["name"], "dir": p["dir"], "config": p["config"],
                                "manifest": p["manifest"]} for v, p in members]}, f, indent=2)
    log = gpath[:-5] + ".log"
    for _, p in members:   # each variant dir still has its log
        link = os.path.join(p["dir"], f"log_elev{int(elev):03d}deg.txt")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.relpath(log, p["dir"]), link)
    return [sys.executable, RUNNERS["stochastic"], "--elevation", str(elev),
            "--gpu_id", str(gpu), "--group", gpath,
            "--resume" if resume else "--no-resume"], log


def plan_groups(variants, base_cfg, base_manifest=None):
    """[(variants, elev, n_scenes)] over all elevations, without writing anything."""
    groups = {}
    for v in variants:
        cfg, m = variant_inputs(v, base_cfg, base_manifest)
        key = solve_key(cfg, m) if m is not None else ("site", v["name"])
        n = len(m["scenes"]) if m is not None else 1
        for e in v["elevations"]:
            groups.setdefault((key, e), ([], n))[0].append(v)
    return [(vs, e, n) for (_, e), (vs, n) in groups.items()]


def describe_groups(jobs):
    for members, e in jobs:
        names = ", ".join(v["name"] for v, _ in members)
        print(f"  elev={e:>4}  {len(members)} variant(s): {names}")


# Keys run_stochastic_sim reads with no default: checked before anything is queued.
REQUIRED_KEYS = ("FREQUENCY", "BASE_SAMPLES_PER_SRC", "BASE_MAX_PATHS")


def queue_entries(sweep_dir, variants, base_cfg, base_manifest=None, resume=True):
    """Pending jobs as plain dicts {cmd, log, tag, results}; cmd's GPU is the '{gpu}' slot."""
    for v in variants:
        if v.get("runner", "stochastic") == "stochastic":
            missing = [k for k in REQUIRED_KEYS if k not in {**base_cfg, **v.get("cfg_overrides", {})}]
            if missing:
                raise KeyError(f"{v['name']}: config lacks {missing}; nothing queued "
                               f"(stale BASE_CFG? re-run the config cell)")
    out = []
    for members, e in group_jobs(sweep_dir, variants, base_cfg, base_manifest):
        cmd, log = _group_cmd(sweep_dir, members, e, "{gpu}", resume)
        out.append({"cmd": cmd, "log": log, "elev": e,
                    "tag": f"{'+'.join(v['name'] for v, _ in members)} elev={e}",
                    "results": [results_path(p["dir"], e) for _, p in members]})
    return out


def run_queue(entries, gpus=(0,), block=True, poll=30):
    """Run queue entries, one process per free GPU; skips entries whose results all exist."""
    jobs, free, running, t0 = list(entries), list(gpus), {}, time.perf_counter()
    print(f"{len(jobs)} job(s) queued on GPUs {list(gpus)}", flush=True)

    def _reap():
        for g, (p, tag) in list(running.items()):
            if p.poll() is not None:
                print(f"  [{time.strftime('%H:%M:%S')}] done {tag}  rc={p.returncode}", flush=True)
                del running[g]
                free.append(g)

    while jobs or (block and running):
        _reap()
        while jobs and free:
            j = jobs.pop(0)
            if all(os.path.exists(r) for r in j["results"]):
                continue
            g = free.pop(0)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
            log = open(j["log"], "w")
            p = subprocess.Popen([str(g) if c == "{gpu}" else c for c in j["cmd"]], env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            running[g] = (p, j["tag"])
            print(f"  [{time.strftime('%H:%M:%S')}] launched {j['tag']} on GPU {g} "
                  f"(PID {p.pid}); {len(jobs)} left", flush=True)
        if jobs or (block and running):
            time.sleep(poll)
    if block:
        print(f"Sweep done in {(time.perf_counter() - t0) / 3600:.2f} h", flush=True)
    return [p for p, _ in running.values()]


def launch_grid(sweep_dir, variants, base_cfg, base_manifest=None, gpus=(0,),
                resume=True, block=True, poll=30):
    """Run every pending job from this process, one per free GPU; returns live procs.

    Jobs are (solve group, elevation): variants that differ only in UE-side keys
    share one runner and one ray trace per scene (see solve_key). The jobs die
    with this process; use launch_detached to survive a closed session.
    """
    return run_queue(queue_entries(sweep_dir, variants, base_cfg, base_manifest, resume),
                     gpus, block, poll)


def start_detached(entries, queue_dir, gpus=(0,), session="sweep", poll=30):
    """Write queue entries to <queue_dir>/_queue.json and run them in a detached tmux session.

    The scheduler (`python -m ...sweep_utils <queue>`) outlives the notebook and the
    SSH login. Relaunching after a crash is safe: finished jobs are skipped and the
    runners resume from their checkpoints.
    """
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError(f"tmux session {session!r} already exists: attach with "
                           f"`tmux attach -t {session}`, or kill it first")
    os.makedirs(queue_dir, exist_ok=True)
    qpath = os.path.join(queue_dir, "_queue.json")
    with open(qpath, "w") as f:
        json.dump({"gpus": list(gpus), "poll": poll, "jobs": list(entries)}, f, indent=2)
    log = os.path.join(queue_dir, "_queue.log")
    root = os.path.dirname(os.path.dirname(os.path.dirname(_SIM_DIR)))   # parent of Ray_Tracing/
    inner = (f"cd {shlex.quote(root)} && {shlex.quote(sys.executable)} -u -m "
             f"Ray_Tracing.src.utils.sweep_utils {shlex.quote(qpath)} 2>&1 | tee -a {shlex.quote(log)}")
    subprocess.run(["tmux", "new-session", "-d", "-s", session, inner], check=True)
    print(f"{len(entries)} job(s) queued -> {qpath}\n"
          f"running in tmux session {session!r}: `tmux attach -t {session}` "
          f"(detach: Ctrl-b d), or `tail -f {log}`")
    return session


def launch_detached(sweep_dir, variants, base_cfg, base_manifest=None, gpus=(0,),
                    resume=True, session=None, poll=30):
    """launch_grid in a detached tmux session (see start_detached); returns the session name."""
    session = session or f"sweep_{os.path.basename(os.path.normpath(sweep_dir))}"
    return start_detached(queue_entries(sweep_dir, variants, base_cfg, base_manifest, resume),
                          sweep_dir, gpus, session, poll)


def sweep_status(sweep_dir, variants):
    for v in variants:
        print(f"── {v['name']}")
        simulation_status(variant_dir(sweep_dir, v), v["elevations"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run a sweep job queue written by launch_detached.")
    ap.add_argument("queue")
    a = ap.parse_args()
    with open(a.queue) as f:
        q = json.load(f)
    run_queue(q["jobs"], q["gpus"], block=True, poll=q.get("poll", 30))
