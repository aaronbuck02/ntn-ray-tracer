# ntn_3gpp: ray-traced update of the TR 38.811 tables

This folder holds the campaign's code, run inputs and processed dataset. It is self-contained: it runs on its own copy of the repo modules (`Ray_Tracing/src/`, as of 2026-10-14) and the patched Sionna in `site/`, not on the repo's `src/`. Run everything from this folder.

The ray-tracing sweep covers:

- 2 scenarios (urban, rural)
- 3 bands (S 2 GHz, Ku 12 GHz, Ka 20 GHz)
- 9 elevations (10°–90°)

Each run uses 10 random scenes × 1000 UEs. Each scenario, band and elevation writes one `results_elevXXXdeg.pkl` per output folder.

| PC | Runs |
|---|---|
| First PC (2 × 10 GB GPUs) | urban, all bands |
| Second PC | rural, all bands |

Both PCs run the same code from this repo: the runner, the repo modules it imports, and the patched Sionna RT 1.2.1 in `site/`. Do not swap in another Sionna. The project's patches fix the line-of-sight test and the ray launch toward a distant satellite, and the patch check below must pass.

## 1. Prepare the environment

**Needs**
- Linux.
- NVIDIA GPU with ≥ 10 GB. CUDA is required; the patched ray launcher does not run on CPU.
- Python 3.12.
- About 220 GB of free disk for rural (results about 110 GB, plus checkpoints of the same size until deleted).

```sh
git clone -b sionna-ntn-3gpp https://github.com/aaronbuck02/ntn-ray-tracer.git
cd ntn-ray-tracer/ntn_3gpp
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=$PWD:$PWD/site python -c "import Ray_Tracing.src.sionna_patch_check as c; print(c.summary_line(c.require()))"
```

The check must print:

```
sionna 1.2.1: beamforming=yes guard=1.0 mm cone=exact cuda=yes archive=2026-10-14
```

If it shows an error, `cuda=no` or an older archive date, stop and fix the setup first.

**Smoke test** (about 5 min, 1 scene × 50 UEs):

```sh
PYTHONPATH=$PWD:$PWD/site python sim/make_runs.py smoke --smoke
RUNS=$PWD/smoke ./run.sh 0 rural Ka 30
grep -E "sionna 1.2.1|candidate cap hit|scene=1/1" smoke/logs/rural_Ka_e30.log
ls smoke/rural_Ka/results_elev030deg.pkl
```

It passes if the log shows `cone=exact`, there is no `candidate cap hit` line, and the results file exists. Then delete `smoke/`.

## 2. Run

`./run.sh GPU SCENARIO BAND [ELEVATION...]`. Elevations default to 10–90.

Second PC (rural) with 2 GPUs, about 26 h each:

```sh
. .venv/bin/activate
nohup sh -c './run.sh 0 rural S;  ./run.sh 0 rural Ku 10 20 30 40'    > runs/gpu0.out 2>&1 &
nohup sh -c './run.sh 1 rural Ka; ./run.sh 1 rural Ku 50 60 70 80 90' > runs/gpu1.out 2>&1 &
```

With 1 GPU, about 52 h:

```sh
nohup sh -c './run.sh 0 rural S; ./run.sh 0 rural Ka; ./run.sh 0 rural Ku' > runs/gpu0.out 2>&1 &
```

**Rules**
- **To resume** after a crash or reboot, run the same command again. Finished elevations are skipped, and a partly done elevation restarts from its last finished scene.
- **Never run the same scenario, band and elevation on two GPUs at once.** Both would write the same checkpoint.
- **Known crash:** Dr.Jit sometimes aborts with `Assertion failed ... queue.cpp:357`. The elevation then appears in `runs/logs/failed.txt`; rerun the command.

**Check while running**

```sh
cat runs/logs/failed.txt                          # failures, plus a "finished" line per band
grep -c "candidate cap hit" runs/logs/*.log       # must be 0, otherwise paths were dropped
grep -o "scene=[0-9]*/10" runs/logs/rural_S_e10.log | tail -1
ls runs/rural_*/results_elev*.pkl | wc -l         # 36 when rural is done
```

`ground ray missing N/N` in the logs is expected for rough ground, which has no mirror-like ground bounce. The check only means something when N is small and the count is not N/N.

## 3. Collect the data

Layout, the same on both PCs:

```
runs/<scenario>_<band>[_<pol>]/results_elevXXXdeg.pkl     the data: keep
runs/<scenario>_<band>[_<pol>]/checkpoint_elevXXXdeg.pkl  same content as results: delete once results exist
runs/<scenario>_<band>[_<pol>]/scratch_elevXXX/           generated scene files: delete
runs/logs/<scenario>_<band>_eXX.log                       keep
```

Output folders, each with 9 results files:

- urban: `urban_S_V`, `urban_S_H`, `urban_Ka`, `urban_Ku`
- rural: `rural_S_V`, `rural_S_H`, `rural_Ka`, `rural_Ku`

S band has one folder per UE port (V, H), both from the same ray trace.

**Check every results file before copying.** Each must hold 10000 UEs. The patch date must be 2026-10-14, except for urban S and urban Ka 10°–70° (2026-09-24; see "Settings worth knowing"):

```sh
PYTHONPATH=$PWD:$PWD/site python - <<'EOF'
import glob, pickle
for f in sorted(glob.glob("runs/*/results_elev*.pkl")):
    r = pickle.load(open(f, "rb"))
    print(f, len(r["P_los"]), r["cfg_snapshot"]["_SIONNA_PATCH"]["archive_rev"])
EOF
```

**Gather everything on the second PC for post-processing.** Pull the first PC's urban results into the second PC's `runs/`:

```sh
rsync -av --include='urban_*/' --include='urban_*/results_elev*.pkl' \
      --include='logs/' --include='logs/urban_*.log' --exclude='*' \
      USER@FIRST_PC:/path/to/ntn_3gpp/runs/ runs/
```

`runs/` then holds all 72 results files. The rsync filter copies only `urban_*`, so it cannot overwrite rural data.

**What a results file holds**
- **Per-UE values** (lists of 10000) in top-level keys, for example `los_probability`, `P_los`, `P_nlos_2us`, `DS_ns`, `sigma_nu_Hz`, `P_nlos_panel`, `heading_deg` and `scene_seeds`.
- **Full paths for 1 UE in 50** in `raw_results` (200 per file).
- The run settings in `cfg_snapshot`.

## Contents

- `run.sh`: the queue for one GPU, one scenario and one band.
- `sim/run_stochastic_sim.py`: the stochastic runner, with the per-UE values of this campaign computed during the run.
- `sim/make_runs.py` and `sim/base_config.json`: rebuild `runs/` exactly (configs and scene lists).
- `runs/`: run inputs for both scenarios. Group files carry an `@ROOT@` placeholder that `run.sh` fills in.
- `Ray_Tracing/src/`: the repo modules the runner imports, as of 2026-10-14. `Ray_Tracing/sionna_patch/` holds the patch archive the check verifies against.
- `site/sionna/`: Sionna RT 1.2.1 with the three project patches.
- `post/`: post-processing and plot scripts; each takes `RUNS` (a run folder, or `dataset/<run>`) and writes to `RUNS/figures/`.
- `dataset/`: the processed dataset (see Dataset below).

**Settings worth knowing**
- **Rural solves 15 UEs at a time.** The rough rural ground (10 cm) makes about 0.5M diffuse paths per UE, and Sionna's 10M-path buffer drops paths without an error when it overflows; a chunk that fills the buffer is redone at half the size.
- **Patch fixed 2026-10-14.** Before then, the beam-cone width was rounded in float32. That biased diffuse power by about 1–2 dB at 10° and ≤ 0.2 dB elsewhere. Results with `archive_rev` 2026-09-24 have this bias.
- **Urban results made with the old patch were not rerun.** They are urban S (all elevations) and urban Ka 10°–70°. Urban Ka 80°–90°, urban Ku and all rural use the fixed patch. In post-processing, keep in mind the diffuse-power bias at 10° (and to a lesser degree 20°) in urban S and Ka. Each file's `archive_rev` shows which patch made it. The results-file check in section 3 therefore expects 2026-09-24 for those urban files.

## Dataset

**Processed, in git (5 MB):** `dataset/with_diffuse/figures/` (the main run) and `dataset/without_diffuse/figures/` (the same scenes and UEs with diffuse scattering off). The `figures/` level is kept because the post scripts take `dataset/<run>` as `RUNS` and read `RUNS/figures/`.

| Files (`_urban`/`_rural` per scenario, `_Ku`/`_Ka` per band, no band suffix for S) | Content |
|---|---|
| `metrics_*.csv` | P_LoS, K and delay spread per state vs elevation, with the 25/75 % spread between scenes |
| `k_3gpp_*.csv`, `ds_3gpp_*.csv` | K and log10 delay spread, 3GPP definition (each UE's value, then the mean over UEs per state) |
| `table2_extra_*.csv`, `table2_elev30.csv` | angle spreads, clutter loss, shadow fading, cluster count, Doppler spread; the summary table at 30 deg |
| `generator_panel_k_*.csv` | K at the UE panel: ray tracing, and the TR 38.811 generator fed with the ray-traced K, ASA and ZSA, without and with the mean zenith AoA offset from the LoS |
| `model_*.csv` | the geometry-stochastic model's values, for comparison |
| `pdp_*.csv`, `mech_doppler_*.csv` | power delay profiles; power share by mechanism and Doppler spread |
| `*.png` | the figures made from these files |

**Raw per-UE results, not in git (216 GB):** with diffuse (211 GB) and without (5.1 GB), laid out as in section 3 (`<scenario>_<band>[_<pol>]/results_elevXXXdeg.pkl`). Available from the authors on request.

**External inputs for some post scripts (not included):** the geometry-stochastic model in `chiu_roy_model/` at the repo root (`post/model_panel_k.py`, `post/model_ds_3gpp.py`, `post/plot_mech_doppler.py`, `post/plot_pdp.py`), and an ns-3 source tree for the TR 38.811 tables (`NS3_ROOT`, default `./ns-3-dev`; `Ray_Tracing/src/utils/ns3_reference.py`).
