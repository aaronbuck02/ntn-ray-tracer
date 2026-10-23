# NTN channel modeling with Sionna RT

Ray-traced LEO satellite non-terrestrial-network channels, distilled into
3GPP TR 38.811-style large-scale parameters.

The pipeline is generic: give it a scene and a satellite geometry, and it
returns per-receiver channel statistics in the parameter domains 3GPP tabulates,
plus a scored comparison against the TR 38.811 reference tables. Scenes come
either from a fixed 3-D model or from a stochastic generator.

```
scene  ──►  RT solve  ──►  per-path CIR  ──►  LSP distillation  ──►  vs TR 38.811
             (Sionna)      channel_utils      lsp_stats            validate_3gpp
```

## Relation to IEEE SA P1944

This code and dataset support the NTN channel modeling subgroup of
IEEE SA P1944 (Standard for Channel Model for Wireless Systems).
- Project page: https://standards.ieee.org/ieee/1944/11261/
- FunLab P1944 page: https://wp.ece.uw.edu/funlab/sa-p1944/

## Branch `sionna-ntn-3gpp`: ray-traced update of the TR 38.811 tables

This branch is `sionna-ntn-dev` plus `ntn_3gpp/`: the ray-tracing campaign that
re-derives the TR 38.811 channel tables (K-factor, delay spread, angle spreads,
clutter loss, shadow fading, cluster count) and fills the gaps 6G needs (Ku band,
K at a directional UE antenna, Doppler spread from satellite motion).

| Path | What it is |
|---|---|
| `ntn_3gpp/sim/`, `ntn_3gpp/run.sh`, `ntn_3gpp/runs/` | the runner, its launcher, and the exact run inputs (configs and scene lists) |
| `ntn_3gpp/Ray_Tracing/`, `ntn_3gpp/site/` | the repo modules (as of 2026-10-14) and the patched Sionna RT 1.2.1 the runs used; the folder is self-contained |
| `ntn_3gpp/post/` | post-processing and plots: per-UE K and delay spread (3GPP definition), the summary table, K at the UE panel against the TR 38.811 generator, the comparison with the geometry-stochastic model |
| `ntn_3gpp/dataset/` | the processed dataset (below) |
| `ntn_3gpp/README.md` | how to run, check and collect a campaign |

**Campaign:** urban and rural scenes from ITU-R P.1410 statistics, 10 scenes x
1000 UEs, S 2 GHz / Ku 12 GHz / Ka 20 GHz, elevation 10-90 deg, satellite at
600 km; Sionna RT with up to 3 bounces, edge diffraction and diffuse scattering,
plus a second run with diffuse scattering off.

**Dataset**
- **In git, processed (5 MB):** `ntn_3gpp/dataset/with_diffuse/figures/` and
  `ntn_3gpp/dataset/without_diffuse/figures/`: every CSV the figures and the
  summary table are made from (per scenario, band, elevation and LoS/NLoS
  state), and the figures. The `figures/` level is kept because the post
  scripts take `dataset/<run>` as their `RUNS` argument and read `RUNS/figures/`.
- **Not in git, raw per-UE results (216 GB):** with diffuse (211 GB) and
  without (5.1 GB). One `results_elevXXXdeg.pkl` per scenario, band (S: one
  folder per UE port, V and H) and elevation, each with the per-UE values of
  10,000 UEs and the full paths of every 50th UE. Available from the authors
  on request.

## Install

**Sionna must be patched first.** On a stock `sionna-rt 1.2.1` the pipeline
produces wrong line-of-sight statistics — silently, with no error. A
reinstall reverts the patch, so this step repeats after any dependency change.

```bash
pip install sionna-rt==1.2.1 mitsuba drjit numpy scipy matplotlib pandas sgp4
SP=$(python -c "import sionna, os; print(os.path.dirname(sionna.__file__))")
cp -r sionna_patch/sionna-1.2.1/rt "$SP"/
python -m Ray_Tracing.src.sionna_patch_check          # must end with ok
```

Every runner calls `sionna_patch_check.require()` at startup and refuses to run
if either patch is missing. A CUDA Dr.Jit backend is required — the beamforming
cone path is CUDA-only. See [docs/sionna-patches.md](docs/sionna-patches.md).

Modules import as `Ray_Tracing.src.*`, so the **parent** of this repository must
be on `sys.path`. Entry-point scripts arrange that themselves; for interactive
use, run from the parent directory or add it to `PYTHONPATH`.

## Running

### Fixed scene, swept over elevation

```bash
CUDA_VISIBLE_DEVICES=0 python src/simulations/run_elevation_sim.py \
    --elevation 30 --gpu_id 0 \
    --config src/simulation_configs/sim_config.json \
    --rx_file src/simulation_configs/rx_positions.pkl \
    --output ./Results/my_run
```

`--config`, `--rx_file` and `--output` are required. One process per elevation,
one GPU each; `CUDA_VISIBLE_DEVICES` must be set before the process starts so
Dr.Jit takes full control of the device. `--resume` / `--no-resume` control
checkpoint reuse.

Pass `--waypoints_file` and `--waypoint_index` to place the transmitter from a
real SGP4-propagated pass instead of a synthetic elevation. Without them there
is no transmitter velocity, so no Doppler is computed.

### Generated scenes, ensemble

```bash
python src/utils/scene_gen_ppp.py --out ./ppp_scenes --size 500 500 \
    --cell-res 30 --alpha 0.3 --gamma 20 --seed 0

CUDA_VISIBLE_DEVICES=0 python src/simulations/run_stochastic_sim.py \
    --elevation 30 --gpu_id 0 \
    --config src/simulation_configs/sim_config.json \
    --scenes_manifest ppp_scenes/scenes_manifest.pkl \
    --output ./Results/my_ensemble
```

`--scenes_manifest` replaces `--rx_file`. The ensemble varies over both scene
morphology and receiver position; the seeding contract keeps a given receiver's
statistics pairable across elevations. See
[docs/pipelines.md](docs/pipelines.md).

### Real satellite passes

For a fixed-UE Canary Wharf sweep selecting real TLE passes near 30°/60°/90° peaks, full-pass RX
SNR and 10° elevation-bin threshold percentages, see [docs/pass-snr.md](docs/pass-snr.md).

```bash
python src/utils/fetch_tle.py                       # cache the TLE catalog
python src/utils/generate_pass_waypoints.py --reference --min_elev 10 \
    --elev_step 10 --output waypoints.pkl
```

`fetch_vtec.py` caches IGS ionosphere maps, needed only if the ionospheric
model is enabled.

### Scoring against TR 38.811

```bash
python -m Ray_Tracing.src.utils.validate_3gpp
python -m Ray_Tracing.src.diagnostics.test_cp_smoke
python -m Ray_Tracing.src.sionna_patch_check --selftest
```

## Documentation

| Doc | Covers |
|---|---|
| [docs/pipelines.md](docs/pipelines.md) | the runners, scene generation, the seeding contract, results schema |
| [docs/physics.md](docs/physics.md) | orbital geometry and frames, ionosphere, troposphere, polarization |
| [docs/analysis.md](docs/analysis.md) | LSP distillation, clustering, the TR 38.811 reference parser and its traps |
| [docs/sionna-patches.md](docs/sionna-patches.md) | the occlusion-guard defect, ray-cone beamforming, ITU range widening |
| [docs/scene_classification.md](docs/scene_classification.md) | the NTN scene-classification dataset: classes, per-scene draws, launcher, dataset schema, analysis |
| [sionna_patch/README.md](sionna_patch/README.md) | patch archive layout and re-apply procedure |
| [validation/METHODS.md](validation/METHODS.md) | validation procedures and acceptance criteria |

## Modules

**`src/simulations/`** — `run_elevation_sim.py` (one fixed scene per elevation),
`run_stochastic_sim.py` (generated-scene ensemble per elevation).

**`src/utils/`** — scene: `scene_gen_ppp.py` (generator, block and wall
primitives), `scene_utils.py` (loading, occupancy grid, receiver placement,
antennas). Physics: `orbital_utils.py` (SGP4, frames, passes, waypoints),
`ionospheric_utils.py`, `atmospheric_utils.py` (both off by default),
`circular_polarization.py`, `channel_utils.py` (CIR statistics, power split, TX
placement). Distillation: `lsp_stats.py`, `cluster_stats.py`, `cluster_id.py`.
Reference and scoring: `ns3_reference.py` (TR 38.811 tables parsed from ns-3
source), `validate_3gpp.py`. Inputs: `fetch_tle.py`, `fetch_vtec.py`,
`generate_pass_waypoints.py`. Infrastructure: `checkpoint_utils.py`,
`itu_range_patch.py`. Scene-classification sweeps: `scene_classes.py` (class
medians, per-scene draws, calibration gate), `variant_manifest.py` (streamed
scene manifests, LHS draws, ITU material anchors), `orbit_sampling.py`
(per-scene real-TLE passes), `sweep_utils.py` (multi-GPU launcher),
`scene_dataset.py` (per-link and per-scene LSP tables), `scene_classify.py`
(scene-grouped classifiers), `sensitivity.py` (results loaders, scene bootstrap,
38.811 distances).

**`src/`** — `sionna_patch_check.py` (verifies the patched Sionna is loaded);
`custom_drivers/scene_class_dataset.ipynb` drives the dataset sweeps and
`analysis/scene_classification.ipynb` analyses them (see
[docs/scene_classification.md](docs/scene_classification.md)).
**`src/diagnostics/`** — `test_cp_smoke.py` (the executable check for the
polarization patterns).

Also in this repository, outside the general pipeline and not documented here:
`src/scenario_generation/` (builders for specific measured scenes),
`src/ns3_integration/` (ns-3 export), `chiu_validation/` (validation against an
analytical reference model), and `matlab_ntn_rt/` (a MATLAB port).

## Configuration

One JSON per campaign under `src/simulation_configs/`. The keys that matter:

| Key | Example | Note |
|---|---|---|
| `FREQUENCY` | 20e9 | the TR 38.811 band is chosen by a hard 13 GHz threshold |
| `SLANT_DIST_M` | 500e3 | fixed range; set `SWEEP_ALTITUDE_KM` for the true per-elevation range |
| `SCENE_RADIUS` | 230.5 | also sets the beamforming half-angle |
| `RX_HEIGHT` | 1.5 | |
| `N_RX_POSITIONS` | 10000 | `N_RX_PER_SCENE` for the ensemble runner |
| `MAX_DEPTH` | 3 | interaction order |
| `SCATTERING_COEFF` | 0.4 | **Sionna defaults this to zero if unset** — a silent-failure trap |
| `TX_POL_TYPE` / `RX_POL_TYPE` | RHCP | |
| `BEAMFORMING` | true | requires the patched Sionna |
| `IONOSPHERE_ENABLED` / `ATMOSPHERE_ENABLED` | false | off when comparing to TR 38.811, which carries them as separate link-budget terms |
| `DOPPLER_ENABLED` | false | gated on `--waypoints_file`; a static sweep has no TX velocity |
| `CHECKPOINT_INTERVAL` | 200 | |
| `RNG_SEED` | 42 | does **not** seed Sionna's solver — see [docs/analysis.md](docs/analysis.md) |

## Results

Each run writes its own directory containing, per elevation,
`checkpoint_elev{NNN}deg.pkl` (resumable) and `results_elev{NNN}deg.pkl`
(final). Both carry the same per-receiver arrays plus a full config snapshot;
the schema is in [docs/pipelines.md](docs/pipelines.md).

Results also carry a `sionna_patch` provenance record, so a file produced
without the patches is identifiable afterwards. **Patched and unpatched results
are not comparable.**

## License

Apache License 2.0, the same license as Sionna RT; see [LICENSE](LICENSE).
The bundled Sionna RT 1.2.1 sources (`sionna_patch/sionna-1.2.1/` and, on the
`sionna-ntn-3gpp` branch, `ntn_3gpp/site/sionna/`) are Copyright (c) 2021-2025
NVIDIA CORPORATION & AFFILIATES and are modified by the patches in
`sionna_patch/diffs/`.
