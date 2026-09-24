# NTN channel modeling with Sionna RT
Extended Sionna RT-based implementation that enables NTN RT simulations. Configured for local GPU utilization, but should be portable to wide variety of configurations.

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
be on `sys.path`, and this repository's directory name must be `Ray_Tracing`.
Clone with an explicit target name:

```bash
git clone <url> Ray_Tracing
```

Entry-point scripts add the parent to `sys.path` themselves; for interactive
use, run from the parent directory or add it to `PYTHONPATH`.

## Running

The command-line runners in `src/simulations/` are the production entry
points. Two notebooks next to them, [src/site_simulation.ipynb](src/site_simulation.ipynb)
and [src/stochastic_simulation.ipynb](src/stochastic_simulation.ipynb), run
exactly the same pipeline interactively and are the recommended way to iterate
on scene setup and per-elevation configuration before launching a full sweep
from the CLI.

| Notebook | Corresponds to |
|---|---|
| [src/site_simulation.ipynb](src/site_simulation.ipynb) | `run_elevation_sim.py` — fixed-scene elevation sweep |
| [src/stochastic_simulation.ipynb](src/stochastic_simulation.ipynb) | `run_stochastic_sim.py` — PPP-ensemble sweep |

### Fixed scene, swept over elevation

```bash
CUDA_VISIBLE_DEVICES=0 python src/simulations/run_elevation_sim.py \
    --elevation 30 --gpu_id 0 \
    --config path/to/sim_config.json \
    --rx_file path/to/rx_positions.pkl \
    --output ./Results/my_run
```

`--config`, `--rx_file` and `--output` are required (bring your own — a config
JSON and a receiver-positions pickle in the schema described below).
One process per elevation,
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
    --config path/to/sim_config.json \
    --scenes_manifest ppp_scenes/scenes_manifest.pkl \
    --output ./Results/my_ensemble
```

`--scenes_manifest` replaces `--rx_file`. The ensemble varies over both scene
morphology and receiver position; the seeding contract keeps a given receiver's
statistics pairable across elevations. See
[docs/pipelines.md](docs/pipelines.md).

### Real satellite passes

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
python -m Ray_Tracing.src.sionna_patch_check --selftest
```

## Documentation

| Doc | Covers |
|---|---|
| [docs/pipelines.md](docs/pipelines.md) | the runners, scene generation, the seeding contract, results schema |
| [docs/physics.md](docs/physics.md) | orbital geometry and frames, ionosphere, troposphere, polarization |
| [docs/analysis.md](docs/analysis.md) | LSP distillation, clustering, the TR 38.811 reference parser and its traps |
| [docs/sionna-patches.md](docs/sionna-patches.md) | the occlusion-guard defect, ray-cone beamforming, ITU range widening |
| [sionna_patch/README.md](sionna_patch/README.md) | patch archive layout and re-apply procedure |

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
`itu_range_patch.py`.

**`src/`** — `sionna_patch_check.py` (verifies the patched Sionna is loaded),
`site_simulation.ipynb` and `stochastic_simulation.ipynb` (the driver
notebooks — see the section above).

## Configuration

One JSON per campaign, passed via `--config`. The keys that matter:

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
