# Pipelines

Three entry points, all writing the same results layout so every downstream
consumer (`checkpoint_utils.load_results`, `lsp_stats`, `cluster_stats`,
`plot_utils`) is shared.

| Runner | Scene | Used for |
|---|---|---|
| `src/simulations/run_elevation_sim.py` | one fixed scene, loaded from disk | a specific site, swept over elevation |
| `src/simulations/run_stochastic_sim.py` | a pre-built ensemble of generated scenes | statistics over a *class* of environments |

Both verify the patched Sionna at startup via
`sionna_patch_check.require()` and record the report in their results — see
[sionna-patches.md](sionna-patches.md).

---

## Fixed scene — `run_elevation_sim.py`

One process per elevation, one GPU each, launched as a subprocess by the site
notebooks. `CUDA_VISIBLE_DEVICES` must be set **before** any GPU import so
Dr.Jit/Mitsuba take full control of that device.

```bash
CUDA_VISIBLE_DEVICES=0 python run_elevation_sim.py \
    --elevation 10 --gpu_id 0 \
    --config sim_config.json \
    --rx_file rx_positions_sieg.pkl \
    --output ./results_sieg_2GHz
```

`--config`, `--rx_file` and `--output` are required. `--waypoints_file` /
`--waypoint_index` switch from a synthetic elevation sweep to a sample of a real
SGP4-propagated pass; without them there is no transmitter velocity, so Doppler
is not computed.

Flow: read config and RX pickle → resume from a per-angle checkpoint if present
→ batched solver loop → checkpoint every `CHECKPOINT_INTERVAL` batches → final
results file.

## Generated ensemble — `run_stochastic_sim.py`

Same structure, but instead of one fixed scene it walks a **pre-generated
ensemble** of independent random scenes (built once by
`scene_gen_ppp.generate_scene_ensemble`) and places `N_RX_PER_SCENE` receivers
drawn uniformly from each scene's free space.

The ensemble therefore varies over **both** urban morphology (scene to scene)
and intra-scene position — the nested sampling the RT-to-stochastic literature
calls for, rather than one UE per scene. A single central UE would be biased:
the generator's origin gate guarantees the scene centre is outdoors, so a
centre-fixed UE samples only unobstructed positions and inflates LoS
probability.

```bash
CUDA_VISIBLE_DEVICES=0 python run_stochastic_sim.py \
    --elevation 10 --gpu_id 0 \
    --config sim_config.json \
    --scenes_manifest ppp_scenes/scenes_manifest.pkl \
    --output ./results_stochastic
```

`--scenes_manifest` replaces `--rx_file`.

### Seeding contract

The UE RNG is seeded from the scene seed alone, never from elevation or carrier:

```
rng_UE = default_rng(RNG_SEED + 7919 * scene_seed)
```

So every elevation and both carriers see an *identical* UE set in each scene.
That is what makes per-UE statistics pairable across a sweep: a receiver's
K-factor change between 10° and 80° is a property of the geometry change, not of
a resampled position.

---

## Scene generation — `src/utils/scene_gen_ppp.py`

Two primitives, selected by `primitive=`:

### `"block"` — building footprints

A 2-D PPP over a regular grid with ITU-R P.1410 heights:

1. Footprint `Lx × Ly` centred on the origin, tiled by square cells of side
   `cell_res` (area `A_c`).
2. Homogeneous PPP of intensity `lam`: `N ~ Poisson(lam · Lx · Ly)` points
   uniform in the rectangle; `k_ij` points land in cell (i, j).
3. Each point carries a fixed unit area `a_p`; the built area of a cell
   saturates: `A_build(i,j) = min(k_ij · a_p, max_fill · A_c)`. Before
   saturation the expected built-up fraction is exactly `alpha = lam · a_p`, the
   P.1410 land-cover parameter. `max_fill < 1` leaves a hairline gap between
   neighbouring saturated cells so adjacent building walls never become
   coplanar — which would give the ray tracer degenerate surfaces.
4. Each occupied cell emits one axis-aligned block of side `sqrt(A_build)` and
   Rayleigh height `p(h) = (h/γ²)·exp(−h²/2γ²)`, optionally clipped.

### `"wall"` — wall segments

An alternative to solid footprints, for comparing against analytical models
that derive their blockage and diffraction terms for a field of finite walls.

`N ~ Poisson(λ_b · π · R²)` wall centres uniform in a disc; each wall an upright
rectangle of fixed length `ℓ_f`, orientation `ξ ~ U[0, π)`, untruncated Rayleigh
height. No exclusion zone about the origin — a wall a few metres from the
terminal is what blocks LoS at high elevation.

Walls are emitted as slabs of finite `thickness`, not zero-thickness plates:
the analytics' UTD coefficient assumes a right-angle roof wedge (n = 3/2), while
a zero-thickness plate is a half-plane (n = 2) and would bias every diffracted
amplitude. Keep thickness well below `ℓ_f` so the slab stays a wall rather than
becoming a building.

Both write a Mitsuba 3 / Sionna RT scene: one ASCII `.ply` per primitive plus a
ground plane, from a `.xml` declaring ITU radio materials with the `mat-` id
prefix Sionna expects. Walls, roofs and ground are separate meshes so each can
carry its own material.

```bash
python src/utils/scene_gen_ppp.py --out ./ppp_scene_001 --size 1000 1000 \
    --cell-res 50 --alpha 0.35 --points-per-cell 8 --gamma 20 --seed 0
```

---

## Results layout — `src/utils/checkpoint_utils.py`

Each elevation runs independently and writes, into its own output directory:

```
checkpoint_elev{angle:03d}deg.pkl   resumable progress
results_elev{angle:03d}deg.pkl      final
```

Both carry the same per-RX arrays, all of length `completed_rx` (or
`n_rx_positions` when finished), plus run identity and a full config snapshot:

| Group | Keys |
|---|---|
| identity | `elevation_deg`, `completed_rx`/`total_rx`, `batch_size`, `timestamp`, `frequency`, `tx_pos`, `slant_dist_m`, `cfg_snapshot`, `perturb_half_m` (results only) |
| power split | `P_los`, `P_nlos`, `P_specular`, `P_diffuse`, `P_diffraction`, `P_refraction`, `K_power_ratio`, `K_moments`, `los_probability` |
| spreads | `tau_rms_mean_s`, `ASD_deg`, `ASA_deg`, `ZSD_deg`, `ZSA_deg` |
| doppler | `doppler_mean_hz`, `doppler_rms_hz`, and the `_td` time-domain pair |
| propagation | `omega_F`, `vtec_sample`, `A_gas_db`, `A_rain_db`, `scint_db`, `atm_total_db` (zero unless the ionosphere/atmosphere models are enabled) |
| scene | `nominal_rx_positions`, `scene_seeds` (stochastic only), `cluster_samples` (site only) |
| raw | one full CIR dict every `RAW_SAVE_INTERVAL` receivers: `rx_nominal`, `taus`, `amps`, `inter_d0`, the power keys, `tau_rms_s`, `h_env`, `has_los` |

Two quirks: `K_moments` is written but never populated with a real value, and
`P_scat` inside `raw_results` duplicates `P_nlos`, kept only so older result
files still load.

Results directories are named per run — `results_sieg_2GHz`,
`results_wharf_du_20GHz`, `chiu_validation/results_urban_cp` and so on — not
collected under a single `results/`.


New sensitivity keys, all absent from older files: `n_paths`, `P_by_depth`
(incoherent power of paths with exactly d interactions, d = 0..`MAX_DEPTH`;
sums to `P_los + P_nlos`), `P_ground`/`P_wall`/`P_roof` (power of paths touching
that surface role at **any** depth, from `paths.objects` and the PPP mesh names;
not additive, zero for site scenes), `K_fit_dB` (moment-method Rician K over
|H(f)|, below) and `rx_height`. Checkpoints that pre-date them resume with NaN
padding.

---

## Scene-classification sweeps — `src/custom_drivers/scene_class_dataset.ipynb`

Drives `run_stochastic_sim.py` over a grid of **variants** × elevations, one
process per free GPU (`sweep_utils.queue_entries` → `start_detached`, tmux
session `scene_class`). Finished (variant, elevation) pairs are skipped and
interrupted ones resume from their checkpoint. The design (classes, per-scene
draws, datasets) is in [scene_classification.md](scene_classification.md);
analysis is in `src/analysis/scene_classification.ipynb`
([analysis.md](analysis.md#scene-classification--scene_datasetpy-scene_classifypy)).

A variant is a dict; the dataset driver uses one per (dataset, carrier, class),
each pointing at that class's manifest (`scene_classes.class_manifest`):

```python
{"name": "2GHz__urban", "elevations": [10, 20, 30, 40, 50, 60, 70, 80],
 "cfg_overrides": {"FREQUENCY": 2e9},                        # any run-wide key
 "manifest": "Results/scene_class/base/_manifests/urban.pkl"}
# or "material_overrides": {...} / "lhs": {"factors": {...}} / "oat": {...}
```

Each writes `Results/scene_class/<dataset>/<name>/` holding `config.json`,
`scenes_manifest.pkl`, `variant.json` and the usual per-elevation files.

**Streaming manifests.** `variant_manifest.virtual_manifest` writes a manifest
with `streaming=True` whose entries describe scenes instead of pointing to files:

```python
{"seed": 17, "sample_kwargs": {...},            # per-scene sampler draws (alpha, cell_res, gamma)
 "cfg_overrides": {"SCENE_EAST_HEADING_DEG": 42.0, ...},
 "ground_params": {...},                        # optional per-scene role params
 "orbit": {10.0: waypoint, 30.0: waypoint, ...},  # per-elevation real-TLE state
 "ground_station": (lat, lon)}
```

For such entries `run_stochastic_sim.py` calls `generate_scene_streaming` into
`<output>/scratch_elevNNN/scene_<seed>/`, solves and deletes it, so no geometry
is cached and the notebook kernel never touches Mitsuba. The same (seed, kwargs)
always gives the same scene, so every variant sees the same scenes and UEs as the
baseline. Manifests with `xml_path` entries still work as before.

**Per-scene TX.** When every entry has an `orbit` waypoint for the run's
elevation, the TX position, beam half-angle and velocity are computed per scene
(`tx_position_from_pass_sample` with that scene's `SCENE_EAST_HEADING_DEG`)
instead of once per run, and `DOPPLER_ENABLED` is honoured without
`--waypoints_file`. Waypoints come from `orbit_sampling.build_orbit_table`
([scene_classification.md § 2.2](scene_classification.md#22-satellite-pass--real-tles-at-a-sampled-peak-elevation)).

**Material override keys**, per role (`wall`, `roof`, `ground`) or `all`:
`base` swaps in an ITU-R P.2040 anchor's (a, b, c, d) from
`variant_manifest.MATERIAL_ANCHORS` and is applied first; `eps_scale`,
`sigma_scale` multiply P.2040 `a`/`c` (so the frequency law is kept); `scattering_coefficient`, `xpd_coefficient`, `thickness`,
`scattering_pattern` (`directive`|`lambertian`|`backscattering`), `alpha_r`,
`alpha_i`, `lambda_` are set absolutely.

**Slab thickness.** The PPP role defaults carry `thickness = 2.0` m. Sionna's
0.1 m default slab adds a thin-film term that moves reflection loss by several
dB depending on ε_r — it would contaminate every ε_r/σ sensitivity. Manifests
from before this default lack the key; `variant_manifest` injects it.

**Common random numbers.** With `SOLVER_SEED_OFFSET` set, the solver seed is a
pure function of (`RNG_SEED`, scene seed, elevation, offset). Variants sharing
an offset see the same ray sample: an identity variant reproduces the baseline
to ~1e-5 relative (float32 reduction order on the GPU), with identical path
sets. Replicates differ only in the offset; their spread is the solver noise
floor. Unset, the legacy per-run RNG stream is used unchanged.

**Per-scene overrides** (`lhs` variants). A manifest entry may carry its own
`wall/roof/ground_params` and `cfg_overrides`; the runner applies them to a
per-scene copy of the config. Only keys read per scene may vary:
`RX_HEIGHT`, `SCATTERING_COEFF`, `RX_ORIENTATION_DEG`, `RX_PITCH_DEG`,
`RX_YAW_DEG`, `RX_BASE_PATTERN`, `POL_BASE_PATTERN`, `TX_POL_TYPE`,
`RX_POL_TYPE`, `SCENE_EAST_HEADING_DEG`; anything else raises.
`variant_manifest.lhs_manifest` layers one Latin-hypercube draw per scene on top
of the entry's existing draws (`oat_uniform_manifest` is the one-factor case).
Factor names route by prefix: `<role>.` / `all.` (material), `cfg.` (config),
`geo.` (scene sampler; `geo.beta` becomes `cell_res`, streaming manifests only).
Specs are `(lo, hi)` for U[lo, hi], `(lo, hi, "int")` for discrete-uniform,
`(lo, hi, "log")` for log-uniform, or `(choices, weights, "cat")` for a weighted
categorical (`weights=None` is uniform; the design column holds the choice
index). `cfg.POL_PAIR` takes `"TX/RX"` choices and sets both polarization keys;
`<role>.base` takes anchor names. The design matrices are concatenated in the
manifest's `design`.

**Link keys** read by `build_scene_batch`: `RX_BASE_PATTERN` (UE element,
default `POL_BASE_PATTERN`; also honoured for linear polarization) and
`RX_ORIENTATION_DEG` (yaw, pitch, roll; pitch −90° turns the array boresight to
zenith). Non-iso UE patterns are only meaningful with the receive-pattern fix
described in [sionna-patches.md](sionna-patches.md): stock Sionna 1.2.1 evaluates
the RX pattern at the last interaction's incidence direction, so every NLoS path
inherits the LoS gain (iso patterns are unaffected to 3e-5). The satellite element pattern is not swept: from 350–550 km the scene
subtends ~0.05°, so it is a constant gain that cannot change K or spreads.
