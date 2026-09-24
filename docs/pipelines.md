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

