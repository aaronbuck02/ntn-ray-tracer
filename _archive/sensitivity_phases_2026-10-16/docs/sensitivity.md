# Sensitivity-analysis pipeline

This pipeline measures how the RT → TR 38.811 large-scale parameters (K-factor,
delay and angular spreads, LoS probability) respond to four groups of inputs:
solver settings (interaction depth, mechanisms), materials (ε_r, σ, scattering
coefficient S, scattering lobe α_R, XPD), link parameters (polarization, UE
antenna, UE height), and the environment itself (P.1410 building density, size
and height; street orientation; satellite pass).

Three design rules run through it:

- **Every sampled parameter is uniform.** Continuous factors are U[lo, hi].
  Integers and categories are discrete-uniform (equal mass per value). Draws are
  Latin-hypercube, one per scene.
- **Scenes and orbits are sampled, not fixed.** Each scene has its own P.1410
  (α, β, γ), street heading and area-uniform ground location. At each elevation
  it gets a *real* Starlink pass (cached TLE catalog, SGP4) that peaks at a
  sampled elevation.
- **Scenes are streamed, never cached.** A manifest only *describes* scenes. Each
  runner subprocess builds a scene in scratch, solves it and deletes it. The
  notebook kernel never imports Mitsuba or Sionna, which is what used to crash it.

It is built on the stochastic pipeline (`scene_gen_ppp` → `run_stochastic_sim`
→ `lsp_stats`). Mechanics are in
[pipelines.md § Sensitivity sweeps](pipelines.md#sensitivity-sweeps--srccustom_driverssensitivity_simulationipynb)
and statistical methods in [analysis.md § Sensitivity analysis](analysis.md#sensitivity-analysis--sensitivitypy).

```
custom_drivers/sensitivity_simulation.ipynb (kernel: no Mitsuba/Sionna)
  factor tables (all uniform) ─► make_base: virtual_manifest + lhs_manifest(scene draws)
                                   + orbit table  ◄── orbit_sampling CLI (CPU pool, real TLEs)
  phaseN() -> [variant, ...]  ─► sweep_utils.launch_grid ── write_variant (config, manifest)
                                        │  one subprocess per GPU, per variant × elevation
                                        ▼
  run_stochastic_sim.py: for each scene entry
      stream scene into scratch ─► per-scene TX from its orbit ─► solve ─► delete scene
        └─► Results/sensitivity/<phase>/<variant>/results_elevNNNdeg.pkl
                                        │
analysis/sensitivity_analysis.ipynb ◄───┘  load_sweep, noise floors, oat_response,
                                           geometry_variance, sobol_from_design, dist_metrics
```

---

## 1. The problem the design solves

Any LSP statistic, such as μ_K at 30°, is a random variable whose spread has
several sources:

```
Var(LSP) =  Var due to the material/link parameters we sweep   <- what we want to measure
          + Var due to the environment (scene draw, orbit)      <- which city, which pass
          + Var due to UE position within a scene               <- where in that city
          + Var due to the ray-tracer's Monte Carlo             <- which rays were launched
```

A naive study would redraw everything for each parameter value. A 0.5 dB
"effect" of ground ε_r could then be just a different city, pass, UE set or
ray sample. The design controls each source separately:

| Source | How it is randomised | How the sweep controls it | Where it is measured |
|---|---|---|---|
| Environment | Uniform P.1410 draws, heading and real-TLE pass per scene (§2.1, §2.2) | **Frozen per scene**: every variant streams the identical scene (same seed and kwargs) with the identical pass | Phase 3 (explained vs residual between-scene share), Phase 4 (Sobol), all CIs |
| UE position | Uniform over each scene's free space, seeded by scene seed (§2.3) | **Frozen**: same UEs in every variant, elevation and carrier | Phase 3 (within-scene share) |
| Ray sampling | Sionna shoot-and-bounce Monte Carlo (§2.4) | **Common random numbers** via `SOLVER_SEED_OFFSET` | Phase 0 (noise floor) |
| Swept parameter | Uniform per scene: one factor at a time (Phase 2) or all jointly (Phase 4) (§2.5) | the thing under test | Phases 2, 4 |

Because the nuisance sources are frozen across variants, variant − baseline
differences are **paired**. The leftover ray-sampling noise is measured in
Phase 0 and used as a gate.

---

## 2. Randomisation processes

### 2.1 Scene geometry — uniform over P.1410 environments

Every scene, in every phase, draws its environment from these uniform ranges
([`lhs_manifest`](../src/utils/variant_manifest.py#L146) over `SCENE_FACTORS`,
simulation notebook §3):

| Factor | Distribution | Basis |
|---|---|---|
| `geo.alpha` built-up fraction | U[0.10, 0.50] | ITU-R P.1410 suburban 0.1 to dense urban 0.5 |
| `geo.beta` buildings per km² | U[300, 750] | P.1410 dense urban 300 to suburban 750 |
| `geo.gamma` Rayleigh height scale | U[8, 20] m | P.1410 suburban 8, urban 15, dense urban 20 |
| `cfg.SCENE_EAST_HEADING_DEG` street-grid orientation vs north | U[0, 90)° | Real grids are not north-aligned, and the grid repeats every 90° |

Given those, [`sample_ppp_scene`](../src/utils/scene_gen_ppp.py#L86) realises the
scene from `default_rng(seed)`:

1. **Random grid shift.** The grid is offset by U(0, cell_res) in x and y.
2. **Point process.** N ~ Poisson(λ · Lx · Ly) uniform points, with 4 per cell on
   average. A cell receiving k points gets built area min(k·a_p, 0.8·A_c), where
   a_p = α·A_c/4. So E[built fraction] = α until cells saturate.
3. **Building count.** β is set through the grid spacing:
   `cell_res = sqrt((1 − e⁻⁴)·10⁶/β)` (36–57 m), since 1 − e⁻⁴ ≈ 98% of cells
   hold a building ([`_geo_kwargs`](../src/utils/variant_manifest.py#L137)).
4. **Heights.** Rayleigh(γ), truncated to [4, 120] m by inverse-CDF sampling
   ([`rayleigh_heights`](../src/utils/scene_gen_ppp.py#L197)).

**Why 4 points per cell and a 120 m cap.** With the earlier 1 point per cell,
α saturated (nominal 0.45 gave 0.377 realised) and only building *size* varied;
the count was fixed at 63% of cells. A 60 m cap also clipped 13.5% of heights at
γ = 30. With the current settings, realised values are unbiased. Over 200
500 m scenes:

| | Mean offset from the drawn value | Per-scene spread |
|---|---|---|
| α | −0.004 | 0.015 |
| β | −0.1% | 4.8% |
| fitted γ | +2.4%, from the 4 m floor at low γ | — |

The remaining spread is genuine finite-scene Poisson variation. The runner
records the realised `itu_alpha`, `itu_beta` and `itu_gamma_fit`
([`itu_parameters`](../src/utils/scene_gen_ppp.py#L292)) for every UE.

Materials are not baked into the geometry. The XML names a dummy ITU material per
role, and the EM parameters come from the manifest at solve time. That is what
lets material variants reuse the same scenes.

### 2.2 Satellite pass — real TLEs at a sampled peak elevation

The old setup placed the satellite at a fixed azimuth (180°) and a fixed
500 km slant range, with no velocity. Combined with axis-aligned streets, that
locked every result to one street orientation, and the range was unphysical at
low elevation. Now each (scene, elevation E) gets a real pass
([`orbit_sampling.py`](../src/utils/orbit_sampling.py)):

| Draw | Distribution |
|---|---|
| Ground location | **Area-uniform** over \|lat\| ≤ 60°: lat = arcsin(U[−sin 60°, sin 60°]), lon ~ U[−180°, 180°) ([`sample_ground_points`](../src/utils/orbit_sampling.py#L24)). One per scene, shared by its elevations. |
| Peak elevation | U[max(30°, E + 1°), 90°] per (scene, E) ([`sample_peaks_and_legs`](../src/utils/orbit_sampling.py#L31)) |
| Pass leg | discrete-uniform {rising, setting} |

Then [`scene_passes`](../src/utils/orbit_sampling.py#L74) runs **one catalog
scan per scene**: the [`make_real_target_passes`](../src/utils/pass_snr.py#L6)
over the cached `tle_catalog/starlink.tle`, using the catalog's median TLE epoch,
a 6 h window and a 1° tolerance. For each target peak it picks the real
satellite pass nearest it.
[`waypoint_at_elevation`](../src/utils/orbit_sampling.py#L48) then solves the
exact instant el(t) = E on the drawn leg (`brentq` over SGP4) and returns that
satellite's azimuth, slant range and velocity.

The satellite state is therefore not drawn directly: it follows from a real orbit.

- **Coverage.** Every elevation 10–80° gets every scene. The +1° margin ensures
  the matched pass (within 1°) still reaches E.
- **Failure handling.** If no pass is within tolerance, the window widens to 24 h.
  If that still fails, the location is redrawn from a seeded stream, and the
  redraw count is stored.
- **Measured behaviour.**
  - About 10 s per scene scan.
  - Peaks within 0.12° of target, up to 88°.
  - el = E to 1e-6°.
  - Altitudes are the real Starlink shells, 350–550 km, including direct-to-cell.
  - Inclinations of 43°, 53°, 70° and 97.6° all appear.
- **Run-time check.** The LoS ray arrives at the waypoint's azimuth after the
  heading rotation (to 0.01°), and LoS Doppler matches −f·ṙ/c from SGP4 to 3 Hz.

[`build_orbit_table`](../src/utils/orbit_sampling.py#L108) parallelises the scans
over a CPU process pool in a CLI subprocess and writes a small `orbit_table_*.pkl`
holding the catalog's SHA-256. The table holds orbits, not scenes, so caching it
is safe. It is built once per ensemble and shared by all variants.

### 2.3 UE placement

Per scene, an exact outdoor mask is built from the footprints
([`free_space_grid`](../src/utils/scene_gen_ppp.py#L853)): a 2 m grid, 1 m wall
clearance and a 20 m edge margin. `N_RX_PER_SCENE` cells are drawn, each jittered
by ±1 m ([`sample_rx_positions`](../src/utils/scene_utils.py#L422)), with the RNG
seeded by the **scene seed alone**
([`run_stochastic_sim.py:943`](../src/simulations/run_stochastic_sim.py#L943)).
Every variant, elevation and carrier therefore sees the same UEs. Streamed and
file-based scenes give the same UEs, as checked.

### 2.4 Ray sampling and common random numbers

Sionna's solver is Monte Carlo, so a single UE's K can move by up to ~10 dB
between seeds ([analysis.md](analysis.md#solver-monte-carlo-noise)). With
`SOLVER_SEED_OFFSET` set, the seed is a hash of identity
([L998](../src/simulations/run_stochastic_sim.py#L998)):

```python
SeedSequence([RNG_SEED, scene_seed, round(100·elev), SOLVER_SEED_OFFSET])
```

- **Same offset:** the same rays per scene (common random numbers). An identity
  variant reproduces the baseline to ~1e-5.
- **Different offsets:** independent replicates, whose spread is the noise floor.
- **Resume-safe**, because the seed depends only on identity.

### 2.5 Material, link and propagation-environment parameters — uniform

| Factor | Distribution | Baseline (x0) |
|---|---|---|
| `{ground,wall,roof}.eps_scale` (× P.2040 `a`) | U[0.5, 1.5] | 1 |
| `{ground,wall,roof}.sigma_scale` (× P.2040 `c`) | U[0.5, 1.5] | 1 |
| `{ground,wall,roof}.scattering_coefficient` | U[0, 0.6] | 0.4 |
| `all.alpha_r` | discrete-uniform {1, …, 10} | 1 |
| `all.xpd_coefficient` | U[0, 0.5] | 0 |
| `cfg.RX_HEIGHT` | U[1, 10] m | 1.5 m |
| `cfg.RX_PITCH_DEG` / `cfg.RX_YAW_DEG` (tr38901 UE) | U[0, 90]° / U[0, 360)° | off until `ORIENTATION_READY` |
| `cfg.VTEC_MEAN_TECU` (mean of the per-scene log-normal VTEC, σ 5 TECU) | U[2, 60] TECU | 10 |
| `cfg.RAIN_R_MM_H` | U[0, 50] mm/h | 0 |
| `cfg.ATM_RHO_G_M3` (surface water vapour) | U[3, 20] g/m³ | 7.5 |

Notes on specific factors:
- **α_R** is integer-only, because Sionna casts it with `int()`. The LHS maps
  u → `floor(lo + u·(hi − lo + 1))`, so each integer has mass 1/10. The earlier
  rounding gave the endpoints half weight.
  [`to_unit`](../src/utils/sensitivity.py#L320) maps each integer back to the
  centre of its stratum for the PCE.
- **Ionosphere and troposphere** are on in every variant (`docs/physics.md`).
  Each scene draws its own VTEC (log-normal) and scintillation fade. The draws
  are seeded by (scene seed, elevation, stream) alone, so they are identical
  across variants, carriers and resumes. VTEC sets Faraday rotation, a roll of
  the TX before the trace, so it is part of the solve key. Rain, water vapour and
  scintillation scale every path equally after the trace, so they are UE-side
  and share one solve. Uniform losses leave K, delay spread and angular spreads
  exactly unchanged, and CP is invariant to Faraday rotation, so for the RHCP
  baseline these factors show only in received power: `PG_dB`, the incoherent
  path gain after all corrections. `phase2_env` adds a V/V VTEC sweep, where
  Faraday rotation can move K. 38.811 carries these losses separately, so
  compare its tables on the LSPs, not on `PG_dB`.
- **Pitch/yaw** are applied by the `scene_utils_sensitivity` fork but need the Sionna receive-pattern fix
  (§7). The runner turns them into Sionna orientation [yaw, −pitch, 0].

**Not sampled continuously, and why:**
- **Categorical settings:** polarization pair, UE element type and scattering-pattern
  kind. Each category is one equal-weight variant (discrete-uniform over
  categories). They are kept out of the PCE, which needs ordered inputs.
- **Numerical settings:** `MAX_DEPTH`, diffuse/diffraction on/off, and slab
  thickness. These are convergence checks, not uncertainties: Phase 1, plus one
  0.1 m thickness check.
- **Scenario axes:** carrier (2 and 20 GHz) and target elevation. These are
  separate studies, as 38.811 tabulates them.

**How draws are layered.** [`lhs_design`](../src/utils/variant_manifest.py#L111)
gives an n_scenes × d Latin hypercube in which each factor's range is cut into
n_scenes strata, each hit once. [`_route`](../src/utils/variant_manifest.py#L122)
sends each factor by name prefix (`geo.`, `cfg.`, `<role>.`/`all.`).
[`lhs_manifest`](../src/utils/variant_manifest.py#L146) layers new draws on top of
the entries' existing scene draws, and concatenates the design matrices so the
analysis sees every factor of every scene. The runner applies per-scene keys
through [`_scene_cfg`](../src/simulations/run_stochastic_sim.py#L442). Only keys
in `PER_SCENE_CFG_KEYS` ([L198](../src/simulations/run_stochastic_sim.py#L198))
may vary per scene; anything else raises.

---

## 3. Streaming scenes (the kernel-crash fix)

Pre-generating an ensemble ran `verify_scene`, a Mitsuba load, for every scene
inside the notebook kernel, and the kernel died. Now:

- **Manifests describe scenes; they never point to files.**
  [`virtual_manifest`](../src/utils/variant_manifest.py#L37) writes entries of
  `{seed, sample_kwargs, cfg_overrides, role params, orbit, ground_station}`.
- **The runner streams each entry.**
  [`_scene_geometry`](../src/simulations/run_stochastic_sim.py#L476) calls
  [`generate_scene_streaming`](../src/utils/scene_gen_ppp.py#L686) into
  `<variant>/scratch_elevNNN/scene_<seed>/`, which is unique per process. The
  spec comes straight back for UE placement. The scene is deleted after the solve
  ([L1034](../src/simulations/run_stochastic_sim.py#L1034)), and the scratch root
  at the end of the run.
- **Per-scene TX.** [`_scene_tx`](../src/simulations/run_stochastic_sim.py#L465)
  computes the TX position, beam half-angle and velocity for each scene from its
  orbit waypoint and heading
  ([`tx_position_from_pass_sample`](../src/utils/channel_utils.py#L228),
  [`tx_velocity_scene_frame`](../src/utils/channel_utils.py#L243)). Doppler can
  be enabled with `DOPPLER_ENABLED`; it is off by default because time steps
  multiply GPU memory.
- **Verification** runs on the first streamed scene of each run (`PPP_VERIFY_EVERY`
  adds more), inside the subprocess.

Measured: streaming costs 0.02 s per 500 m scene (1 s with verification), against
about 55 s of ray tracing. A streamed scene reproduces its file-based twin to
≤7e-5 with identical UEs. Over 50 streamed scenes the runner's RSS stayed flat
(673 → 702 MB), and scratch was empty at the end. File-based manifests still work
unchanged, so `stochastic_simulation.ipynb` can adopt `virtual_manifest` to fix
the same crash.

---

## 4. Experiment phases

The phases are defined in [`sensitivity_simulation.ipynb`](../src/custom_drivers/sensitivity_simulation.ipynb)
§5, one function per phase; `per_freq()` runs each at 2 and 20 GHz. All phases
use a base ensemble of sampled scenes with real passes (`make_base`, §4 of the
notebook).

| Phase | Question | Design | Analysis |
|---|---|---|---|
| `phase0` | Solver noise? CI widths? Estimator artefact? Distance to 38.811? | Base ensemble at 8 elevations + 4 seed replicates at 10/30/60° | `noise_floor`, `noise_floor_scene`, `summary_table`, `estimator_table`, `dist_metrics(by_class)` |
| `phase1` | Where does interaction depth converge? | `MAX_DEPTH` 1–5 × diffuse × diffraction | power by depth, μ_K, μ_lgDS, n_paths vs depth |
| `phase2_materials` | How far does each material uncertainty move the LSPs? | **Uniform one-at-a-time**: one variant per factor in §2.5, drawn per scene with the others at baseline; plus Lambertian and 0.1 m thickness checks | `oat_response` swings, response curves |
| `phase2_env` | How much do ionosphere, rain and water vapour move received power (and K under linear pol)? | Uniform OAT on VTEC mean, rain rate and water vapour, plus VTEC under V/V | `oat_response` on `PG_dB`, K |
| `phase2_link` | Antenna/polarization/height vs environment? | Uniform OAT on UE height (and orientation when ready); one variant per polarization pair and UE element | `oat_response`, `oat_effects` |
| `phase3` | Where does the variance come from? | Base ensemble with 300 UEs per scene | `geometry_variance`, `variance_components` |
| `phase4` | Global ranking and interactions | 1000 sampled scenes; every continuous factor drawn jointly | `sobol_from_design` |

**Uniform OAT instead of fixed levels.** Each scene draws the factor from its
whole range, so Phase 2 measures a *response curve* Δ(x) rather than two points.
It also costs one variant per factor instead of two or three.

**Satellite antenna not swept.** From 350–550 km the scene subtends ~0.05°, so
the satellite pattern only contributes a constant gain.

**Launching detached.** The notebook's launch cell queues every job of the phases
in `RUN` and, with `DETACH`, hands the queue to
[`start_detached`](../src/utils/sweep_utils.py): it writes
`Results/sensitivity/_queue.json` and runs the scheduler
(`python -m Ray_Tracing.src.utils.sweep_utils <queue>`) in the tmux session
`sweep`. The sweep then outlives the notebook kernel and the SSH login. Follow it
with `tail -f Results/sensitivity/_queue.log` or `tmux attach -t sweep`. Re-running
the cell after a crash skips finished jobs and resumes the rest from checkpoints.

**Shared solves.** [`launch_grid`](../src/utils/sweep_utils.py) groups jobs by
[`solve_key`](../src/utils/sweep_utils.py): a hash of the config and manifest
with the UE-side keys (`UE_KEYS`: TX/RX polarization, RX height, orientation,
`N_RX_PER_SCENE`) removed. Variants with the same key and elevation run as one
runner process (`--group`, files in `<sweep>/_groups/`). Each scene is streamed
and ray-traced once for all of them, and every variant still gets its own
`results_elevNNNdeg.pkl` (the log is a symlink to the group log). In `phase2_link`
the height OAT and the four `cat_pol_*` variants form one group; `cat_ue_*`
change the element pattern and stay separate. Other phases change the seed,
materials or solver settings, so each of their variants is its own group.

- **Receivers.** Every member's UEs go into one receiver list. UEs with the same
  position and orientation are solved once. UE xy positions depend only on the
  scene seed, so the pol variants share their 100 receivers.
- **Polarization.** If a group has a single TX/RX polarization pair, the
  arrays are built exactly as before. If a side has several, that side gets a
  dual-pol `[V, H]` element. Each member then reads its pair out of the 2×2
  coefficients ([`_pol_weights`](../src/simulations/run_stochastic_sim.py#L501)):
  linear in the TX Jones vector, and conjugate-linear in the handedness-flipped
  RX vector. With dual-pol, a path counts as present if it exists for any
  pattern pair, so `has_los` is geometric even for cross-pol pairs.
- **Chunks.** Receivers are solved in chunks of `RX_CHUNK` (0 means all at once),
  all with the scene's solver seed. A Dr.Jit OOM halves the chunk and retries.
  The scene is built once per chunk, which costs ~0.2 s; the solve itself is
  0.2–0.7 s with merged shapes. Each scene's log line gives the cap fill and the
  build/solve/readout seconds of every chunk.
  Sionna pads its per-path tensors to the busiest receiver, which is what ran
  the 10° baseline out of memory. That chunk size then holds for the rest of
  the run.
- **Candidate cap.** `BASE_MAX_PATHS` caps the (ray, receiver) candidates of
  the whole solve, not of each receiver. Past the cap Sionna drops paths
  silently, and the result then depends on the chunk size. Each scene logs
  every chunk's cap fill, and the runner warns when the cap is hit.

**Merged shapes (`MERGE_SHAPES`).** Streamed scenes write one PLY per wall and
roof, about 400 shapes. Unmerged, every solve spends ~22 s in Dr.Jit code
generation dispatching over them, while the GPU runs for 0.6 s. All kernels are
cache hits, so nothing is compiled; the cost is regenerating the per-shape
dispatch code. `MERGE_SHAPES = True` merges by radio material into 3 shapes, and
the solve takes 0.7 s. Measured on 2 scenes × 100 UEs at 20°: identical to
unmerged within the solver's run-to-run noise (median per-UE relative difference
≤ 3e-5, LoS exact). Merged walls and roofs lose their names, so the runner maps
roles by their distinct dummy materials (`_scene_roles`) and refuses
`MERGE_SHAPES` when the materials are not distinct.

**Cost.** `gpu_hours` in the notebook costs each solve group per scene as
~0.7 s + 0.45 s per UE at 10° (about 1e6 paths per UE) or 0.07 s per UE at 20° and
above, ×1.5 at depth 5 (measured whole-scene wall time, RTX 3080). Per-UE path
extraction dominates, not the ray trace, so cost scales with UEs, and 10° is
about three quarters of any phase that includes it. Phases 1–2 are capped at the
first 50 scenes (`n_scenes`) and, per scene, the first 50 of phase0's 100 UEs, 25 at
10° (`N_RX_KEEP_BY_ELEV = {"10": 25, "*": 50}`). They are the same UEs phase0 uses,
so `oat_response` pairs them UE for UE through `_same_ues`. That is 6,250 links per
variant; all phases total 1.33 M links, ≈ 64 GPU-h. `BASE_SAMPLES_PER_SRC` = 1e6 would be ~3×
faster again; decide it from phase0's noise floor.

**Moment-fit K is off** (`K_FIT_N = 0`, `K_fit_dB` stored as NaN). It cost ~1.3 s
per UE and was undefined (NaN) for 75–90 % of UEs.

---

## 5. Analysis — how effects are judged

[`sensitivity_analysis.ipynb`](../src/analysis/sensitivity_analysis.ipynb) writes
CSVs to `Results/sensitivity/analysis/`.

| Function | What it computes, and why |
|---|---|
| [`load_sweep`](../src/utils/sensitivity.py#L75) / [`load_results_frame`](../src/utils/sensitivity.py#L41) | Per-RX frame with every design column joined by scene seed, realised P.1410 values, satellite azimuth, slant range, pass peak, leg and NORAD id |
| [`scene_bootstrap`](../src/utils/sensitivity.py#L167) | CIs from resampling whole scenes; UEs in a scene are correlated |
| [`noise_floor`](../src/utils/sensitivity.py#L242) / [`noise_floor_scene`](../src/utils/sensitivity.py#L397) | Std across seed replicates of ensemble statistics and of scene-mean outputs |
| [`oat_response`](../src/utils/sensitivity.py#L415) | For a uniform OAT variant: per-scene paired Δ = variant − baseline (same scene, pass, UEs and rays), fitted as Δ = b₁(x − x₀) + b₂(x − x₀)² through Δ(x₀) = 0. **Swing** is the fit's max − min over the range, i.e. what the factor's full uncertainty does. It also reports the slope at baseline. CIs come from a scene bootstrap. A swing is ≥ 0 by construction, so a swing CI cannot test for zero. Instead, a factor is significant if the fitted Δ at either end of its range has a CI excluding 0 *and* swing > `k_floor` × the scene noise floor. On a null factor the swing CI was [0.010, 0.129] (above 0), but neither end's CI excluded 0. |
| [`oat_effects`](../src/utils/sensitivity.py#L253) | Categorical variants: paired Δ of ensemble statistics, with the same gate |
| [`geometry_variance`](../src/utils/sensitivity.py#L463) | Splits total variance into within-scene (UE), between-scene *explained* by realised α, β, γ and satellite azimuth relative to the grid (quadratic regression, CV R²), and between-scene *residual*. Built on [`variance_components`](../src/utils/sensitivity.py#L196). |
| [`sobol_from_design`](../src/utils/sensitivity.py#L375) | Legendre PCE on scene means over the joint uniform design, with indices taken analytically from the coefficients. R²<sub>cv</sub> is the share the factors explain. |
| [`p1410_class`](../src/utils/sensitivity.py#L498) + [`dist_metrics(scenario="by_class")`](../src/utils/sensitivity.py#L519) | Scenes span suburban to dense urban, so each is binned to the nearest P.1410 class by realised (α, β, γ). Each class is then scored (KS, W1) against its own 38.811 table: Suburban, Urban or DenseUrban. |
| [`estimator_table`](../src/utils/sensitivity.py#L558) | K four ways, per class: mean of dB, median of dB, ratio of means, CFR moment fit |

Per-UE outputs come from `channel_utils.py`: power by depth and by surface role,
and the moment-fit K. They are stored through `EXTRA_KEYS`
([L427](../src/simulations/run_stochastic_sim.py#L427)).

---

## 6. Verification status

| Check | Result |
|---|---|
| Design columns vs uniform (1000-row LHS) | KS p ≈ 1 for every column; α_R exactly 100 per integer |
| Ground points | sin(lat) and lon uniform (KS p ≈ 1) |
| Realised vs drawn α, β (500 m scenes) | unbiased: −0.004 and −0.1%, per-scene sd 0.015 and 4.8% |
| Orbit table | peaks within 0.12° of target; el = E to 1e-6°; real shells 350–550 km |
| Per-scene TX from orbit | LoS arrival azimuth = waypoint azimuth after heading rotation (0.01°); LoS Doppler = SGP4 −f·ṙ/c to 3 Hz |
| Streamed vs file-based scene | ≤7e-5 relative, identical UEs |
| 50 streamed scenes | RSS flat (673 → 702 MB), scratch empty |
| `oat_response` on a known quadratic | swing 2.077 vs true 2.083; slope −1.99 vs −2 |
| `geometry_variance` | shares sum to 1; explained share 0.72 vs expected 0.715 |
| Sobol (Ishigami), variance components, bootstrap coverage, Rician fit | as before: ±0.01, 4.37/0.98 for 4/1, 93%, 6.3 dB for 6 dB |

## 7. Open items

- **Doppler.** The sweeps use the path-domain estimator only (`DOPPLER_ENABLED=True`,
  `DOPPLER_NUM_TIME_STEPS=1`): each scene's orbit velocity gives per-path Doppler,
  summarised as the mean shift and the power-weighted RMS spread. On Phase 0
  scenes at 30°, the mean matches SGP4's −f·ṙ/c to within 1 Hz, and the spread for
  static UEs is a median 0.32 Hz (10–90%: 0.13–1.37 Hz). The time-domain estimator
  is not usable here. It samples at 10× the maximum Doppler for N steps, so its
  bins are kHz wide, and it reported a median 14.4 kHz "spread", which is leakage.
  It also costs about 4.4× per scene. Doppler spectra for figures come from the
  per-path `doppler_hz` and `amps` in `raw_results`. The Doppler-less Phase 0 of
  2026-10-14 is kept in `phase0_nodoppler_2026-10-14/`.

- **Phase 0 of 2026-10-13 is invalid.** It ran before the `beam_cone_precision`
  Sionna patch: 10° beams collapsed onto the scene centre and every other
  elevation carried ±0.5–2.4 dB of diffuse bias
  ([sionna-patches.md](sionna-patches.md#beam_cone_precision--exact-cone-width-at-long-slant-ranges)).
  The results are kept in `Results/sensitivity/phase0_conebug_2026-10-13/` and
  must not be analysed. Re-run Phase 0 in full.

- **`scene_utils` fork.** `scene_utils.py` is left unchanged,
  so the runner imports [`scene_utils_sensitivity.py`](../src/utils/scene_utils_sensitivity.py)
  instead: a copy with [`tools/scene_utils_sensitivity.patch`](../tools/scene_utils_sensitivity.patch)
  applied. It honours `alpha_r`/`scattering_pattern`, `RX_BASE_PATTERN`,
  `RX_ORIENTATION_DEG`/pitch/yaw and non-iso linear patterns. With none of those
  set it matches `scene_utils` to 1e-5. Re-sync it when `scene_utils.py` changes:
  `patch -p1 -o src/utils/scene_utils_sensitivity.py src/utils/scene_utils.py < tools/scene_utils_sensitivity.patch`,
  then re-add the fork note to its docstring.
- **Sionna 1.2.1 receive-pattern defect** ([sionna-patches.md](sionna-patches.md#known-upstream-defect--receive-pattern-direction-not-yet-patched)).
  Directional UE patterns are wrong until the fix is archived. Keep
  `ORIENTATION_READY = False` until then.
- **TLE catalog epoch.** Passes are historical predictions at the catalog's
  median epoch (August 2026), not measured tracks. Refresh with
  `fetch_tle.py --force` for a current campaign; the SHA-256 in each orbit table
  records which catalog was used.
