# Scene-classification dataset pipeline

This pipeline builds a forward dataset for one question: **can distilled
large-scale parameters (LSPs) classify an NTN terminal's scene as rural,
suburban or urban?** The LSPs are K-factor, delay and angular spreads, excess
loss, LoS state and Doppler spread. Each class is a stochastic P.1410 block
scene family. Each scene perturbs its class geometry and draws class-weighted
materials. Each link draws class-blind nuisances (polarization, orbit, UE
position). The RT outputs are distilled per UE and per scene, then classified.

It replaces the phase-structured sensitivity sweeps (Phases 0–4). Their results
stay in `Results/sensitivity/`. The code and docs as they were on 2026-10-16
are archived in `_archive/sensitivity_phases_2026-10-16/`.

Design rules:

- **Only geometry medians and material mixtures depend on the class.** Every
  other draw has the same distribution for all three classes, so it cannot leak
  the label.
- **One Latin-hypercube row per scene**, covering geometry, materials and
  polarization. With LHS, categorical proportions are exact rather than
  binomial.
- **Scenes are streamed, never cached**, as before: a manifest describes the
  scenes, and each runner builds one in scratch, solves it and deletes it.
- **Folds are grouped by scene.** A scene appears at every elevation and
  carrier, so row-level folds would leak its geometry.

```
custom_drivers/scene_class_dataset.ipynb (kernel: no Mitsuba/Sionna)
  scene_classes: CLASS_MEDIANS, class_factors ─► class_manifest(cls, n, orbit table)
     calibration gate (CPU): block P_LoS vs Chiu eq. (27)
     orbit_sampling CLI (CPU pool, real TLEs) ─► Results/scene_class/orbit_table.pkl
  one variant per (dataset, carrier, class) ─► sweep_utils.queue_entries / start_detached
                                        │  tmux "scene_class", one process per GPU
                                        ▼
  run_stochastic_sim.py: stream scene ─► per-scene TX from orbit ─► dual-pol solve ─► delete
        └─► Results/scene_class/<dataset>/<freq>__<class>/results_elevNNNdeg.pkl
                                        │
analysis/scene_classification.ipynb ◄───┘ scene_dataset.build_dataset ─► links.pkl / scenes.pkl
                                          scene_classify: grouped CV, transfer, importance
```

---

## 1. Classes: Chiu & Roy wall parameters → P.1410 block scenes

The class baselines are the rural and urban parameters of the Chiu & Roy
validation ([`WALL_SCENARIOS`](../src/utils/scene_gen_ppp.py#L222),
`chiu_validation/`). That model is a wall process: wall density λ_b, wall
length l_f and Rayleigh height scale γ, with λ_1D = 2λ_b·l_f/π walls crossed per
metre. The dataset uses the P.1410 **block** sampler, so the wall parameters
are mapped onto (α, β, γ) by three rules:

1. **Same height law.** γ_block = γ_Chiu.
2. **Mean footprint side = wall length.** √(α/β) = l_f.
3. **Same blocking along a ray, after calibration.** √(αβ) = κ·λ_1D.

This gives **α = κ·λ_1D·l_f** and **β = κ·λ_1D/l_f**
([`block_medians`](../src/utils/scene_classes.py)).

κ corrects for the block geometry. At κ = 1, block scenes block far more than
Chiu's zero-width walls: P_LoS at 10° was 0.68 against 0.77 (rural) and 0.13
against 0.30 (urban). There are two reasons. A square's mean projected width
is 4L/π, not L. And UEs stand in streets right next to buildings, whereas
Chiu's terminal has no exclusion zone. κ was fitted per class so that the
block P_LoS from [`calibration_table`](../src/utils/scene_classes.py) (a ray /
box test over 20–24 scenes × 150–200 UEs, uniform azimuth) matches Chiu's
closed form eq. (27) on mean log P_LoS over 10–50°, then rounded.

| Class | Source | γ [m] | λ_1D [1/m] | l_f [m] | κ | **α** | **β [bldg/km²]** | Median / mean bldg height [m] | Buildings per 500×500 m |
|---|---|---|---|---|---|---|---|---|---|
| Rural | Chiu rural | **3.0** | 0.020 | 10 | 0.72 | **0.144** | **1440** | 3.5 / 3.8 | ~360 |
| Suburban | interpolated | **5.0** | 0.022 | 15 | 0.66 | **0.218** | **968** | 5.9 / 6.3 | ~240 |
| Urban | Chiu urban | **8.0** | 0.025 | 20 | 0.60 | **0.300** | **750** | 9.4 / 10.0 | ~190 |

**The suburban point.** Chiu's eq. (27) exponent scales with λ_1D·γ, which is
0.06 for rural and 0.20 for urban. Suburban (γ = 5 m, λ_1D = 0.022, l_f = 15 m)
gives 0.11, close to their geometric mean, so its P_LoS lies between theirs at
every elevation. γ = 5 m corresponds to 1–2-storey housing.

**Calibration gate result** (κ as above, 20 scenes × 200 UEs per class, as the
driver runs it; block minus Chiu, in points of P_LoS):

| Elevation | 10° | 20° | 30° | 40° | 50° | 60° | 70° | 80° |
|---|---|---|---|---|---|---|---|---|
| Rural | −2.0 | 0.0 | +1.0 | +1.6 | +1.2 | +1.5 | +1.3 | +0.8 |
| Suburban | −2.4 | +0.2 | 0.0 | +1.1 | +1.7 | +2.0 | +1.9 | +1.6 |
| Urban | +3.4 | −2.0 | −1.3 | +0.3 | +0.7 | +1.6 | +1.8 | +2.3 |

The worst case is 3.4 points. The Monte-Carlo uncertainty on one cell is about
±2 points; it is dominated by scene-to-scene variation. The driver notebook
re-runs the gate (§3 of the notebook) and asserts two things: |diff| ≤ 5 points
at 10–50°, and rural ≥ suburban ≥ urban at every elevation. Above 50° the block
P_LoS approaches 1 slightly faster than eq. (27).

**Caveat for any write-up.** The Chiu reference README says its "urban" values
were tuned to the TR 38.811 *suburban/rural* LoS row. The class names are Chiu's
labels, not 38.811's.

---

## 2. Per-scene draws

### 2.1 Geometry — perturbed around the class median

Each factor is **log-uniform on [median/(1+p), median·(1+p)]**, so the median is
exactly the class value ([`geo_factors`](../src/utils/scene_classes.py)).
`PERTURB` p = 0.20 is the default; the `hard` dataset uses 0.30.

| Factor | Distribution | Rural (p = 0.20) | Suburban | Urban |
|---|---|---|---|---|
| `geo.alpha` built fraction | log-U ×/÷1.2 | 0.120–0.173 | 0.182–0.261 | 0.250–0.360 |
| `geo.beta` buildings/km² | log-U ×/÷1.2 | 1200–1728 | 807–1162 | 625–900 |
| `geo.gamma` Rayleigh scale [m] | log-U ×/÷1.2 | 2.50–3.60 | 4.17–6.00 | 6.67–9.60 |
| `cfg.SCENE_EAST_HEADING_DEG` | U[0°, 90°] | same | same | same |

The fixed sampler settings are `PPP_KW`: 500×500 m, 4 points per cell,
`max_fill` 0.8, `min_side` 3 m, heights truncated to [0.5, 120] m. Chiu's heights
are untruncated; the 0.5 m floor only stops near-zero boxes from sitting
coplanar with the ground.

[`sample_ppp_scene`](../src/utils/scene_gen_ppp.py#L86) realises each scene
from `default_rng(seed)`:

1. The grid gets a random shift.
2. Poisson points are binned into cells; a cell with k points gets built area
   min(k·a_p, 0.8·A_c).
3. β sets the cell size: `cell_res = sqrt((1 − e⁻⁴)·10⁶/β)`
   ([`_geo_kwargs`](../src/utils/variant_manifest.py)).
4. Each building gets one Rayleigh(γ) height.

Realised α, β and fitted γ match the medians. In the gate run they were
0.144 / 1418 / 3.03 for rural, 0.218 / 1001 / 5.05 for suburban and
0.300 / 765 / 7.90 for urban. The runner records `itu_alpha`, `itu_beta` and `itu_gamma_fit`
for every UE.

### 2.2 Satellite pass — real TLEs at a sampled peak elevation

Each (scene, elevation E) gets a real pass from
[`orbit_sampling.py`](../src/utils/orbit_sampling.py):

| Draw | Distribution |
|---|---|
| Ground location | **Area-uniform** over \|lat\| ≤ 60°: lat = arcsin(U[−sin 60°, sin 60°]), lon ~ U[−180°, 180°). One per scene, shared by its elevations. |
| Peak elevation | U[max(30°, E + 1°), 90°] per (scene, E) |
| Pass leg | discrete-uniform {rising, setting} |

[`scene_passes`](../src/utils/orbit_sampling.py) runs **one catalog scan per
scene**. It calls `make_real_target_passes` over the cached
`tle_catalog/starlink.tle`, using the catalog's median TLE epoch, a 6 h window
and a 1° tolerance, and takes the real pass nearest each target peak.
[`waypoint_at_elevation`](../src/utils/orbit_sampling.py) then solves the
instant el(t) = E on the drawn leg (`brentq` over SGP4). That gives the
satellite's azimuth, slant range and velocity, so the satellite state follows
from a real orbit rather than being drawn directly.

- **Failure handling.** If no pass is found within tolerance, the window widens
  to 24 h. If that fails too, the ground point is redrawn from a seeded stream.
- **Measured behaviour.**
  - Peaks land within 0.12° of target.
  - el = E to within 1e-6°.
  - Shells are the real 350–550 km ones.
  - The LoS Doppler matches SGP4's −f·ṙ/c to 3 Hz.
- **90° is not in the dataset.** A peak above E + 1° cannot be drawn there.

**One table covers every dataset.** The driver builds a single
`Results/scene_class/orbit_table.pkl` over all seed blocks. The CLI takes
several `first:count` blocks. The orbit draw is class-blind, and the table
records the catalog's SHA-256.

### 2.3 UE placement

`N_RX_PER_SCENE` = 20 UEs are drawn per scene. The exact outdoor mask
([`free_space_grid`](../src/utils/scene_gen_ppp.py#L853)) uses a 2 m grid, 1 m
clearance and a 20 m edge margin, and each UE is jittered by ±1 m. The RNG is
seeded by the scene seed alone, so the UEs are identical at every elevation and
carrier.

**Walking UEs (`walk` dataset).** With `UE_TRAJ_STEPS` > 1, each of
`N_RX_PER_SCENE` = 10 UEs walks a straight line
([`sample_rx_trajectories`](../src/utils/scene_utils_sensitivity.py)):
- Start: a uniform free-space point, as above.
- Heading: U[0°, 360°), counter-clockwise from scene +x.
- Speed: U[`UE_SPEED_RANGE_MPS`] = U[1.0, 1.6] m/s.
- Snapshots: `UE_TRAJ_STEPS` = 11, every `UE_TRAJ_DT_S` = 1 s, so each walk is 10–16 m.
- Each snapshot is solved as its own receiver, with `Receiver.velocity` set. Velocity affects
  only Doppler; LoS and DS come from the snapshot geometry.
- **The satellite is frozen during the walk.** It stays at the elevation's orbit
  draw. In 10 s a LEO satellite moves ~75 km (several degrees of elevation), which
  would blur the elevation bins.
- **Feasibility.** The whole path, checked every 0.25 m, must stay outdoors under the same
  analytic mask, clearance and margin. A blocked walk redraws heading and speed (up to 50
  tries), then the start. Over 200 walks per class, headings folded mod 90° stay flat, even
  in urban, so the conditioning toward street-aligned headings is negligible at these
  street widths.
- Walks use their own RNG stream, `[RNG_SEED, scene seed, 7919, 1]`. They are identical at
  every elevation and carrier, and UE i depends only on the draws before it.

### 2.4 Ray sampling

The solver seed is `SeedSequence([RNG_SEED, scene_seed, round(100·elev),
SOLVER_SEED_OFFSET])`, so it is resume-safe and identical for both carriers.
A single UE's K can move by up to ~10 dB between solver seeds
([analysis.md](analysis.md#solver-monte-carlo-noise)). Scene-level features
average over 20 UEs, and classification accuracy is reported against the
number of UEs aggregated.

### 2.5 Materials — class-conditioned ITU anchors

Every scene has **three material roles: ground, walls and roof**. Each role is
one material, shared by every building in the scene. `MERGE_SHAPES` needs three
distinct radio materials, and it is what keeps a solve at ~0.7 s rather than
~22 s of Dr.Jit dispatch. The XML names (`itu_concrete` / `itu_brick` /
`itu_plasterboard`) are dummy identities, distinct per role, so two roles may
carry the same anchor.

Each role is drawn per scene in two stages
([`material_factors`](../src/utils/scene_classes.py)):

1. **Anchor.** A categorical draw `<role>.base` from the class weights below.
   [`apply_material_override`](../src/utils/variant_manifest.py) swaps in the
   anchor's P.2040 (a, b, c, d) *before* any scaling.
2. **Jitter.** `eps_scale` and `sigma_scale`, each log-uniform ×/÷1.25,
   multiply `a` and `c`. That is narrow enough to keep the anchors distinct
   (wood ε_r 1.6–2.5 against brick 3.1–4.9) and wide enough to span variants
   within a type, such as asphalt against concrete.

ITU-R P.2040 anchors ([`MATERIAL_ANCHORS`](../src/utils/variant_manifest.py)),
with ε_r = a·f^b and σ = c·f^d [S/m]:

| Anchor | a | b | c | d | ε_r / σ @2 GHz | ε_r / σ @20 GHz | Stands in for |
|---|---|---|---|---|---|---|---|
| concrete | 5.24 | 0 | 0.0462 | 0.7822 | 5.24 / 0.079 | 5.24 / 0.48 | concrete, asphalt, paving |
| brick | 3.91 | 0 | 0.0238 | 0.16 | 3.91 / 0.027 | 3.91 / 0.038 | brick walls, clay-tile roofs |
| wood | 1.99 | 0 | 0.0047 | 1.0718 | 1.99 / 0.010 | 1.99 / 0.12 | timber frame, wood-shingle roofs |
| glass | 6.31 | 0 | 0.0036 | 1.3394 | 6.31 / 0.009 | 6.31 / 0.20 | curtain-wall facades |
| metal | 1 | 0 | 1e7 | 0 | 1 / 1e7 | 1 / 1e7 | metal and membrane-on-deck roofs, sheds |
| medium_dry_ground | 15 | −0.1 | 0.035 | 1.63 | 14.0 / 0.11 | 11.1 / 4.6 | dry soil, fields |
| wet_ground | 30 | −0.4 | 0.15 | 1.30 | 22.7 / 0.37 | 9.1 / 7.4 | damp soil, grass |

P.2040 specifies the ground anchors only over 1–10 GHz. At 20 GHz they are
extrapolated through the `widen()` ITU-range patch.

Anchor weights (`MATERIAL_WEIGHTS`):

| Ground | concrete | medium_dry_ground | wet_ground |
|---|---|---|---|
| Urban | **0.80** | 0.12 | 0.08 |
| Suburban | 0.35 | 0.35 | 0.30 |
| Rural | 0.05 | **0.50** | **0.45** |

| Roof | metal | concrete | brick (tile) | wood |
|---|---|---|---|---|
| Urban | **0.60** | 0.30 | 0.10 | 0 |
| Suburban | 0.10 | 0.05 | 0.25 | **0.60** |
| Rural | 0.15 | 0 | 0.15 | **0.70** |

| Wall | concrete | glass | brick | wood |
|---|---|---|---|---|
| Urban | **0.65** | 0.25 | 0.10 | 0 |
| Suburban | 0.15 | 0 | 0.40 | **0.45** |
| Rural | 0.10 | 0 | 0.30 | **0.60** |

The rural metal-roof weight of 0.15 stands for corrugated barns and sheds.

Class-blind material factors:

| Factor | Distribution |
|---|---|
| `{wall,roof,ground}.scattering_coefficient` | U[0, 0.6] |
| `all.alpha_r` (directive lobe) | discrete-U {1…10} |
| `all.xpd_coefficient` | U[0, 0.5] |
| thickness, scattering pattern | fixed: 2 m, directive |

That is 14 material draws per scene: 3 anchors and 11 continuous values. The
anchors carry class information, so the classifier sees geometry *and*
materials. The `matctl` dataset draws every class's anchors from the pooled
mean weights (`pooled=True`), which isolates geometry.

### 2.6 Link nuisances (class-blind)

| Factor | Distribution |
|---|---|
| Polarization pair | **fixed RHCP/RHCP** in every dataset (driver `POL_PAIR`, since 2026-10-19) |
| Orbit, ground point | §2.2 |
| UE positions | §2.3 |
| RX height | fixed at 1.5 m |
| Ionosphere / troposphere | on, at the defaults in `BASE_CFG` (VTEC log-N 10 ± 5 TECU, rain 0, ρ 7.5 g/m³, scintillation), seeded per (scene, elevation) |

**Why the pair is fixed.** The first `base` run drew the pair per scene, over
{RHCP/RHCP, RHCP/LHCP, V/V, H/H, RHCP/V}. That run is kept as
`Results/scene_class/base_mixedpol`. In it, K was not comparable across pairs:
- RHCP/RHCP rejects the ground bounce, giving 10–13 dB above 40°.
- H/H keeps the ground bounce, giving 2–8 dB.
- V/V peaks near the ground's Brewster angle.

The pair explained 57–76% of the scene-to-scene K variance, so pooled K
described no real link (see `report/base_vs_chiu`). RHCP/LHCP direct power is
~1e-18, so its K is undefined.

The LHS still draws `cfg.POL_PAIR`, and the driver overwrites it with
[`fixed_pol_manifest`](../src/utils/scene_classes.py). Scene, material, orbit
and solver-seed draws therefore match `base_mixedpol` scene for scene. Set
`POL_PAIR = None` to restore drawn pairs.

**Dual-pol readout is inexact for diffuse paths.** With mixed pairs, the runner
solves a dual-pol `[V, H]` element and reads each pair out of the 2×2 with
[`_pol_weights`](../src/simulations/run_stochastic_sim.py#L515). This is exact
for specular and diffraction paths. Sionna's diffuse Jones matrix, however, is
scaled by γ = |R_spec·e_i| / |e_i|, which depends on the incident polarization.
Diffuse scattering is therefore not linear in the TX polarization.

For a CP pair, the readout combines γ_V and γ_H like an arithmetic mean,
whereas a native solve gives their RMS. In 28 same-scene RHCP/RHCP checks (rural,
10° and 20°), the native solve had 1.3–1.4× the diffuse power, with specular
within 0.1 points, and K 0.2–0.4 dB lower. CP pairs in `base_mixedpol` are
biased accordingly. V/V and H/H read out one port and are exact.

Use native single-pol solves (one pair per dataset), not readout, for any CP
statistic.

**Swept, not drawn** (both are known to the terminal and enter the classifier
as context):
- Elevation: 10–80° in steps of 10°.
- Carrier: 2 and 20 GHz.

### 2.7 How close the classes come to overlapping

The table gives the fraction of each class's log-range shared with its
neighbour. The last column is the pairwise Bayes error of an ideal classifier
that sees the true blocking index √(αβ)·γ, the quantity that drives P_LoS
(exact, by Monte Carlo):

| p | α overlap R–S / S–U | β overlap R–S / S–U | γ overlap R–S / S–U | Blocking-index Bayes error R–S / S–U |
|---|---|---|---|---|
| 0.15 | 0 / 0 | 0 / 9 % | 0 / 0 | 0.02 % / 0.07 % |
| **0.20 (default)** | 0 / 12 % | 0 / 30 % | 0 / 0 | **1.6 % / 2.0 %** |
| 0.30 (`hard`) | 21 % / 39 % | 24 % / 51 % | 3 % / 10 % | 8.6 % / 9.4 % |

At p = 0.20 the geometry nearly separates the classes. What remains hard comes
from five sources:
- material mixtures overlapping
- polarization
- orbit
- UE placement
- how many UEs are aggregated

That is what the experiment is testing.

---

## 3. Solve mechanics

These are unchanged from the sensitivity sweeps; see
[pipelines.md](pipelines.md#scene-classification-sweeps--srccustom_driversscene_class_datasetipynb).

- **Streaming.** [`_scene_geometry`](../src/simulations/run_stochastic_sim.py#L490)
  calls `generate_scene_streaming` into `<variant>/scratch_elevNNN/` and deletes
  the scene after the solve. Over 50 streamed scenes the runner's RSS stayed
  flat.
- **Per-scene TX.** [`_scene_tx`](../src/simulations/run_stochastic_sim.py#L479)
  places the TX from the scene's orbit waypoint and heading.
- **Doppler.** Path-domain only (`DOPPLER_NUM_TIME_STEPS = 1`): the mean shift
  and the power-weighted RMS spread over paths. The time-domain estimator gives
  kHz-wide bins at LEO and reports leakage as spread.
- **Merged shapes.** There are three shapes, one per role. Roles are mapped back
  by their distinct dummy materials (`_scene_roles`).
- **Chunks and cap.** Receivers are solved in chunks of `RX_CHUNK` (20 UEs fit
  in one chunk); an OOM halves the chunk. `BASE_MAX_PATHS` caps the candidates
  per solve, and the runner warns when the cap is hit.
- **Cost.** About 0.7 s per scene, plus ~0.45 s per UE at 10° or ~0.07 s per UE
  at 20° and above, before the dual-pol overhead. In the smoke test (1e6 rays,
  4 UEs, 30°), dual-pol scenes took 2–3 s each. The driver's estimate assumes
  `DUAL_POL_FACTOR = 1.5` until the pilot measures it.

---

## 4. Datasets and launch

| Dataset | Scenes/class | Carriers | p | Anchor weights | Seed offset | ≈ GPU-h |
|---|---|---|---|---|---|---|
| `pilot` | 30 | 2 GHz | 0.20 | per class | 0 | ~1 |
| `base` | 400 | 2, 20 GHz | 0.20 | per class | 0 | ~25 |
| `hard` | 150 | 2 GHz | 0.30 | per class | 20 000 | ~5 |
| `matctl` | 150 | 2 GHz | 0.20 | pooled | 40 000 | ~5 |
| `walk` | 400 | 2, 20 GHz | 0.20 | per class | `base`'s manifests | ~35 (measured ~6.5 s/scene; ~17 h on 2 GPUs) |

`walk` copies `base`'s class manifests (`scenes_from`) rather than redrawing them,
because the LHS draws depend on n. Its UE-side keys (`N_RX_PER_SCENE` = 10,
`UE_TRAJ_*`, `UE_SPEED_RANGE_MPS`) are in `sweep_utils.UE_KEYS`, so they are not part of
the solve key. `walk_smoke` is the first 3 scenes per class, at 2 GHz and 10°/40°.

The GPU-hours include the assumed ×1.5 dual-pol overhead, except for `walk`. Its
estimate is measured: ~6–10 s per scene with 110 receivers on one 3080.

**Seeds.** Each class has its own block: rural 0…, suburban 100 000…, urban
200 000…. Each dataset adds its offset, and each class LHS is seeded from its
own base seed. `pilot` shares `base`'s seeds and orbits, but its LHS has a
different size, so its draws differ.

**Outputs.**
- Manifests are cached in `Results/scene_class/<dataset>/_manifests/<class>.pkl`.
- Variants are `<freq>__<class>`, with results in
  `Results/scene_class/<dataset>/<freq>__<class>/`.

**Launch.** The launch cell refuses to start while any `run_stochastic_sim.py`
is alive, unless `ALLOW_CONCURRENT` is set. Another queue, or another user's
runs, would then share the 10 GB GPUs. The cell prints the `config.json` files
actually written, since that is what the runners read. It then starts the tmux
session `scene_class`. Follow it with `tail -f Results/scene_class/_queue.log`.

---

## 5. Dataset schema and leakage rules

[`scene_dataset.build_dataset`](../src/utils/scene_dataset.py) writes three
files to `Results/scene_class/dataset/<dataset>/`: `links.pkl`, `scenes.pkl` and
`columns.json`. They are pandas pickles; pyarrow needs NumPy ≥ 2, but this venv
pins 1.26.

- **`links`**: one row per (scene, UE, elevation, carrier, walk snapshot `traj_t_s`;
  this is 0 for static UEs). It holds the per-UE
  LSPs from `per_rx_lsp_frame`, plus:
  - `excess_loss_dB` = −PG_dB − FSPL(slant range, f)
  - `lg_doppler_rms_hz`
  - the scene's labels and draws, joined from the variant's `scenes_manifest.pkl`
    by scene seed
- **`scenes`**: one row per (scene, elevation, carrier). It holds:
  - P_LoS over the scene's UEs
  - K as a ratio of means over its LoS UEs
  - for each LSP and each state (all / LoS / NLoS): mean, std, q10, q50 and q90
  - every column that is constant within the scene (labels, draws, orbit,
    realised P.1410)
  
  `scene_features(links, n_ue=k)` rebuilds this table from k UEs per scene.

Every column has a role (`columns.json`, `scene_dataset._role`):

| Role | Columns | May feed a classifier |
|---|---|---|
| `feature` | `is_los`, `K_dB`, `lgDS`, `lgASA`, `lgZSA`, `lgASD`, `lgZSD`, `excess_loss_dB`, `lg_doppler_rms_hz`; scene-level `P_LoS`, `f_*` | yes |
| `context` | `elevation`, `frequency`, `pol_pair` | yes (known to the terminal) |
| `label` | `scene_class`, `geo_alpha`, `geo_beta`, `geo_gamma` | target only |
| `id` | `sweep`, `scene_seed`, `ue`, `traj_t_s` | no |
| `nuisance` | material anchors and draws, heading, orbit, UE position, `ue_speed_mps`, `ue_heading_deg`, ionosphere/atmosphere samples | no |
| `oracle` | `P_ground`/`P_wall`/`P_roof`, `P_by_depth`, `itu_*`, `n_paths`, `SF_dB`, raw spreads and Doppler | no |

`SF_dB` is tagged oracle on purpose. It is a residual about the mean of its
results file, and each file holds one class.

---

## 6. Analysis

[`scene_classification.ipynb`](../src/analysis/scene_classification.ipynb), with
helpers in [`scene_classify.py`](../src/utils/scene_classify.py), writes CSVs to
`Results/scene_class/analysis/`. It covers:

1. **Class sanity.**
   - RT P_LoS per class against Chiu eq. (27).
   - Per-scene LSP medians and 10–90 % bands against elevation.
   - Per-UE W1/KS distance to the matching 38.811 table (rural → Rural,
     suburban → Suburban, urban → Urban).
2. **Classification.**
   - Models: gradient boosting (NaN-native) and multinomial logit.
   - `StratifiedGroupKFold` by scene seed.
   - Accuracy per elevation × carrier, and confusion matrices.
   - A shuffled-label control (permuted at scene level) must come out at chance.
3. **UEs needed.** Accuracy against the number of UEs aggregated per scene
   (1, 2, 5, 10, 20).
4. **LSP importance.** Permutation importance on held-out scenes, summed per LSP
   family.
5. **Polarization transfer.** Train on CP pairs and test on linear pairs, and
   the reverse.
6. **Geometry against materials.**
   - `base` accuracy against `matctl` (geometry only) and `hard` (p = 0.30).
   - Accuracy conditioned on the drawn anchors, for example urban scenes that
     drew non-metal roofs.
7. **MLP on elevation curves** ([`scene_mlp_classifier.ipynb`](../src/analysis/scene_mlp_classifier.ipynb),
   [`scene_mlp.py`](../src/utils/scene_mlp.py), `make_model("mlp")`). Sections 2–6 above are the
   scene-level baselines.
   - **Rows.** One per (scene, UE, carrier), built by `scene_dataset.elevation_curves`.
   - **Inputs.**
     - Two LSPs (`scene_dataset.MLP_LSPS`, since 2026-10-20; notebook `LSPS`) at each elevation:
       LoS state and delay spread, i.e. `is_los@10 … @80` and `lgDS@10 … @80`. The `base` runs
       before that also used `K_dB`. Above 30°, K and DS track the class-blind ground scattering
       coefficient.
     - Walking UEs (`TRAJ`):
       - `full`: `<lsp>@<elev>@t<k>`, 2 × 8 × 11 = 176 inputs
       - `summary`: `<lsp>~mean@<elev>` and `<lsp>~std@<elev>`
       - `t0`: the first snapshot only, a static-UE ablation

       Importance gains a per-snapshot (`time`) view.
     - Why so few: on `base`, is_los, K and DS matched all nine link LSPs (0.792 against 0.795 at 20 UEs).
       The angular spreads (ASA, ZSA, ASD, ZSD) need antenna arrays or are satellite-side, and
       neither they, excess loss nor Doppler adds accuracy after aggregation.
     - Plus carrier and pol pair.
     - Elevation is the curve's axis, not an input.
     - UEs are seeded by the scene alone, and the results files list the same UEs in the same
       order at every elevation (checked: identical `nominal_rx_positions`). So a row is one
       terminal's sweep.
     - There are no scene-level `f_*` or `P_LoS` aggregates. An assert rejects any column whose
       role is not `feature` or `context`.
   - **Preprocessing.** Fitted on the training fold:
     - imputation (a missing scene × elevation, or K at NLoS elevations), plus missing indicators
     - standardization
     - one-hot carrier and pol pair
   - **Network.** PyTorch, run in the `mlp-venv` kernel (`/path/to/mlp-venv`: CUDA torch, no Sionna/TF). `device="all"` spreads CV folds over both GPUs, one thread per GPU; seeding is per device, so runs are reproducible.
     - Hidden layers are Linear → [LayerNorm] → activation → Dropout, with configurable depth and
       widths.
     - A 3-way softmax head is trained with cross-entropy.
     - Early stopping uses scene seeds held out from the training fold.
   - **Settings.** Every model, preprocessing, input-subset and test setting is in the notebook's
     first code cell. The input subsets are ELEVATIONS and CONTEXT; the LSP set is fixed.
   - **UEs aggregated** (`ue_aggregated_accuracy`). Draw k UE curves of a held-out (scene, carrier),
     average their log-posteriors and take the argmax. There is no retraining per k.
   - **Shuffled control.** It runs over several permutation seeds. A single permutation of N scene
     labels matches the truth at 1/3 ± √(2/9N), and a strong model inherits that agreement.
   - **Importance** (`curve_importance`). Grouped permutation, in two views:
     - one LSP's whole curve
     - every LSP at one elevation
   - **Polarization transfer** (`pol_transfer`).
     - With `PAIR_DIR`, the two sides are native RHCP/RHCP (`base`) and V/V + H/H (`base_mixedpol`,
       same scenes). Without it, both sides come from one mixed-pol sweep.
     - Folds split scene seeds, so a test scene is unseen in both families.
     - `pol_pair` is dropped from the inputs.

---

## 7. Verification status

| Check | Result |
|---|---|
| `block_medians` / class medians | reproduce the §1 table |
| LHS medians of α, β, γ (2000 scenes per class) | within 1 % of the class median |
| Anchor and polarization proportions (2000 scenes) | exact: LHS-stratified, within ±2 points required |
| `base` applied before `eps_scale` | ε_r = a_anchor × scale, as checked per role |
| Class-blind factors | two-sample KS across classes, all p > 0.01 |
| Seed blocks | disjoint across classes |
| Calibration gate | §1; within 3.4 points at 10–80°, classes ordered |
| Driver notebook, dry run | small orbit table, manifests and `config.json` written as intended (depth 3, diffuse, path-domain Doppler, 20 UEs); launch stubbed |
| GPU smoke test (3 urban scenes, 4 UEs, 30°, 1e6 rays) | dual-pol `VH` arrays, metal roofs solve, finite LSPs, ≈3 dB mismatch loss on RHCP/V |
| `build_dataset` on Phase 0 results with a class manifest | 20 000 links → 200 scene rows; no label, oracle or material column among the features |
| Analysis notebook on a synthetic three-class set | runs end to end; shuffled control 0.34 |
| `sample_rx_trajectories`, 200 walks per class (CPU) | every point outdoors; steps equal speed × dt; same seed gives the same walks; a k-walk draw is a prefix of the n-walk draw |
| `walk_smoke` (9 scenes, 10°/40°, 2 GHz) | 10 × 11 rows per scene; spacing 1.00–1.58 m; walks identical across elevations; motion Doppler spread ~1 Hz (≤ f/c·v = 10.7 Hz) vs ~0.1 Hz static; t = 0 P_LoS / DS consistent with `base`; MLP notebook runs in full, summary and t0 modes |

Not yet run: the pilot itself. It is the first real measurement of dual-pol
cost, per-class LSP separation and the classifier.

## 8. Open items

- **Status (2026-10-19).**
  - `pilot` done, with mixed pairs.
  - The mixed-pol `base` is done and archived as `base_mixedpol`.
  - `base` is re-running at RHCP/RHCP.
  - `hard` and `matctl` have RHCP/RHCP manifests and have not run.
  - The `pilot/_manifests` cache was rewritten to RHCP/RHCP by the driver. Its
    results still carry their per-variant `scenes_manifest.pkl` with the drawn
    pairs.
- **`walk` (2026-10-20).** Launched: 48 jobs. The MLP input is now `is_los` and `lgDS`
  only (`MLP_LSPS`; K dropped by user decision). Planned comparisons: `walk/full`,
  `walk/summary` and `walk/t0`, against `base` with the same two LSPs.
- **20 GHz ground anchors** are extrapolated beyond P.2040's 1–10 GHz range.
- **`scene_utils` fork.** The runner imports `scene_utils_sensitivity.py`, a
  copy of `scene_utils.py`. Re-sync with `tools/scene_utils_sensitivity.patch`
  when `scene_utils.py` changes, then re-add the per-RX velocities and
  `sample_rx_trajectories`, which are not in the patch.
- **Receive-pattern defect** (Sionna 1.2.1). UE patterns are `iso`, so the
  dataset is unaffected; keep it that way until the fix is archived.
- **TLE epoch.** Passes are predictions at the catalog's median epoch; the
  SHA-256 is in the orbit table.
