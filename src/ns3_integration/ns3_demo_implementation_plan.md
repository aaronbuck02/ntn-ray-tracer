# ns-3 / Sionna RT Real-TLE Demo — Implementation Plan (v2)

Supersedes the earlier `ns3_demo_implementation_plan.md`. That version was
based on parameter values read out of `sim_config.json` (a leftover from a
previous, circularly-polarized run) and a canned `pass_waypoints.pkl`. This
version is grounded directly in `simulation.ipynb` as currently written, plus
`run_elevation_sim.py` (the actual script the notebook launches, which turns
out to diverge from the notebook's own cell-10 comments in one important
way -- see Step 2). It also switches from reusing an old pass to fetching a
live TLE, per your request.

## What changed from v1, and why it matters

Reading `run_elevation_sim.py` (not just the notebook cells) surfaced two
corrections that materially change the plan:

1. **Beam angle must be recomputed per-sample**, not set once for the whole
   pass. `run_elevation_sim.py` computes `BEAM_HALF_ANGLE_RAD =
   arctan(SCENE_RADIUS / range_at_this_sample)` fresh for every single run,
   and says so explicitly in a comment: this replaced an earlier design
   using one fixed beam angle for the whole sweep, because a fixed angle
   *"under-illuminated the scene at low elevation, starving NLoS ray
   statistics there and artificially inflating K-factor."* That is almost
   certainly the real explanation for the earlier ns-3 demo's rising-leg-
   works/setting-leg-blocked asymmetry -- not solver noise, a genuine
   config gap in how the ns-3 example was calling the solver.
2. **Ray budget is 10x what was benchmarked.** `samples_per_src` and
   `max_num_paths_per_src` are both `BASE_SAMPLES_PER_SRC` /
   `BASE_MAX_PATHS` = **10,000,000**, passed directly into every
   `PathSolver()` call. The earlier runtime benchmark (~17.6 s/call) used
   Sionna's own default of 1,000,000 -- 10x lower. This needs re-measuring,
   not scaling by assumption, before picking a sample count.

## Current notebook parameters (as of this read, `simulation.ipynb` cells 2/4/6/8/10)

| Parameter | Value | Notes |
|---|---|---|
| `SCENE_XML` | `/path/to/wharf_scene_v3/canarywharf3_modified.xml` | unchanged from before |
| `SCENE_CENTER` | `[0, 0]` | |
| `SCENE_RADIUS` | `sqrt(503^2 + 306^2)` = 588.77 m | |
| `FREQUENCY` | 11e9 Hz | unchanged |
| `TX_POWER_DBM` | 43.0 | unchanged |
| `RX_HEIGHT` | **0.5 m** | was 1.5 in the examples built so far -- fix |
| `MATERIAL_NAMES` | `["itu_concrete"]`, `SCATTERING_COEFF=0.4` | still unsupported by the ns-3 branch, unchanged gap |
| `MAX_DEPTH` | **3** | was 2 -- fix |
| `LOS` / `SPECULAR_REFLECTION` / `DIFFUSE_REFLECTION` / `DIFFRACTION` | all `True` | matches current examples |
| `EDGE_DIFFRACTION` | **False** | was True -- fix |
| `REFRACTION` | **False** | was True -- fix |
| `BEAMFORMING` | `True` | matches the fix already applied |
| `BEAM_ANGLE` | computed **per-sample**, not fixed | see Step 2 -- correction needed |
| `BASE_SAMPLES_PER_SRC` / `BASE_MAX_PATHS` | **10,000,000 / 10,000,000** | not currently exposed in the ns-3 branch at all -- add |
| `TX_POL_TYPE` / `RX_POL_TYPE` | **"H" / "H"** | was left at ns-3 defaults -- fix (see Step 1) |
| `SIM_LAT` / `SIM_LON` | 51.5 / -0.1 (London) | unchanged, matches Canary Wharf |
| `SCENE_EAST_HEADING_DEG` | 90.0 | matches `export_ns3_waypoints.py` already |
| `SATELLITE_SOURCE` | `"synthetic_sweep"` in the cell as currently written | **you'll flip this to `"real_search"`** -- see Step 0 |
| `PASS_EPOCH` | `"2026-07-28T00:00:00"` | stale -- today is 2026-08-16; update before running |

---

## Step 0 — Fetch a live TLE and generate a fresh real pass

The notebook's `real_search` mode (cell 10) already does exactly this via
`fetch_tle.py`'s `fetch_tle_catalog()`:

```python
from fetch_tle import fetch_tle_catalog
TLE_CATALOG_FILE = fetch_tle_catalog("starlink", cache_dir="./tle_catalog", max_age_hours=6.0)
```

Downloads `https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=TLE`
(no auth needed), caches to `./tle_catalog/starlink.tle`, re-fetches if the
cache is older than `max_age_hours` (LEO TLEs go stale within hours).

1. In the notebook, set `SATELLITE_SOURCE = "real_search"` (cell 10),
   `PASS_EPOCH = "2026-08-16T00:00:00"` (or leave `None` and let it use
   "now" if you add that convenience -- currently it hardcodes a default of
   `2026-01-01` if `PASS_EPOCH` is `None`, so set it explicitly to today or
   a near-future time). Leave `TLE_GROUP="starlink"`,
   `CATALOG_SELECT="highest_elev"`, `PASS_GEOMETRY="overhead"`,
   `PASS_MIN_ELEV_DEG=10.0`, `PASS_SEARCH_HOURS=24.0` as-is unless no
   qualifying pass turns up (widen the search window first).
2. Run cells 2, 4, 6, 8, 10 in order (2/4 are environment+imports, 6/8 set
   the RF/atmosphere parameters cell 10 needs). This is the only part of
   the notebook you need for the ns-3 demo -- cells 14+ (occupancy grid, RX
   sampling, multi-GPU launch) drive the *offline* elevation sweep and
   aren't needed here.
3. Cell 10 will print the selected real satellite (name + NORAD ID) and
   write a fresh `pass_waypoints.pkl` to
   `<OUTPUT_DIR>/pass_waypoints.pkl` -- note `OUTPUT_DIR` is currently
   `"./results_sat_HH"` in cell 6; consider changing it to something
   demo-specific, e.g. `"./results_demo_tle"`, so this run doesn't get
   mixed up with the HH-polarization elevation-sweep results already in
   that directory.
4. Record the chosen satellite's name/NORAD ID somewhere visible (e.g. in
   the demo's README or narration) -- "this is a live, currently-orbiting
   Starlink satellite, NORAD #XXXXX, whose real published TLE we fetched
   just now" is a much stronger demo line than a silent file path.

**Checkpoint:** a fresh `pass_waypoints.pkl` exists, printed console output
names a real, specific satellite, and `pass_dict["rows"]` (the dense table)
covers a full rise-to-set pass above 10 deg elevation.

---

## Step 1 — Match ns-3 `RtPathSolverConfig` to the notebook exactly

In both `sionna-rt-canarywharf-example.cc` and
`sionna-rt-sgp4-mobility-example.cc`:

1. `RtPathSolverConfig.maxDepth = 3;` (was 2)
2. `RtPathSolverConfig.edgeDiffraction = false;` (was true)
3. `RtPathSolverConfig.refraction = false;` (was true)
4. Add two new fields to `RtPathSolverConfig` in `sionna-rt-channel-model.h`
   -- `int samplesPerSrc = -1;` and `int maxNumPathsPerSrc = -1;` (sentinel
   for "use Sionna's own default"), following the exact `>= 0` pattern
   already used for `beamAngle`. Wire both into the `>= 1.2.0` branch of
   `CalculatePaths()` in `sionna-rt-channel-model.cc` as
   `py::arg("samples_per_src")` / `py::arg("max_num_paths_per_src")`, same
   conditional-cast approach as `beamAngleArg`.
5. In the examples, set both to `10000000` to match
   `BASE_SAMPLES_PER_SRC`/`BASE_MAX_PATHS`.
6. `RX_HEIGHT`: change the RX position in
   `sionna-rt-sgp4-mobility-example.cc` from `Vector(0.0, 0.0, 1.5)` to
   `Vector(0.0, 0.0, 0.5)`.
7. Polarization: `SionnaRtChannelModel::GetPolarizationFromPolSlantAngle()`
   maps a single-port array's `PolSlantAngle` attribute of `0.0` -> `"V"`
   and `M_PI/2` -> `"H"`. To match `TX_POL_TYPE="H"`/`RX_POL_TYPE="H"`, set
   `PolSlantAngle` to `M_PI/2` on both the TX and RX
   `UniformPlanarArray` objects (check the attribute exists via
   `./ns3 run "... --PrintAttributes=ns3::UniformPlanarArray"` if unsure of
   the exact name/units).

**Checkpoint:** every solver-facing parameter in your ns-3 example matches
the value currently in `simulation.ipynb`, not an earlier guess.

---

## Step 2 — Fix beam angle to be computed per-sample (likely resolves the earlier asymmetry)

This is the one correction likely to change demo *behavior*, not just
fidelity.

1. In `export_ns3_waypoints.py`, add a `beam_angle_rad` column to the
   output CSV: `beam_angle_rad = atan(SCENE_RADIUS / range_m)` computed
   per-row from each dense sample's real slant range (`row["range_km"] *
   1e3`) -- literally the same formula
   `tx_position_from_pass_sample`/`tx_position_from_elevation` already
   return as their second output, so this is just capturing a value you're
   already computing and currently discarding.
2. In `sionna-rt-sgp4-mobility-example.cc`, `RtPathSolverConfig` currently
   gets set once via `m_spectrumLossModel->SetRtPathSolverConfig(...)`
   before `Simulator::Run()`. Change this so beam angle updates per
   sample: either (a) call `SetRtPathSolverConfig()` again with a freshly
   computed `beamAngle` immediately before each `ComputeSnr` call (add a
   small wrapper scheduled alongside it, or fold the update into
   `ComputeSnr` itself before it calls `CalcRxPowerSpectralDensity`), or
   (b) extend `ComputeSnrParams` to carry the per-sample `beamAngle` and
   have `ComputeSnr` call `SetRtPathSolverConfig` at the top before doing
   anything else. Either way, look up `beam_angle_rad` from the matching
   CSV row for that call's timestamp.
3. Re-run the pass sweep from Step 3's benchmark-informed sample count and
   check whether the setting leg (previously uniformly blocked) now shows
   the same kind of graceful signal falloff the rising leg does.
4. If the asymmetry persists even with correct per-sample beam angles, at
   that point it's more likely to be genuine building-layout-dependent
   blockage -- worth a scene render at a couple of setting-leg timestamps
   to confirm visually (same as the v1 plan's Step 1 fallback).

**Checkpoint:** SNR trace shows a believable, roughly-symmetric-in-*kind*
(not necessarily identical, since Canary Wharf isn't azimuthally symmetric)
opening and closing of the link across rise and set, with any remaining
asymmetry attributable to visible building geometry rather than a config
gap.

---

## Step 3 — Re-benchmark and pick a sample count (do this after Steps 1-2, not before)

The earlier ~17.6 s/call estimate used the *wrong* ray budget (1M vs. the
real 10M). Before committing to a `numSamples` value for the final demo
run:

1. Apply Steps 1-2 first (they change the solver call's cost).
2. Time `--numSamples=1` and `--numSamples=5` the same way as before
   (wall-clock the whole `./ns3 run` invocation, subtract, divide by 4 for
   the marginal per-call cost).
3. Multiply out to a full-pass estimate at a few candidate `numSamples`
   values, same table format as v1's Step 3.
4. If the 10M-ray cost turns out impractical for a live demo (plausible --
   10x the ray count is not necessarily 10x the wall time, but budget for
   worse, not better):
   - **Best fix, if you have the time:** get the CUDA toolkit installed so
     Dr.Jit uses one of the 2 GPUs already visible to TensorFlow in
     `sionna-venv` instead of falling back to LLVM/CPU (no `nvcc` was found
     at ns-3 configure time in Phase 0). This is the only option that gets
     you full production-fidelity ray counts *and* a fast demo.
   - **Fastest fix:** temporarily lower `samplesPerSrc`/`maxNumPathsPerSrc`
     for the demo run specifically (e.g. 1-2 million), and say so plainly
     in the demo narration -- "reduced ray budget for live demo purposes;
     production runs use 10M" is an honest, defensible caveat.
   - **No-code-change fix:** just reduce `numSamples` (fewer points across
     the pass) to keep the full 10M budget within a tolerable wait.
5. Pick one, and confirm the resulting wall-clock time before you're
   relying on it live.

**Checkpoint:** a `numSamples` + ray-budget combination with a *measured*
(not assumed) total runtime you're comfortable with for however you're
presenting this (live vs. pre-recorded).

---

## Step 4 — Smooth the trajectory (unchanged from v1)

You already have position *and* velocity at every dense sample from
`sample_pass_table`. Fit a `scipy.interpolate.CubicHermiteSpline`
per-axis in `export_ns3_waypoints.py` and upsample to a finer time grid
(e.g. 1 s) before writing the CSV, so `WaypointMobilityModel`'s linear
interpolation between consecutive CSV rows is smoothing an already-smooth
curve rather than chording a raw 5 s-spaced polyline. Re-derive
`vx,vy,vz` from the spline's derivative at the finer grid, don't just
repeat the original 5 s velocity values.

Also upsample the new `beam_angle_rad` column consistently (either by
including it in the same per-axis fit against range, or just recomputing
it directly from the upsampled x/y/z's implied range at each finer time
step -- cheaper and just as correct, since it's a closed-form function of
position, not something that itself needs fitting).

---

## Step 5 — Visualization (unchanged from v1)

Two-panel plot from the trace output: SNR vs. elevation, and a polar sky
plot (azimuth vs. `90-elevation`, colored by SNR) showing the live-fetched
satellite's actual track. Log `elev_deg`/`az_deg` alongside SNR in
`ComputeSnr()`'s trace output line so the plotting script doesn't have to
re-derive geometry. Mark blocked (`-inf`) samples distinctly rather than
dropping them.

---

## Steps 6/7 (optional, unchanged from v1)

Rendered scene hero shots (`IsImageRenderedEnabled`) and a Doppler readout
(`range_rate_km_s` is already sitting in the dense pass rows, unused --
`(FREQUENCY/c) * range_rate` is a one-line addition once it's threaded
through the CSV and trace output). Still not required for a working demo;
add if time allows.

## Not needed for the demo (unchanged from v1)

RX antenna orientation/tracking, material scattering-coefficient parity
with the offline pipeline, and Phase 4's anchor-cadence throttling are all
still out of scope for a standalone demo.

---

## Suggested order of operations

1. **Step 0** -- fetch the live TLE, generate the real pass. Do this first;
   everything downstream depends on knowing the actual satellite and pass
   geometry.
2. **Step 1** -- parameter fidelity (cheap, mechanical, do before any more
   test runs so you're not re-running with wrong config).
3. **Step 2** -- per-sample beam angle (the fix most likely to change
   observed behavior, not just match config on paper).
4. **Step 3** -- re-benchmark now that 1-2 are in place, pick sample count
   and ray budget against a measured runtime.
5. **Step 4** -- trajectory smoothing (independent, do anytime before the
   final run).
6. Final full run.
7. **Step 5** -- plot the result.
8. **Steps 6/7** -- optional polish if time allows.
