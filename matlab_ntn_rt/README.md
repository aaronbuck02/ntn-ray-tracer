# matlab_ntn_rt — MATLAB equivalent of the Sionna NTN site-specific pipeline

A MATLAB implementation of the `rician_urban_sim` satellite-to-urban ray-tracing
pipeline, built for a MathWorks feature conversation. It is faithful to the
Sionna pipeline everywhere MATLAB can be, so that the places it cannot be are
concrete, measured, and reproducible in front of them.

Out of scope by design: ray-cone beamforming, the circular-polarisation
`AntennaPattern` subclass, and in-solver Faraday rotation. These are unreachable
through MATLAB's public API and appear here only as *measured gaps*.

---

## Verification status — read this first

| Component | Status |
|---|---|
| Scene export (PLY → STL + material map) | **Verified.** Sieg bbox, centre `[13.23, −1.27]` and `radius_xy = 230.5 m` match `sim_config.json` exactly. |
| Occupancy-grid algorithm | **Verified exactly.** The rasterisation reproduces Mitsuba's raycast grid cell for cell — `n_free` 59539/59539, `n_occupied` 24230/24230, occupied-cell set Jaccard **1.0000**. See `tools/validate_grid_algorithm.py`. |
| Ported ITU constants | **Audited.** All 57 P.676-12 coefficients and every weather/rain/TEC preset matched against the Python source (`tools/audit_constants.py`). The P.838-3 table is *generated*, not transcribed. |
| Ray-density argument | **Verified numerically** (Python, independent of MATLAB). |
| **All MATLAB code** | **NOT EXECUTED.** See below. |

**MATLAB could not be run in the environment where this was written.** R2026a
requires an interactive MathWorks account sign-in (`Please enter your MathWorks
Account email address`), which a non-interactive shell cannot answer, and the
R2021b install's license has expired (`License Manager Error -10`). Every `.m`
file here is therefore unrun. Two static bug classes were found and fixed by
review (`struct('f',[1 2 3])` building a struct *array*; `rows(end+1,:)` on a
`0×0` cell), but assume more remain.

**Your first action** is to sign in and run the self-test:

```matlab
cd matlab_ntn_rt; addpath(pwd); tools/selftest
```

It checks the toolboxes, the scene, the occupancy grid against the Python
reference, and all 1,744 frozen ITU reference values — in seconds, without a
ray trace.

---

## The argument, in three numbers

At the Sieg scene (radius 230.5 m) with a TX at the configured 500 km slant range:

1. **0.877 rays** land on the *entire scene* at MATLAB's finest permitted
   `AngularSeparation` (0.05°). Ray spacing there is **436 m** — wider than the
   461 m scene. Site-specific NTN is not reachable in SBR mode at all.
2. Sionna's containing cone half-angle for this geometry is **0.0264°** —
   *finer than MATLAB's finest permitted angular separation*. MATLAB's angular
   resolution limit is coarser than the entire angular extent of the problem.
3. Even with the proxy-TX workaround at 20 km, MATLAB puts **548 rays** on the
   scene against Sionna's 10⁷ in-cone — an **18,000×** ray-density deficit.

`ntnrt.gapReport(cfg)` recomputes all of this live from the installed release.

## Setup

```bash
# one-off, needs the sionna venv (numpy + scipy only)
cd tools
/path/to/sionna-venv/bin/python export_matlab_scene.py      # PLY -> STL + materials
/path/to/sionna-venv/bin/python export_matlab_inputs.py     # UEs + baseline sweeps
/path/to/sionna-venv/bin/python gen_golden.py               # frozen ITU references
```

Already run; outputs are in `data/`.

## Layout

```
ntn_rt_demo.m          driver + presentation narrative (open in Live Editor, save as .mlx)
+ntnrt/
  loadConfig.m         reads the SAME sim_config.json; collects unhonourable keys
  gapReport.m          gap table, queried live from the installed release
  runElevation.m       one elevation end to end (mirrors run_elevation_sim.py)
  +scene/  buildSceneModel, loadMaterialMap
  +geom/   occupancyGrid, sampleRxPositions, importRxPositions
  +orbit/  slantRange, passWaypoints, compassToSceneAzimuth, rotateEnuToScene
  +atmos/  P.676-12 / P.838-3 / P.618-13 port, p838Table (generated), p618Compare
  +iono/   P.531-15 port (MATLAB ships no ionospheric model at all)
  +mat/    materialFromParams, buildFaceIndex, faceAtPoint, perInteractionMaterials
  +tx/     txFromElevation, beamHalfAngle, proxyRangeAndExcessFspl
  +rt/     runRayTrace, runRayTraceProxy, runRayTraceImage
  +chan/   complexChannel, splitByInteractionType, riceanK, spreads, Doppler
  +io/     saveResults, loadResults, importSionnaResults
  +plot/   palette, compareCdfs, mechanismSplit, metricsVsElevation, costTable
tools/     exporters, golden-value generator, selftest.m, algorithm validators
data/      generated STL, material maps, UE sets, baselines, golden values
```

Conventions: one-line function headers; comments only where the reason is
non-obvious; every ported function carries a provenance line naming its Python
original; every MATLAB-imposed compromise carries a `% GAP:` marker, so
`grep -rn "% GAP:" +ntnrt` regenerates the gap list from the source itself.

## Still to do

- **The ablation — do not present without it.** Re-run Sionna with
  `DIFFUSE_REFLECTION=False, MAX_DEPTH=2`, export it, and pass it as
  `AblatedTag` to `ntnrt.plot.compareCdfs`. If MATLAB converges onto that curve,
  the gap is *proven* to be the missing features rather than an implementation
  bug. This is the difference between a complaint and a feature request.
- Full sweeps: 9 elevations × 10 000 UEs, both scenes, both strategies.
- Wharf DU stress case (310 shapes): record whether the image method completes —
  "it did not finish" is itself a result.
- `+geom/sampleRxPositions` cannot reproduce numpy's `Generator` draw, so the
  headline comparison uses `importRxPositions`. The sampler is there so the
  module stands alone, and is validated against the imported set, not bit-matched.
