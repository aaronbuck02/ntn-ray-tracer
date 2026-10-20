# Sionna RT patches

Why this project cannot run on a stock `sionna-rt 1.2.1`, and what the three
patches do. Operational instructions — layout, checksums, how to re-apply —
live in [`../sionna_patch/README.md`](../sionna_patch/README.md).

| Patch | Where | Mechanism |
|---|---|---|
| `epsilon_guard` | `rt/utils/ray_tracing.py` | patched source |
| `beamforming` | `ray_tracing.py`, `path_solver.py`, `sb_candidate_generator.py`, `field_calculator.py` | patched source |
| ITU frequency range | `src/utils/itu_range_patch.py` | runtime, mutates a dict |

The first two are archived under `sionna_patch/` and must be copied into
site-packages by hand. `sionna_patch_check.require()` asserts they are present
and is called from `src/utils/scene_utils.py` `build_scene_batch`, the chokepoint every
path-solving entry point passes through.

---

## `epsilon_guard` — the occlusion-ray dead zone

### The defect

`sionna/rt/utils/ray_tracing.py`, inside `spawn_ray_to`:

```python
maxt *= (1. - EPSILON_FLOAT)        # EPSILON_FLOAT = 100 * 2**-24 = 5.96e-06
```

Every occlusion ray is shortened by a **relative** epsilon. As a
self-intersection guard measured in units of float32 precision this is well
designed — it holds the guard at a constant ~50–90 ULPs whatever the ray length
— and at terrestrial range it is sub-millimetre and invisible. Over a LEO link
it is not:

| Slant range | Elevation | Guard |
|---|---|---|
| 1816 km | 10° | 10.82 m |
| 558 km | 80° | 3.32 m |

`sb_candidate_generator.py` spawns the direct-path test as
`spawn_ray_to(origins=TX, targets=RX)`, so the shortening lands at the
**receiver**. Any occluder nearer than the guard falls outside the ray's search
interval, `ray_test` returns "clear", and a blocked link is reported as LoS.

### Measured impact

Quantified on a 2 km-radius generated wall ensemble with no exclusion zone about
the terminal, so near-terminal blockers are common: **83 of 800 scenes were
falsely reported LoS at 10° elevation**, and in every case the nearest blocker
lay inside the predicted dead zone, the boundary matching prediction to about
1%. Consequences: LoS probability ~0.09 too high, and K-factor inflated by
1.16 dB, because the falsely-included scenes are heavily shadowed and carry
~8× less non-direct power than genuinely clear ones.

The size of the error depends on how much geometry sits within the dead zone of
the receiver, so it is worst in dense scenes at low elevation.

### The fix

Cap the guard's absolute size at the point of use:

```python
maxt = dr.maximum(maxt - dr.minimum(maxt * EPSILON_FLOAT, MAX_RAY_GUARD_M), 0.)
```

`EPSILON_FLOAT` itself is left alone: it is overloaded, serving also as a length
tolerance in metres (`utils/wedges.py`) and as a dimensionless cosine tolerance
(`path_solvers/image_method.py`).

**Nor would shrinking it work.** The guard would still be relative. Bringing the
10.82 m dead zone under a centimetre at 1816 km needs `EPSILON_FLOAT < 5.5e-9`,
below float32 machine epsilon — so `maxt * (1 - eps)` would round to `maxt` at
*every* range and the guard would vanish from the short segments where it does
real work. No single relative value is right at both 1 m and 1816 km. That is
the defect.

With `MAX_RAY_GUARD_M = 1e-3` the cap binds only above 168 m of ray length:

| Ray length | Guard before | Guard after | float32 ULP |
|---|---|---|---|
| 1 km | 6.0 mm | 1.0 mm | 0.1 mm |
| 10 km | 60 mm | 1.0 mm | 1.0 mm |
| 558 km | 3.31 m | below ULP | 62.5 mm |
| 1816 km | 10.88 m | below ULP | 125.0 mm |

Over a long free-space segment the subtraction falls beneath the ULP, leaving
`maxt` at full distance — correct, since neither endpoint is on a surface and
there is nothing to self-intersect. The dead zone drops from 10.88 m to half an
ULP, 62.5 mm.

### What this does not change

- **Path length, delay and power.** Those come from the vertex geometry in
  `field_calculator.py` (`tau = path_length / c`); `maxt` only limits how far a
  visibility or vertex-finding ray may travel.
- **Bounce-segment occlusion.** `image_method.py` calls `spawn_ray_to` with the
  origin on the surface and the target the source or its image, so the
  shortening lands ~1816 km away in empty space. The surface end is the origin,
  protected by `offset_p` and untouched by `maxt`.

The direct-path test is therefore the only place in an NTN configuration where a
long segment terminates in dense geometry, which is why the defect shows up
purely as false LoS.

### Verification

`python -m Ray_Tracing.src.sionna_patch_check --selftest` ray-tests 360 azimuths per elevation
against the scene mesh, with the full-length ray as ground truth:

```
beta=  10 deg  slant=  1816 km  dead zone  10.82 m -> 1.0 mm | 360 directions: wrong before  12, after   0  OK
beta=  40 deg  slant=   812 km  dead zone   4.84 m -> 1.0 mm | 360 directions: wrong before  25, after   0  OK
beta=  80 deg  slant=   558 km  dead zone   3.32 m -> 1.0 mm | 360 directions: wrong before   0, after   0  OK
```

**Patched and unpatched results are not comparable.** Runs before this fix
overstate LoS probability and K-factor at low elevation.

### History

This began as `sionna_shadow_patch.py`, a runtime monkey-patch that rebound
`spawn_ray_to` across four modules (consumers had done
`from sionna.rt.utils import spawn_ray_to`, so patching the defining module
alone missed them). It is now folded into the patched source, which removes that
import-order hazard. The module was retired in favour of
`src/sionna_patch_check.py`, which asserts rather than applies.

---

## `beamforming` — ray-cone launching

An isotropic launch from 550 km puts essentially no rays on a 2 km scene, and
Mitsuba caps the ray count at `2**32 - 1`, so the density cannot be recovered by
launching more. The patch confines the ray budget to a cone of half-angle
ψ about boresight, and rescales the ray-tube solid angle to match:

```
Omega_cone = 2*pi*(1 - cos(psi))          initial_solid_angle = Omega_cone / samples_per_src
```

against `4*pi / samples_per_src` for the isotropic case. Without the rescale the
cone would carry the wrong total transmit power.

Callers set `psi = arctan(SCENE_RADIUS / d)` per trajectory sample — recomputed
every sample, not fixed per run, because holding ψ constant under a varying
slant range under-illuminates the scene at low elevation and inflates the
K-factor exactly where the model is most stressed.

The four files and their roles are tabulated in
[`../sionna_patch/README.md`](../sionna_patch/README.md).

The original cone call hardcoded `dr.cuda.ad.Float`. `beam_cone_precision`
(below) replaced that call, so the sampler is no longer CUDA-only. The pipeline is
still validated on CUDA only.

---

## `beam_cone_precision` — exact cone width at long slant ranges

### The defect

The `beamforming` patch sampled the cone with
`mi.warp.square_to_uniform_cone(samples, cos(psi))`, with `cos(psi)` evaluated in
float32. A beam sized to a few-hundred-metre scene from LEO is ψ ≈ 1e-4 rad, so
cos ψ = 1 − ψ²/2 lies within one float32 step (5.96e-8) of 1. Mitsuba's first
operation is `1 - cos_cutoff`, and that difference was lost to rounding:

| slant / scene radius | float32 cos ψ | effect |
|---|---|---|
| < ~1360 | several steps below 1 | cone correct to within a few % |
| ~1360 – 4096 | snapped to the nearest step | cone the wrong size, e.g. 464 m instead of 354 m at 1345 km |
| > 4096 (2¹²) | exactly 1.0 | **cone collapses**: every ray hits the scene centre |

The field calculator meanwhile computed the solid angle in float64 from the true
ψ. In the middle regime, ray weights and actual coverage disagree, and diffuse
power is biased by 10·log₁₀(ψ²/ψ_eff²). The bias is positive when the cone snaps
narrow and negative when it snaps wide. The model predicts the measured bias per
scene to within ~0.05 dB.

Measured on the Phase 0 sensitivity run of 2026-10-13 (500 m PPP scenes, real
Starlink passes):
- **10°:** 88% of scenes collapsed. Per-UE mean K was 38 dB instead of ~6 dB, and
  64% of LoS UEs had no ground-interacting power.
- **20–80°:** diffuse power biased by −2.4 to +1.75 dB per scene. Only 0–63% of
  scenes per elevation were within ±0.2 dB.
- **Older runs:** sieg at a fixed 500 km (slant/radius 2169) carries about
  −0.5 dB of diffuse bias.

### The fix

Never form cos ψ. `beam_cone_t_max(psi)` returns 1 − cos ψ as 2 sin²(ψ/2), which
float32 holds to full relative precision. `square_to_uniform_cone_t(sample, t_max)`
is Mitsuba's own concentric-disk cone mapping with that value substituted for
`1 - cos_cutoff`. On the GPU, Mitsuba's mapping matches this formula
bit-for-bit, so the lattice stratification is unchanged:

```
p = square_to_uniform_disk_concentric(sample);  t = t_max * |p|^2
direction = ( p * sqrt(t_max * (2 - t)),  1 - t )        # +z boresight, then rotated to +x
```

The field calculator's solid angle becomes `2*pi*beam_cone_t_max(psi)`, the same
number the sampler uses.

### Verification (2026-10-14)

- **Coverage:** at slant/radius 500 to 10,000, rays fill the scene disc with exactly
  25.0% inside R/2 and a maximum radius of R. Nothing collapses.
- **Regression where the old cone was fine** (slant/radius 707): per-UE NLoS power
  changes by a median 0.03 dB, against 0.14 dB between solver seeds.
- **Collapsed 10° scenes:** mean K 40.8 → 5.6 dB; missing-ground fraction 0.64 → 0.
- **Energy:** doubling ψ changes ensemble NLoS power by −0.02 dB (seed noise
  0.01 dB), although only a quarter of the paths reach the scene.

`sionna_patch_check` gates on this patch. It requires `beam_cone_precision` in
`__RUS_PATCH__` and a `beam_cone_t_max` accurate to 1e-6 at slant/radius 5000, and
rejects a sampler that still calls `square_to_uniform_cone`. Its manifest
comparison also works again: `ARCHIVE` had pointed at `src/sionna_patch/`.

---

## ITU material frequency ranges

`itu_range_patch.widen()` rewrites the `(lo, hi)` frequency key of
`very_dry_ground`, `medium_dry_ground` and `wet_ground` in
`ITU_MATERIALS_PROPERTIES` from 1–10 GHz to 1–100 GHz, so 11 and 20 GHz
campaigns will load. It is a runtime dict mutation, idempotent, applied from
`src/utils/scene_utils.py` `build_scene_batch` before `scene.frequency` is set — the frequency
setter fires each material's callback and raises for out-of-band materials, so
ordering matters.

This **extrapolates** the ITU-R P.2040 `a*f^b` and `c*f^d` laws past their
validated band: 1.1× at 11 GHz, 2× at 20 GHz. Building facades and roofs are
unaffected — concrete and metal are characterised to 100 GHz — so the
extrapolation bears on the ground-reflected contribution specifically. It is
not archived under `sionna_patch/` because it changes no Sionna source.

---

## Known upstream defect — receive-pattern direction (not yet patched)

In `rt/path_solvers/field_calculator.py::_compute_cir`, 1.2.1 updates
`ki_world = dr.select(active, ko_world, ki_world)` *after* `active` has been
cleared on a path's last interaction, so `ki_world` is never advanced to the
final segment. The receive pattern is then evaluated at `-ki_world`: the
reversed **incidence direction onto the last interaction**, not the arrival
direction at the RX. sionna-rt 2.1.0 has the two lines in the other order.

For a satellite every first segment is parallel to the LoS, so under 1.2.1
each single-bounce NLoS path gets the LoS direction's pattern gain. Measured on
a PPP smoke scene (2 GHz, 45°, RHCP): a zenith-pointing `tr38901` UE gives
mean K 11.5 dB with the defect and 27.8 dB with the reorder (ground-path power
−21 dB). `iso` patterns, CP and linear, are unchanged to ≤3e-5 relative, so
existing iso-antenna results stand. The fix is the one-line reorder; it is not
in the archive yet, and `sionna_patch_check` does not test for it.
