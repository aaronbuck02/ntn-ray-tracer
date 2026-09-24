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

**Known limitation:** `ray_tracing.py` hardcodes `dr.cuda.ad.Float` in the cone
call, so the path requires a CUDA Dr.Jit variant. On an LLVM/CPU build the patch
is present but broken, which introspection alone cannot detect — this is why
`sionna_patch_check` reports CUDA availability separately.

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
